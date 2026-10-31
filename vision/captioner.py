"""Per-face speech captions via Gemini generateContent (chunked, utterance by utterance).

The audio thread calls feed(face_id, block, speaking, t) every hop with the focused beam
for that face. Segmentation happens inline (cheap); finished utterances are transcribed
on a thread pool. Models are tried in order; a model that answers 503/429/timeout is
marked busy and skipped for BUSY_S seconds.

The API key is read from the GOOGLE_API_KEY environment variable or, failing that, from
%LOCALAPPDATA%/Ika/simulator/.env (the ika-simulator config). It is never logged.
"""
from __future__ import annotations

import base64
import collections
import io
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FS = 48000
PREROLL_S = 0.4
HANGOVER_S = 0.7
MAX_UTTERANCE_S = 10.0
MIN_SPEECH_S = 0.4
REQUEST_TIMEOUT_S = 20.0
BUSY_S = 60.0
TRANSIENT = {429, 500, 502, 503, 504}
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = ("Transcribe the speech in this audio verbatim, in the language spoken{hint}. "
          "The speaker may be quiet, whispering or partly masked by background noise or other voices; "
          "transcribe only the main, closest voice. Output only the spoken words, with no labels, "
          "timestamps, quotes or commentary. If there is no intelligible speech, output nothing.")
_NOISE = re.compile(r"^\s*[\[(<]?\s*(no speech|silence|inaudible|no intelligible speech|"
                    r"unintelligible|music|noise|empty)\s*[\])>]?\s*\.?\s*$", re.I)
_TAGS = re.compile(r"[\[(<](no speech|silence|inaudible|unintelligible|noise|music)[\])>]", re.I)


class TransportError(Exception):
    def __init__(self, code: int | None, message: str):
        super().__init__(message)
        self.code = code            # HTTP status, or None for timeouts / network errors

    @property
    def transient(self) -> bool:
        return self.code is None or self.code in TRANSIENT


# transport(model, wav_bytes, prompt, timeout) -> text ; raises TransportError
Transport = Callable[[str, bytes, str, float], str]


@dataclass
class Caption:
    face_id: int
    text: str            # "" while pending
    t_start: float       # monotonic seconds (from the t passed to feed)
    t_end: float
    pending: bool
    model: str = ""      # model that produced it
    error: str = ""      # non-empty if all models failed


def load_api_key(env_file: Path | None = None) -> str:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    if env_file is None:
        env_file = Path(os.getenv("LOCALAPPDATA", Path.home())) / "Ika" / "simulator" / ".env"
    try:
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GOOGLE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def http_transport(api_key: str) -> Transport:
    def send(model: str, wav: bytes, prompt: str, timeout: float) -> str:
        body = {"contents": [{"parts": [
            {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav).decode()}},
            {"text": prompt}]}],
            "generationConfig": {"temperature": 0}}
        req = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode(),
                                     headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", {}).get("message", "")
            except Exception:
                msg = ""
            raise TransportError(e.code, f"HTTP {e.code} {msg[:120]}".strip()) from None
        except (socket.timeout, TimeoutError):
            raise TransportError(None, "timeout") from None
        except (urllib.error.URLError, OSError) as e:
            raise TransportError(None, f"network: {getattr(e, 'reason', e)}") from None
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise TransportError(400, f"empty response ({reason})")
        parts = (cands[0].get("content") or {}).get("parts")
        if parts is None:   # e.g. gemini-3.5-transcribe answers STOP with no content: not a usable result
            raise TransportError(204, f"no content (finish {cands[0].get('finishReason')})")
        return "".join(p.get("text", "") for p in parts)
    return send


def clean_text(text: str) -> str:
    text = (text or "").strip().strip('"').strip()
    if _NOISE.match(text):
        return ""
    return re.sub(r"\s{2,}", " ", _TAGS.sub("", text)).strip()


def encode_wav(audio48: np.ndarray) -> bytes:
    """48 kHz float -> 16 kHz 16-bit WAV, normalized to ~-20 dBFS RMS without clipping."""
    y = resample_poly(np.asarray(audio48, np.float64), 1, 3)
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if rms > 1e-9:
        gain = min(0.1 / rms, 0.97 / max(peak, 1e-9))
        y = y * gain
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), 16000, "PCM_16", format="WAV")
    return buf.getvalue()


class _Segment:
    __slots__ = ("pre", "blocks", "active", "speech", "silence", "t_start", "t_last")

    def __init__(self, preroll_blocks: int):
        self.pre = collections.deque(maxlen=preroll_blocks)
        self.blocks: list[np.ndarray] = []
        self.active = False
        self.speech = 0           # samples flagged speaking in the open utterance
        self.silence = 0          # consecutive non-speaking samples
        self.t_start = 0.0
        self.t_last = 0.0


class Captioner:
    def __init__(self, model: str = "gemini-3.8-flash",
                 fallbacks=("gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-transcribe"),
                 env_file: Path | None = None,
                 language_hint: str = "", max_workers: int = 3,
                 transport: Transport | None = None, retry_rounds: int = 1,
                 retry_backoff_s: tuple[float, ...] = (20.0, 40.0, 60.0)):
        self.models = [model] + [m for m in fallbacks if m != model]
        self.prompt = PROMPT.format(hint=f" (probably {language_hint})" if language_hint else "")
        self.retry_rounds = max(1, retry_rounds)
        self.retry_backoff_s = retry_backoff_s
        self._lock = threading.Lock()
        self._captions: list[Caption] = []
        self._segments: dict[int, _Segment] = {}
        self._busy_until: dict[str, float] = {}
        self._now = 0.0
        self._pre_blocks = max(1, round(PREROLL_S * FS / 512))
        self._closed = False
        if transport is None:
            key = load_api_key(env_file)
            transport = http_transport(key) if key else None
        self._transport = transport
        self.enabled = transport is not None
        self.status = f"ready ({self.models[0]})" if self.enabled else "no API key (GOOGLE_API_KEY)"
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="caption") if self.enabled else None

    # ---- audio thread -------------------------------------------------------------
    def feed(self, face_id: int, block: np.ndarray, speaking: bool, t: float) -> None:
        if not self.enabled or self._closed:
            return
        try:
            self._now = t
            seg = self._segments.get(face_id)
            if seg is None:
                seg = self._segments[face_id] = _Segment(self._pre_blocks)
            n = len(block)
            if not seg.active:
                if speaking:
                    seg.active = True
                    seg.blocks = list(seg.pre)
                    seg.t_start = t - len(seg.blocks) * n / FS
                    seg.speech = seg.silence = 0
                    seg.pre.clear()
                else:
                    seg.pre.append(np.array(block, np.float32, copy=True))
                    return
            seg.blocks.append(np.array(block, np.float32, copy=True))
            seg.t_last = t + n / FS
            if speaking:
                seg.speech += n
                seg.silence = 0
            else:
                seg.silence += n
            length = len(seg.blocks) * n / FS
            if seg.silence >= HANGOVER_S * FS:
                self._close(face_id, seg, trim=seg.silence)
            elif length >= MAX_UTTERANCE_S:
                self._close(face_id, seg, trim=0)
                if speaking:           # keep going without a gap
                    seg.active, seg.blocks, seg.t_start = True, [], seg.t_last
                    seg.speech = seg.silence = 0
        except Exception as e:         # never raise into the audio thread
            self.status = f"feed error: {e}"

    def _close(self, face_id: int, seg: _Segment, trim: int):
        blocks, speech = seg.blocks, seg.speech
        t_start, t_end = seg.t_start, seg.t_last - trim / FS
        seg.active, seg.blocks, seg.speech, seg.silence = False, [], 0, 0
        if speech < MIN_SPEECH_S * FS:
            return
        # keep ~0.25 s of the trailing silence so the last word is not cut
        keep = max(0, trim - int(0.25 * FS))
        audio = np.concatenate(blocks)
        if keep:
            audio = audio[:-keep]
        cap = Caption(face_id, "", t_start, t_end, True)
        with self._lock:
            self._captions.append(cap)
            if len(self._captions) > 500:
                self._captions = [c for c in self._captions if c.pending] + \
                    [c for c in self._captions if not c.pending][-300:]
        self._pool.submit(self._transcribe, cap, audio)

    # ---- workers -----------------------------------------------------------------
    def _order(self) -> list[str]:
        now = time.monotonic()
        free = [m for m in self.models if self._busy_until.get(m, 0) <= now]
        if free:
            return free
        return sorted(self.models, key=lambda m: self._busy_until.get(m, 0))[:1]

    def _transcribe(self, cap: Caption, audio: np.ndarray):
        try:
            wav = encode_wav(audio)
        except Exception as e:
            cap.error, cap.pending = f"encode: {e}", False
            return
        errors, empty = [], []
        for round_ in range(self.retry_rounds):
            if self._closed:
                break
            if empty:   # some model answered "nothing here" and none produced text: accept silence
                cap.text, cap.model, cap.pending = "", empty[0], False
                return
            if round_:
                time.sleep(self.retry_backoff_s[min(round_ - 1, len(self.retry_backoff_s) - 1)])
            for model in self._order():
                try:
                    text = self._transport(model, wav, self.prompt, REQUEST_TIMEOUT_S)
                except TransportError as e:
                    if e.code == 204:          # answered without content: ask the next model
                        empty.append(model)
                        continue
                    errors.append(f"{model}: {e}")
                    if e.transient:
                        self._busy_until[model] = time.monotonic() + BUSY_S
                    else:
                        self._busy_until[model] = time.monotonic() + 10 * BUSY_S
                    continue
                except Exception as e:
                    errors.append(f"{model}: {e}")
                    continue
                cap.text, cap.model, cap.pending = clean_text(text), model, False
                if model == self.models[0]:
                    self.status = f"ready ({model})"
                else:
                    self.status = f"{_short(self.models[0])} busy -> using {_short(model)}"
                return
        if empty:
            cap.text, cap.model, cap.pending = "", empty[0], False
            return
        cap.error, cap.pending = ("; ".join(errors[-3:]) or "failed"), False
        self.status = f"captions failing: {errors[-1] if errors else 'unknown'}"

    # ---- readers -----------------------------------------------------------------
    def captions(self, face_id: int | None = None, max_age_s: float = 30.0) -> list[Caption]:
        cutoff = self._now - max_age_s
        with self._lock:
            out = [c for c in self._captions
                   if (face_id is None or c.face_id == face_id) and (c.pending or c.t_end >= cutoff)]
        return sorted(out, key=lambda c: c.t_start)

    def all_captions(self) -> list[Caption]:
        with self._lock:
            return sorted(self._captions, key=lambda c: c.t_start)

    def flush(self, t: float | None = None) -> None:
        """Close any open utterance (end of a recording / app shutdown)."""
        if not self.enabled:
            return
        for fid, seg in list(self._segments.items()):
            if seg.active:
                self._close(fid, seg, trim=seg.silence)

    def close(self) -> None:
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)


def _short(model: str) -> str:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", model)
    return m.group(1) if m else model
