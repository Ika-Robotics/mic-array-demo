"""Ika focus demo: find me, beam the 16-mic array at my mouth, compare with the general mix.

Run "Start Beam Demo.cmd" from File Explorer with earbuds/headphones plugged in
(laptop speakers would feed back into the array). Close the wizard, MicArrayFocus and
the other demos first: the camera and the ASIO driver serve one app at a time.

Keys (click the video window first; press h to show/hide this list on screen):
  click a face   listen to that person (focused)
  e / E          enroll the selected face as ME / forget me (ME is then followed automatically)
  space          general <-> focused     a   auto A/B every 3 s
  c              quick calibration: start, then click/whistle at your mouth while moving
                 your head; c again saves when good (Esc cancels)
  r              record start/stop (raw 16 ch + faces); playback is prepared on stop
  p              play / stop the last recording; space switches general/focused while playing
  b              beam type das / mvdr / lcmv      n   noise filter on/off
  t              captions on/off (Gemini)           , .  volume
  x y            flip array direction (only when not calibrated)
  Esc/q          quit
"""
from __future__ import annotations

import os

os.environ.setdefault("SD_ENABLE_ASIO", "1")

import argparse
import collections
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

from beamformer import (CH, FS, HOP, KINDS, Beamformer, PostFilter, SrpMap, StreamProcessor, level_db,
                        mix_weights)
from calibration import CalibrationSession, EventDetector
from face_tracker import FaceTracker, Settings
from identity import Identity
from pair_render import render_pair, save_pair
from vision_demo import Camera, draw, find_camera

try:
    from captioner import Captioner
except ImportError:          # captions are optional
    Captioner = None

HERE = Path(__file__).parent
AB_PERIOD_S = 3.0
WINDOW = "Ika focus"
DEFAULT_GAIN_DB = 35.0       # raw speech at 1 m is only about -55 dBFS on this array


def pick_devices(in_hint: str, out_hint: str | None):
    devices = sd.query_devices()
    api = lambda d: sd.query_hostapis(d["hostapi"])["name"]
    ins = [i for i, d in enumerate(devices) if d["max_input_channels"] >= CH and in_hint.lower() in d["name"].lower()]
    if not ins:
        raise RuntimeError(f"No 16-channel input matching '{in_hint}'. Is the UMA-16 plugged in?")
    outs = [(i, d) for i, d in enumerate(devices) if d["max_output_channels"] >= 2
            and api(d) == "Windows WASAPI" and not any(k in d["name"] for k in ("MCHStreamer", "UMA16"))]
    if out_hint is not None:
        if out_hint.isdigit():
            return ins[0], int(out_hint)
        outs = [(i, d) for i, d in outs if out_hint.lower() in d["name"].lower()]
    if not outs:
        raise RuntimeError("No usable headphone/speaker output found. Use --output.")
    rank = lambda d: next((k for k, w in enumerate(("earbud", "headphone", "headset")) if w in d["name"].lower()), 9)
    return ins[0], min(outs, key=lambda x: rank(x[1]))[0]


class Fifo:
    """Mono sample FIFO between the processing thread and the output callback."""
    PRIME = 2 * HOP

    def __init__(self):
        self.buf = collections.deque()
        self.size = 0
        self.lock = threading.Lock()
        self.underruns = 0
        self.primed = False            # play silence until PRIME samples are buffered

    def push(self, x: np.ndarray, max_size: int):
        with self.lock:
            self.buf.append(x)
            self.size += len(x)
            while self.size > max_size:            # clock drift / hiccup: drop oldest, keep latency bounded
                self.size -= len(self.buf.popleft())

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        with self.lock:
            if not self.primed:
                if self.size < self.PRIME + n:
                    return out
                self.primed = True
            k = 0
            while k < n and self.buf:
                head = self.buf[0]
                take = min(n - k, len(head))
                out[k:k + take] = head[:take]
                k += take
                if take == len(head):
                    self.buf.popleft()
                else:
                    self.buf[0] = head[take:]
                self.size -= take
            if k < n:
                self.underruns += 1
                self.primed = False    # rebuild a small cushion instead of stuttering
        return out


class Player:
    """Plays a rendered recording; general/focused switch keeps the playback position."""

    def __init__(self, pair: dict):
        self.pair, self.pos, self.which, self.playing = pair, 0, "focused", False
        self.length = len(pair["general"])

    def pull(self, n: int, gain: float) -> np.ndarray:
        out = np.zeros(n, np.float32)
        if not self.playing:
            return out
        chunk = self.pair[self.which][self.pos:self.pos + n]
        out[:len(chunk)] = chunk * gain
        self.pos += n
        if self.pos >= self.length:
            self.playing, self.pos = False, 0
        return np.tanh(out)


class AudioEngine:
    def __init__(self, in_dev: int, out_dev: int):
        self.bf = Beamformer()
        self.proc_out = StreamProcessor(self.bf)
        self.srp = SrpMap(self.bf)
        self.calib = EventDetector(self.bf)
        self.inq: queue.Queue = queue.Queue(maxsize=400)
        self.fifo = Fifo()
        self.lock = threading.Lock()
        # Shared state (main thread writes, processing thread reads)
        self.faces: dict[int, dict] = {}           # id -> {"u": unit vector, "speaking": bool}
        self.mode = "general"                      # "general" | "focused"
        self.kind = "mvdr"
        self.postfilter = True
        self.target_id: int | None = None
        self.target_u = None                       # last known direction, held if the face drops out
        self.gain_db = DEFAULT_GAIN_DB
        self.ab_since: float | None = None
        self.calibrating = False
        self.player: Player | None = None
        self.captioner = None
        # Telemetry (processing thread writes)
        self.general_db = -120.0
        self.face_db: dict[int, float] = {}
        self.listen_db = -120.0
        self.peak = None                           # ((sx, sy), contrast)
        self.overruns = 0
        self.samples = 0
        self.recorder: sf.SoundFile | None = None
        self.running = True
        self._post_out = PostFilter()
        self._face_streams: dict[int, StreamProcessor] = {}

        self.inp = sd.InputStream(device=in_dev, channels=CH, samplerate=FS, blocksize=HOP,
                                  dtype="float32", callback=self._on_input)
        self.out = sd.OutputStream(device=out_dev, channels=2, samplerate=FS, blocksize=HOP,
                                   dtype="float32", callback=self._on_output, latency="low")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.in_name = sd.query_devices(in_dev)["name"]
        self.out_name = sd.query_devices(out_dev)["name"]

    def start(self):
        self.thread.start()
        self.inp.start()
        self.out.start()

    def stop(self):
        self.running = False
        for s in (self.inp, self.out):
            try:
                s.stop(); s.close()
            except Exception:
                pass
        self.thread.join(timeout=1)
        self.set_recording(None)

    def _on_input(self, indata, frames, t, status):
        if status.input_overflow:
            self.overruns += 1
        try:
            self.inq.put_nowait((indata.copy(), time.monotonic() - frames / FS))
        except queue.Full:
            self.overruns += 1

    def _on_output(self, outdata, frames, t, status):
        live = self.fifo.pull(frames)
        player = self.player
        mono = player.pull(frames, 10 ** ((self.gain_db - DEFAULT_GAIN_DB) / 20)) if player and player.playing else live
        outdata[:, 0] = mono
        outdata[:, 1] = mono

    def listening(self) -> tuple[str, int | None]:
        """Effective (mode, face id), taking auto A/B into account."""
        with self.lock:
            mode, target, since = self.mode, self.target_id, self.ab_since
        if since is not None:
            mode = "focused" if int((time.monotonic() - since) / AB_PERIOD_S) % 2 == 0 else "general"
        return mode, target

    def _run(self):
        hop = 0
        smooth = lambda old, new: 0.8 * old + 0.2 * new
        wmix = mix_weights()
        while self.running:
            try:
                block, t_block = self.inq.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.recorder is not None:
                self.recorder.write(block)
            spectra = self.proc_out.spectra(block)
            with self.lock:
                faces, kind, post_on = dict(self.faces), self.kind, self.postfilter
                captioner = self.captioner
            mode, target = self.listening()

            # Per-face focused streams: meters + captions
            weights = {}
            for fid, info in faces.items():
                others = [v["u"] for k, v in faces.items() if k != fid]
                weights[fid] = self.bf.weights(kind, info["u"], others)
                if captioner is not None:
                    # Captions get the beam WITHOUT the noise filter: the filter sounds cleaner
                    # to people but removed words for the speech model on real recordings.
                    proc = self._face_streams.setdefault(fid, StreamProcessor(self.bf))
                    y = proc.render(spectra, weights[fid], 0)
                    captioner.feed(fid, y.astype(np.float32), bool(info["speaking"]), t_block)
                else:
                    y = np.fft.irfft(Beamformer.apply(weights[fid], spectra))
                self.face_db[fid] = smooth(self.face_db.get(fid, level_db(y)), level_db(y))
            for fid in list(self.face_db):
                if fid not in faces:
                    del self.face_db[fid]
                    self._face_streams.pop(fid, None)
            self.general_db = smooth(self.general_db, level_db(block.mean(axis=1)))

            if target is not None and target in faces:
                self.target_u = faces[target]["u"]
            if mode == "focused" and self.target_u is not None:
                others = [v["u"] for k, v in faces.items() if k != target]
                w = weights.get(target) if target in faces else self.bf.weights(kind, self.target_u, others)
                out = self.proc_out.render(spectra, w, (kind, target, post_on), self._post_out if post_on else None)
            else:
                out = self.proc_out.render(spectra, wmix, "general")
            self.listen_db = smooth(self.listen_db, level_db(out))
            out = np.tanh(out * 10 ** (self.gain_db / 20)).astype(np.float32)   # soft limiter protects ears
            self.fifo.push(out, max_size=8 * HOP)

            if self.calibrating:
                self.calib.feed(spectra, t_block)
            if hop % 4 == 0:
                self.srp.add(spectra)
            if hop % 16 == 0:
                self.peak = self.srp.take()
            hop += 1
            self.samples += len(block)

    def set_recording(self, folder: Path | None):
        old, self.recorder = self.recorder, None
        if old is not None:
            time.sleep(0.05)
            old.close()
        if folder is not None:
            self.recorder = sf.SoundFile(str(folder / "mics.wav"), "w", FS, CH, "PCM_24")


class Recording:
    def __init__(self, engine: AudioEngine, tracker: FaceTracker, camera_name: str):
        self.folder = HERE / "recordings" / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.folder.mkdir(parents=True)
        (self.folder / "info.json").write_text(json.dumps({
            "sample_rate": FS, "channels": CH, "general": "mean of 16 mics",
            "input": engine.in_name, "camera": camera_name,
            "direction_source": tracker.mapper.status, "camera_hfov_deg": tracker.settings.camera_hfov_deg,
            "flip_x": tracker.mapper.flip_x, "flip_y": tracker.mapper.flip_y}, indent=1))
        self.log = open(self.folder / "faces.jsonl", "w", encoding="utf-8")
        self.start_sample = engine.samples
        engine.set_recording(self.folder)
        self.engine = engine

    def write(self, faces, me_id):
        mode, target = self.engine.listening()
        self.log.write(json.dumps({
            "sample": self.engine.samples - self.start_sample, "listening": mode, "target": target,
            "me": me_id, "kind": self.engine.kind, "postfilter": self.engine.postfilter, "faces": [{
                "id": f.id, "u": [round(v, 4) for v in f.direction.unit_vector],
                "mouth": [round(v, 4) for v in f.mouth],
                "mouth_moving": f.mouth_moving, "looking": f.looking_at_camera} for f in faces]}) + "\n")

    def close(self):
        self.engine.set_recording(None)
        self.log.close()


class App:
    def __init__(self, args):
        in_dev, out_dev = pick_devices(args.input, args.output)
        self.tracker = FaceTracker(Settings(max_faces=3, camera_hfov_deg=args.hfov))
        self.identity = Identity()
        self.engine = AudioEngine(in_dev, out_dev)
        source, self.cam_name = find_camera(args.camera_name, args.camera_index)
        self.cam = Camera(source, 1280, 720)
        self.faces, self.size = [], (1280, 720)
        self.history: collections.deque = collections.deque(maxlen=90)   # (t, {id: mouth}) for calibration
        self.selected: int | None = None
        self.follow_me = True
        self.recording: Recording | None = None
        self.last_folder: Path | None = None
        self.calib: CalibrationSession | None = None
        self.show_help = False
        self.captions_on = False
        self.notice, self.notice_until = "", 0.0
        self.render_thread: threading.Thread | None = None
        self.caption_status = "captions: module not available" if Captioner is None else "captions off (t)"
        self.captioner = None
        self.args = args

    # ---------- helpers ----------
    def say(self, msg: str, seconds: float = 4.0):
        self.notice, self.notice_until = msg, time.monotonic() + seconds
        print(msg)

    def me_id(self):
        return self.identity.me_id

    def subject(self):
        """Who we focus on: ME if enrolled and visible (and following), else the clicked face."""
        ids = {f.id for f in self.faces}
        if self.follow_me and self.identity.me_id in ids:
            return self.identity.me_id
        if self.selected in ids:
            return self.selected
        return self.selected if self.selected is not None else None

    def face_at(self, x, y):
        w, h = self.size
        for f in self.faces:
            x0, y0, x1, y1 = f.bbox[0] * w, f.bbox[1] * h, f.bbox[2] * w, f.bbox[3] * h
            pad = 0.25 * (x1 - x0)
            if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + 2 * pad:
                return f
        return None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            f = self.face_at(x, y)
            if f is not None:
                self.selected = f.id
                self.follow_me = f.id == self.identity.me_id
                with self.engine.lock:
                    self.engine.mode, self.engine.ab_since = "focused", None
                self.say(f"listening to face #{f.id}" + (" (ME)" if f.id == self.identity.me_id else ""))

    # ---------- main loop ----------
    def run(self):
        e = self.engine
        print(f"Audio in: {e.in_name}\nAudio out: {e.out_name}\nCamera: {self.cam_name}\n{self.tracker.mapper.status}")
        if not any(w in e.out_name.lower() for w in ("earbud", "headphone", "headset")):
            print("WARNING: output is not headphones; keep the volume low to avoid feedback into the array.")
        if self.identity.enrolled:
            self.say("ME is enrolled from a previous session; will follow automatically.")
        e.start()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                img = self.step()
                if img is not None:
                    cv2.imshow(WINDOW, img)
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key != 255 and not self.on_key(key):
                    break
        finally:
            if self.recording:
                self.recording.close()
                print(f"saved {self.recording.folder}")
            if self.captioner is not None:
                self.captioner.flush()
                self.captioner.close()
            e.stop()
            self.cam.close()
            self.tracker.close()
            cv2.destroyAllWindows()

    _last_seq, _fps, _frame_no = -1, 0.0, 0

    def step(self):
        """Process the newest camera frame (if any); returns the annotated image."""
        e = self.engine
        frame, t, seq = self.cam.latest()
        if frame is None or seq == self._last_seq:
            return None
        t0 = time.monotonic()
        self.size = (frame.shape[1], frame.shape[0])
        self.faces = [f for f in self.tracker.process(frame, t) if f.confirmed]
        self.identity.update(frame, self.faces, run_embeddings=self._frame_no % 5 == 0)
        self._frame_no += 1
        self.history.append((t, {f.id: f.mouth for f in self.faces}))
        target = self.subject()
        with e.lock:
            e.faces = {f.id: {"u": f.direction.unit_vector, "speaking": f.mouth_moving} for f in self.faces}
            e.target_id = target
        if self.recording:
            self.recording.write(self.faces, self.identity.me_id)
        self.poll_calibration()
        dt = time.monotonic() - t0
        self._fps = 0.9 * self._fps + 0.1 / max(dt, 1e-3) if self._fps else 1 / max(dt, 1e-3)
        self._last_seq = seq
        img = draw(frame, self.faces, self.tracker, self._fps, show_lips=True, mirror=False)
        return self.overlay(img)

    def on_key(self, key) -> bool:
        e = self.engine
        ch = chr(key) if key < 128 else ""
        if key == 27:
            if self.calib is not None:
                self.stop_calibration(save=False)
                return True
            return False
        if ch == "q":
            return False
        player = e.player
        with e.lock:
            if ch == " ":
                if player and player.playing:
                    player.which = "general" if player.which == "focused" else "focused"
                else:
                    e.mode, e.ab_since = ("general" if e.mode == "focused" else "focused"), None
            elif ch == "a":
                e.ab_since = None if e.ab_since is not None else time.monotonic()
            elif ch == "b":
                e.kind = KINDS[(KINDS.index(e.kind) + 1) % len(KINDS)]
            elif ch == "n":
                e.postfilter = not e.postfilter
            elif ch == ",":
                e.gain_db = max(0.0, e.gain_db - 3)
            elif ch == ".":
                e.gain_db = min(60.0, e.gain_db + 3)
            elif ch in "xy" and ch and self.tracker.mapper.affine is None:
                setattr(self.tracker.mapper, f"flip_{ch}", not getattr(self.tracker.mapper, f"flip_{ch}"))
        if ch == "e":
            f = next((f for f in self.faces if f.id == self.subject()), None) or (self.faces[0] if len(self.faces) == 1 else None)
            if f is None:
                self.say("Click your face first, then press e to enroll it as ME.")
            else:
                frame, _, _ = self.cam.latest()
                self.identity.enroll(frame, f)
                self.follow_me, self.selected = True, f.id
                with e.lock:
                    e.mode = "focused"
                self.say(f"Enrolled face #{f.id} as ME. It will be followed automatically.")
        elif ch == "E":
            self.identity.forget()
            self.say("Forgot ME.")
        elif ch == "c":
            if self.calib is None:
                self.start_calibration()
            else:
                self.stop_calibration(save=True)
        elif ch == "r":
            self.toggle_recording()
        elif ch == "p":
            if player is None:
                self.say("Nothing to play yet: record with r first." if self.render_thread is None else "Still preparing playback...")
            else:
                player.playing = not player.playing
                if player.playing and player.pos == 0:
                    self.say("Playing recording. space = general/focused, p = stop.")
        elif ch == "t":
            self.toggle_captions()
        elif ch == "h":
            self.show_help = not self.show_help
        return True

    # ---------- calibration ----------
    def start_calibration(self):
        if not self.faces:
            self.say("Calibration needs your face in view.")
            return
        self.calib = CalibrationSession()
        while not self.engine.calib.results.empty():
            self.engine.calib.results.get_nowait()
        self.engine.calibrating = True
        self.say("Calibration: click your tongue / whistle / clap at your mouth, moving your head around.", 6)

    def stop_calibration(self, save: bool):
        s, self.calib = self.calib, None
        self.engine.calibrating = False
        if save and s is not None and s.good:
            data = s.save(self.cam_name, self.size)
            self.tracker.mapper.set_calibration(data["affine"], data)
            self.say(f"Calibration saved ({data['points']} points, error {data['rms']:.3f}).", 6)
        elif save and s is not None:
            self.say(f"Calibration not good enough yet, not saved: {s.message}", 6)
        else:
            self.say("Calibration cancelled.")

    def poll_calibration(self):
        if self.calib is None:
            return
        q = self.engine.calib.results
        while not q.empty():
            ev = q.get_nowait()
            snap = min(self.history, key=lambda h: abs(h[0] - ev.t), default=None)
            mouth = None
            if snap is not None and abs(snap[0] - ev.t) < 0.3:
                mouths = snap[1]
                who = self.subject()
                if who in mouths:
                    mouth = mouths[who]
                elif len(mouths) == 1:
                    mouth = next(iter(mouths.values()))
            self.calib.add(ev, mouth, self.size[0] / self.size[1])

    # ---------- recording / playback ----------
    def toggle_recording(self):
        if self.recording is None:
            self.recording = Recording(self.engine, self.tracker, self.cam_name)
            self.say(f"Recording to {self.recording.folder.name} (r to stop)")
            return
        self.recording.close()
        folder, self.recording = self.recording.folder, None
        self.last_folder = folder
        self.say(f"Saved {folder.name}. Preparing general vs focused playback...")

        def work():
            try:
                pair = render_pair(folder)
                save_pair(folder, pair)
                self.engine.player = Player(pair)
                self.say("Playback ready: p to play, space to switch general/focused.", 6)
            except Exception as ex:      # keep the live demo running
                self.say(f"Could not prepare playback: {ex}", 6)
            finally:
                self.render_thread = None

        self.render_thread = threading.Thread(target=work, daemon=True)
        self.render_thread.start()

    # ---------- captions ----------
    def toggle_captions(self):
        if Captioner is None:
            self.say("Captions module not available yet.")
            return
        if self.captioner is None:
            try:
                self.captioner = Captioner(model=self.args.caption_model)
            except Exception as ex:
                self.say(f"Captions failed to start: {ex}")
                return
        self.captions_on = not self.captions_on
        with self.engine.lock:
            self.engine.captioner = self.captioner if self.captions_on else None
        self.say("Captions on" if self.captions_on else "Captions off")

    # ---------- drawing ----------
    def overlay(self, img):
        e = self.engine
        h, w = img.shape[:2]
        k = max(w / 1280, 0.9)
        font, s, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, max(1, int(round(1.5 * k)))
        mode, target = e.listening()
        player = e.player
        playing = player is not None and player.playing

        def text(msg, x, y, col=(255, 255, 255), scale=s):
            cv2.putText(img, msg, (int(x), int(y)), font, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, msg, (int(x), int(y)), font, scale, col, th, cv2.LINE_AA)

        me = self.identity.me_id
        for f in self.faces:
            x0, y0, x1, y1 = f.bbox[0] * w, f.bbox[1] * h, f.bbox[2] * w, f.bbox[3] * h
            if f.id == me:
                text("ME", x0, y0 - 34 * k, (0, 255, 255), s * 1.4)
            if f.id == target:
                col = (255, 0, 255) if mode == "focused" and not playing else (140, 140, 140)
                pad = 10 * k
                cv2.rectangle(img, (int(x0 - pad), int(y0 - pad)), (int(x1 + pad), int(y1 + pad)), col, 3 * th)
                text("FOCUS", x0 + 55 * k, y0 - 2 * pad, col)
            if self.captioner is not None and self.captions_on:
                caps = self.captioner.captions(f.id, max_age_s=12)
                lines = [c.text if c.text else ("..." if c.pending else f"[{c.error[:30]}]") for c in caps[-2:]]
                yy = y1 + 150 * k
                for line in lines:
                    for chunk in [line[i:i + 48] for i in range(0, len(line), 48)][-2:]:
                        text(chunk, x0 - 40 * k, yy, (255, 255, 255), s * 1.05)
                        yy += 24 * k

        if e.peak is not None and self.calib is None:   # loudest acoustic direction
            (sx, sy), contrast = e.peak
            if contrast > 2.0 and e.general_db > -60:
                px, py = self.tracker.mapper.project(sx, sy, w / h)
                c = (int(px * w), int(py * h))
                cv2.circle(img, c, int(18 * k), (0, 165, 255), th + 1)
                cv2.line(img, (c[0] - int(28 * k), c[1]), (c[0] + int(28 * k), c[1]), (0, 165, 255), th)
                cv2.line(img, (c[0], c[1] - int(28 * k)), (c[0], c[1] + int(28 * k)), (0, 165, 255), th)

        if self.calib is not None:
            cal = self.calib
            for sx, sy, nx, ny in cal.points:
                cv2.circle(img, (int(nx * w), int(ny * h)), int(5 * k), (0, 255, 0), -1)
                if cal.fit is not None:
                    a = cal.fit[0]
                    px, py = a[0] * sx + a[1] * sy + a[2], a[3] * sx + a[4] * sy + a[5]
                    cv2.line(img, (int(nx * w), int(ny * h)), (int(px * w), int(py * h)), (0, 200, 255), 1)
            det = e.calib
            bar = min(max((det.level_db - det.floor_db) / 30, 0), 1)
            cv2.rectangle(img, (12, int(60 * k)), (12 + int(300 * k), int(76 * k)), (60, 60, 60), -1)
            cv2.rectangle(img, (12, int(60 * k)), (12 + int(300 * k * bar), int(76 * k)), (0, 255, 0), -1)
            xt = 12 + int(300 * k * 12 / 30)
            cv2.line(img, (xt, int(56 * k)), (xt, int(80 * k)), (255, 255, 255), 2)
            text(f"CALIBRATING  {cal.message}", 12, int(100 * k), (0, 255, 0) if cal.good else (0, 220, 255))
            text(f"points {len(cal.points)}  rejected {cal.rejected}  spread {cal.spread():.0%}   c = save, Esc = cancel",
                 12, int(126 * k), (220, 220, 220), s * 0.85)

        # Bottom panel
        panel_h = int(150 * k)
        cv2.rectangle(img, (0, h - panel_h), (w, h), (25, 25, 25), -1)
        y = h - panel_h + int(30 * k)
        if playing:
            frac = player.pos / max(player.length, 1)
            what = f"PLAYBACK {player.which.upper()}  {player.pos / FS:5.1f}/{player.length / FS:.1f} s   (space switches, p stops)"
            col = (255, 0, 255) if player.which == "focused" else (0, 220, 255)
            cv2.rectangle(img, (0, h - panel_h - 6), (int(w * frac), h - panel_h), col, -1)
        elif mode == "focused" and target is not None:
            who = "ME" if target == me else f"face #{target}"
            what = f"FOCUSED on {who}: {e.kind.upper()} beam{' + noise filter' if e.postfilter else ''}"
            col = (255, 0, 255)
        elif mode == "focused":
            what = "FOCUSED: click a face (or enroll ME with e)"
            col = (255, 0, 255)
        else:
            what = "GENERAL: plain mix of all 16 mics"
            col = (0, 220, 255)
        ab = "   AUTO A/B" if e.ab_since is not None else ""
        text(f"{what}{ab}", 12, y, col, s * 1.15)
        y += int(32 * k)

        def meter(label, db, x, y, color):
            bw = int(200 * k)
            frac = min(max((db + 80) / 60, 0), 1)
            cv2.rectangle(img, (x, y - int(14 * k)), (x + bw, y), (60, 60, 60), -1)
            cv2.rectangle(img, (x, y - int(14 * k)), (x + int(bw * frac), y), color, -1)
            text(f"{label} {db:5.1f}", x + bw + 8, y)

        meter("general ", e.general_db, 12, y, (0, 220, 255))
        meter("you hear", e.listen_db, 12 + int(420 * k), y, col)
        y += int(28 * k)
        for i, f in enumerate(self.faces[:3]):
            meter(f"{'ME' if f.id == me else '#' + str(f.id)} beam", e.face_db.get(f.id, -120), 12 + i * int(420 * k), y, (255, 0, 255))
        y += int(32 * k)
        rec = "  REC" if self.recording else ""
        cap = (self.captioner.status if self.captioner is not None else self.caption_status) if self.captions_on or self.captioner is None else "captions off (t)"
        text(f"vol +{e.gain_db:.0f}  out: {e.out_name[:28]}  overruns {e.overruns}  underruns {e.fifo.underruns}  "
             f"{self.tracker.mapper.status[:40]}  {cap}{rec}   h = help", 12, y, (200, 200, 200), s * 0.8)
        if self.recording:
            cv2.circle(img, (w - int(30 * k), int(30 * k)), int(12 * k), (0, 0, 255), -1)
            text("REC", w - int(90 * k), int(38 * k), (0, 0, 255))
        if time.monotonic() < self.notice_until:
            text(self.notice, 12, h - panel_h - int(16 * k), (255, 255, 0), s * 1.05)
        if self.show_help:
            lines = [l for l in __doc__.splitlines()[5:] if l.strip()]
            box_h = int((len(lines) + 1) * 24 * k)
            cv2.rectangle(img, (int(w * 0.18), int(60 * k)), (int(w * 0.82), int(60 * k) + box_h), (20, 20, 20), -1)
            for i, line in enumerate(lines):
                text(line, int(w * 0.19), int(60 * k) + int((i + 1) * 24 * k), (230, 230, 230), s * 0.85)
        return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-name", default="icspring")
    ap.add_argument("--camera-index", type=int)
    ap.add_argument("--hfov", type=float, default=80.0)
    ap.add_argument("--input", default="miniDSP ASIO", help="input device name substring")
    ap.add_argument("--output", help="output device name substring or index (default: earbuds/headphones)")
    ap.add_argument("--caption-model", default="gemini-3.8-flash")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    App(args).run()


if __name__ == "__main__":
    main()
