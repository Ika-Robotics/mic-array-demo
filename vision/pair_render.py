"""Render a recording into the two things we compare: general vs focused.

Uses the same streaming classes as the live app, so playback sounds like what was
heard live. Focused follows the recording's target (the person being listened to /
'me'), falling back to whoever's lips moved most.

  python pair_render.py recordings/<time>     -> recordings/<time>/two/{general,focused}.wav
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from beamformer import HOP, NFFT, Beamformer, PostFilter, StreamProcessor, mix_weights
from render_recording import angle, group_people


def _loud_db(y: np.ndarray, fs: int) -> float:
    n = int(0.05 * fs)
    lv = [10 * np.log10(np.mean(y[i:i + n] ** 2) + 1e-15) for i in range(0, max(len(y) - n, 1), n // 2)]
    return float(np.percentile(lv, 95))


def render_pair(folder: Path, kind: str | None = None, postfilter: bool | None = None) -> dict:
    """Returns {"general", "focused", "fs", "target", "kind", "postfilter"}; voice peaks of
    both are matched and normalised to about -20 dBFS (same gain for both)."""
    mics, fs = sf.read(str(folder / "mics.wav"), dtype="float32")
    log = [json.loads(line) for line in (folder / "faces.jsonl").read_text().splitlines() if line.strip()]
    if not log:
        raise ValueError("no face snapshots in this recording")
    kind = kind or log[-1].get("kind", "mvdr")
    postfilter = log[-1].get("postfilter", True) if postfilter is None else postfilter

    # Who to follow: the logged target per snapshot; else the person who talked most.
    people = group_people(log)
    talk = [sum(f["mouth_moving"] for e in log for f in e["faces"] if f["id"] in p["ids"]) for p in people]
    fallback = people[int(np.argmax(talk))] if people else None
    targets = [e.get("target") for e in log]
    samples = np.array([e["sample"] for e in log])

    bf = Beamformer()
    proc_g, proc_f, post = StreamProcessor(bf), StreamProcessor(bf), PostFilter() if postfilter else None
    wmix = mix_weights()
    general, focused = [], []
    last_u = tuple(fallback["u"]) if fallback else (0.0, 0.0, 1.0)
    for start in range(0, len(mics) - HOP + 1, HOP):
        i = max(int(np.searchsorted(samples, start, side="right")) - 1, 0)
        now = {f["id"]: tuple(f["u"]) for f in log[i]["faces"]}
        tid = targets[i]
        if tid in now:
            last_u = now[tid]
        elif fallback is not None:
            mine = [u for k, u in now.items() if k in fallback["ids"]]
            if tid is None and mine:
                last_u = mine[0]
        others = [u for k, u in now.items() if angle(u, last_u) > 3]
        block = mics[start:start + HOP]
        general.append(proc_g.render(proc_g.spectra(block), wmix, 0))
        focused.append(proc_f.render(proc_f.spectra(block), bf.weights(kind, last_u, others), 0, post))
    d = NFFT - HOP
    g = np.concatenate(general)[d:]
    f = np.concatenate(focused)[d:]
    f *= 10 ** ((_loud_db(g, fs) - _loud_db(f, fs)) / 20)          # same voice loudness
    gain = 10 ** ((-20 - _loud_db(g, fs)) / 20)
    gain /= max(max(np.abs(g).max(), np.abs(f).max()) * gain / 0.98, 1)
    return {"general": (g * gain).astype(np.float32), "focused": (f * gain).astype(np.float32),
            "fs": fs, "kind": kind, "postfilter": bool(postfilter)}


def save_pair(folder: Path, pair: dict) -> Path:
    out = folder / "two"
    out.mkdir(exist_ok=True)
    for name in ("general", "focused"):
        sf.write(str(out / f"{name}.wav"), pair[name], pair["fs"], "PCM_16")
    return out


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    pair = render_pair(folder)
    print(f"focused = {pair['kind']} beam{' + noise filter' if pair['postfilter'] else ''}")
    print(f"-> {save_pair(folder, pair)}")
