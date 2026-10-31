"""Render a beam_demo recording into listenable WAVs.

  python render_recording.py recordings/20261030-193000 [--kinds das mvdr lcmv]

Writes into <folder>/rendered/:
  0_reference.wav                 the single reference mic (what a normal mic hears)
  person<N>_az<deg>_<kind>.wav    beam following person N, per beam type
Tracker IDs that are the same person (same direction, not visible at the same time)
are merged. Every file gets the SAME gain (reference speech peaks at about -20 dBFS),
so the files are audible and directly comparable in any audio player.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from beamformer import HOP, KINDS, NFFT, REFERENCE_MIC, Beamformer, StreamProcessor

MERGE_DEG = 8.0      # IDs whose mean directions are this close are the same person
MIN_FRAMES = 5       # ignore tracks seen in fewer face snapshots (false detections)


def angle(u, v):
    return math.degrees(math.acos(float(np.clip(np.dot(u, v), -1, 1))))


def group_people(log):
    tracks = {}
    for e in log:
        for f in e["faces"]:
            tr = tracks.setdefault(f["id"], {"samples": [], "u": []})
            tr["samples"].append(e["sample"]); tr["u"].append(f["u"])
    people = []      # each: {"ids": set, "u": mean unit vector, "spans": [(a, b)]}
    for tid, tr in sorted(tracks.items(), key=lambda kv: kv[1]["samples"][0]):
        if len(tr["samples"]) < MIN_FRAMES:
            continue
        u = np.mean(tr["u"], axis=0); u /= np.linalg.norm(u)
        span = (tr["samples"][0], tr["samples"][-1])
        for p in people:
            overlap = any(span[0] < b and a < span[1] for a, b in p["spans"])
            if not overlap and angle(u, p["u"]) < MERGE_DEG:
                p["ids"].add(tid); p["spans"].append(span)
                p["u"] = p["u"] + u; p["u"] /= np.linalg.norm(p["u"])
                break
        else:
            people.append({"ids": {tid}, "u": u, "spans": [span]})
    people.sort(key=lambda p: p["u"][0])     # left to right
    return people


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    args = ap.parse_args()

    mics, fs = sf.read(str(args.folder / "mics.wav"), dtype="float32")
    log = [json.loads(line) for line in (args.folder / "faces.jsonl").read_text().splitlines() if line.strip()]
    if not log:
        raise SystemExit("faces.jsonl is empty: no face was visible during the recording")
    samples = np.array([e["sample"] for e in log])
    people = group_people(log)
    print(f"{len(mics) / fs:.1f} s, {len(log)} face snapshots, {len(people)} people:")
    for n, p in enumerate(people, 1):
        az = math.degrees(math.atan2(p["u"][0], p["u"][2]))
        p["name"] = f"person{n}_az{az:+.0f}"
        print(f"  {p['name']}  (tracker ids {sorted(p['ids'])})")

    def directions_at(n):
        i = max(int(np.searchsorted(samples, n, side="right")) - 1, 0)
        return {f["id"]: tuple(f["u"]) for f in log[i]["faces"]}

    bf = Beamformer()
    jobs = [("0_reference", None, None)] + [(f"{p['name']}_{k}", k, p) for p in people for k in args.kinds]
    procs = {name: StreamProcessor(bf) for name, _, _ in jobs}
    outs = {name: [] for name, _, _ in jobs}
    last_u = {}
    for start in range(0, len(mics) - HOP + 1, HOP):
        block = mics[start:start + HOP]
        now = directions_at(start)
        for name, kind, person in jobs:
            p = procs[name]
            spec = p.spectra(block)
            w = None
            if kind is not None:
                pid = id(person)
                mine = [u for tid, u in now.items() if tid in person["ids"]]
                last_u[pid] = mine[0] if mine else last_u.get(pid, tuple(person["u"]))
                others = [u for tid, u in now.items() if tid not in person["ids"]
                          and angle(u, last_u[pid]) > 3]
                w = bf.weights(kind, last_u[pid], others)
            outs[name].append(p.render(spec, w, 0))

    delay = NFFT - HOP
    signals = {name: np.concatenate(chunks)[delay:] for name, chunks in outs.items()}
    ref = signals["0_reference"]
    frames = [np.mean(ref[i:i + 4800] ** 2) for i in range(0, len(ref) - 4800, 4800)]
    loud_db = 10 * np.log10(np.percentile(frames, 98) + 1e-12)
    gain_db = -20 - loud_db
    peak = max(np.abs(y).max() for y in signals.values()) * 10 ** (gain_db / 20)
    if peak > 0.98:
        gain_db -= 20 * np.log10(peak / 0.98)
    out_dir = args.folder / "rendered"
    out_dir.mkdir(exist_ok=True)
    print(f"\ncommon gain +{gain_db:.1f} dB applied to every file (reference mic {REFERENCE_MIC})")
    for name, y in signals.items():
        y = y * 10 ** (gain_db / 20)
        sf.write(str(out_dir / f"{name}.wav"), y.astype(np.float32), fs, "PCM_16")
        print(f"  {name}.wav  {10 * np.log10(max(np.mean(y ** 2), 1e-12)):6.1f} dBFS")
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
