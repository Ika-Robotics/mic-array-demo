"""Caption a beam_demo recording: general mix vs focused beam, side by side.

  python caption_recording.py recordings/20261030-195311 [--source focused|general|both]
                              [--person N] [--kind lcmv|mvdr|das] [--model gemini-3.8-flash]

The main speaker is the person whose lips moved most (or --person, 1-based left to right).
Utterances are cut using that person's lip movement from faces.jsonl, so both sources are
segmented identically and the transcripts compare only the audio quality.
Writes <folder>/captions.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from beamformer import HOP, KINDS, NFFT, Beamformer, StreamProcessor
from captioner import Captioner
from make_two import noise_suppress
from render_recording import angle, group_people

SOURCES = {"general": 0, "focused": 1}


def focused_beam(mics, log, me, kind):
    bf = Beamformer()
    samples = np.array([e["sample"] for e in log])
    proc, out, last = StreamProcessor(bf), [], tuple(me["u"])
    for start in range(0, len(mics) - HOP + 1, HOP):
        e = log[max(int(np.searchsorted(samples, start, side="right")) - 1, 0)]
        now = {f["id"]: tuple(f["u"]) for f in e["faces"]}
        mine = [u for tid, u in now.items() if tid in me["ids"]]
        last = mine[0] if mine else last
        others = [u for tid, u in now.items() if tid not in me["ids"] and angle(u, last) > 3]
        out.append(proc.render(proc.spectra(mics[start:start + HOP]), bf.weights(kind, last, others), 0))
    y = np.concatenate(out)[NFFT - HOP:]
    return np.pad(y, (0, len(mics) - len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--source", choices=["focused", "general", "both"], default="both")
    ap.add_argument("--person", type=int)
    ap.add_argument("--kind", choices=KINDS, default="lcmv")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--language", default="")
    ap.add_argument("--wait", type=float, default=300, help="max seconds to wait for Gemini")
    args = ap.parse_args()

    mics, fs = sf.read(str(args.folder / "mics.wav"), dtype="float32")
    log = [json.loads(line) for line in (args.folder / "faces.jsonl").read_text().splitlines() if line.strip()]
    people = group_people(log)
    if not people:
        raise SystemExit("no person visible long enough in this recording")
    talk = [sum(f["mouth_moving"] for e in log for f in e["faces"] if f["id"] in p["ids"]) for p in people]
    idx = args.person - 1 if args.person else int(np.argmax(talk))
    me = people[idx]
    az = math.degrees(math.atan2(me["u"][0], me["u"][2]))
    print(f"speaker: person {idx + 1}/{len(people)} (az {az:+.0f} deg), {len(mics) / fs:.1f} s")

    signals = {}
    if args.source in ("general", "both"):
        signals["general"] = mics.mean(axis=1)
    if args.source in ("focused", "both"):
        signals["focused"] = noise_suppress(focused_beam(mics, log, me, args.kind), fs)

    # speaking mask per hop from the speaker's lip movement
    samples = np.array([e["sample"] for e in log])
    def speaking_at(n):
        e = log[max(int(np.searchsorted(samples, n, side="right")) - 1, 0)]
        return any(f["mouth_moving"] for f in e["faces"] if f["id"] in me["ids"])

    cap = Captioner(model=args.model, language_hint=args.language, retry_rounds=4,
                    retry_backoff_s=(20.0, 40.0, 60.0))
    if not cap.enabled:
        raise SystemExit(f"captions disabled: {cap.status}")
    calls = []
    send = cap._transport

    def timed(model, wav, prompt, timeout):
        t = time.time()
        try:
            text = send(model, wav, prompt, timeout)
            calls.append((model, time.time() - t, "ok"))
            return text
        except Exception as e:
            calls.append((model, time.time() - t, str(e)[:50]))
            raise
    cap._transport = timed
    for start in range(0, len(mics) - HOP + 1, HOP):
        speaking, t = speaking_at(start), start / fs
        for name, y in signals.items():
            cap.feed(SOURCES[name], y[start:start + HOP], speaking, t)
    cap.flush()
    utterances = cap.all_captions()
    print(f"{len(utterances) // max(len(signals), 1)} utterance(s) per source; waiting for Gemini...")
    t0 = time.time()
    while any(c.pending for c in cap.all_captions()) and time.time() - t0 < args.wait:
        time.sleep(0.5)
    cap.close()

    names = {v: k for k, v in SOURCES.items()}
    rows = []
    for c in cap.all_captions():
        rows.append({"source": names[c.face_id], "t_start": round(c.t_start, 2), "t_end": round(c.t_end, 2),
                     "text": c.text, "model": c.model, "error": c.error, "pending": c.pending})
    for name in signals:
        print(f"\n== {name} ==")
        for r in (r for r in rows if r["source"] == name):
            what = r["text"] or ("(pending)" if r["pending"] else f"(error: {r['error'][:80]})" if r["error"] else "(no speech)")
            print(f"  [{r['t_start']:5.1f}-{r['t_end']:5.1f}s] {r['model'] or '-':18} {what}")
    out = args.folder / "captions.json"
    out.write_text(json.dumps({"speaker_person": idx + 1, "speaker_az_deg": round(az, 1), "beam": args.kind,
                               "captions": rows}, indent=1), encoding="utf-8")
    print("\nrequests:")
    for model, dt, result in calls:
        print(f"  {model:22} {dt:5.1f}s  {result}")
    print(f"\nstatus: {cap.status}\n-> {out}")


if __name__ == "__main__":
    main()
