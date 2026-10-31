"""Two files from a recording: the general mix vs. the focused voice.

  python make_two.py recordings/20261030-195311 [--person N] [--kind mvdr|lcmv|das]

Writes <folder>/two/:
  general.wav             average of all 16 mics: what the array hears with no focusing
  focused.wav             beam on the speaker + noise suppression post-filter
  focused_beam_only.wav   the beam without the post-filter (to hear what each part does)
The speaker defaults to the person whose lips moved most. All files share one gain.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from beamformer import HOP, KINDS, NFFT, Beamformer, StreamProcessor
from render_recording import angle, group_people


def noise_suppress(y, fs, floor_db=-18.0, over=1.5, nfft=1024, hop=256):
    """Wiener post-filter. Noise spectrum = 15th percentile of each frequency's power over
    the file (stationary fans/rumble/hiss). Gain floor limits 'musical noise'."""
    win = np.sqrt(np.hanning(nfft + 1)[:nfft])
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(y, (nfft, nfft)), nfft)[::hop] * win
    spec = np.fft.rfft(frames, axis=1)
    power = np.abs(spec) ** 2
    noise = np.percentile(power, 15, axis=0)
    snr = np.maximum(power / (over * noise + 1e-20) - 1, 0)
    # light smoothing over time keeps the gain from fluttering (less "musical noise")
    for t in range(1, len(snr)):
        snr[t] = 0.3 * snr[t - 1] + 0.7 * snr[t]
    gain = np.maximum(snr / (1 + snr), 10 ** (floor_db / 20))
    out_frames = np.fft.irfft(spec * gain, nfft, axis=1) * win
    out = np.zeros(len(y) + 2 * nfft + nfft)
    for i, f in enumerate(out_frames):
        out[i * hop:i * hop + nfft] += f
    norm = np.zeros_like(out)
    for i in range(len(out_frames)):
        norm[i * hop:i * hop + nfft] += win ** 2
    out = out / np.maximum(norm, 1e-8)
    return out[nfft:nfft + len(y)]


def speech_range_db(y, fs):
    """Loud (p95) minus floor (p10) of 50 ms levels in 300-3500 Hz: rough speech-vs-background."""
    yb = sosfiltfilt(butter(4, [300, 3500], "bandpass", fs=fs, output="sos"), y)
    n = int(0.05 * fs)
    lv = np.array([10 * np.log10(np.mean(yb[i:i + n] ** 2) + 1e-15) for i in range(0, len(yb) - n, n // 2)])
    return float(np.percentile(lv, 95) - np.percentile(lv, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--person", type=int, help="1-based, left to right (default: who talked most)")
    ap.add_argument("--kind", choices=KINDS, help="beam type (default: best measured)")
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
    print(f"speaker: person {idx + 1} of {len(people)} (az {az:+.0f} deg, lips moving in {talk[idx]} snapshots)")

    samples = np.array([e["sample"] for e in log])
    bf = Beamformer()

    def beam(kind):
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

    general = mics.mean(axis=1)
    kinds = [args.kind] if args.kind else ["mvdr", "lcmv"]
    best = None
    for k in kinds:
        b = beam(k)
        f = noise_suppress(b, fs)
        score = speech_range_db(f, fs)
        print(f"  {k}: beam {speech_range_db(b, fs):.1f} dB, beam+filter {score:.1f} dB")
        if best is None or score > best[0]:
            best = (score, k, b, f)
    _, kind, beam_only, focused = best

    level = lambda y: 10 * np.log10(np.mean(y ** 2) + 1e-15)
    # Match loudness on the voice (loud moments) so the comparison is about background, not volume
    gen_loud = np.percentile([level(general[i:i + 2400]) for i in range(0, len(general) - 2400, 1200)], 95)
    for_loud = np.percentile([level(focused[i:i + 2400]) for i in range(0, len(focused) - 2400, 1200)], 95)
    match = 10 ** ((gen_loud - for_loud) / 20)
    focused, beam_only = focused * match, beam_only * 10 ** ((gen_loud - np.percentile(
        [level(beam_only[i:i + 2400]) for i in range(0, len(beam_only) - 2400, 1200)], 95)) / 20)
    gain = 10 ** ((-20 - gen_loud) / 20)
    peak = max(np.abs(y).max() for y in (general, focused, beam_only)) * gain
    gain /= max(peak / 0.98, 1)

    out = args.folder / "two"
    out.mkdir(exist_ok=True)
    print(f"\nspeech-vs-background (300-3500 Hz, higher is better):")
    for name, y in [("general", general), ("focused_beam_only", beam_only), ("focused", focused)]:
        sf.write(str(out / f"{name}.wav"), (y * gain).astype(np.float32), fs, "PCM_16")
        print(f"  {name + '.wav':24} {speech_range_db(y, fs):5.1f} dB")
    print(f"focused = {kind} beam + noise suppression\n-> {out}")


if __name__ == "__main__":
    main()
