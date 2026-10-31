"""Offline beamformer checks with simulated plane waves: python beam_selftest.py

Prints, per beam type, how much each component changes vs. the single reference mic.
Free-field simulation: real rooms (reflections) will give noticeably less than this.
"""
import math

import numpy as np

from beamformer import (CH, FS, HOP, KINDS, NFFT, REFERENCE_MIC, SPEED, Beamformer, SrpMap,
                        StreamProcessor, level_db, mic_positions)

rng = np.random.default_rng(0)
POS = mic_positions()
SECONDS = 3
N = FS * SECONDS
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}: {name} {detail}")


def ray(az_deg, el_deg=0.0):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])


def speechish():
    """Band-limited noise with a 4 Hz syllable envelope, 100-7000 Hz."""
    spec = np.fft.rfft(rng.normal(size=N))
    f = np.fft.rfftfreq(N, 1 / FS)
    spec[(f < 100) | (f > 7000)] = 0
    spec /= np.maximum(f / 500, 1) ** 0.5            # speech-like spectral tilt
    s = np.fft.irfft(spec, N)
    s *= 0.6 + 0.4 * np.sin(2 * np.pi * 4 * np.arange(N) / FS)
    return s / s.std()


def plane_wave(signal, u):
    """(N,16) mic signals for a far-field source in direction u (fractional delays)."""
    f = np.fft.rfftfreq(N, 1 / FS)
    tau = -(POS @ u) / SPEED
    spec = np.fft.rfft(signal)
    return np.fft.irfft(spec[None] * np.exp(-2j * np.pi * f[None] * tau[:, None]), N, axis=1).T


def diffuse(count=200):
    out = np.zeros((N, CH))
    for _ in range(count):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        out += plane_wave(speechish(), v)
    return out / math.sqrt(count)


def run(proc_input, w):
    p = StreamProcessor(bf)
    out = [p.render(p.spectra(proc_input[i:i + HOP].astype(np.float32)), w, 0)
           for i in range(0, N - HOP + 1, HOP)]
    return np.concatenate(out)[NFFT:]                 # skip warm-up


bf = Beamformer()
target_u, other_u = ray(-25, 5), ray(25, 0)
target = plane_wave(speechish(), target_u)
other = plane_wave(speechish(), other_u)
noise = diffuse()

# Passthrough: reference path reproduces the reference mic (delayed by NFFT-HOP)
p = StreamProcessor(bf)
x = rng.normal(size=(N, CH)).astype(np.float32)
y = np.concatenate([p.render(p.spectra(x[i:i + HOP]), None, 0) for i in range(0, N - HOP + 1, HOP)])
d = NFFT - HOP
err = np.max(np.abs(y[NFFT:] - x[NFFT - d:len(y) - d, REFERENCE_MIC - 1]))
check("reference path is a clean passthrough", err < 1e-4, f"(max err {err:.1e})")

ref = {name: level_db(run(sig, None)) for name, sig in
       [("target", target), ("other", other), ("noise", noise)]}
print(f"\n{'beam':6} {'target':>8} {'other talker':>13} {'diffuse noise':>14}   (dB vs single mic)")
gains = {}
for kind in KINDS:
    w = bf.weights(kind, target_u, [other_u])
    g = {name: level_db(run(sig, w)) - ref[name] for name, sig in
         [("target", target), ("other", other), ("noise", noise)]}
    gains[kind] = g
    print(f"{kind:6} {g['target']:+8.1f} {g['other']:+13.1f} {g['noise']:+14.1f}")
print()
for kind in KINDS:
    check(f"{kind}: target level kept (distortionless)", abs(gains[kind]["target"]) < 1.0,
          f"({gains[kind]['target']:+.1f} dB)")
check("das reduces the other talker", gains["das"]["other"] < -3, f"({gains['das']['other']:+.1f} dB)")
check("mvdr reduces diffuse noise more than das", gains["mvdr"]["noise"] < gains["das"]["noise"],
      f"({gains['mvdr']['noise']:+.1f} vs {gains['das']['noise']:+.1f} dB)")
check("lcmv nulls the other talker hardest", gains["lcmv"]["other"] < min(gains["das"]["other"], gains["mvdr"]["other"]) and gains["lcmv"]["other"] < -15,
      f"({gains['lcmv']['other']:+.1f} dB)")

# SRP-PHAT finds the talker
srp = SrpMap(bf)
p = StreamProcessor(bf)
for i in range(0, FS // 2, HOP):
    srp.add(p.spectra((target + 0.1 * noise)[i:i + HOP].astype(np.float32)))
(sx, sy), contrast = srp.take()
true_sx, true_sy = target_u[0] / target_u[2], target_u[1] / target_u[2]
check("SRP-PHAT peak at the talker", abs(sx - true_sx) < 0.12 and abs(sy - true_sy) < 0.12,
      f"(found {sx:+.2f},{sy:+.2f} true {true_sx:+.2f},{true_sy:+.2f}, contrast {contrast:.1f})")

# Real-time budget: one hop with 3 beams + SRP must take well under HOP/FS = 10.7 ms
import time
p = StreamProcessor(bf)
ws = [bf.weights("lcmv", ray(a), [ray(b) for b in (-30, 0, 30) if b != a]) for a in (-30, 0, 30)]
blk = x[:HOP]
t0 = time.perf_counter()
for k in range(200):
    s = p.spectra(blk)
    for w in ws:
        Beamformer.apply(w, s)
    p.render(s, ws[0], 0)
    if k % 4 == 0:          # the app updates the direction map every 4th hop
        srp.add(s)
ms = (time.perf_counter() - t0) / 200 * 1000
check("processing per hop well under real time", ms < 2, f"({ms:.2f} ms per {1000 * HOP / FS:.1f} ms hop)")

print(f"\n{sum(results)}/{len(results)} passed")
if __name__ == "__main__":
    raise SystemExit(0 if all(results) else 1)
