"""Offline calibration checks with simulated clicks/whistles: python calibration_selftest.py"""
import math
import time

import numpy as np

from beamformer import CH, FS, HOP, SPEED, Beamformer, StreamProcessor, mic_positions
from calibration import CalibrationSession, EventDetector, fit_similarity

rng = np.random.default_rng(3)
POS = mic_positions()
ASPECT = 16 / 9
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}: {name} {detail}")


# A "true" camera->array relation: mirrored, rotated 7 deg, scaled, offset.
ang = math.radians(7)
TRUE = {"scale": 0.62, "rot": ang, "mirror": True, "off": (0.52, 0.47)}


def true_project(sx, sy):
    z = complex(sx, -sy) if TRUE["mirror"] else complex(sx, sy)
    w = TRUE["scale"] * complex(math.cos(TRUE["rot"]), math.sin(TRUE["rot"])) * z
    return TRUE["off"][0] + w.real / ASPECT, TRUE["off"][1] + w.imag


def plane(signal, sx, sy):
    u = np.array([sx, sy, 1.0]); u /= np.linalg.norm(u)
    n = len(signal)
    f = np.fft.rfftfreq(n, 1 / FS)
    tau = -(POS @ u) / SPEED
    return np.fft.irfft(np.fft.rfft(signal)[None] * np.exp(-2j * np.pi * f[None] * tau[:, None]), n, axis=1).T


def sound(kind, n):
    t = np.arange(n) / FS
    if kind == "click":
        s = np.zeros(n); s[n // 3:n // 3 + 40] = rng.normal(0, 1, 40) * np.hanning(40)
        return s * 0.3
    f0 = rng.uniform(1200, 2500)            # whistle
    env = np.clip(np.sin(np.pi * np.clip((t - 0.02) / 0.06, 0, 1)), 0, 1)
    return 0.05 * np.sin(2 * np.pi * f0 * t) * env


# 1. Fit recovers a known mirrored/rotated mapping from noisy points with outliers
pts = []
for _ in range(14):
    sx, sy = rng.uniform(-0.6, 0.6), rng.uniform(-0.35, 0.35)
    nx, ny = true_project(sx, sy)
    pts.append((sx + rng.normal(0, 0.01), sy + rng.normal(0, 0.01), nx, ny))
pts += [(0.5, -0.3, 0.1, 0.9), (-0.4, 0.2, 0.9, 0.2)]          # two wild outliers
affine, rms, keep = fit_similarity(np.array(pts), ASPECT)
err = max(abs((affine[0] * sx + affine[1] * sy + affine[2]) - true_project(sx, sy)[0]) +
          abs((affine[3] * sx + affine[4] * sy + affine[5]) - true_project(sx, sy)[1])
          for sx, sy in [(-0.5, -0.3), (0.5, 0.3), (0, 0), (0.4, -0.2)])
check("fit recovers mirrored+rotated mapping", err < 0.03, f"(max err {err:.3f}, rms {rms:.3f})")
check("outliers rejected", not keep[-1] and not keep[-2] and keep[:-2].sum() >= 12, f"({keep.sum()}/{len(keep)} kept)")

# 2. Fit works with mostly horizontal head movement (realistic: people move sideways more)
pts_h = [(sx, sy, *true_project(sx, sy)) for sx, sy in zip(np.linspace(-0.6, 0.6, 8), rng.normal(0, 0.03, 8))]
a2, rms2, _ = fit_similarity(np.array(pts_h), ASPECT)
up = (a2[3] * 0 + a2[4] * 0.4 + a2[5]) - true_project(0, 0.4)[1]
check("horizontal-only spread still extrapolates vertically", abs(up) < 0.05, f"(err {up:.3f})")

# 3. End to end: simulated clicks + whistles through EventDetector -> session -> fit
bf = Beamformer()
det = EventDetector(bf)
proc = StreamProcessor(bf)
session = CalibrationSession()
background = rng.normal(0, 0.002, (FS * 14, CH))
truth = []
t = 0.0
mouths = {}
for k in range(12):
    sx, sy = rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3)
    mouth = true_project(sx, sy)
    start = int((0.5 + k * 1.0) * FS)
    ev = plane(sound("click" if k % 2 else "whistle", int(0.12 * FS)), sx, sy)
    background[start:start + len(ev)] += ev
    truth.append((start / FS, sx, sy, mouth))
events = []
worker_ms = []
for i in range(0, len(background) - HOP + 1, HOP):
    det.feed(proc.spectra(background[i:i + HOP].astype(np.float32)), i / FS)
    if not det.jobs.empty():           # pace like real time would: let the worker finish each event
        t0 = time.perf_counter()
        while not det.jobs.empty() or det.results.qsize() + len(events) < len(worker_ms) + 1:
            time.sleep(0.002)
            if time.perf_counter() - t0 > 2:
                break
        worker_ms.append((time.perf_counter() - t0) * 1000)
        while not det.results.empty():
            events.append(det.results.get())
time.sleep(0.5)
while not det.results.empty():
    events.append(det.results.get())
check("worker keeps up with sounds 0.15 s apart", worker_ms and max(worker_ms) < 150,
      f"(max {max(worker_ms):.0f} ms per event)")
check("every click/whistle detected once", len(events) == len(truth), f"({len(events)} events for {len(truth)} sounds)")
dir_err = []
for ev in events:
    tt, sx, sy, mouth = min(truth, key=lambda q: abs(q[0] - ev.t))
    dir_err.append(math.hypot(ev.slopes[0] - sx, ev.slopes[1] - sy))
    session.add(ev, mouth, ASPECT)
check("acoustic directions accurate", dir_err and np.median(dir_err) < 0.05, f"(median slope error {np.median(dir_err):.3f})")
check("session reaches a good calibration", session.good, f"({session.message})")
if session.fit:
    a = session.fit[0]
    e = max(abs(a[0] * sx + a[1] * sy + a[2] - true_project(sx, sy)[0]) + abs(a[3] * sx + a[4] * sy + a[5] - true_project(sx, sy)[1])
            for sx, sy in [(-0.5, -0.25), (0.5, 0.25), (0.0, 0.0)])
    check("end-to-end mapping matches truth", e < 0.05, f"(max err {e:.3f} of image)")

# 4. Detector stays quiet on steady background
det2, p2 = EventDetector(bf), StreamProcessor(bf)
quiet = rng.normal(0, 0.002, (FS * 4, CH)).astype(np.float32)
for i in range(0, len(quiet) - HOP + 1, HOP):
    det2.feed(p2.spectra(quiet[i:i + HOP]), i / FS)
time.sleep(0.3)
check("no false events on steady noise", det2.results.empty(), f"({det2.results.qsize()})")

t0 = time.perf_counter()
s = proc.spectra(background[:HOP].astype(np.float32))
for _ in range(500):
    det2.feed(s, 0.0)
ms = (time.perf_counter() - t0) / 500 * 1000
check("audio-thread cost per hop small", ms < 0.5, f"({ms:.3f} ms)")

print(f"\n{sum(results)}/{len(results)} passed")
if __name__ == "__main__":
    raise SystemExit(0 if all(results) else 1)
