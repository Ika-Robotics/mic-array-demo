"""Offline checks (no camera): python selftest.py"""
import math
import time
from pathlib import Path

import cv2
import numpy as np

from face_tracker import DirectionMapper, FaceTracker, Settings, TrackedFace, _Track, load_profile

HERE = Path(__file__).parent
PORTRAIT = cv2.imread(str(HERE / "testdata" / "portrait.jpg"))
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}: {name} {detail}")


def three_faces(shift=0, scale=0.4):
    """1280x720 canvas, three portraits side by side. scale 0.4 -> ~77 px faces,
    about what an 80 degree camera sees of a person 1.5 m away."""
    canvas = np.full((720, 1280, 3), 90, np.uint8)
    face = cv2.resize(PORTRAIT, None, fx=scale, fy=scale)
    fh, fw = face.shape[:2]
    for x in (150, 480, 810):
        canvas[180:180 + fh, x + shift:x + shift + fw] = face
    return canvas


# 1. Single face, mirror symmetry of yaw and azimuth
ft = FaceTracker()
def placed(nx, scale=0.6):
    """Portrait scaled into a 1280x720 frame with the mouth at (nx, 0.5)."""
    m0 = FaceTracker().process(PORTRAIT, 0)[0].mouth
    tx = nx * 1280 - m0[0] * PORTRAIT.shape[1] * scale
    ty = 360 - m0[1] * PORTRAIT.shape[0] * scale
    return cv2.warpAffine(PORTRAIT, np.float32([[scale, 0, tx], [0, scale, ty]]), (1280, 720),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(90, 90, 90))
CENTERED = placed(0.5)
a = FaceTracker().process(CENTERED, 0)
b = FaceTracker().process(cv2.flip(CENTERED, 1), 0)
check("portrait detected", len(a) == 1 and len(b) == 1)
if a and b:
    fa, fb = a[0], b[0]
    check("yaw mirrors under flip", abs(fa.yaw_deg + fb.yaw_deg) < 6, f"({fa.yaw_deg:.1f} vs {fb.yaw_deg:.1f})")
    check("azimuth mirrors under flip", abs(fa.direction.azimuth_deg + fb.direction.azimuth_deg) < 1,
          f"({fa.direction.azimuth_deg:.1f} vs {fb.direction.azimuth_deg:.1f})")
    check("frontal face at image centre is near camera axis", fa.off_axis_deg < 15, f"(off-axis {fa.off_axis_deg:.1f})")
    edge = FaceTracker().process(placed(0.85), 0)[0]
    grow = edge.off_axis_deg - fa.off_axis_deg
    check("same head pose at right edge reads as looking past the camera",
          abs(grow - edge.direction.azimuth_deg) < 10 and not edge.looking_at_camera,
          f"(off-axis {edge.off_axis_deg:.1f}, azimuth {edge.direction.azimuth_deg:.1f}, yaw {edge.yaw_deg:.1f})")

# 2. In-plane rotation shows up as roll, not yaw
h, w = PORTRAIT.shape[:2]
rot = cv2.warpAffine(PORTRAIT, cv2.getRotationMatrix2D((w / 2, h / 2), 20, 1), (w, h))
r = ft.process(rot, 0)
if r and a:
    check("20 deg image rotation -> roll ~20", abs(abs(r[0].roll_deg - a[0].roll_deg) - 20) < 6,
          f"(roll {r[0].roll_deg:.1f}, yaw {r[0].yaw_deg:.1f})")
ft.close()

# 3. Three faces: count, left-to-right order, stable IDs while moving
vt = FaceTracker(Settings(max_faces=3))
ids, azs, times = None, None, []
stable = True
for k in range(15):
    t0 = time.perf_counter()
    faces = vt.process(three_faces(shift=k * 4), k / 30)
    times.append(time.perf_counter() - t0)
    if k == 0:
        ids = [f.id for f in faces]
        azs = [f.direction.azimuth_deg for f in faces]
    elif [f.id for f in faces] != ids:
        stable = False
check("three faces detected", ids is not None and len(ids) == 3, f"({ids})")
check("azimuth increases left to right", azs is not None and azs == sorted(azs), f"({[round(v, 1) for v in azs or []]})")
check("IDs stable across 15 moving frames", stable)
ms = 1000 * float(np.median(times[3:]))
check("3-face processing >= 20 fps", ms < 50, f"(median {ms:.1f} ms/frame at 1280x720)")
vt.process(three_faces(scale=0.22), 1.0)  # scene change: following fails, detector reruns next frame
small = vt.process(three_faces(scale=0.22), 1.03)
check("three ~42 px faces (~2.7 m away) detected", len(small) == 3, f"({len(small)})")
vt.close()

# 3b. Distance scales inversely with face size
dt = FaceTracker()
def one_face(scale):
    canvas = np.full((720, 1280, 3), 90, np.uint8)
    face = cv2.resize(PORTRAIT, None, fx=scale, fy=scale)[:700]
    canvas[:face.shape[0], 400:400 + face.shape[1]] = face
    return canvas
near = FaceTracker().process(one_face(0.6), 0)
far = FaceTracker().process(one_face(0.3), 0)
if near and far:
    ratio = far[0].distance_m / near[0].distance_m
    check("half-size face reads ~2x distance", 1.8 < ratio < 2.2,
          f"({near[0].distance_m:.2f} m -> {far[0].distance_m:.2f} m)")

# 3c. Still face with camera jitter/noise must not look like talking
rng = np.random.default_rng(1)
base = three_faces()
peak = 0.0
for k in range(45):
    m = cv2.getRotationMatrix2D((640, 360), rng.uniform(-0.5, 0.5), 1 + rng.uniform(-0.01, 0.01))
    m[:, 2] += rng.uniform(-1.5, 1.5, 2)
    frame = cv2.warpAffine(base, m, (1280, 720))
    frame = np.clip(frame + rng.normal(0, 4, frame.shape), 0, 255).astype(np.uint8)
    faces = dt.process(frame, k / 30)
    if k > 20:
        peak = max([peak] + [f.lip_activity for f in faces])
check("still faces + sensor noise stay below lip threshold", peak < Settings().activity_off,
      f"(peak lip_activity {peak:.4f} vs off {Settings().activity_off})")
dt.close()

# 4. Lip activity logic on synthetic mouth-opening traces
def run_trace(values, fps=30):
    s = Settings()
    tr, moving = _Track(1, (0.5, 0.5), 0), []
    probe = FaceTracker.__new__(FaceTracker)
    probe.settings = s
    for i, v in enumerate(values):
        f = TrackedFace(id=1, bbox=(0, 0, 1, 1), mouth=(0.5, 0.5), landmarks=np.zeros((1, 3)),
                        mouth_open=v, off_axis_deg=10)
        probe._update_signals(tr, f, i / fps)
        moving.append(f.mouth_moving)
    return moving

closed = run_trace([0.02] * 60)
open_static = run_trace([0.02] * 15 + [0.35] * 75)
talking = run_trace([0.15 + 0.12 * math.sin(2 * math.pi * 4 * i / 30) for i in range(60)])
pause = run_trace([0.15 + 0.12 * math.sin(2 * math.pi * 4 * i / 30) for i in range(45)] + [0.03] * 45)
check("closed mouth -> not moving", not any(closed))
check("held-open mouth (yawn/smile) settles to not moving", not any(open_static[-30:]))
check("4 Hz lip oscillation -> moving", all(talking[20:]))
check("stops ~1 s after speech ends", pause[50] and not any(pause[-20:]))

# 5. Profile parsing and unprojection agree with the wizard's affine map
prof = HERE.parent / "test-output" / "workflow" / "setup-profile.ini"
affine, size = load_profile(prof)
m = DirectionMapper(Settings(), prof)
err = 0.0
for nx, ny in [(0.2, 0.3), (0.5, 0.5), (0.8, 0.7)]:
    d = m(nx, ny, 16 / 9)
    sx, sy = d.unit_vector[0] / d.unit_vector[2], d.unit_vector[1] / d.unit_vector[2]
    px = affine[0] * sx + affine[1] * sy + affine[2]
    py = affine[3] * sx + affine[4] * sy + affine[5]
    err = max(err, abs(px - nx), abs(py - ny))
check("profile unproject round-trips the wizard projection", err < 1e-9 and m.affine is not None, f"({size}, err {err:.1e})")

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
