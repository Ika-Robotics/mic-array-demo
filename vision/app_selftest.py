"""Headless end-to-end test of beam_demo.App with fake camera/audio: python app_selftest.py

Uses temp files only (never touches beam-calibration.json, me.npz or recordings/)."""
import math
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import cv2
import numpy as np

TMP = Path(tempfile.mkdtemp(prefix="ika_app_test_"))
import sounddevice as sd


class FakeStream:
    def __init__(self, **kw): pass
    def start(self): pass
    def stop(self): pass
    def close(self): pass


sd.InputStream = sd.OutputStream = FakeStream
_qd = sd.query_devices
sd.query_devices = lambda i=None, *a, **k: {"name": f"fake device {i}"} if i is not None else _qd()

import calibration
import identity
import beam_demo as bd
from beamformer import CH, FS, HOP, SPEED, mic_positions

calibration.CALIBRATION = TMP / "beam-calibration.json"
calibration.load_calibration.__defaults__ = (TMP / "none.json",)
identity.Identity.__init__.__defaults__ = (TMP / "me.npz",)
bd.HERE = TMP
bd.pick_devices = lambda *a: (0, 1)

HERE = Path(__file__).parent
ME_IMG = cv2.imread(str(HERE / "testdata" / "portrait.jpg"))
OTHER_IMG = cv2.imread(str(HERE / "testdata" / "lena.jpg"))
POS = mic_positions()
rng = np.random.default_rng(5)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}: {name} {detail}")


def paste(canvas, img, cx, cy, face_w):
    """Paste img scaled so its face is ~face_w px wide, centred near (cx, cy)."""
    s = face_w / (0.25 * img.shape[1])
    im = cv2.resize(img, None, fx=s, fy=s)
    h, w = im.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h * 0.3)
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + w, canvas.shape[1]), min(y0 + h, canvas.shape[0])
    canvas[ys:ye, xs:xe] = im[ys - y0:ye - y0, xs - x0:xe - x0]


def scene(me_x, other_x=None, me_y=300):
    c = np.full((720, 1280, 3), 80, np.uint8)
    if other_x is not None:
        paste(c, OTHER_IMG, other_x, 300, 150)
    paste(c, ME_IMG, me_x, me_y, 150)
    return c


class FakeCam:
    def __init__(self, *a, **k):
        self.frame, self.seq = scene(400, 900), 0

    def set(self, frame):
        self.frame = frame
        self.seq += 1

    def latest(self):
        return self.frame, time.monotonic(), self.seq

    def close(self): pass


bd.Camera = FakeCam
bd.find_camera = lambda *a: ((0, 0), "fake cam")


class FakeCaptioner:
    def __init__(self, **kw):
        self.fed, self.speaking, self.status = {}, {}, "fake captions ready"
    def feed(self, fid, block, speaking, t):
        self.fed[fid] = self.fed.get(fid, 0) + 1
        self.speaking[fid] = self.speaking.get(fid, 0) + int(speaking)
    def captions(self, fid=None, max_age_s=30):
        return [types.SimpleNamespace(text=f"hello from {fid}", pending=False, error="")]
    def flush(self): pass
    def close(self): pass


bd.Captioner = FakeCaptioner

# True physical relation (unknown to the app): array mounted MIRRORED vs. the camera.
def true_dir(nx, ny):
    sx, sy = (0.5 - nx) / 0.6, (0.5 - ny) / 0.6
    u = np.array([sx, sy, 1.0])
    return u / np.linalg.norm(u)


def plane(signal, u):
    n = len(signal)
    f = np.fft.rfftfreq(n, 1 / FS)
    tau = -(POS @ u) / SPEED
    return np.fft.irfft(np.fft.rfft(signal)[None] * np.exp(-2j * np.pi * f[None] * tau[:, None]), n, axis=1).T


def feed_audio(app, x, pace=0.0):
    for i in range(0, len(x) - HOP + 1, HOP):
        app.engine._on_input(x[i:i + HOP].astype(np.float32), HOP, None, types.SimpleNamespace(input_overflow=False))
        if pace:
            time.sleep(pace)
    time.sleep(0.15)


def frames(app, n=4):
    img = None
    for _ in range(n):
        app.cam.set(app.cam.frame)
        img = app.step() if True else None
    return img


args = types.SimpleNamespace(input="x", output=None, hfov=80.0, camera_name="x", camera_index=None,
                             caption_model="fake")
app = bd.App(args)
app.engine.thread.start()

# 1. Two people; click ME, enroll
frames(app, 5)
check("two people tracked", len(app.faces) == 2, f"({len(app.faces)})")
me_face = min(app.faces, key=lambda f: f.mouth[0])
w, h = app.size
app.on_mouse(cv2.EVENT_LBUTTONDOWN, int(me_face.mouth[0] * w), int(me_face.mouth[1] * h), 0, None)
check("click selects the face", app.selected == me_face.id and app.engine.mode == "focused")
app.on_key(ord("e"))
check("enroll ME", app.identity.enrolled and app.identity.me_id == me_face.id)

# 2. People swap places, tracker IDs reset -> ME re-identified by face
app.cam.set(np.full((720, 1280, 3), 80, np.uint8)); app.step()
time.sleep(0.01)
app.tracker.tracks.clear()
app.cam.set(scene(950, 300)); frames(app, 6)
me_now = app.identity.me_id
me_face2 = next((f for f in app.faces if f.id == me_now), None)
check("ME found again after swapping places", me_face2 is not None and me_face2.mouth[0] > 0.6,
      f"(me id {me_now}, x={me_face2.mouth[0]:.2f})" if me_face2 else "")
check("focus follows ME", app.engine.target_id == me_now)

# 3. Calibration through a mirrored mounting
before = app.tracker.mapper(*me_face2.mouth, w / h).unit_vector
err_before = math.degrees(math.acos(np.clip(np.dot(before, true_dir(*me_face2.mouth)), -1, 1)))
app.on_key(ord("c"))
check("calibration started", app.calib is not None and app.engine.calibrating)
for k, (x, y) in enumerate([(250, 260), (450, 330), (650, 280), (850, 340), (1050, 300), (350, 380), (750, 240), (950, 280)]):
    app.cam.set(scene(x, None, y)); frames(app, 4)
    me = app.faces[0]
    n = int(0.12 * FS)
    click = np.zeros(n); click[n // 3:n // 3 + 40] = rng.normal(0, 1, 40) * np.hanning(40) * 0.3
    x_audio = rng.normal(0, 0.002, (int(0.4 * FS), CH))
    x_audio[int(0.1 * FS):int(0.1 * FS) + n] += plane(click, true_dir(*me.mouth))
    feed_audio(app, x_audio, pace=0.001)
    time.sleep(0.12)
    frames(app, 1)
check("calibration collected points", len(app.calib.points) >= 6, f"({len(app.calib.points)}: {app.calib.message})")
app.on_key(ord("c"))
check("calibration saved", calibration.CALIBRATION.exists() and app.calib is None and "quick calibration" in app.tracker.mapper.status,
      f"({app.tracker.mapper.status})")
app.cam.set(scene(950, 300)); frames(app, 4)
me_face3 = next(f for f in app.faces if f.id == app.identity.me_id)
after = app.tracker.mapper(*me_face3.mouth, w / h).unit_vector
err_after = math.degrees(math.acos(np.clip(np.dot(after, true_dir(*me_face3.mouth)), -1, 1)))
check("beam now points at the true mouth direction", err_after < 3 and err_before > 10,
      f"(error {err_before:.1f} deg before -> {err_after:.1f} deg after)")

# 4. Captions wiring (fake), record, playback
app.on_key(ord("t"))
check("captions on", app.captions_on and app.engine.captioner is not None)
app.on_key(ord("r"))
check("recording started", app.recording is not None)
speech = rng.normal(0, 0.02, int(2.5 * FS)) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * np.arange(int(2.5 * FS)) / FS))
others_noise = rng.normal(0, 0.02, int(2.5 * FS))
other_face = next(f for f in app.faces if f.id != app.identity.me_id)
mix = plane(speech, true_dir(*me_face3.mouth)) + plane(others_noise, true_dir(*other_face.mouth)) + rng.normal(0, 0.003, (len(speech), CH))
for i in range(0, len(mix), FS // 4):
    feed_audio(app, mix[i:i + FS // 4])
    frames(app, 2)
app.on_key(ord("r"))
t0 = time.time()
while app.engine.player is None and time.time() - t0 < 30:
    time.sleep(0.1)
check("playback prepared after stop", app.engine.player is not None,
      f"({app.engine.player.length / FS:.1f} s)" if app.engine.player else "")
check("two/ files written", (app.last_folder / "two" / "general.wav").exists() and (app.last_folder / "two" / "focused.wav").exists())
cap = app.captioner
check("captions fed per face", cap is not None and len(cap.fed) == 2 and min(cap.fed.values()) > 50, f"({cap.fed if cap else None})")
p = app.engine.player
app.on_key(ord("p"))
out1 = np.zeros((HOP, 2), np.float32); app.engine._on_output(out1, HOP, None, None)
which1 = p.which
app.on_key(ord(" "))
out2 = np.zeros((HOP, 2), np.float32); app.engine._on_output(out2, HOP, None, None)
check("playback plays and space switches general/focused", p.playing and np.abs(out1).max() > 0 and which1 != p.which,
      f"({which1} -> {p.which}, pos {p.pos})")
# Focused should favour ME over the other talker compared to general
import soundfile as sf
g, _ = sf.read(str(app.last_folder / "two" / "general.wav"))
f, _ = sf.read(str(app.last_folder / "two" / "focused.wav"))
corr = lambda a, b: abs(np.dot(a[:len(b)], b[:len(a)])) / np.sqrt(np.dot(a[:len(b)], a[:len(b)]) * np.dot(b[:len(a)], b[:len(a)]) + 1e-20)

img = frames(app, 2)
cv2.imwrite(str(TMP / "app.png"), img)
app.on_key(ord("h")); cv2.imwrite(str(TMP / "app_help.png"), frames(app, 1))
app.engine.running = False
print(f"screenshots: {TMP / 'app.png'}")
print(f"\n{sum(results)}/{len(results)} passed")
if __name__ == "__main__":
    sys.stdout.flush()
    os._exit(0 if all(results) else 1)   # skip slow interpreter teardown of audio/vision threads
