"""Live face / lip / head-pose / direction demo for the icspring camera.

Run from File Explorer via "Start Vision Demo.cmd" (the coding session cannot open
the camera). Close the wizard/basic demo first if they hold the camera.

Keys (click the window first):
  [ / ]   lower / raise lip-activity threshold   (more / less sensitive)
  - / +   adjust assumed camera horizontal FOV by 5 degrees
  l       toggle lip outline         m   mirror display (overlay only)
  Esc/q   quit

Writes vision/ika-vision-status.json (~10 Hz, atomic replace) unless --no-json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from face_tracker import LIPS_OUTLINE, FaceTracker, Settings, TrackedFace

HERE = Path(__file__).parent
STATUS = HERE / "ika-vision-status.json"
COLORS = [(80, 200, 255), (120, 255, 120), (255, 160, 80), (220, 120, 255), (90, 220, 220)]


class Camera:
    """Background grabber that always hands out the newest frame (no queue lag)."""

    def __init__(self, source, width: int, height: int):
        self.cap = cv2.VideoCapture(*source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {source}. Close other apps using it and check "
                               "Windows camera privacy settings for desktop apps.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame, self.t, self.seq, self.error = None, 0.0, 0, ""
        self.lock, self.running = threading.Lock(), True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.error = "camera read failed"
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame, self.t, self.error = frame, time.monotonic(), ""
                self.seq += 1

    def latest(self):
        with self.lock:
            return self.frame, self.t, self.seq

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def find_camera(name_part: str, index: int | None):
    if index is not None:
        return (index, cv2.CAP_MSMF), f"index {index}"
    from cv2_enumerate_cameras import enumerate_cameras
    cams = list(enumerate_cameras(cv2.CAP_MSMF))
    for c in cams:
        if name_part.lower() in c.name.lower():
            return (c.index, cv2.CAP_MSMF), c.name
    names = ", ".join(f"{c.index}:{c.name}" for c in cams) or "none"
    raise RuntimeError(f"No camera containing '{name_part}' (found: {names}). "
                       "Use --camera-index N to pick one explicitly.")


def write_status(faces: list[TrackedFace], tracker: FaceTracker, frame_seq: int, fresh: bool,
                 fps: float, size, error: str):
    data = {
        "schema_version": 1,
        "timestamp_ms": int(time.time() * 1000),
        "monotonic_ms": int(time.monotonic() * 1000),
        "process_id": os.getpid(),
        "frame_sequence": frame_seq,
        "camera_fresh": fresh,
        "camera_error": error,
        "fps": round(fps, 1),
        "frame_size": list(size),
        "direction_source": tracker.mapper.status,
        "camera_hfov_deg": tracker.settings.camera_hfov_deg,
        "faces": [f.to_json() for f in faces] if fresh else [],
    }
    # Windows refuses the replace while any reader (antivirus, indexer, editor,
    # a consumer polling the file) has it open. Retry briefly, then skip this
    # snapshot: the next one follows in ~100 ms. Never take the live view down.
    tmp = STATUS.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(tmp, STATUS)
                return True
            except PermissionError:
                time.sleep(0.002 * (attempt + 1))
    except OSError:
        pass
    return False


def draw(frame: np.ndarray, faces: list[TrackedFace], tracker: FaceTracker, fps: float,
         show_lips: bool, mirror: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    out = cv2.flip(frame, 1) if mirror else frame.copy()
    fx = (lambda x: 1 - x) if mirror else (lambda x: x)
    k = max(w / 1280, 0.9)
    font, s = cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k
    th = max(1, int(round(1.5 * k)))
    st = tracker.settings

    for f in faces:
        col = COLORS[(f.id - 1) % len(COLORS)]
        x0, x1 = sorted((fx(f.bbox[0]) * w, fx(f.bbox[2]) * w))
        y0, y1 = f.bbox[1] * h, f.bbox[3] * h
        p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
        cv2.rectangle(out, p0, p1, (0, 230, 0) if f.addressing_camera_visual else col,
                      3 * th if f.addressing_camera_visual else th)

        if show_lips:
            pts = np.array([(fx(f.landmarks[i, 0]) * w, f.landmarks[i, 1] * h) for i in LIPS_OUTLINE], np.int32)
            cv2.polylines(out, [pts], True, (0, 0, 255) if f.mouth_moving else (200, 200, 200), th)

        # Head forward arrow from the nose tip.
        nose = (fx(f.landmarks[1, 0]) * w, f.landmarks[1, 1] * h)
        length = (x1 - x0) * 0.8
        dx = math.sin(math.radians(f.yaw_deg)) * length * (-1 if mirror else 1)
        dy = -math.sin(math.radians(f.pitch_deg)) * length
        cv2.arrowedLine(out, (int(nose[0]), int(nose[1])), (int(nose[0] + dx), int(nose[1] + dy)),
                        (0, 255, 255) if f.looking_at_camera else (0, 120, 255), th + 1, tipLength=0.25)

        # Lip activity bar with on/off thresholds.
        bx, by, bw, bh = p0[0], p1[1] + int(8 * k), max(p1[0] - p0[0], int(120 * k)), int(10 * k)
        full = 0.1
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        cv2.rectangle(out, (bx, by), (bx + int(bw * min(f.lip_activity / full, 1)), by + bh),
                      (0, 0, 255) if f.mouth_moving else (180, 180, 180), -1)
        for thr in (st.activity_off, st.activity_on):
            xx = bx + int(bw * thr / full)
            cv2.line(out, (xx, by - 2), (xx, by + bh + 2), (255, 255, 255), 1)

        d = f.direction
        lines = [
            f"#{f.id}  {'SPEAKING TO CAMERA' if f.addressing_camera_visual else ('lips moving' if f.mouth_moving else 'still')}",
            f"{'LOOKING' if f.looking_at_camera else 'away'}  off-axis {f.off_axis_deg:4.0f}  yaw {f.yaw_deg:+4.0f} pitch {f.pitch_deg:+4.0f}",
            f"az {d.azimuth_deg:+5.1f}  el {d.elevation_deg:+5.1f}  dist {f.distance_m or 0:.1f} m" if d else "",
            f"lip {f.lip_activity:.3f}  open {f.mouth_open:.2f}  jaw {f.jaw_open:.2f}",
        ]
        y = by + bh + int(20 * k)
        for line in lines:
            if line:
                cv2.putText(out, line, (bx, y), font, s, (0, 0, 0), th + 2, cv2.LINE_AA)
                cv2.putText(out, line, (bx, y), font, s, col, th, cv2.LINE_AA)
                y += int(20 * k)

    hud = (f"{fps:4.1f} fps  faces {len(faces)}  lip thr {st.activity_on:.3f}  "
           f"hfov {st.camera_hfov_deg:.0f}  dir: {tracker.mapper.status}")
    cv2.putText(out, hud, (10, int(24 * k)), font, s, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(out, hud, (10, int(24 * k)), font, s, (255, 255, 255), th, cv2.LINE_AA)
    return out


def run_live(args):
    settings = Settings(max_faces=args.max_faces, camera_hfov_deg=args.hfov)
    source, name = find_camera(args.camera_name, args.camera_index)
    cam = Camera(source, args.width, args.height)
    tracker = FaceTracker(settings)
    print(f"Camera: {name}. {tracker.mapper.status}")
    cv2.namedWindow("Ika vision", cv2.WINDOW_NORMAL)
    show_lips, mirror = True, False
    last_seq, last_json, fps, faces, size = -1, 0.0, 0.0, [], (0, 0)
    skipped = 0
    try:
        while True:
            frame, t, seq = cam.latest()
            fresh = frame is not None and time.monotonic() - t < 1.0
            if frame is not None and seq != last_seq:
                t0 = time.monotonic()
                faces = tracker.process(frame, t)
                dt = time.monotonic() - t0
                fps = 0.9 * fps + 0.1 / max(dt, 1e-3) if fps else 1 / max(dt, 1e-3)
                last_seq, size = seq, (frame.shape[1], frame.shape[0])
                cv2.imshow("Ika vision", draw(frame, faces, tracker, fps, show_lips, mirror))
            if not args.no_json and time.monotonic() - last_json > 0.1:
                if write_status(faces, tracker, seq, fresh, fps, size, cam.error):
                    skipped = 0
                else:
                    skipped += 1
                    if skipped % 50 == 1:
                        print(f"status file locked by another program; skipped {skipped} update(s), still running")
                last_json = time.monotonic()
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty("Ika vision", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("["), ord("]")):
                f = 0.85 if key == ord("[") else 1 / 0.85
                settings.activity_on *= f
                settings.activity_off *= f
            elif key in (ord("-"), ord("+"), ord("=")):
                settings.camera_hfov_deg = min(150, max(30, settings.camera_hfov_deg + (-5 if key == ord("-") else 5)))
            elif key == ord("l"):
                show_lips = not show_lips
            elif key == ord("m"):
                mirror = not mirror
    finally:
        if not args.no_json:
            write_status([], tracker, last_seq, False, 0, size, "stopped")
        cam.close()
        tracker.close()
        cv2.destroyAllWindows()


def run_file(args):
    """Offline: annotate an image or video file (for tuning without the camera)."""
    settings = Settings(max_faces=args.max_faces, camera_hfov_deg=args.hfov)
    src = Path(args.input)
    if src.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        tracker = FaceTracker(settings)
        frame = cv2.imread(str(src))
        faces = tracker.process(frame, 0.0)
        out = Path(args.output or src.with_name(src.stem + "-annotated.png"))
        cv2.imwrite(str(out), draw(frame, faces, tracker, 0, True, False))
        print(json.dumps([f.to_json() for f in faces], indent=1))
        print(f"wrote {out}")
        return
    tracker = FaceTracker(settings)
    cap = cv2.VideoCapture(str(src))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    writer, i = None, 0
    out = Path(args.output or src.with_name(src.stem + "-annotated.mp4"))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = tracker.process(frame, i / fps_in)
        img = draw(frame, faces, tracker, fps_in, True, False)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, img.shape[1::-1])
        writer.write(img)
        i += 1
    if writer:
        writer.release()
    print(f"wrote {out} ({i} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-name", default="icspring")
    ap.add_argument("--camera-index", type=int)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=80.0, help="camera horizontal field of view, degrees")
    ap.add_argument("--max-faces", type=int, default=3)
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--input", help="image or video file instead of the live camera")
    ap.add_argument("--output")
    args = ap.parse_args()
    run_file(args) if args.input else run_live(args)


if __name__ == "__main__":
    main()
