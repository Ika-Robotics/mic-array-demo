"""Face detection, lip activity, head pose and array direction for up to N faces.

No camera or UI code here: feed BGR frames + timestamps into FaceTracker.process()
and read back a list of TrackedFace. The audio pipeline can import this directly.

Signals per face
  - mouth_open:      inner-lip gap / inter-ocular distance (3D, yaw-robust)
  - lip_activity:    std-dev of mouth_open over a short window (speech = oscillation;
                     a static open mouth, smile or yawn stays low)
  - mouth_moving:    lip_activity with hysteresis + hold, i.e. "visually speaking"
  - head pose:       yaw/pitch/roll and off_axis_deg = angle between the face's
                     forward axis and the line from the face to the camera
                     (corrects for faces at the edge of the frame)
  - looking_at_camera: off_axis_deg below threshold, with hysteresis
  - array direction: mouth position projected to an array ray, using the
                     wizard's setup-profile.ini affine map when available,
                     otherwise an assumed pinhole field of view
"""
from __future__ import annotations

import math
import shlex
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).with_name("models") / "face_landmarker.task"
DETECTOR = Path(__file__).with_name("models") / "face_detection_yunet_2023mar.onnx"
PROFILE = Path(__file__).resolve().parents[1] / "setup-profile.ini"

# MediaPipe face mesh indices
LIP_TOP, LIP_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
EYE_LEFT, EYE_RIGHT = 33, 263
EYE_CORNER_SPAN_M = 0.09   # typical adult outer eye-corner distance, for range
LIPS_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                409, 270, 269, 267, 0, 37, 39, 40, 185]


@dataclass
class Settings:
    max_faces: int = 3
    activity_window_s: float = 0.6
    activity_on: float = 0.035     # lip_activity to start "mouth_moving"
    activity_off: float = 0.020    # ...and to stop it
    moving_hold_s: float = 0.35    # bridge short pauses between syllables
    look_on_deg: float = 20.0      # off_axis below this -> looking
    look_off_deg: float = 28.0     # off_axis above this -> not looking
    track_max_jump: float = 0.15   # normalized image distance for ID matching
    track_timeout_s: float = 2.5   # keep a lost person's ID this long (turning away, occlusion)
    confirm_frames: int = 3        # frames before a new face counts (drops one-frame false detections)
    detect_every: int = 4          # full-frame detector period; crops follow landmarks between
    camera_hfov_deg: float = 80.0  # icspring lens; also the array fallback without a profile


@dataclass
class Direction:
    unit_vector: tuple[float, float, float]
    azimuth_deg: float     # + = right in the camera image
    elevation_deg: float   # + = up
    source: str            # "setup-profile" | "assumed-fov"


@dataclass
class TrackedFace:
    id: int
    bbox: tuple[float, float, float, float]   # normalized x0,y0,x1,y1
    mouth: tuple[float, float]                # normalized image point
    landmarks: np.ndarray                     # (478,3) normalized x,y,z
    mouth_open: float = 0.0
    jaw_open: float = 0.0                     # blendshape 0..1
    lip_activity: float = 0.0
    mouth_moving: bool = False
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    off_axis_deg: float = 90.0
    distance_m: float | None = None
    looking_at_camera: bool = False
    direction: Direction | None = None
    confirmed: bool = True

    @property
    def addressing_camera_visual(self) -> bool:
        """Looking at the camera and lips moving. Combine with audio VAD in the beam."""
        return self.looking_at_camera and self.mouth_moving

    def to_json(self) -> dict:
        r = lambda v, n=3: None if v is None else round(float(v), n)
        d = self.direction
        return {
            "id": self.id,
            "bbox_norm": [r(v) for v in self.bbox],
            "mouth_norm": [r(v) for v in self.mouth],
            "mouth_open": r(self.mouth_open), "jaw_open": r(self.jaw_open),
            "lip_activity": r(self.lip_activity, 4), "mouth_moving": self.mouth_moving,
            "head_pose": {"yaw_deg": r(self.yaw_deg, 1), "pitch_deg": r(self.pitch_deg, 1),
                          "roll_deg": r(self.roll_deg, 1), "off_axis_deg": r(self.off_axis_deg, 1)},
            "looking_at_camera": self.looking_at_camera,
            "addressing_camera_visual": self.addressing_camera_visual,
            "distance_m_est": r(self.distance_m, 2),
            "array_direction": None if d is None else {
                "unit_vector": [r(v, 4) for v in d.unit_vector],
                "azimuth_deg": r(d.azimuth_deg, 1), "elevation_deg": r(d.elevation_deg, 1),
                "source": d.source},
        }


class DirectionMapper:
    """Normalized image point -> array ray, mirroring Projection::unproject in setup_model.h."""

    def __init__(self, settings: Settings, profile_path: Path = PROFILE):
        self.settings = settings
        self.affine = None
        self.profile_size = None
        # Without a profile the array is assumed to look where the camera looks; these
        # flips fix a mirrored/upside-down mounting. The profile already encodes both.
        self.flip_x = self.flip_y = False
        self.status = "no setup-profile.ini; using assumed field of view"
        if profile_path.exists():
            try:
                self.affine, self.profile_size = load_profile(profile_path)
                self.status = f"setup-profile.ini ({self.profile_size[0]}x{self.profile_size[1]})"
            except ValueError as e:
                self.status = f"setup-profile.ini ignored: {e}"
        # Quick acoustic calibration (calibration.py) wins over the wizard profile
        from calibration import load_calibration
        affine, info = load_calibration()
        if affine is not None:
            self.set_calibration(affine, info)

    def set_calibration(self, affine, info=None):
        self.affine = list(affine)
        when = (info or {}).get("created", "now")
        self.status = f"quick calibration ({when}, error {(info or {}).get('rms', 0):.3f})"

    def __call__(self, nx: float, ny: float, aspect: float) -> Direction:
        if self.affine is not None:
            a = self.affine
            det = a[0] * a[4] - a[1] * a[3]
            sx = (a[4] * (nx - a[2]) - a[1] * (ny - a[5])) / det
            sy = (-a[3] * (nx - a[2]) + a[0] * (ny - a[5])) / det
            source = "setup-profile"
        else:
            sx, sy = camera_slopes(nx, ny, aspect, self.settings.camera_hfov_deg)
            sx, sy = (-sx if self.flip_x else sx), (-sy if self.flip_y else sy)
            source = "assumed-fov"
        n = math.sqrt(sx * sx + sy * sy + 1)
        return Direction((sx / n, sy / n, 1 / n), math.degrees(math.atan(sx)),
                         math.degrees(math.atan2(sy, math.sqrt(1 + sx * sx))), source)


    def project(self, sx: float, sy: float, aspect: float) -> tuple[float, float]:
        """Array ray slopes -> normalized image point (inverse of __call__)."""
        if self.affine is not None:
            a = self.affine
            return a[0] * sx + a[1] * sy + a[2], a[3] * sx + a[4] * sy + a[5]
        sx, sy = (-sx if self.flip_x else sx), (-sy if self.flip_y else sy)
        t = math.tan(math.radians(self.settings.camera_hfov_deg) / 2)
        return 0.5 + sx / (2 * t), 0.5 - sy * aspect / (2 * t)


def camera_slopes(nx: float, ny: float, aspect: float, hfov_deg: float) -> tuple[float, float]:
    """Pinhole ray slopes (x right, y up) for a normalized image point."""
    t = math.tan(math.radians(hfov_deg) / 2)
    return (nx - 0.5) * 2 * t, -(ny - 0.5) * 2 * t / aspect


def load_profile(path: Path):
    """Parse the wizard's IKA_MIC_SETUP 1 format (see setup_model.h serialize)."""
    tokens = shlex.split(path.read_text(encoding="utf-8"), posix=True)
    if len(tokens) != 2 + 2 + 2 + 5 + 16 + 7 + 4 or tokens[:2] != ["IKA_MIC_SETUP", "1"]:
        raise ValueError("unexpected format")
    width, height = int(tokens[4]), int(tokens[5])
    affine = [float(v) for v in tokens[27:33]]
    det = affine[0] * affine[4] - affine[1] * affine[3]
    if not all(math.isfinite(v) for v in affine) or abs(det) < 0.005:
        raise ValueError("degenerate projection")
    return affine, (width, height)


class _Track:
    def __init__(self, tid: int, center, t: float):
        self.id, self.center, self.last_seen = tid, center, t
        self.hits = 0
        self.history: deque[tuple[float, float]] = deque()
        self.moving, self.moving_until = False, -1.0
        self.looking = False


class FaceTracker:
    """Two stages: YuNet finds faces in the full frame (works down to ~25 px wide),
    then FaceLandmarker runs on a square crop around each face. Running the
    landmarker on the whole frame misses faces narrower than ~130 px at 1280x720,
    i.e. anyone beyond ~1 m with a wide-angle camera."""

    CROP = 256          # landmarker input size
    CROP_SCALE = 2.0    # crop side relative to the detected face box

    def __init__(self, settings: Settings | None = None, model_path: Path = MODEL,
                 detector_path: Path = DETECTOR):
        self.settings = settings or Settings()
        self.mapper = DirectionMapper(self.settings)
        self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.6, 0.3, 50)
        self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.5,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))
        self.tracks: list[_Track] = []
        self.next_id = 1
        self._frame = 0
        self._follow: list[tuple[float, float, float, float, float]] = []

    def close(self):
        self.landmarker.close()

    def detect_boxes(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Face boxes (x, y, w, h, score) in frame pixels, best first, at most max_faces."""
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, 960 / w)
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale) if scale < 1 else frame_bgr
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, found = self.detector.detect(small)
        if found is None:
            return []
        boxes = [(*(found[i, :4] / scale), float(found[i, 14])) for i in range(len(found))]
        return sorted(boxes, key=lambda b: -b[4])[: self.settings.max_faces]

    def process(self, frame_bgr: np.ndarray, t: float) -> list[TrackedFace]:
        """t = capture time in seconds (monotonic). Returns faces sorted left to right."""
        h, w = frame_bgr.shape[:2]
        f_px = (w / 2) / math.tan(math.radians(self.settings.camera_hfov_deg) / 2)
        redetect = self._frame % self.settings.detect_every == 0 or not self._follow
        boxes = self.detect_boxes(frame_bgr) if redetect else self._follow
        self._frame += 1
        faces, follow = [], []
        for bx, by, bw, bh, _ in boxes:
            side = self.CROP_SCALE * max(bw, bh)
            x0, y0 = bx + bw / 2 - side / 2, by + bh / 2 - side / 2
            k = self.CROP / side
            warp = np.float32([[k, 0, -x0 * k], [0, k, -y0 * k]])
            crop = cv2.warpAffine(frame_bgr, warp, (self.CROP, self.CROP), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop[:, :, ::-1]))
            result = self.landmarker.detect(image)
            if not result.face_landmarks:
                continue
            c = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float64)
            pts = np.column_stack(((x0 + c[:, 0] * side) / w, (y0 + c[:, 1] * side) / h, c[:, 2] * side / w))
            cx, cy = pts[:, 0].mean() * w, pts[:, 1].mean() * h
            if not (bx <= cx <= bx + bw and by <= cy <= by + bh):
                continue  # landmarker locked onto a neighbour inside the crop
            face = TrackedFace(id=0, bbox=(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()),
                               mouth=tuple((pts[LIP_TOP, :2] + pts[LIP_BOTTOM, :2]) / 2), landmarks=pts)
            px = pts * np.array([w, h, w])  # z uses the same scale as x
            eye_span = np.linalg.norm(px[EYE_LEFT] - px[EYE_RIGHT])
            face.mouth_open = float(np.linalg.norm(px[LIP_TOP] - px[LIP_BOTTOM]) / max(eye_span, 1e-6))
            face.distance_m = EYE_CORNER_SPAN_M * f_px / max(eye_span, 1e-6)
            if result.face_blendshapes:
                face.jaw_open = next((b.score for b in result.face_blendshapes[0] if b.category_name == "jawOpen"), 0.0)
            if result.facial_transformation_matrixes:
                sx, sy = camera_slopes(face.mouth[0], face.mouth[1], w / h, self.settings.camera_hfov_deg)
                _head_pose(face, np.asarray(result.facial_transformation_matrixes[0]), sx, sy)
            face.direction = self.mapper(face.mouth[0], face.mouth[1], w / h)
            faces.append(face)
            lx0, ly0, lx1, ly1 = face.bbox
            follow.append((lx0 * w, ly0 * h, (lx1 - lx0) * w, (ly1 - ly0) * h, 1.0))

        # Lost a face while following -> force a detector pass next frame.
        self._follow = follow if len(follow) == len(boxes) else []
        self._associate(faces, t)
        return sorted(faces, key=lambda f: f.mouth[0])

    def _associate(self, faces: list[TrackedFace], t: float):
        s = self.settings
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= s.track_timeout_s]
        pairs = sorted(((math.dist(f.mouth, tr.center), fi, ti)
                        for fi, f in enumerate(faces) for ti, tr in enumerate(self.tracks)))
        used_f, used_t = set(), set()
        assigned: dict[int, _Track] = {}
        for d, fi, ti in pairs:
            if d > s.track_max_jump or fi in used_f or ti in used_t:
                continue
            used_f.add(fi); used_t.add(ti); assigned[fi] = self.tracks[ti]
        for fi, f in enumerate(faces):
            tr = assigned.get(fi)
            if tr is None:
                tr = _Track(self.next_id, f.mouth, t)
                self.next_id += 1
                self.tracks.append(tr)
            tr.center, tr.last_seen = f.mouth, t
            f.id = tr.id
            tr.hits += 1
            f.confirmed = tr.hits >= self.settings.confirm_frames
            self._update_signals(tr, f, t)

    def _update_signals(self, tr: _Track, f: TrackedFace, t: float):
        s = self.settings
        tr.history.append((t, f.mouth_open))
        while tr.history and t - tr.history[0][0] > s.activity_window_s:
            tr.history.popleft()
        values = np.array([v for _, v in tr.history])
        f.lip_activity = float(values.std()) if len(values) >= 4 else 0.0

        if f.lip_activity >= s.activity_on:
            tr.moving, tr.moving_until = True, t + s.moving_hold_s
        elif f.lip_activity >= s.activity_off and tr.moving:
            tr.moving_until = t + s.moving_hold_s
        elif t > tr.moving_until:
            tr.moving = False
        f.mouth_moving = tr.moving

        if f.off_axis_deg <= s.look_on_deg:
            tr.looking = True
        elif f.off_axis_deg >= s.look_off_deg:
            tr.looking = False
        f.looking_at_camera = tr.looking


def _head_pose(face: TrackedFace, m: np.ndarray, sx: float, sy: float):
    """MediaPipe metric space: x right, y up, z towards the viewer; camera at origin.

    The face->camera line comes from our camera FOV (sx, sy), not from MediaPipe's
    translation, which assumes a fixed 63 degree vertical FOV."""
    r = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
    forward = r @ np.array([0.0, 0.0, 1.0])
    face.yaw_deg = math.degrees(math.atan2(forward[0], forward[2]))
    face.pitch_deg = math.degrees(math.atan2(forward[1], math.hypot(forward[0], forward[2])))
    up = r @ np.array([0.0, 1.0, 0.0])
    face.roll_deg = math.degrees(math.atan2(-up[0], up[1]))
    to_camera = -np.array([sx, sy, -1.0]) / math.sqrt(sx * sx + sy * sy + 1)
    face.off_axis_deg = math.degrees(math.acos(float(np.clip(forward @ to_camera, -1, 1))))
