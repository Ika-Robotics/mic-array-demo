"""'Me' re-identification: enroll a face once, then find it again among tracked faces.

SFace (OpenCV zoo) face embeddings, aligned with the MediaPipe landmarks we already
have. Glasses/caps are fine: the embedding uses eyes/nose/mouth geometry and texture.
The enrolled gallery persists in me.npz so it survives restarts.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).with_name("models") / "face_recognition_sface_2021dec.onnx"
GALLERY = Path(__file__).with_name("me.npz")
MATCH = 0.40          # cosine; OpenCV's suggested same-person threshold is 0.363
SWITCH_MARGIN = 0.08  # another face must beat the current 'me' by this much to take over
MAX_GALLERY = 12


def yunet_row(f, w: int, h: int) -> np.ndarray:
    """YuNet-style 15-value row from MediaPipe landmarks, for FaceRecognizerSF.alignCrop.
    Order: box, right eye (image left), left eye, nose tip, right/left mouth corners, score."""
    lm = f.landmarks[:, :2] * (w, h)
    x0, y0, x1, y1 = f.bbox[0] * w, f.bbox[1] * h, f.bbox[2] * w, f.bbox[3] * h
    pts = [lm[[33, 133]].mean(0), lm[[362, 263]].mean(0), lm[1], lm[61], lm[291]]
    return np.array([x0, y0, x1 - x0, y1 - y0, *np.concatenate(pts), 1.0], np.float32)


class Identity:
    def __init__(self, gallery_path: Path = GALLERY):
        self.rec = cv2.FaceRecognizerSF.create(str(MODEL), "")
        self.path = gallery_path
        self.gallery: list[np.ndarray] = []
        if gallery_path.exists():
            try:
                self.gallery = list(np.load(gallery_path)["emb"])
            except Exception:
                self.gallery = []
        self.me_id: int | None = None
        self.scores: dict[int, float] = {}

    @property
    def enrolled(self) -> bool:
        return bool(self.gallery)

    def embed(self, frame, f) -> np.ndarray:
        h, w = frame.shape[:2]
        crop = self.rec.alignCrop(frame, yunet_row(f, w, h))
        e = self.rec.feature(crop).ravel()
        return e / max(np.linalg.norm(e), 1e-9)

    def enroll(self, frame, f, reset: bool = True):
        e = self.embed(frame, f)
        if reset:
            self.gallery = []
        self.gallery.append(e)
        self.gallery = self.gallery[-MAX_GALLERY:]
        self.me_id = f.id
        self._save()

    def forget(self):
        self.gallery, self.me_id, self.scores = [], None, {}
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _save(self):
        try:
            np.savez(self.path, emb=np.array(self.gallery))
        except OSError:
            pass

    def score(self, e: np.ndarray) -> float:
        return float(max(g @ e for g in self.gallery)) if self.gallery else 0.0

    def update(self, frame, faces, run_embeddings: bool) -> int | None:
        """Returns the tracker ID that is 'me' (or None). Embeddings only when run_embeddings
        (every few frames) or when the current 'me' is no longer visible."""
        if not self.gallery:
            self.me_id = None
            return None
        ids = {f.id for f in faces}
        if self.me_id not in ids:
            self.me_id = None
            run_embeddings = True
        if run_embeddings and faces:
            self.scores = {f.id: self.score(self.embed(frame, f)) for f in faces}
            best = max(self.scores, key=self.scores.get)
            current = self.scores.get(self.me_id, 0.0) if self.me_id is not None else 0.0
            if self.scores[best] >= MATCH and (self.me_id is None or self.scores[best] > current + SWITCH_MARGIN):
                self.me_id = best
            elif self.me_id is not None and current < MATCH - 0.1:
                self.me_id = None   # what we were following clearly isn't me
            # Grow the gallery with confident new views (different pose/light), slowly
            if self.me_id is not None and MATCH + 0.12 < self.scores[self.me_id] < 0.75 and len(self.gallery) < MAX_GALLERY:
                f = next(f for f in faces if f.id == self.me_id)
                self.gallery.append(self.embed(frame, f))
                self._save()
        return self.me_id
