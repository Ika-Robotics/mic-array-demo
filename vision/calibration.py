"""Quick acoustic calibration: mouth clicks / whistles / claps at a known face.

The face tracker knows where the mouth is in the image; each short loud sound gives
an acoustic direction (array ray slopes). Pairs of (slopes -> image point) are fitted
with a similarity transform (scale, rotation, optional mirror, offset), robust to
outliers. Result is the same affine form as the wizard (setup_model.h Projection), so
DirectionMapper uses it directly. Saved in beam-calibration.json (wizard file untouched).
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beamformer import FS, HOP, SPEED, Beamformer

CALIBRATION = Path(__file__).with_name("beam-calibration.json")
EVENT_DB = 12.0          # sound must be this far above the background
EVENT_HOPS = 6           # ~64 ms of audio per event
MIN_GAP_S = 0.15
BAND = (300.0, 4000.0)   # above ~4 kHz the 42 mm grid aliases


@dataclass
class AcousticEvent:
    t: float
    slopes: tuple[float, float]
    contrast: float
    bins: int


class EventDetector:
    """Audio thread side (cheap): spots transients, hands spectra to a worker for SRP."""

    def __init__(self, bf: Beamformer):
        self.bf = bf
        self.band = np.where((bf.freqs >= BAND[0]) & (bf.freqs <= BAND[1]))[0]
        self.floor = None                  # band energy background
        self.noise_psd = None              # per-bin background (for the SNR mask)
        self.active: list[np.ndarray] = []
        self.t0 = 0.0
        self.last_event = -1.0
        self.jobs: queue.Queue = queue.Queue(maxsize=8)
        self.results: queue.Queue = queue.Queue()
        self.level_db = -120.0
        self.floor_db = -120.0
        self._fine = None
        threading.Thread(target=self._worker, daemon=True).start()
        coarse = np.stack(np.meshgrid(np.linspace(-1.3, 1.3, 27), np.linspace(-0.9, 0.9, 19)), -1).reshape(-1, 2)
        self._coarse = coarse
        self._coarse_tau = self._taus(coarse)

    def _taus(self, slopes):
        rays = np.column_stack((slopes, np.ones(len(slopes))))
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return -(rays @ self.bf.pos.T) / SPEED                         # (G,16)

    def feed(self, spectra: np.ndarray, t: float):
        band = spectra[:, self.band]
        psd = (np.abs(band) ** 2).mean(axis=0)                         # (Fb,)
        e = float(psd.sum())
        if self.floor is None:
            self.floor, self.noise_psd = e, psd.copy()
        rise = 10 ** (1.0 / 10 / (FS / HOP))                           # floor rises 1 dB/s
        self.floor = min(self.floor * rise, 0.9 * self.floor + 0.1 * e) if not self.active else self.floor
        if not self.active:
            self.noise_psd = np.minimum(self.noise_psd * rise, 0.9 * self.noise_psd + 0.1 * psd)
        self.level_db = 10 * math.log10(e + 1e-20)
        self.floor_db = 10 * math.log10(self.floor + 1e-20)
        loud = self.level_db > self.floor_db + EVENT_DB
        if loud and not self.active and t - self.last_event > MIN_GAP_S:
            self.active, self.t0 = [band], t
        elif self.active:
            self.active.append(band)
            if len(self.active) >= EVENT_HOPS or not loud:
                try:
                    self.jobs.put_nowait((self.t0, np.array(self.active), self.noise_psd.copy()))
                except queue.Full:
                    pass
                self.active, self.last_event = [], t

    def _worker(self):
        f = self.bf.freqs[self.band]
        while True:
            t, specs, noise = self.jobs.get()
            # SNR mask: only bins where the event clearly stands out (whistle = few bins, click = many)
            pw = (np.abs(specs) ** 2).mean(axis=1)                     # (hops, Fb)
            mask = pw > 6.0 * noise[None]
            nb = int(mask.any(axis=0).sum())
            if nb < 2:       # a pure whistle can sit in just 2-3 bins; phase still gives direction
                continue
            x = specs / np.maximum(np.abs(specs), 1e-12)                # PHAT
            x = x * mask[:, None, :]

            def srp(tau):
                st = np.exp(-2j * np.pi * f[None, :, None] * tau[:, None, :])     # (G,Fb,16)
                y = np.einsum("gfm,hmf->ghf", st.conj(), x)
                return (np.abs(y) ** 2).sum(axis=(1, 2))

            p = srp(self._coarse_tau)
            i = int(p.argmax())
            contrast = float((p[i] - p.mean()) / max(p.std(), 1e-12))
            c = self._coarse[i]
            fine = np.stack(np.meshgrid(np.linspace(c[0] - 0.12, c[0] + 0.12, 25),
                                        np.linspace(c[1] - 0.12, c[1] + 0.12, 25)), -1).reshape(-1, 2)
            pf = srp(self._taus(fine))
            best = fine[int(pf.argmax())]
            self.results.put(AcousticEvent(t, (float(best[0]), float(best[1])), contrast, nb))


def fit_similarity(points: np.ndarray, aspect: float):
    """points: (N,4) = sx, sy, nx, ny. Returns (affine[6], rms residual in normalized image
    units, inlier mask). Fits w = a*z + b and w = a*conj(z) + b (mirrored), keeps the better."""
    z = points[:, 0] + 1j * points[:, 1]
    w = points[:, 2] * aspect + 1j * points[:, 3]      # isotropic image units

    def solve(zz, keep):
        A = np.column_stack((zz[keep], np.ones(keep.sum())))
        (a, b), *_ = np.linalg.lstsq(A, w[keep], rcond=None)
        return a, b, np.abs(a * zz + b - w)

    best = None
    for mirrored in (False, True):
        zz = np.conj(z) if mirrored else z
        keep = np.ones(len(z), bool)
        for _ in range(3):
            a, b, r = solve(zz, keep)
            thr = max(2.5 * np.median(r[keep]), 0.02 * aspect)
            new = r < thr
            if new.sum() < 4 or (new == keep).all():
                break
            keep = new
        rms = float(np.sqrt(np.mean(r[keep] ** 2)))
        if best is None or rms < best[0]:
            best = (rms, mirrored, a, b, keep)
    rms, mirrored, a, b, keep = best
    ar, ai, br, bi = a.real, a.imag, b.real, b.imag
    s = -1 if mirrored else 1
    # nx*aspect = ar*sx - s*ai*sy + br ; ny = ai*sx + s*ar*sy + bi
    affine = [ar / aspect, -s * ai / aspect, br / aspect, ai, s * ar, bi]
    return affine, rms / aspect, keep


class CalibrationSession:
    MIN_POINTS = 6
    MAX_RMS = 0.06       # normalized image units
    MIN_SPREAD = 0.15    # image fraction the mouth must cover (largest axis)

    def __init__(self):
        self.points: list[tuple[float, float, float, float]] = []
        self.rejected = 0
        self.started = time.monotonic()
        self.fit = None
        self.message = "Make sharp clicks/whistles at your mouth. Move your head left, right, up, down."

    def add(self, ev: AcousticEvent, mouth: tuple[float, float] | None, aspect: float):
        if mouth is None or ev.contrast < 1.5:   # narrowband whistles score ~1.7-2.7; the robust fit drops bad ones
            self.rejected += 1
            return
        self.points.append((ev.slopes[0], ev.slopes[1], mouth[0], mouth[1]))
        self._refit(aspect)

    def spread(self) -> float:
        if not self.points:
            return 0.0
        p = np.array(self.points)
        return float(max(np.ptp(p[:, 2]), np.ptp(p[:, 3])))

    def _refit(self, aspect):
        n, sp = len(self.points), self.spread()
        if n < self.MIN_POINTS:
            self.message = f"{n}/{self.MIN_POINTS} sounds. Keep clicking while moving your head."
            return
        affine, rms, keep = fit_similarity(np.array(self.points), aspect)
        self.fit = (affine, rms, keep)
        if sp < self.MIN_SPREAD:
            self.message = f"Move further across the image (spread {sp:.0%}, need {self.MIN_SPREAD:.0%})."
        elif rms > self.MAX_RMS:
            self.message = f"Inconsistent (error {rms:.3f}). Keep going; outliers are dropped."
        else:
            self.message = f"Good: {keep.sum()}/{n} points, error {rms:.3f}. Press c to save."

    @property
    def good(self) -> bool:
        return (self.fit is not None and len(self.points) >= self.MIN_POINTS
                and self.spread() >= self.MIN_SPREAD and self.fit[1] <= self.MAX_RMS)

    def save(self, camera: str, size) -> dict:
        affine, rms, keep = self.fit
        data = {"version": 1, "affine": affine, "rms": rms, "points": int(keep.sum()),
                "camera": camera, "frame_size": list(size), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = CALIBRATION.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(CALIBRATION)
        return data


def load_calibration(path: Path = CALIBRATION):
    try:
        d = json.loads(path.read_text())
        a = [float(v) for v in d["affine"]]
        if len(a) == 6 and abs(a[0] * a[4] - a[1] * a[3]) > 1e-4:
            return a, d
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None, None
