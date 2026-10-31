"""Fixed (geometry-only) beamformers for the UMA-16 v2, steered by face directions.

Conventions match demo.cpp: mic positions from its positions(), and a unit vector u
points from the array towards the source, so mic m hears the wave at tau_m = -p_m.u/c.

All beams are distortionless towards the target (unit gain), so the target voice keeps
the level of a single microphone and only background / other talkers change.
  das   delay-and-sum
  mvdr  superdirective MVDR (diffuse-noise coherence + diagonal loading)
  lcmv  MVDR plus nulls towards the other faces
"""
from __future__ import annotations

import numpy as np

FS = 48000
SPEED = 343.0
NFFT = 1024
HOP = 512
CH = 16
REFERENCE_MIC = 5          # inner mic, used as "plain single microphone" reference
KINDS = ("das", "mvdr", "lcmv")


def mic_positions() -> np.ndarray:
    """(16,3) metres. Same table and mirroring as positions() in demo.cpp."""
    layout = [[8, 7, 10, 9], [6, 5, 12, 11], [4, 3, 14, 13], [2, 1, 16, 15]]
    p = np.zeros((CH, 3))
    for r in range(4):
        for c in range(4):
            p[layout[r][c] - 1] = ((1.5 - c) * 0.042, (1.5 - r) * 0.042, 0.0)
    return p


def window() -> np.ndarray:
    """sqrt-Hann (periodic): analysis x synthesis = Hann, sums to 1 at 50% overlap."""
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT))


class Beamformer:
    def __init__(self, loading: float = 0.05, null_loading: float = 0.02):
        self.pos = mic_positions()
        self.freqs = np.fft.rfftfreq(NFFT, 1 / FS)                      # (F,)
        dist = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=2)
        coherence = np.sinc(2 * self.freqs[:, None, None] * dist[None] / SPEED)  # (F,16,16)
        eye = np.eye(CH)[None]
        self.inv_gamma = np.linalg.inv(coherence + loading * eye)
        self.null_loading = null_loading
        self._cache: dict = {}

    def steering(self, u) -> np.ndarray:
        """(F,16) phase-only steering vector; d^H x aligns all mics on direction u."""
        u = np.asarray(u, float)
        tau = -(self.pos @ u) / SPEED                                   # arrival times (16,)
        return np.exp(-2j * np.pi * self.freqs[:, None] * tau[None])

    def weights(self, kind: str, target, others=()) -> np.ndarray:
        """(F,16) weights; output Y = sum(conj(w) * X)."""
        key = (kind, _q(target), tuple(sorted(_q(o) for o in others)) if kind == "lcmv" else ())
        w = self._cache.get(key)
        if w is not None:
            return w
        d = self.steering(target)
        if kind == "das":
            w = d / CH
        elif kind == "mvdr" or (kind == "lcmv" and not others):
            g = np.einsum("fij,fj->fi", self.inv_gamma, d)
            w = g / np.einsum("fi,fi->f", d.conj(), g)[:, None]
        elif kind == "lcmv":
            cons = np.stack([d] + [self.steering(o) for o in others], axis=2)        # (F,16,K)
            gc = self.inv_gamma @ cons                                               # (F,16,K)
            m = cons.conj().transpose(0, 2, 1) @ gc                                  # (F,K,K)
            k = cons.shape[2]
            m += self.null_loading * np.trace(m, axis1=1, axis2=2).real[:, None, None] / k * np.eye(k)
            resp = np.zeros(k); resp[0] = 1
            coef = np.linalg.solve(m, np.broadcast_to(resp, (len(self.freqs), k))[..., None])[..., 0]
            w = np.einsum("fmk,fk->fm", gc, coef)
        else:
            raise ValueError(kind)
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = w
        return w

    @staticmethod
    def apply(w: np.ndarray, spectra: np.ndarray) -> np.ndarray:
        """spectra (16,F) -> beam spectrum (F,)."""
        return np.einsum("fm,mf->f", w.conj(), spectra)


class SrpMap:
    """SRP-PHAT over a grid of ray slopes; returns the loudest direction for display."""

    def __init__(self, bf: Beamformer, fmin=300.0, fmax=3500.0, nx=25, ny=17, span=(1.25, 0.85),
                 bin_step=2):
        self.band = np.where((bf.freqs >= fmin) & (bf.freqs <= fmax))[0][::bin_step]
        sx, sy = np.meshgrid(np.linspace(-span[0], span[0], nx), np.linspace(-span[1], span[1], ny))
        self.slopes = np.column_stack((sx.ravel(), sy.ravel()))
        rays = np.column_stack((self.slopes, np.ones(len(self.slopes))))
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        tau = -(rays @ bf.pos.T) / SPEED                                # (G,16)
        f = bf.freqs[self.band]
        self.steer = np.exp(-2j * np.pi * f[None, :, None] * tau[:, None, :])   # (G,Fb,16)
        self.acc = np.zeros(len(rays))

    def add(self, spectra: np.ndarray):
        x = spectra[:, self.band].T                                     # (Fb,16)
        x = x / np.maximum(np.abs(x), 1e-12)
        y = np.einsum("gfm,fm->gf", self.steer.conj(), x)
        self.acc += (np.abs(y) ** 2).sum(axis=1)

    def take(self):
        """Peak slopes (sx, sy) and contrast (peak vs mean, in std-devs); resets accumulator."""
        a, self.acc = self.acc, np.zeros_like(self.acc)
        i = int(a.argmax())
        contrast = (a[i] - a.mean()) / max(a.std(), 1e-12)
        return tuple(self.slopes[i]), float(contrast)


class PostFilter:
    """Streaming Wiener noise suppression on a beam spectrum (single-channel post-filter).
    over=4 compensates the downward bias of minimum tracking (tuned: -9 dB background,
    speech kept).

    Noise spectrum follows the minimum of the smoothed power: it drops immediately and
    rises slowly (~1.5 dB/s), so speech does not leak into the estimate but a change of
    background (fan turning on) is learned within seconds."""

    def __init__(self, bins: int = NFFT // 2 + 1, floor_db: float = -18.0, over: float = 4.0):
        self.smooth = None
        self.noise = None
        self.snr = np.zeros(bins)
        self.floor = 10 ** (floor_db / 20)
        self.over = over
        self.rise = 10 ** (1.5 / 10 / (FS / HOP))   # +1.5 dB per second

    def __call__(self, y: np.ndarray) -> np.ndarray:
        p = np.abs(y) ** 2
        if self.smooth is None:
            self.smooth, self.noise = p.copy(), p.copy()
        self.smooth = 0.8 * self.smooth + 0.2 * p
        self.noise = np.minimum(self.noise * self.rise, self.smooth) + 1e-20
        snr = np.maximum(p / (self.over * self.noise) - 1, 0)
        self.snr = 0.3 * self.snr + 0.7 * snr
        return y * np.maximum(self.snr / (1 + self.snr), self.floor)


def mix_weights() -> np.ndarray:
    """'General' audio: plain average of the 16 microphones (no steering)."""
    return np.full((NFFT // 2 + 1, CH), 1.0 / CH, complex)


class StreamProcessor:
    """Streaming STFT -> beams -> overlap-add. Feed HOP-sample blocks of (HOP,16)."""

    def __init__(self, bf: Beamformer):
        self.bf = bf
        self.win = window()
        self.inbuf = np.zeros((NFFT, CH), np.float32)
        self.outbuf = np.zeros(NFFT)
        self.prev_key = None

    def spectra(self, block: np.ndarray) -> np.ndarray:
        self.inbuf[:-HOP] = self.inbuf[HOP:]
        self.inbuf[-HOP:] = block
        return np.fft.rfft(self.inbuf.T * self.win, axis=1)             # (16,F)

    def render(self, spectra: np.ndarray, w: np.ndarray | None, key, post: PostFilter | None = None) -> np.ndarray:
        """w=None -> reference mic. post -> noise suppression on the output.
        Crossfades over one hop when the selection changes."""
        def time_frame(weights, filt=None):
            y = spectra[REFERENCE_MIC - 1] if weights is None else Beamformer.apply(weights, spectra)
            if filt is not None:
                y = filt(y)
            return np.fft.irfft(y, NFFT) * self.win
        frame = time_frame(w, post)
        if self.prev_key is not None and key != self.prev_key[0]:
            old = time_frame(self.prev_key[1])
            ramp = np.linspace(0, 1, NFFT)
            frame = ramp * frame + (1 - ramp) * old
        self.prev_key = (key, w)
        self.outbuf += frame
        out = self.outbuf[:HOP].copy()
        self.outbuf[:-HOP] = self.outbuf[HOP:]
        self.outbuf[-HOP:] = 0
        return out


def _q(u) -> tuple:
    return tuple(np.round(np.asarray(u, float), 2))


def level_db(x: np.ndarray) -> float:
    return float(10 * np.log10(max(np.mean(np.square(x)), 1e-12)))
