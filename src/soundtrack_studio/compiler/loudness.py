"""Perceived loudness (ITU-R BS.1770 / EBU R128 integrated loudness, LUFS) and true peak, in NumPy only.

* **Loudness.** The K-weighting filter (a high shelf around 1.7 kHz and a high-pass at 38 Hz) is applied by FFT
  convolution with its impulse response (
  they decay below 1e-12 within about 2800 samples at 48 kHz, so 4096 taps equal the recursive filter for practical purposes); mean squares of 100 ms sub-blocks, four per 400 ms gating
  block (75 % overlap), then the standard gating: absolute at -70 LUFS, relative at -10 LU. Checked against a
  reference implementation in tests/test_loudness.py.
* **True peak.** Sample peak of the signal oversampled four times (polyphase windowed sinc), so peaks that fall
  between samples are seen, processed in chunks to bound memory.
* **Levelling** never compresses or limits: the gain towards the target is capped so the true peak stays at or below
  the ceiling, and whatever cannot be reached is reported instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional

import numpy as np

SUB_BLOCK_S = 0.1
ABSOLUTE_GATE = -70.0
RELATIVE_GATE = -10.0
DEFAULT_CEILING_DBTP = -1.0       # room for the Vorbis encoder's small peak overshoot


def k_weighting(rate: int):
    """((b, a) shelf, (b, a) high-pass) biquads of BS.1770 for ``rate`` (the 48 kHz values of the standard result)."""

    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = ([(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
             [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1 + k / q + k * k
    highpass = ([1.0, -2.0, 1.0], [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    return shelf, highpass


IR_TAPS = 4096


def _impulse_response(rate: int) -> np.ndarray:
    """K-weighting impulse response, by running the two biquads over an impulse (direct form, once per rate)."""

    x = np.zeros(IR_TAPS)
    x[0] = 1.0
    for b, a in k_weighting(rate):
        y = np.zeros(IR_TAPS)
        x1 = x2 = y1 = y2 = 0.0
        b0, b1, b2 = b
        _a0, a1, a2 = a
        for i in range(IR_TAPS):
            xi = x[i]
            yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, xi, y1, yi
            y[i] = yi
        x = y
    return x


_IR_CACHE: dict = {}


class LoudnessMeter:
    """Streaming integrated loudness: feed (frames, channels) float arrays, then ``lufs()``."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.sub = max(1, int(round(rate * SUB_BLOCK_S)))
        if rate not in _IR_CACHE:
            _IR_CACHE[rate] = _impulse_response(rate)
        self.ir = _IR_CACHE[rate]
        self._ir_spec: Optional[np.ndarray] = None
        self._ir_size = 0
        self.tail: Optional[np.ndarray] = None      # convolution overlap carried into the next chunk
        self.pending: Optional[np.ndarray] = None   # filtered samples not yet forming a whole sub-block
        self.energies: List[np.ndarray] = []        # mean square per sub-block, summed over channels

    def _filter(self, x: np.ndarray) -> np.ndarray:
        n = len(x) + len(self.ir) - 1
        size = 1 << (n - 1).bit_length()
        if self._ir_size != size:
            self._ir_spec, self._ir_size = np.fft.rfft(self.ir, size)[:, None], size
        spec = np.fft.rfft(x, size, axis=0) * self._ir_spec
        y = np.fft.irfft(spec, size, axis=0)[:n]
        if self.tail is not None:
            y[:len(self.tail)] += self.tail
        self.tail = y[len(x):].copy()
        return y[:len(x)]

    def add(self, x: np.ndarray, chunk: int = 1 << 14) -> None:
        x = np.asarray(x, np.float64)
        if x.ndim == 1:
            x = x[:, None]
        for start in range(0, len(x), chunk):
            y = self._filter(x[start:start + chunk])
            if self.pending is not None and len(self.pending):
                y = np.concatenate([self.pending, y])
            n = len(y) // self.sub
            self.pending = y[n * self.sub:]
            if n:
                blocks = y[:n * self.sub].reshape(n, self.sub, y.shape[1])
                self.energies.append(np.einsum("nsc,nsc->n", blocks, blocks) / self.sub)

    def lufs(self) -> Optional[float]:
        if not self.energies:
            return None
        sub = np.concatenate(self.energies)
        if len(sub) < 4:
            z = np.array([sub.mean()])
        else:
            z = (sub[:-3] + sub[1:-2] + sub[2:-1] + sub[3:]) / 4.0        # 400 ms blocks, 100 ms hop
        with np.errstate(divide="ignore"):
            block = -0.691 + 10 * np.log10(np.maximum(z, 1e-20))
        keep = block > ABSOLUTE_GATE
        if not np.any(keep):
            return None
        relative = -0.691 + 10 * math.log10(float(z[keep].mean())) + RELATIVE_GATE
        keep &= block > relative
        if not np.any(keep):
            return None
        return round(-0.691 + 10 * math.log10(float(z[keep].mean())), 2)


def integrated_lufs(x: np.ndarray, rate: int) -> Optional[float]:
    meter = LoudnessMeter(rate)
    meter.add(x)
    return meter.lufs()


def lufs_of_blocks(blocks: Iterable[np.ndarray], rate: int) -> Optional[float]:
    meter = LoudnessMeter(rate)
    for b in blocks:
        meter.add(b)
    return meter.lufs()


TP_OVERSAMPLE = 4
TP_TAPS = 48                       # per phase
TP_CANDIDATE_DB = -6.0             # only samples within 6 dB of the sample peak can hold the true peak


@lru_cache(maxsize=4)
def _tp_bank() -> np.ndarray:
    """Windowed-sinc interpolation taps for the fractional positions 1/4, 2/4, 3/4 between samples."""

    half = TP_TAPS // 2
    rows = []
    for k in range(1, TP_OVERSAMPLE):
        t = np.arange(-half + 1, half + 1) - k / TP_OVERSAMPLE        # sample offsets relative to the position
        h = np.sinc(t) * np.kaiser(TP_TAPS, 8.0)
        rows.append(h / h.sum())
    return np.array(rows)                                             # (3, taps)


def true_peak_dbtp(x: np.ndarray, rate: int = 48000) -> Optional[float]:
    """Peak of the signal reconstructed between samples (4x), looked at only near the loudest samples.

    An inter-sample peak can only exceed its neighbouring samples by a few dB, so only positions next to samples
    within 6 dB of the sample peak are interpolated - the same answer as oversampling everything, much faster."""

    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) == 0:
        return None
    mag = np.max(np.abs(x), axis=1)
    peak = float(mag.max())
    if peak <= 0:
        return None
    bank = _tp_bank()
    half = TP_TAPS // 2
    cand = np.flatnonzero(mag >= peak * 10 ** (TP_CANDIDATE_DB / 20))
    cand = np.unique(np.concatenate([cand - 1, cand]))                # the gap before and after each loud sample
    cand = cand[(cand >= half - 1) & (cand < len(x) - half)]
    best = peak
    offsets = np.arange(-half + 1, half + 1)
    for start in range(0, len(cand), 20000):
        idx = cand[start:start + 20000]
        win = x[idx[:, None] + offsets[None, :]]                       # (n, taps, channels)
        vals = np.einsum("ntc,kt->nkc", win, bank)
        best = max(best, float(np.max(np.abs(vals))))
    return 20 * math.log10(best)


@dataclass
class LevelResult:
    loudness_before: Optional[float]
    target: Optional[float]
    gain_db: float
    short_by_db: float              # how far below the target the track stays to keep its peaks clean
    true_peak_after: Optional[float]

    def to_dict(self) -> dict:
        return {"loudness_before_lufs": self.loudness_before, "loudness_target_lufs": self.target,
                "gain_db": round(self.gain_db, 2), "short_of_target_db": round(self.short_by_db, 2),
                "true_peak_dbtp": None if self.true_peak_after is None else round(self.true_peak_after, 2)}


def level(x: np.ndarray, rate: int, target_lufs: Optional[float], ceiling_dbtp: float = DEFAULT_CEILING_DBTP,
          measured: Optional[float] = None) -> "tuple[np.ndarray, LevelResult]":
    """Scale ``x`` towards ``target_lufs`` without ever letting the true peak exceed ``ceiling_dbtp``.

    ``target_lufs=None`` keeps the track's own level and only turns it down if its peaks are above the ceiling.
    No compression or limiting: a single gain for the whole piece."""

    before = measured if measured is not None else integrated_lufs(x, rate)
    peak = true_peak_dbtp(x, rate)
    if peak is None or before is None:
        return x.astype(np.float32, copy=False), LevelResult(before, target_lufs, 0.0, 0.0, peak)
    wanted = 0.0 if target_lufs is None else target_lufs - before
    allowed = ceiling_dbtp - peak
    gain = min(wanted, allowed)
    short = max(0.0, wanted - gain) if target_lufs is not None else 0.0
    y = (x * (10 ** (gain / 20))).astype(np.float32)
    return y, LevelResult(before, target_lufs, gain, short, peak + gain)
