"""Replacement audio rendering: decode, resample, map channels, fit to a timeline, normalise.

The user's file is only read. Everything is produced in memory and written
into the build workspace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

from ..app_paths import os_path
from .archive import CompileError

TARGET_RATE = 48000
FADE_IN_S = 0.02
CROSSFADE_S = 2.0


@dataclass
class FitSettings:
    fit_mode: str = "auto"          # auto | trim | loop | pad
    start_offset_s: float = 0.0
    normalize: bool = True
    target_rms_dbfs: float = -18.0
    peak_limit_dbfs: float = -1.0


def read_audio(path: Path, start_s: float = 0.0, max_s: float = 0.0) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    try:
        with sf.SoundFile(os_path(path)) as f:
            rate = f.samplerate
            start = min(int(start_s * rate), f.frames)
            f.seek(start)
            frames = f.frames - start if not max_s else min(f.frames - start, int(max_s * rate) + rate)
            data = f.read(frames, dtype="float32", always_2d=True)
    except (RuntimeError, OSError, getattr(sf, "SoundFileError", RuntimeError)) as exc:
        raise CompileError("A music file could not be read for the build.", hint="Check that the file still exists.",
                           details=f"{path}: {exc}") from exc
    return data, rate


@lru_cache(maxsize=16)
def _filter_bank(up: int, down: int, taps_per_phase: int = 32) -> np.ndarray:
    cutoff = 0.5 / max(up, down) * 0.95  # anti-aliasing low-pass (normalised to the upsampled rate)
    n = taps_per_phase * up
    t = np.arange(n) - (n - 1) / 2.0
    h = 2 * cutoff * np.sinc(2 * cutoff * t) * np.kaiser(n, 8.6)
    h *= up / h.sum()
    return h.reshape(taps_per_phase, up).T[:, ::-1].astype(np.float32)   # (phases, taps)


def resample(x: np.ndarray, rate_in: int, rate_out: int = TARGET_RATE) -> np.ndarray:
    """Polyphase windowed-sinc resampling of (frames, channels) audio."""

    if rate_in == rate_out or len(x) == 0:
        return x.astype(np.float32, copy=False)
    g = math.gcd(rate_in, rate_out)
    up, down = rate_out // g, rate_in // g
    bank = _filter_bank(up, down)
    taps = bank.shape[1]
    pad = taps
    xp = np.concatenate([np.zeros((pad, x.shape[1]), np.float32), x.astype(np.float32),
                         np.zeros((pad, x.shape[1]), np.float32)])
    n_out = int(len(x) * up // down)
    out = np.empty((n_out, x.shape[1]), np.float32)
    block = 65536
    for start in range(0, n_out, block):
        k = np.arange(start, min(n_out, start + block))
        pos = k * down
        phase = pos % up
        base = pos // up + pad - taps // 2
        idx = base[:, None] + np.arange(taps)[None, :]
        coeffs = bank[phase]                                   # (n, taps)
        for c in range(x.shape[1]):
            out[k, c] = np.einsum("nt,nt->n", xp[idx, c], coeffs)
    return out


def map_channels(x: np.ndarray, channels: int) -> np.ndarray:
    have = x.shape[1]
    if have == channels:
        return x
    if channels == 1:
        return x.mean(axis=1, keepdims=True)
    stereo = np.repeat(x, 2, axis=1) if have == 1 else x[:, :2]
    if channels == 2:
        return stereo
    out = np.zeros((len(x), channels), np.float32)
    out[:, :2] = stereo
    if channels >= 4:
        out[:, 2:4] = stereo * 0.707   # rears get the front signal at -3 dB
    return out


def _fade(x: np.ndarray, start: int, length: int, fade_in: bool) -> None:
    length = max(0, min(length, len(x) - start))
    if length <= 0:
        return
    ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
    if not fade_in:
        ramp = ramp[::-1]
    x[start:start + length] *= ramp[:, None]


def fit_to_length(x: np.ndarray, frames: int, mode: str, rate: int = TARGET_RATE) -> Tuple[np.ndarray, str]:
    """Fit audio to exactly ``frames``. Returns (audio, description)."""

    if len(x) == 0:
        return np.zeros((frames, x.shape[1] if x.ndim == 2 else 2), np.float32), "silence (empty source)"
    fade_out = int(min(3.0, frames / rate * 0.05 + 0.5) * rate)
    if mode == "auto":
        mode = "trim" if len(x) >= frames else "loop"
    out = np.zeros((frames, x.shape[1]), np.float32)
    if len(x) >= frames:
        out[:] = x[:frames]
        _fade(out, frames - fade_out, fade_out, fade_in=False)
        how = "trimmed with a fade-out" if len(x) > frames + rate // 10 else "fits"
    elif mode == "loop":
        # overlap-add copies with equal-length crossfades between repetitions
        xf = int(min(CROSSFADE_S, len(x) / rate / 4) * rate)
        step = max(1, len(x) - xf)
        starts = list(range(0, frames, step))
        for i, s0 in enumerate(starts):
            piece = x.copy()
            if i > 0 and xf:
                _fade(piece, 0, xf, fade_in=True)
            if s0 + len(piece) < frames and xf:
                _fade(piece, len(piece) - xf, xf, fade_in=False)
            n = min(len(piece), frames - s0)
            out[s0:s0 + n] += piece[:n]
        _fade(out, frames - fade_out, fade_out, fade_in=False)
        how = f"looped {frames / len(x):.1f}× with crossfades"
    else:  # trim (shorter source) or pad: play once, then silence
        out[:len(x)] = x
        _fade(out, len(x) - min(fade_out, len(x) // 4), min(fade_out, len(x) // 4), fade_in=False)
        how = "played once, then silence"
    _fade(out, 0, int(FADE_IN_S * rate), fade_in=True)
    return out, how


def normalize(x: np.ndarray, target_rms_dbfs: float, peak_limit_dbfs: float) -> Tuple[np.ndarray, float]:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    if rms < 1e-6:
        return x, 0.0
    gain = 10 ** (target_rms_dbfs / 20) / rms
    y = x * gain
    limit = 10 ** (peak_limit_dbfs / 20)
    peak = float(np.max(np.abs(y)))
    if peak > limit:
        # soft knee: tanh saturation above the limit keeps loud passages from clipping
        y = np.where(np.abs(y) > limit * 0.8,
                     np.sign(y) * (limit * 0.8 + (limit * 0.2) * np.tanh((np.abs(y) - limit * 0.8) / (limit * 0.2))), y)
    return y.astype(np.float32), round(20 * math.log10(gain), 2)


def render_timeline(path: Path, duration_s: float, channels: int, fit: FitSettings) -> Tuple[np.ndarray, dict]:
    """The user's track rendered onto a segment timeline of ``duration_s`` at 48 kHz."""

    frames = int(round(duration_s * TARGET_RATE))
    needed = duration_s + 5.0 if fit.fit_mode in ("auto", "trim", "pad") else 0.0
    data, rate = read_audio(path, fit.start_offset_s, needed)
    data = map_channels(resample(data, rate, TARGET_RATE), channels)
    out, how = fit_to_length(data, frames, fit.fit_mode)
    gain = 0.0
    if fit.normalize:
        out, gain = normalize(out, fit.target_rms_dbfs, fit.peak_limit_dbfs)
    return out, {"fit": how, "gain_db": gain, "source_rate": rate, "frames": frames}
