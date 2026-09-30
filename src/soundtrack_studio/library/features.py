"""Deterministic audio measurements.

Everything here is plain signal processing on the decoded PCM, computed in
fixed-size blocks so long files never need to fit in memory. Each value is
reported with how it was obtained; values that are *estimates* are labelled
as such and are never presented as authoritative:

* measured:  duration, RMS level, peak level, crest factor, level spread,
             spectral centroid / roll-off / flatness, band energy split,
             zero-crossing rate, stereo width, silence ratio
* estimate:  tempo (autocorrelation of an onset envelope; ``None`` when the
             periodicity is too weak), onset rate
* heuristic: ``energy_index`` and ``brightness_index`` - simple, documented
             0..1 summaries of the measurements for sorting and later
             matching; not perceptual truth

Musical key and vocal presence are deliberately *not* computed: simple
algorithms for them are unreliable, and a wrong value presented as fact is
worse than no value. Later phases can add them (or ask the user) explicitly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

ANALYSIS_VERSION = 1
FRAME = 2048
HOP = 1024
SILENCE_DB = -60.0
EPS = 1e-12


@dataclass
class AudioFeatures:
    analysis_version: int = ANALYSIS_VERSION
    decoded_duration_s: float = 0.0
    rms_dbfs: Optional[float] = None
    peak_dbfs: Optional[float] = None
    crest_db: Optional[float] = None
    level_spread_db: Optional[float] = None         # 95th - 10th percentile of 400 ms RMS (non-silent)
    silence_ratio: float = 0.0                      # share of 400 ms windows below -60 dBFS
    spectral_centroid_hz: Optional[float] = None
    spectral_rolloff_hz: Optional[float] = None     # 85 % energy roll-off
    spectral_flatness: Optional[float] = None       # 0 = tonal, 1 = noise-like
    band_energy: Dict[str, float] = field(default_factory=dict)   # low <250 Hz, mid, high >4 kHz (fractions)
    zero_crossing_rate: Optional[float] = None      # crossings per sample (mono mix)
    stereo_width: Optional[float] = None            # side / (mid + side) energy; None for mono
    tempo_bpm: Optional[float] = None               # estimate; None when not reliably periodic
    tempo_confidence: float = 0.0                   # normalised autocorrelation peak, 0..1
    onset_rate: Optional[float] = None              # estimated onsets per second
    onset_peakiness: Optional[float] = None         # how distinct onsets are (p99/median of onset envelope, capped)
    broadband_onset_ratio: Optional[float] = None   # share of frames where most bands rise at once (attacks)
    energy_index: Optional[float] = None            # heuristic 0..1
    brightness_index: Optional[float] = None        # heuristic 0..1
    notes: List[str] = field(default_factory=list)
    methods: Dict[str, str] = field(default_factory=lambda: dict(METHODS))

    def to_dict(self) -> dict:
        return asdict(self)


METHODS = {
    "rms_dbfs": "measured", "peak_dbfs": "measured (sample peak, not true peak)", "crest_db": "measured",
    "level_spread_db": "measured (not EBU R128 loudness range)", "spectral_centroid_hz": "measured",
    "spectral_rolloff_hz": "measured", "spectral_flatness": "measured", "band_energy": "measured",
    "zero_crossing_rate": "measured", "stereo_width": "measured", "tempo_bpm": "estimate (onset autocorrelation)",
    "onset_rate": "estimate", "onset_peakiness": "measured",
    "broadband_onset_ratio": "measured", "energy_index": "heuristic", "brightness_index": "heuristic",
}
TEMPO_MIN_CONFIDENCE = 0.2
NOVELTY_MIN = 0.002   # mean positive spectral change relative to mean spectral level
BROADBAND_BAND_SHARE = 0.3    # a frame is a broadband onset when >= 30 % of bands rise clearly
BROADBAND_MIN_FRAMES = 0.005  # ...and at least 0.5 % of frames must be such onsets (~1 per 5 s)


def _db(value: float) -> Optional[float]:
    return None if value <= EPS else round(20.0 * math.log10(value), 2)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def analyze_blocks(blocks: Iterable[np.ndarray], sample_rate: int, *,
                   cancel: Optional[Callable[[], bool]] = None) -> AudioFeatures:
    """Analyse float32 blocks shaped (frames, channels) produced with overlap ``FRAME - HOP``."""

    overlap = FRAME - HOP
    window = np.hanning(FRAME).astype(np.float32)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sample_rate).astype(np.float64)
    low_mask, high_mask = freqs < 250.0, freqs >= 4000.0
    bands = _band_matrix(freqs, sample_rate)

    total_samples = 0
    sum_sq = 0.0
    peak = 0.0
    zc = 0
    mid_e = side_e = 0.0
    channels = None
    frame_ms: List[np.ndarray] = []
    flux: List[np.ndarray] = []
    level: List[np.ndarray] = []
    rising: List[np.ndarray] = []
    centroid_sum = rolloff_sum = flat_sum = 0.0
    spectral_frames = 0
    band = np.zeros(3)
    prev_mag: Optional[np.ndarray] = None
    last_sample = None
    first = True

    for block in blocks:
        if cancel and cancel():
            from ..errors import OperationCancelled
            raise OperationCancelled()
        if block.ndim == 1:
            block = block[:, None]
        channels = block.shape[1]
        fresh = block if first else block[overlap:]
        first = False
        if fresh.size:
            total_samples += fresh.shape[0]
            sum_sq += float(np.sum(fresh.astype(np.float64) ** 2)) / channels
            peak = max(peak, float(np.max(np.abs(fresh))))
            mono_fresh = fresh.mean(axis=1)
            signs = np.signbit(mono_fresh)
            zc += int(np.count_nonzero(signs[1:] != signs[:-1]))
            if last_sample is not None and signs.size:
                zc += int(signs[0] != last_sample)
            if signs.size:
                last_sample = signs[-1]
            if channels >= 2:
                left, right = fresh[:, 0].astype(np.float64), fresh[:, 1].astype(np.float64)
                mid_e += float(np.sum(((left + right) * 0.5) ** 2))
                side_e += float(np.sum(((left - right) * 0.5) ** 2))

        mono = block.mean(axis=1)
        n = (len(mono) - FRAME) // HOP + 1
        if n <= 0:
            continue
        idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
        frames = mono[idx]
        ms = np.mean(frames.astype(np.float64) ** 2, axis=1)
        frame_ms.append(ms)
        mag = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float64)
        comp = np.log1p(mag @ bands)  # onset detection on ~32 log-spaced bands, not noisy single bins
        prev = comp[:1] if prev_mag is None else prev_mag[None, :]
        diffs = np.diff(np.vstack([prev, comp]), axis=0)
        flux.append(np.sum(np.maximum(diffs, 0.0), axis=1))
        level.append(np.sum(comp, axis=1))
        rising.append(np.mean(diffs > 0.1, axis=1))  # share of bands getting clearly louder in this frame
        prev_mag = comp[-1]
        loud = ms > 10 ** (SILENCE_DB / 10.0)
        if np.any(loud):
            m = mag[loud]
            power = m ** 2
            msum = m.sum(axis=1) + EPS
            centroid_sum += float(np.sum((m @ freqs) / msum))
            cum = np.cumsum(power, axis=1)
            thresh = 0.85 * cum[:, -1:]
            rolloff_sum += float(np.sum(freqs[np.argmax(cum >= thresh, axis=1)]))
            geo = np.exp(np.mean(np.log(power + EPS), axis=1))
            flat_sum += float(np.sum(geo / (np.mean(power, axis=1) + EPS)))
            ptotal = power.sum(axis=1) + EPS
            band += np.array([np.sum(power[:, low_mask].sum(axis=1) / ptotal),
                              np.sum(power[:, ~(low_mask | high_mask)].sum(axis=1) / ptotal),
                              np.sum(power[:, high_mask].sum(axis=1) / ptotal)])
            spectral_frames += int(np.count_nonzero(loud))

    f = AudioFeatures()
    if total_samples == 0 or channels is None:
        f.notes.append("no audio samples could be decoded")
        return f
    f.decoded_duration_s = round(total_samples / sample_rate, 3)
    rms = math.sqrt(sum_sq / total_samples)
    f.rms_dbfs = _db(rms)
    f.peak_dbfs = _db(peak)
    if f.rms_dbfs is not None and f.peak_dbfs is not None:
        f.crest_db = round(f.peak_dbfs - f.rms_dbfs, 2)
    f.zero_crossing_rate = round(zc / max(1, total_samples - 1), 5)
    if channels >= 2 and mid_e + side_e > EPS:
        f.stereo_width = round(side_e / (mid_e + side_e), 4)
    elif channels >= 2:
        f.stereo_width = None
    if f.rms_dbfs is None:
        f.notes.append("the file is silent")
        f.silence_ratio = 1.0
        return f

    ms_all = np.concatenate(frame_ms) if frame_ms else np.zeros(0)
    per_window = max(1, int(round(0.4 * sample_rate / HOP)))
    if ms_all.size >= per_window:
        usable = ms_all[: (ms_all.size // per_window) * per_window].reshape(-1, per_window).mean(axis=1)
        win_db = 10.0 * np.log10(usable + EPS)
        f.silence_ratio = round(float(np.mean(win_db < SILENCE_DB)), 4)
        loud_db = win_db[win_db >= SILENCE_DB]
        if loud_db.size >= 3:
            f.level_spread_db = round(float(np.percentile(loud_db, 95) - np.percentile(loud_db, 10)), 2)
    if spectral_frames:
        f.spectral_centroid_hz = round(centroid_sum / spectral_frames, 1)
        f.spectral_rolloff_hz = round(rolloff_sum / spectral_frames, 1)
        f.spectral_flatness = round(flat_sum / spectral_frames, 5)
        shares = band / spectral_frames
        f.band_energy = {"low": round(float(shares[0]), 4), "mid": round(float(shares[1]), 4),
                         "high": round(float(shares[2]), 4)}

    envelope = np.concatenate(flux) if flux else np.zeros(0)
    levels = np.concatenate(level) if level else np.zeros(0)
    env_rate = sample_rate / HOP
    novelty = float(envelope.mean() / (levels.mean() + EPS)) if envelope.size else 0.0
    peakiness = float(np.percentile(envelope, 99) / (np.median(envelope) + EPS)) if envelope.size else 0.0
    f.onset_peakiness = round(min(peakiness, 1000.0), 2)
    rises = np.concatenate(rising) if rising else np.zeros(0)
    broadband = float(np.mean(rises >= BROADBAND_BAND_SHARE)) if rises.size else 0.0
    f.broadband_onset_ratio = round(broadband, 4)
    if novelty < NOVELTY_MIN or broadband < BROADBAND_MIN_FRAMES:
        # Real onsets (drums, plucks, note attacks) raise many bands at once. Stationary sound
        # and beating/modulated partials only move one or two bands; for those the
        # scale-invariant autocorrelation would otherwise "find" a beat in the fluctuation.
        f.onset_rate = 0.0
        f.notes.append("no rhythmic onsets detected; tempo left empty")
    else:
        _tempo(envelope, env_rate, f)

    level = _clamp01((f.rms_dbfs + 40.0) / 32.0)
    activity = _clamp01((f.onset_rate or 0.0) / 6.0)
    f.energy_index = round(0.65 * level + 0.35 * activity, 3)
    if f.spectral_centroid_hz:
        f.brightness_index = round(_clamp01(
            (math.log2(max(f.spectral_centroid_hz, 1.0)) - math.log2(500.0)) / (math.log2(5000.0) - math.log2(500.0))), 3)
    if f.decoded_duration_s < 5:
        f.notes.append("very short file: tempo and level statistics are unreliable")
    return f


def _band_matrix(freqs: np.ndarray, sample_rate: int, count: int = 32) -> np.ndarray:
    edges = np.geomspace(30.0, sample_rate / 2.0, count + 1)
    matrix = np.zeros((freqs.size, count))
    for i in range(count):
        matrix[(freqs >= edges[i]) & (freqs < edges[i + 1]), i] = 1.0
    return matrix[:, matrix.sum(axis=0) > 0]


def _tempo(envelope: np.ndarray, env_rate: float, f: AudioFeatures) -> None:
    duration = envelope.size / env_rate if env_rate else 0
    if envelope.size < env_rate * 8:
        f.notes.append("too short for a tempo estimate")
        return
    env = envelope - np.convolve(envelope, np.ones(16) / 16.0, mode="same")  # remove slow trend
    env = np.maximum(env, 0.0)
    # widen sharp onset peaks so a beat period falling between two integer lags is not lost
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    env = np.convolve(env, kernel / kernel.sum(), mode="same")
    threshold = env.mean() + env.std()
    peaks = (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > threshold)
    f.onset_rate = round(float(np.count_nonzero(peaks)) / duration, 3)
    env = env - env.mean()
    if not np.any(env):
        return
    size = 1 << int(np.ceil(np.log2(2 * env.size)))
    spec = np.fft.rfft(env, size)
    acf = np.fft.irfft(spec * np.conj(spec), size)[: env.size]
    if acf[0] <= EPS:
        return
    acf = acf / acf[0]
    lag_min = int(np.floor(60.0 * env_rate / 200.0))
    lag_max = min(int(np.ceil(60.0 * env_rate / 60.0)), acf.size - 2)
    if lag_max <= lag_min + 2:
        return
    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * env_rate / lags
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 1.0) ** 2)  # mild preference for common tempi (octave choice)
    scores = acf[lags] * prior
    best = int(lags[int(np.argmax(scores))])
    y0, y1, y2 = acf[best - 1], acf[best], acf[best + 1]
    denom = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / denom if abs(denom) > EPS else 0.0
    lag = best + max(-0.5, min(0.5, shift))
    confidence = float(max(0.0, min(1.0, y1)))
    f.tempo_confidence = round(confidence, 3)
    if confidence >= TEMPO_MIN_CONFIDENCE:
        f.tempo_bpm = round(60.0 * env_rate / lag, 1)
    else:
        f.notes.append("no clear steady beat detected; tempo left empty")


def block_parameters() -> Dict[str, int]:
    return {"frame": FRAME, "hop": HOP, "overlap": FRAME - HOP}
