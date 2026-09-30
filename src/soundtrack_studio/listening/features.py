"""CLAP audio front end in NumPy.

A re-implementation of Hugging Face ``transformers``' ``ClapFeatureExtractor``
(``truncation="rand_trunc"``, ``padding="repeatpad"``: the configuration of
LAION's larger CLAP checkpoints) so the portable build needs neither PyTorch
nor ``transformers``. Inputs are mono float waveforms at 48 kHz of at most
``max_length_s`` seconds (the caller picks the excerpt, so no random crop is
involved). ``tests/test_listening_reference.py`` compares the output with the
original implementation in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class FrontEndConfig:
    sampling_rate: int = 48000
    feature_size: int = 64
    fft_window_size: int = 1024
    hop_length: int = 480
    frequency_min: float = 50.0
    frequency_max: float = 14000.0
    max_length_s: float = 10.0
    padding: str = "repeatpad"
    truncation: str = "rand_trunc"

    @classmethod
    def from_preprocessor_config(cls, cfg: Dict[str, Any]) -> "FrontEndConfig":
        c = cls(sampling_rate=int(cfg.get("sampling_rate", 48000)), feature_size=int(cfg.get("feature_size", 64)),
                fft_window_size=int(cfg.get("fft_window_size", 1024)), hop_length=int(cfg.get("hop_length", 480)),
                frequency_min=float(cfg.get("frequency_min", 50)), frequency_max=float(cfg.get("frequency_max", 14000)),
                max_length_s=float(cfg.get("max_length_s", 10)), padding=str(cfg.get("padding", "repeatpad")),
                truncation=str(cfg.get("truncation", "rand_trunc")))
        if c.truncation != "rand_trunc" or c.padding not in ("repeatpad", "repeat", "pad"):
            raise ValueError(f"unsupported CLAP preprocessing: truncation={c.truncation}, padding={c.padding}")
        return c

    @property
    def max_samples(self) -> int:
        return int(round(self.max_length_s * self.sampling_rate))


def _hz_to_mel_slaney(freq):
    freq = np.asarray(freq, dtype=np.float64)
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    log_region = freq >= min_log_hz
    mels = np.where(log_region, min_log_mel + np.log(np.maximum(freq, 1e-10) / min_log_hz) * logstep, mels)
    return mels


def _mel_to_hz_slaney(mels):
    mels = np.asarray(mels, dtype=np.float64)
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    return np.where(log_region, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freq)


def mel_filter_bank(num_frequency_bins: int, num_mel_filters: int, min_frequency: float, max_frequency: float,
                    sampling_rate: int) -> np.ndarray:
    """Slaney-scale, Slaney-normalised triangular filters, shape (num_frequency_bins, num_mel_filters)."""

    mel_min, mel_max = _hz_to_mel_slaney(min_frequency), _hz_to_mel_slaney(max_frequency)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _mel_to_hz_slaney(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / filter_diff[:-1]
    up = slopes[:, 2:] / filter_diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2:num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    return filters * enorm[None, :]


class ClapFrontEnd:
    def __init__(self, config: FrontEndConfig) -> None:
        self.config = config
        n_freq = (config.fft_window_size >> 1) + 1
        self.mel_filters = mel_filter_bank(n_freq, config.feature_size, config.frequency_min, config.frequency_max,
                                           config.sampling_rate)
        # periodic Hann window, as transformers.audio_utils.window_function
        self.window = np.hanning(config.fft_window_size + 1)[:-1].astype(np.float64)

    def log_mel(self, waveform: np.ndarray) -> np.ndarray:
        """(frames, feature_size) log-mel spectrogram in dB (power 2, reflect-padded, centred frames)."""

        n_fft, hop = self.config.fft_window_size, self.config.hop_length
        x = np.pad(np.asarray(waveform, dtype=np.float64), (n_fft // 2, n_fft // 2), mode="reflect")
        n_frames = 1 + (len(x) - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
        frames = x[idx] * self.window[None, :]
        power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2          # (frames, freq)
        mel = np.maximum(1e-10, power @ self.mel_filters)                  # (frames, mels)
        return (10.0 * np.log10(np.maximum(mel, 1e-10))).astype(np.float32)

    def prepare(self, waveform: np.ndarray) -> np.ndarray:
        """One excerpt -> model input of shape (1, frames, feature_size)."""

        x = np.asarray(waveform, dtype=np.float32).reshape(-1)
        max_len = self.config.max_samples
        if len(x) > max_len:
            raise ValueError("excerpt longer than the model's input length; the caller selects excerpts")
        if len(x) == 0:
            x = np.zeros(max_len, np.float32)
        if len(x) < max_len:
            if self.config.padding == "repeat":
                x = np.tile(x, int(max_len / len(x)) + 1)[:max_len]
            elif self.config.padding == "repeatpad":
                x = np.tile(x, int(max_len / len(x)))
                x = np.pad(x, (0, max_len - len(x)), mode="constant")
            else:
                x = np.pad(x, (0, max_len - len(x)), mode="constant")
        return self.log_mel(x)[None, :, :]

    def batch(self, excerpts) -> np.ndarray:
        """List of excerpts -> ``input_features`` of shape (batch, 1, frames, feature_size)."""

        return np.stack([self.prepare(e) for e in excerpts]).astype(np.float32)
