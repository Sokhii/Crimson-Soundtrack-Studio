"""Wwise ``.wem`` writer (PCM) and reader (PCM and Wwise Vorbis).

PCM layout per vgmstream ``meta/wwise.c`` (see docs/research/modding_format.md):
RIFF/WAVE, ``fmt `` chunk of 0x18 bytes with format tag 0xFFFE, 16-bit
little-endian interleaved samples, ``cbSize`` 6, valid bits 16 and the Wwise
channel-config word ``numChannels | configType(1=standard) << 8 | channelMask << 12``.

Wwise Vorbis (format tag 0xFFFF, ``fmt `` 0x42 bytes) is what the game and every working community music mod use.
The Studio does not encode it itself (Wwise does, see ``wwise.py``); it only reads the header: at ``fmt+0x18`` the
sample count, at ``fmt+0x28``/``fmt+0x2C`` the setup and audio offsets inside ``data`` (checked against files from
the game and from Nexus mods).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .archive import CompileError

FORMAT_WWISE_PCM = 0xFFFE
FORMAT_WWISE_VORBIS = 0xFFFF
CHANNEL_MASKS = {1: 0x4, 2: 0x3, 4: 0x33, 6: 0x3F}   # C / FL FR / FL FR BL BR / 5.1


def build_pcm_wem(samples: np.ndarray, sample_rate: int) -> bytes:
    """``samples``: float32 array (frames, channels) in [-1, 1]."""

    if samples.ndim != 2:
        raise ValueError("samples must be (frames, channels)")
    channels = samples.shape[1]
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2").tobytes()
    block_align = channels * 2
    config = channels | (1 << 8) | (CHANNEL_MASKS.get(channels, 0) << 12)
    fmt = struct.pack("<HHIIHHHHI", FORMAT_WWISE_PCM, channels, sample_rate, sample_rate * block_align, block_align, 16,
                      6, 16, config)
    assert len(fmt) == 0x18
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(body)) + body


@dataclass
class WemInfo:
    format_tag: int
    channels: int
    sample_rate: int
    bits_per_sample: int
    fmt_size: int
    data_offset: int
    data_size: int
    samples: Optional[int] = None          # Wwise Vorbis: sample count from the header
    fmt_offset: int = 0
    avg_bytes_per_sec: int = 0

    @property
    def is_vorbis(self) -> bool:
        return self.format_tag == FORMAT_WWISE_VORBIS

    @property
    def frames(self) -> int:
        if self.samples is not None:
            return self.samples
        return self.data_size // max(1, self.channels * self.bits_per_sample // 8)

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def read_wem_info(data: bytes, allow_truncated: bool = False) -> WemInfo:
    """``allow_truncated``: ``data`` is only the beginning of a file (e.g. a prefetch copy inside a bank)."""

    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise CompileError("Not a RIFF/WAVE (.wem) file.")
    riff_size = struct.unpack_from("<I", data, 4)[0]
    if riff_size + 8 != len(data) and not (allow_truncated and riff_size + 8 > len(data)):
        raise CompileError("The .wem file size does not match its header.", details=f"{riff_size + 8} != {len(data)}")
    pos = 12
    fmt = None
    while pos + 8 <= len(data):
        tag = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if tag == b"fmt ":
            fmt = (pos + 8, size)
        elif tag == b"data":
            if fmt is None:
                raise CompileError("The .wem file has audio data before its format.")
            f_off, f_size = fmt
            tag_, channels, rate, avg, _align, bits = struct.unpack_from("<HHIIHH", data, f_off)
            if pos + 8 + size > len(data) and not allow_truncated:
                raise CompileError("The .wem audio data is truncated.")
            samples = None
            if tag_ == FORMAT_WWISE_VORBIS:
                if f_size < 0x2C + 4 or f_off + 0x30 > len(data):
                    raise CompileError("The Wwise Vorbis header is too short.", details=f"fmt size {f_size}")
                samples = struct.unpack_from("<I", data, f_off + 0x18)[0]
            return WemInfo(tag_, channels, rate, bits, f_size, pos + 8, size, samples, f_off, avg)
        pos += 8 + size + (size & 1)
    raise CompileError("The .wem file has no audio data.")


def read_pcm(data: bytes) -> np.ndarray:
    info = read_wem_info(data)
    if info.bits_per_sample != 16:
        raise CompileError("Only 16-bit PCM .wem files can be read back.")
    raw = np.frombuffer(data, dtype="<i2", count=info.data_size // 2, offset=info.data_offset)
    return (raw.reshape(-1, info.channels).astype(np.float32) / 32767.0)


def vorbis_offsets(data: bytes, info: Optional[WemInfo] = None) -> tuple:
    """(setup offset, audio offset) of a Wwise Vorbis file, absolute in ``data``."""

    info = info or read_wem_info(data, allow_truncated=True)
    if not info.is_vorbis:
        raise CompileError("Not a Wwise Vorbis .wem file.")
    setup, audio = struct.unpack_from("<II", data, info.fmt_offset + 0x28)
    return info.data_offset + setup, info.data_offset + audio


PREFETCH_SECONDS = 0.1


def prefetch_prefix(data: bytes, seconds: float = PREFETCH_SECONDS) -> bytes:
    """The bytes a soundbank keeps of a prefetch-streamed source: the beginning of the very same ``.wem`` file.

    In the game's banks and in working community mods the in-bank copy is a plain byte prefix of the streamed
    file (compared byte for byte on Crimson Tamriel V2). Here: the whole header, seek table and setup, plus
    about ``seconds`` of audio, never more than the file."""

    info = read_wem_info(data)
    _setup, audio = vorbis_offsets(data, info)
    extra = int(max(0.0, seconds) * max(info.avg_bytes_per_sec, 1))
    return data[:min(len(data), audio + extra)]


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """16-bit PCM .wav (the input format Wwise converts from)."""

    channels = samples.shape[1]
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate, sample_rate * channels * 2, channels * 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    with open(path, "wb") as handle:
        handle.write(b"RIFF" + struct.pack("<I", len(body)) + body)
