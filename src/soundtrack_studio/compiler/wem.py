"""Wwise PCM ``.wem`` writer and reader.

Layout per vgmstream ``meta/wwise.c`` (see docs/research/modding_format.md):
RIFF/WAVE, ``fmt `` chunk of 0x18 bytes with format tag 0xFFFE, 16-bit
little-endian interleaved samples, ``cbSize`` 6, valid bits 16 and the Wwise
channel-config word ``numChannels | configType(1=standard) << 8 | channelMask << 12``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .archive import CompileError

FORMAT_WWISE_PCM = 0xFFFE
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

    @property
    def frames(self) -> int:
        return self.data_size // max(1, self.channels * self.bits_per_sample // 8)

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def read_wem_info(data: bytes) -> WemInfo:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise CompileError("Not a RIFF/WAVE (.wem) file.")
    riff_size = struct.unpack_from("<I", data, 4)[0]
    if riff_size + 8 != len(data):
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
            tag_, channels, rate, _avg, _align, bits = struct.unpack_from("<HHIIHH", data, f_off)
            if pos + 8 + size > len(data):
                raise CompileError("The .wem audio data is truncated.")
            return WemInfo(tag_, channels, rate, bits, f_size, pos + 8, size)
        pos += 8 + size + (size & 1)
    raise CompileError("The .wem file has no audio data.")


def read_pcm(data: bytes) -> np.ndarray:
    info = read_wem_info(data)
    if info.bits_per_sample != 16:
        raise CompileError("Only 16-bit PCM .wem files can be read back.")
    raw = np.frombuffer(data, dtype="<i2", count=info.data_size // 2, offset=info.data_offset)
    return (raw.reshape(-1, info.channels).astype(np.float32) / 32767.0)
