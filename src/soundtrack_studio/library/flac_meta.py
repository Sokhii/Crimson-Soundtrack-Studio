"""Minimal, strict FLAC metadata reader (STREAMINFO, VORBIS_COMMENT, PICTURE).

Implemented here instead of using a tag library so the bundled application
has no GPL dependency. Reads only the metadata blocks at the start of the
file; the audio stream is decoded separately (libsndfile) for analysis.
Reference: https://xiph.org/flac/format.html
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, List, Optional


class FlacFormatError(ValueError):
    pass


@dataclass
class FlacInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    total_samples: int
    md5: str                                  # hex; all zeros when the encoder did not set it
    min_block: int = 0
    max_block: int = 0
    vendor: str = ""
    tags: Dict[str, List[str]] = field(default_factory=dict)   # upper-case keys
    has_picture: bool = False
    metadata_end: int = 0                     # file offset where audio frames start

    @property
    def duration_s(self) -> Optional[float]:
        if self.total_samples and self.sample_rate:
            return self.total_samples / self.sample_rate
        return None

    @property
    def md5_set(self) -> bool:
        return self.md5 != "0" * 32

    def tag(self, *keys: str) -> Optional[str]:
        for key in keys:
            values = [v for v in self.tags.get(key.upper(), []) if v.strip()]
            if values:
                return "; ".join(dict.fromkeys(v.strip() for v in values))
        return None


MAX_BLOCK = 16 * 1024 * 1024  # 24-bit length field


def _skip_id3v2(handle: BinaryIO) -> None:
    head = handle.read(10)
    if len(head) == 10 and head[:3] == b"ID3":
        size = 0
        for b in head[6:10]:
            if b & 0x80:
                raise FlacFormatError("malformed ID3v2 header")
            size = (size << 7) | b
        footer = 10 if head[5] & 0x10 else 0
        handle.seek(10 + size + footer)
    else:
        handle.seek(0)


def read_flac_info(handle: BinaryIO, file_size: int) -> FlacInfo:
    _skip_id3v2(handle)
    if handle.read(4) != b"fLaC":
        raise FlacFormatError("not a FLAC file (missing 'fLaC' signature)")
    info: Optional[FlacInfo] = None
    first = True
    while True:
        header = handle.read(4)
        if len(header) < 4:
            raise FlacFormatError("truncated metadata block header")
        last = bool(header[0] & 0x80)
        block_type = header[0] & 0x7F
        length = int.from_bytes(header[1:4], "big")
        start = handle.tell()
        if start + length > file_size:
            raise FlacFormatError(f"metadata block {block_type} extends past the end of the file")
        if first and block_type != 0:
            raise FlacFormatError("first metadata block is not STREAMINFO")
        first = False
        if block_type == 127:
            raise FlacFormatError("invalid metadata block type 127")
        if block_type == 0:
            if length != 34:
                raise FlacFormatError("STREAMINFO block has the wrong size")
            info = _parse_streaminfo(handle.read(34))
        elif block_type == 4 and info is not None:
            _parse_vorbis_comment(handle.read(length), info)
        elif block_type == 6 and info is not None:
            info.has_picture = True
            handle.seek(start + length)
        else:
            handle.seek(start + length)
        if last:
            break
    assert info is not None
    info.metadata_end = handle.tell()
    return info


def _parse_streaminfo(data: bytes) -> FlacInfo:
    min_block, max_block = struct.unpack(">HH", data[0:4])
    packed = int.from_bytes(data[10:18], "big")
    sample_rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    bits = ((packed >> 36) & 0x1F) + 1
    total = packed & 0xFFFFFFFFF
    if sample_rate == 0:
        raise FlacFormatError("STREAMINFO sample rate is 0")
    return FlacInfo(sample_rate=sample_rate, channels=channels, bits_per_sample=bits, total_samples=total,
                    md5=data[18:34].hex(), min_block=min_block, max_block=max_block)


def _parse_vorbis_comment(data: bytes, info: FlacInfo) -> None:
    try:
        pos = 0
        (vendor_len,) = struct.unpack_from("<I", data, pos)
        pos += 4
        info.vendor = data[pos:pos + vendor_len].decode("utf-8", "replace")
        pos += vendor_len
        (count,) = struct.unpack_from("<I", data, pos)
        pos += 4
        for _ in range(min(count, 10000)):
            (length,) = struct.unpack_from("<I", data, pos)
            pos += 4
            entry = data[pos:pos + length].decode("utf-8", "replace")
            pos += length
            if "=" in entry:
                key, value = entry.split("=", 1)
                info.tags.setdefault(key.strip().upper(), []).append(value)
    except struct.error:
        # A damaged comment block loses tags but not the file; keep what was read.
        info.tags.setdefault("_WARNING", []).append("vorbis comment block truncated")
