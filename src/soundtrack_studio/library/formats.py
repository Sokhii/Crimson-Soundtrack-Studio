"""Audio format handlers.

Each supported container implements :class:`AudioFormatHandler`. Adding a
format (e.g. WAV, Ogg, MP3) means adding a handler and registering it; the
scanner, cache and analysis code are format-agnostic. Handlers only ever open
files for reading.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol

import numpy as np

from ..app_paths import os_path
from .flac_meta import FlacFormatError, read_flac_info


class AudioReadError(Exception):
    """A file could not be read as audio. ``message`` is user-facing."""


@dataclass
class ProbeResult:
    format: str
    codec: str
    container: str
    sample_rate: int
    channels: int
    bit_depth: Optional[int]
    total_samples: Optional[int]
    duration_s: Optional[float]
    identity: str                              # stable audio identity (tags edits do not change it when possible)
    tags: Dict[str, Optional[str]] = field(default_factory=dict)
    raw_tags: Dict[str, List[str]] = field(default_factory=dict)
    has_cover: bool = False
    warnings: List[str] = field(default_factory=list)


class AudioFormatHandler(Protocol):
    name: str
    extensions: tuple

    def probe(self, path: Path) -> ProbeResult: ...

    def blocks(self, path: Path, block_frames: int, overlap: int) -> Iterator[np.ndarray]:
        """Yield float32 arrays shaped (frames, channels)."""


def _int_prefix(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    m = re.match(r"\s*(\d+)", value)
    return int(m.group(1)) if m else None


def _year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    m = re.search(r"(1[0-9]{3}|20[0-9]{2})", value)
    return int(m.group(1)) if m else None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(os_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FlacHandler:
    name = "flac"
    extensions = (".flac",)

    def probe(self, path: Path) -> ProbeResult:
        try:
            size = Path(os_path(path)).stat().st_size
            with open(os_path(path), "rb") as handle:
                info = read_flac_info(handle, size)
        except FlacFormatError as exc:
            raise AudioReadError(f"This file is not a valid FLAC file ({exc}).") from exc
        except OSError as exc:
            raise AudioReadError(f"The file could not be read ({exc.strerror or exc}).") from exc
        warnings: List[str] = []
        if not info.total_samples:
            warnings.append("FLAC header does not state the length; duration measured while decoding")
        if info.md5_set:
            identity = f"flac-md5:{info.md5}:{info.total_samples}:{info.sample_rate}:{info.channels}"
        else:
            identity = "sha256:" + _hash_file(path)
            warnings.append("FLAC header has no audio checksum; the whole file is hashed for identification")
        if "_WARNING" in info.tags:
            warnings.extend(info.tags.pop("_WARNING"))
        tags = {
            "title": info.tag("TITLE"),
            "artist": info.tag("ARTIST"),
            "album": info.tag("ALBUM"),
            "album_artist": info.tag("ALBUMARTIST", "ALBUM ARTIST", "ALBUM_ARTIST"),
            "composer": info.tag("COMPOSER"),
            "genre": info.tag("GENRE"),
            "year": _year(info.tag("DATE", "YEAR", "ORIGINALDATE")),
            "track_number": _int_prefix(info.tag("TRACKNUMBER", "TRACK")),
            "disc_number": _int_prefix(info.tag("DISCNUMBER", "DISC")),
        }
        return ProbeResult(
            format="flac", codec="FLAC", container="FLAC", sample_rate=info.sample_rate, channels=info.channels,
            bit_depth=info.bits_per_sample, total_samples=info.total_samples or None, duration_s=info.duration_s,
            identity=identity, tags=tags, raw_tags=info.tags, has_cover=info.has_picture, warnings=warnings)

    def blocks(self, path: Path, block_frames: int, overlap: int) -> Iterator[np.ndarray]:
        import soundfile as sf

        try:
            with sf.SoundFile(os_path(path), mode="r") as f:
                yield from f.blocks(blocksize=block_frames, overlap=overlap, dtype="float32", always_2d=True)
        except (RuntimeError, getattr(sf, "SoundFileError", RuntimeError)) as exc:
            raise AudioReadError(f"The audio data could not be decoded; the file may be damaged ({exc}).") from exc


HANDLERS: List[AudioFormatHandler] = [FlacHandler()]


def handler_for(path: Path) -> Optional[AudioFormatHandler]:
    suffix = Path(path).suffix.lower()
    for handler in HANDLERS:
        if suffix in handler.extensions:
            return handler
    return None


def supported_extensions() -> List[str]:
    return sorted({ext for h in HANDLERS for ext in h.extensions})
