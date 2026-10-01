"""Minimal Wwise soundbank (v150) reader/patcher for music source replacement.

Only what the compiler needs (see docs/research/modding_format.md, section 4):

* split a bank into chunks and write it back;
* find a MusicTrack's ``AkBankSourceData`` for a source ID by its exact byte
  pattern (``u32 plugin, u8 stream type, u32 source id, u32 in-memory size,
  u8 source bits``) and cross-check it with the Analyzer's decoded values;
* patch those fixed-size fields in place (object sizes never change, so no
  offsets move);
* rebuild ``DIDX``/``DATA`` (remove, replace) keeping order and alignment.

Anything unexpected raises ``CompileError`` instead of guessing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .archive import CompileError

PLUGIN_PCM = 0x00010001
PLUGIN_ADPCM = 0x00020001
PLUGIN_VORBIS = 0x00040001
STREAM_IN_BANK, STREAM_PREFETCH, STREAM_STREAMING = 0, 1, 2
BIT_PREFETCH = 0x02
HIRC_MUSIC_TRACK = 11
SOURCE_STRUCT = struct.Struct("<IBIIB")  # plugin, stream type, source id, in-memory size, source bits


@dataclass
class Chunk:
    tag: bytes
    payload: bytes


@dataclass
class SourceRef:
    object_id: int
    offset: int          # absolute offset of the AkBankSourceData inside the bank
    plugin_id: int
    stream_type: int
    source_id: int
    in_memory_size: int
    bits: int


def parse_chunks(data: bytes) -> List[Chunk]:
    chunks: List[Chunk] = []
    pos = 0
    while pos < len(data):
        if pos + 8 > len(data):
            raise CompileError("The soundbank is truncated.", details=f"chunk header at {pos}")
        tag = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if pos + 8 + size > len(data):
            raise CompileError("The soundbank is damaged (a chunk runs past the end).", details=f"{tag!r} at {pos}")
        chunks.append(Chunk(tag, data[pos + 8:pos + 8 + size]))
        pos += 8 + size
    if not chunks or chunks[0].tag != b"BKHD":
        raise CompileError("This file is not a Wwise soundbank.", details="missing BKHD")
    return chunks


def write_chunks(chunks: Iterable[Chunk]) -> bytes:
    return b"".join(c.tag + struct.pack("<I", len(c.payload)) + c.payload for c in chunks)


def bank_version(chunks: List[Chunk]) -> int:
    return struct.unpack_from("<I", chunks[0].payload, 0)[0]


def chunk_offsets(chunks: List[Chunk]) -> Dict[bytes, int]:
    """Absolute payload offset of each chunk (first occurrence)."""

    out: Dict[bytes, int] = {}
    pos = 0
    for c in chunks:
        out.setdefault(c.tag, pos + 8)
        pos += 8 + len(c.payload)
    return out


def hirc_objects(chunks: List[Chunk]) -> Dict[int, List[Tuple[int, int, int]]]:
    """object id -> [(type, absolute body offset, body size)] (ids can repeat across types)."""

    offsets = chunk_offsets(chunks)
    hirc = next((c for c in chunks if c.tag == b"HIRC"), None)
    if hirc is None:
        return {}
    base = offsets[b"HIRC"]
    data = hirc.payload
    count = struct.unpack_from("<I", data, 0)[0]
    pos = 4
    out: Dict[int, List[Tuple[int, int, int]]] = {}
    for _ in range(count):
        if pos + 5 > len(data):
            raise CompileError("The soundbank's object list is damaged.", details=f"HIRC at {pos}")
        obj_type = data[pos]
        size = struct.unpack_from("<I", data, pos + 1)[0]
        body = pos + 5
        if body + size > len(data) or size < 4:
            raise CompileError("The soundbank's object list is damaged.", details=f"object at {pos}")
        obj_id = struct.unpack_from("<I", data, body)[0]
        out.setdefault(obj_id, []).append((obj_type, base + body, size))
        pos = body + size
    return out


def find_sources(bank: bytes, chunks: List[Chunk], object_id: int, source_id: int) -> List[SourceRef]:
    """All AkBankSourceData entries for ``source_id`` inside MusicTrack ``object_id``."""

    refs: List[SourceRef] = []
    needle = struct.pack("<I", source_id)
    for obj_type, start, size in hirc_objects(chunks).get(object_id, []):
        if obj_type != HIRC_MUSIC_TRACK:
            continue
        end = start + size
        pos = bank.find(needle, start, end)
        while pos != -1:
            head = pos - 5
            if head >= start + 4 and pos + 9 <= end:
                plugin, stream, sid, inmem, bits = SOURCE_STRUCT.unpack_from(bank, head)
                # a codec plugin id is <plugin number><company 0x000><type 1>: its low 16 bits are exactly 0x0001. (Looser
                # tests matched other fields of the object that merely contain the source id, e.g. the playlist entry.)
                if (plugin & 0xFFFF) == 0x0001 and stream in (0, 1, 2) and sid == source_id:
                    refs.append(SourceRef(object_id, head, plugin, stream, sid, inmem, bits))
            pos = bank.find(needle, pos + 1, end)
    return refs


def didx_entries(chunks: List[Chunk]) -> List[Tuple[int, int, int]]:
    didx = next((c for c in chunks if c.tag == b"DIDX"), None)
    if didx is None:
        return []
    if len(didx.payload) % 12:
        raise CompileError("The soundbank's media index is damaged.", details="DIDX size not a multiple of 12")
    return [struct.unpack_from("<III", didx.payload, i) for i in range(0, len(didx.payload), 12)]


def media_data(chunks: List[Chunk], media_id: int) -> Optional[bytes]:
    data = next((c for c in chunks if c.tag == b"DATA"), None)
    for mid, off, size in didx_entries(chunks):
        if mid == media_id and data is not None:
            return data.payload[off:off + size]
    return None


def rebuild_media(chunks: List[Chunk], changes: Dict[int, Optional[bytes]]) -> List[Chunk]:
    """Return new chunks with DIDX/DATA updated: media id -> new bytes, or None to remove."""

    entries = didx_entries(chunks)
    known = {mid for mid, _o, _s in entries}
    added = sorted(mid for mid, blob in changes.items() if blob is not None and mid not in known)
    if not entries:
        if added:
            raise CompileError("The soundbank has no media section to hold embedded audio.")
        return chunks
    data_chunk = next((c for c in chunks if c.tag == b"DATA"), None)
    if data_chunk is None:
        raise CompileError("The soundbank has a media index but no media data.")
    align = 16 if all(off % 16 == 0 for _m, off, _s in entries) else 1
    new_index = bytearray()
    new_data = bytearray()
    # media ids stay in ascending order (as Wwise writes them); a media id new to this bank is slotted in
    order = sorted([(mid, off, size) for mid, off, size in entries] + [(mid, -1, 0) for mid in added],
                   key=lambda e: e[0]) if added else entries
    for mid, off, size in order:
        if off >= 0 and off + size > len(data_chunk.payload):
            raise CompileError("The soundbank's media index points outside its data.", details=f"media {mid}")
        blob = changes[mid] if mid in changes else data_chunk.payload[off:off + size]
        if blob is None:
            continue
        if align > 1 and len(new_data) % align:
            new_data.extend(b"\x00" * (align - len(new_data) % align))
        new_index += struct.pack("<III", mid, len(new_data), len(blob))
        new_data += blob
    out = []
    for c in chunks:
        if c.tag == b"DIDX":
            if new_index:
                out.append(Chunk(b"DIDX", bytes(new_index)))
        elif c.tag == b"DATA":
            if new_index:
                out.append(Chunk(b"DATA", bytes(new_data)))
        else:
            out.append(c)
    return out


@dataclass
class SourcePatch:
    object_id: int
    source_id: int
    expected_plugin: Optional[int]      # from the Analyzer; None = do not check
    expected_stream: Optional[int]
    expected_in_memory: Optional[int]
    embedded_data: Optional[bytes] = None   # WEM bytes when the source stays inside the bank
    codec: str = "pcm"                      # pcm: switch the source to PCM | vorbis: keep the game's Vorbis codec
    prefetch_data: Optional[bytes] = None   # vorbis + prefetch-streamed: the new in-bank copy (prefix of the .wem)


def patch_bank(bank: bytes, patches: List[SourcePatch]) -> Tuple[bytes, List[str]]:
    """Point the listed sources at the replacement audio.

    ``codec="pcm"``: switch them to PCM (streamed, or in-bank when ``embedded_data`` is given).
    ``codec="vorbis"`` (the replacement is Wwise Vorbis, like the game's own media and every working community mod):
    codec, storage type and flags stay as the game shipped them; only the media changes - the in-bank copy of an
    in-bank source (``embedded_data``), or the prefetch copy of a prefetch-streamed source (``prefetch_data``, which
    must be the beginning of the new streamed file). Plain streamed sources need no bank change at all."""

    chunks = parse_chunks(bank)
    version = bank_version(chunks)
    if version != 150:
        raise CompileError("This soundbank uses a Wwise version the Studio has not been verified with.",
                           hint="The compiler supports bank version 150 (the version Crimson Desert ships).",
                           details=f"bank version {version}")
    notes: List[str] = []
    buf = bytearray(bank)
    media_changes: Dict[int, Optional[bytes]] = {}
    present = {mid for mid, _o, _s in didx_entries(chunks)}
    for p in patches:
        refs = find_sources(bank, chunks, p.object_id, p.source_id)
        if not refs:
            raise CompileError("A music track in the soundbank does not have the expected layout.",
                               hint="The Analyzer database and the game files disagree; re-run the Analyzer.",
                               details=f"MusicTrack {p.object_id}, source {p.source_id}")
        for ref in refs:
            if p.expected_plugin is not None and ref.plugin_id != p.expected_plugin:
                raise CompileError("A music source's codec differs from the Analyzer database.",
                                   details=f"source {p.source_id}: bank 0x{ref.plugin_id:08x}, "
                                           f"database 0x{p.expected_plugin:08x}")
            if p.expected_stream is not None and ref.stream_type != p.expected_stream:
                raise CompileError("A music source's storage type differs from the Analyzer database.",
                                   details=f"source {p.source_id}: bank {ref.stream_type}, database {p.expected_stream}")
            if p.expected_in_memory is not None and ref.in_memory_size != p.expected_in_memory:
                raise CompileError("A music source's size differs from the Analyzer database.",
                                   details=f"source {p.source_id}: bank {ref.in_memory_size}, database {p.expected_in_memory}")
            if p.codec == "vorbis":
                _patch_vorbis(buf, ref, p, media_changes, notes)
            elif p.embedded_data is not None:
                SOURCE_STRUCT.pack_into(buf, ref.offset, PLUGIN_PCM, STREAM_IN_BANK, ref.source_id,
                                        len(p.embedded_data), ref.bits & ~BIT_PREFETCH & 0xFF)
            else:
                SOURCE_STRUCT.pack_into(buf, ref.offset, PLUGIN_PCM, STREAM_STREAMING, ref.source_id, 0,
                                        ref.bits & ~BIT_PREFETCH & 0xFF)
        if p.codec == "vorbis":
            continue
        if p.embedded_data is not None:
            if p.source_id not in present:
                raise CompileError("An in-bank music source has no data in the soundbank.", details=str(p.source_id))
            media_changes[p.source_id] = p.embedded_data
        elif p.source_id in present:
            media_changes[p.source_id] = None   # stale prefetch copy of the old Vorbis stream
            notes.append(f"removed prefetch data of source {p.source_id}")
    patched_chunks = rebuild_media(parse_chunks(bytes(buf)), media_changes)
    return write_chunks(patched_chunks), notes


def _patch_vorbis(buf: bytearray, ref: SourceRef, p: SourcePatch, media_changes: Dict[int, Optional[bytes]],
                  notes: List[str]) -> None:
    if ref.plugin_id != PLUGIN_VORBIS:
        raise CompileError("A music source is not Vorbis in the game's soundbank, so a Vorbis replacement cannot be "
                           "used for it.", details=f"source {p.source_id}: 0x{ref.plugin_id:08x}")
    if ref.stream_type == STREAM_IN_BANK:
        if p.embedded_data is None:
            raise CompileError("An in-bank music source needs its replacement audio.", details=str(p.source_id))
        data = p.embedded_data
    elif ref.stream_type == STREAM_PREFETCH:
        if p.prefetch_data is None:
            raise CompileError("A prefetch-streamed music source needs its prefetch data.", details=str(p.source_id))
        data = p.prefetch_data
    else:
        return                                   # plain streaming: the bank only refers to the .wem file
    SOURCE_STRUCT.pack_into(buf, ref.offset, ref.plugin_id, ref.stream_type, ref.source_id, len(data), ref.bits)
    if media_changes.get(p.source_id, data) != data:
        raise CompileError("Two different replacements for the same in-bank media.", details=str(p.source_id))
    media_changes[p.source_id] = data
    notes.append(f"{'embedded' if ref.stream_type == STREAM_IN_BANK else 'prefetch'} data of source {p.source_id}: "
                 f"{len(data)} bytes")


def describe_sources(bank: bytes) -> Dict[int, List[SourceRef]]:
    """All music-track sources in a bank (used by validation)."""

    chunks = parse_chunks(bank)
    out: Dict[int, List[SourceRef]] = {}
    for obj_id, items in hirc_objects(chunks).items():
        for obj_type, start, size in items:
            if obj_type != HIRC_MUSIC_TRACK:
                continue
            pos = start + 4
            end = start + size
            while pos + SOURCE_STRUCT.size <= end:
                plugin, stream, sid, inmem, bits = SOURCE_STRUCT.unpack_from(bank, pos)
                if plugin in (PLUGIN_PCM, PLUGIN_ADPCM, PLUGIN_VORBIS) and stream in (0, 1, 2) and sid:
                    out.setdefault(sid, []).append(SourceRef(obj_id, pos, plugin, stream, sid, inmem, bits))
                    pos += SOURCE_STRUCT.size
                else:
                    pos += 1
    return out
