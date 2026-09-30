"""Read-only access to original game files needed for compilation.

The Analyzer database already records where each file lives (package, PAZ
index, offset, stored/original size, flags), so no archive index is parsed
here. Entries are decrypted (ChaCha20, key derived from the basename) and
decompressed (LZ4 block) as documented in docs/research/modding_format.md,
then verified against the SHA-1 the Analyzer recorded. Game files are only
ever opened with ``"rb"``.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..app_paths import os_path
from ..errors import StudioError

_MASK = 0xFFFFFFFF
CHACHA20_HASH_INITVAL = 0x000C5EDE
CHACHA20_KEY_XOR = 0x60616263
CHACHA20_XOR_DELTAS = (0x00000000, 0x0A0A0A0A, 0x0C0C0C0C, 0x06060606,
                       0x0E0E0E0E, 0x0A0A0A0A, 0x06060606, 0x02020202)


class CompileError(StudioError):
    title = "Mod build problem"


def _rot(x: int, k: int) -> int:
    return ((x << k) | (x >> (32 - k))) & _MASK


def hashlittle(data: bytes, initval: int = 0) -> int:
    """Bob Jenkins' lookup3 ``hashlittle`` (public domain algorithm)."""

    length = len(data)
    a = b = c = (0xDEADBEEF + length + initval) & _MASK
    off = 0
    remaining = length
    while remaining > 12:
        a = (a + struct.unpack_from("<I", data, off)[0]) & _MASK
        b = (b + struct.unpack_from("<I", data, off + 4)[0]) & _MASK
        c = (c + struct.unpack_from("<I", data, off + 8)[0]) & _MASK
        a = (a - c) & _MASK; a ^= _rot(c, 4); c = (c + b) & _MASK      # noqa: E702
        b = (b - a) & _MASK; b ^= _rot(a, 6); a = (a + c) & _MASK      # noqa: E702
        c = (c - b) & _MASK; c ^= _rot(b, 8); b = (b + a) & _MASK      # noqa: E702
        a = (a - c) & _MASK; a ^= _rot(c, 16); c = (c + b) & _MASK     # noqa: E702
        b = (b - a) & _MASK; b ^= _rot(a, 19); a = (a + c) & _MASK     # noqa: E702
        c = (c - b) & _MASK; c ^= _rot(b, 4); b = (b + a) & _MASK      # noqa: E702
        off += 12
        remaining -= 12
    if remaining == 0:
        return c
    tail = data[off:] + b"\x00" * 12
    words = struct.unpack_from("<III", tail, 0)
    masks = [_MASK >> (8 * max(0, 4 - min(4, remaining - 4 * i))) if remaining > 4 * i else 0 for i in range(3)]
    a = (a + (words[0] & masks[0])) & _MASK
    b = (b + (words[1] & masks[1])) & _MASK
    c = (c + (words[2] & masks[2])) & _MASK
    c ^= b; c = (c - _rot(b, 14)) & _MASK    # noqa: E702
    a ^= c; a = (a - _rot(c, 11)) & _MASK    # noqa: E702
    b ^= a; b = (b - _rot(a, 25)) & _MASK    # noqa: E702
    c ^= b; c = (c - _rot(b, 16)) & _MASK    # noqa: E702
    a ^= c; a = (a - _rot(c, 4)) & _MASK     # noqa: E702
    b ^= a; b = (b - _rot(a, 14)) & _MASK    # noqa: E702
    c ^= b; c = (c - _rot(b, 24)) & _MASK    # noqa: E702
    return c


def chacha20_key_nonce(filename: str) -> Tuple[bytes, bytes]:
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
    seed = hashlittle(basename.encode("utf-8"), CHACHA20_HASH_INITVAL)
    nonce = struct.pack("<I", seed) * 4
    key_base = seed ^ CHACHA20_KEY_XOR
    key = b"".join(struct.pack("<I", key_base ^ d) for d in CHACHA20_XOR_DELTAS)
    return key, nonce


def chacha20_decrypt(data: bytes, filename: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

    key, nonce = chacha20_key_nonce(filename)
    return Cipher(algorithms.ChaCha20(key, nonce), mode=None).decryptor().update(data)


def decode_entry(raw: bytes, flags: int, orig_size: int, vpath: str) -> bytes:
    compression, encryption = flags & 0xF, (flags >> 4) & 0xF
    data = raw
    if encryption == 3:
        data = chacha20_decrypt(data, vpath)
    elif encryption:
        raise CompileError("A game file uses an encryption the Studio cannot read.",
                           details=f"{vpath}: encryption type {encryption}")
    if compression == 2:
        import lz4.block

        try:
            data = lz4.block.decompress(data, uncompressed_size=orig_size)
        except Exception as exc:  # noqa: BLE001 - lz4 raises LZ4BlockError
            raise CompileError("A game file could not be decompressed.", details=f"{vpath}: {exc}") from exc
    elif compression:
        raise CompileError("A game file uses a compression the Studio cannot read.",
                           details=f"{vpath}: compression type {compression}")
    if len(data) != orig_size:
        raise CompileError("A game file has an unexpected size.", details=f"{vpath}: {len(data)} != {orig_size}")
    return data


class GameFileReader:
    """Reads original files by their Analyzer ``asset`` rows, verifying the recorded hash."""

    def __init__(self, game_root: Path, conn) -> None:
        self.root = Path(game_root)
        self.conn = conn   # read-only connection to the Analyzer snapshot

    def entry_for_asset(self, asset_id: int) -> Dict[str, Any]:
        row = self.conn.execute(
            "SELECT a.id, a.origin, a.vpath, a.size, a.content_hash, a.hash_kind, e.package, e.paz_index, e.offset,"
            " e.comp_size, e.orig_size, e.flags FROM asset a LEFT JOIN archive_entry e ON e.id=a.archive_entry_id"
            " WHERE a.id=?", (asset_id,)).fetchone()
        if row is None:
            raise CompileError("The Analyzer database does not know this game file.", details=f"asset {asset_id}")
        return dict(row)

    def package_for_vpath(self, vpath: str, installation_id: int) -> Optional[str]:
        row = self.conn.execute("SELECT package FROM archive_entry WHERE installation_id=? AND vpath=? LIMIT 1",
                                (installation_id, vpath)).fetchone()
        return row[0] if row else None

    def read(self, asset_id: int) -> bytes:
        e = self.entry_for_asset(asset_id)
        if e["origin"] == "archive":
            if e["package"] is None:
                raise CompileError("The Analyzer database has no archive location for a game file.", details=e["vpath"])
            paz = self.root / e["package"] / f"{e['paz_index']}.paz"
            try:
                with open(os_path(paz), "rb") as handle:
                    handle.seek(e["offset"])
                    raw = handle.read(e["comp_size"])
            except OSError as exc:
                raise CompileError("A game archive could not be read.", hint="Check the Crimson Desert folder.",
                                   details=f"{paz}: {exc}") from exc
            if len(raw) != e["comp_size"]:
                raise CompileError("A game archive is shorter than the Analyzer database says.",
                                   hint="The game was probably updated; re-run Crimson Desert Analyzer.", details=str(paz))
            data = decode_entry(raw, e["flags"], e["orig_size"], e["vpath"])
        elif e["origin"] == "loose":
            try:
                data = Path(os_path(self.root / e["vpath"])).read_bytes()
            except OSError as exc:
                raise CompileError("A game file could not be read.", details=f"{e['vpath']}: {exc}") from exc
        else:
            raise CompileError("This game file is stored inside another file and cannot be read directly.",
                               details=f"{e['origin']}: {e['vpath']}")
        if e["hash_kind"] == "sha1" and e["content_hash"] and hashlib.sha1(data).hexdigest() != e["content_hash"]:
            raise CompileError("A game file does not match the Analyzer database.",
                               hint="The game was probably updated after the database was created. Re-run Crimson "
                                    "Desert Analyzer and import the new database.", details=e["vpath"])
        return data
