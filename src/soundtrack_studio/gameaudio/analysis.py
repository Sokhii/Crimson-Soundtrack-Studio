"""Read-only analysis of the game's own music.

For each music source the original ``.wem`` is read through the Analyzer's
records (a streamed file from the game archives, or the copy embedded in a
soundbank), verified against the Analyzer's hash, written to
``temp/gameaudio/``, decoded there with vgmstream, measured with the same
signal analysis as the user's tracks, optionally listened to by the listening
model, and deleted again. Only the numbers are kept, in
``data/cache/game_audio.sqlite3``, keyed by the audio's content hash so every
project (and a re-imported database of the same game version) reuses them.
Nothing is ever written to the game folder, and no game audio is kept.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..app_paths import AppPaths, os_path
from ..compiler.archive import CompileError, GameFileReader
from ..compiler import bnk
from ..errors import OperationCancelled, StudioError
from ..library.features import ANALYSIS_VERSION, FRAME, HOP, analyze_blocks
from .decoder import DecodeError, decode

log = logging.getLogger(__name__)

# bump when the decode/measure pipeline changes in a way that invalidates cached results
PIPELINE_VERSION = 1
MEASURE_VERSION = PIPELINE_VERSION * 1000 + ANALYSIS_VERSION
BLOCK_SECONDS = 10

Listener = Callable[[Path], Dict[str, Any]]


@dataclass
class SourceLocation:
    source_id: int
    asset_id: int
    origin: str                 # archive | loose | embedded
    vpath: str
    size: Optional[int]
    content_hash: Optional[str]
    hash_kind: Optional[str]
    bank_asset_id: Optional[int] = None
    codec: Optional[str] = None
    duration_s: Optional[float] = None

    @property
    def key(self) -> Optional[str]:
        if not self.content_hash:
            return None
        return f"{self.hash_kind or 'hash'}:{self.content_hash}:{self.size or 0}"


@dataclass
class SourceResult:
    source_id: int
    status: str = "pending"            # ok | error | missing | pending
    features: Dict[str, Any] = field(default_factory=dict)
    listening: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    decoded_s: Optional[float] = None
    cached: bool = False
    origin: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"source_id": self.source_id, "status": self.status, "error": self.error, "decoded_s": self.decoded_s,
                "origin": self.origin, "features": self.features, "listening": self.listening}


class GameAudioCache:
    """Shared, thread-safe store of measurements and listening results for game audio."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(os_path(path), check_same_thread=False)
        self.conn.executescript(
            "CREATE TABLE IF NOT EXISTS measurement (key TEXT PRIMARY KEY, version INTEGER NOT NULL, status TEXT NOT NULL,"
            " features_json TEXT, error TEXT, decoded_s REAL);"
            "CREATE TABLE IF NOT EXISTS listening (key TEXT NOT NULL, model_key TEXT NOT NULL, result_json TEXT NOT NULL,"
            " PRIMARY KEY (key, model_key));")
        self.conn.commit()

    def get_measure(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            row = self.conn.execute("SELECT status, features_json, error, decoded_s FROM measurement WHERE key=? AND version=?",
                                    (key, MEASURE_VERSION)).fetchone()
        if row is None:
            return None
        return {"status": row[0], "features": json.loads(row[1]) if row[1] else {}, "error": row[2] or "",
                "decoded_s": row[3]}

    def put_measure(self, key: str, status: str, features: Optional[Dict[str, Any]], error: str = "",
                    decoded_s: Optional[float] = None) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO measurement VALUES (?,?,?,?,?,?)",
                              (key, MEASURE_VERSION, status, json.dumps(features) if features else None, error, decoded_s))
            self.conn.commit()

    def get_listening(self, key: str, model_key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            row = self.conn.execute("SELECT result_json FROM listening WHERE key=? AND model_key=?",
                                    (key, model_key)).fetchone()
        return json.loads(row[0]) if row else None

    def put_listening(self, key: str, model_key: str, result: Dict[str, Any]) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO listening VALUES (?,?,?)", (key, model_key, json.dumps(result)))
            self.conn.commit()

    def forget_errors(self) -> int:
        with self.lock:
            n = self.conn.execute("DELETE FROM measurement WHERE status='error'").rowcount
            self.conn.commit()
        return n

    def close(self) -> None:
        with self.lock:
            self.conn.close()


def _verify(data: bytes, loc: SourceLocation) -> None:
    kind, expected = (loc.hash_kind or ""), (loc.content_hash or "")
    if not expected:
        return
    if kind == "sha1":
        actual = hashlib.sha1(data).hexdigest()
    elif kind.startswith("sha1_prefix_"):
        try:
            n = int(kind.rsplit("_", 1)[1])
        except ValueError:
            return
        actual = hashlib.sha1(data[:n]).hexdigest()
    else:
        return
    if actual != expected:
        raise CompileError("A game audio file does not match the Analyzer database.",
                           hint="The game was probably updated after the database was created. Re-run Crimson Desert "
                                "Analyzer and import the new database.", details=loc.vpath)


class GameAudioAnalyzer:
    def __init__(self, paths: AppPaths, game_root: Path, conn, installation_id: int, cache: GameAudioCache,
                 vgmstream: Optional[Path]) -> None:
        self.paths = paths
        self.reader = GameFileReader(game_root, conn)
        self.conn = conn
        self.installation_id = installation_id
        self.cache = cache
        self.vgmstream = vgmstream
        self._banks: Dict[int, List[bnk.Chunk]] = {}

    # ----------------------------------------------------------------- locate
    def locate(self, source_id: int) -> Optional[SourceLocation]:
        rows = self.conn.execute(
            "SELECT w.container, w.bank_asset_id, w.codec, w.duration_s, a.id, a.origin, a.vpath, a.size, a.content_hash,"
            " a.hash_kind FROM wem w JOIN asset a ON a.id=w.asset_id WHERE a.installation_id=? AND w.source_id=?"
            " AND w.valid=1", (self.installation_id, source_id)).fetchall()
        if not rows:
            return None

        def rank(r) -> tuple:
            # a complete streamed file first; the embedded copy of a streamed sound is only its prefetch start
            return (0 if r[5] in ("archive", "loose") else 1, -(r[7] or 0))

        r = sorted(rows, key=rank)[0]
        return SourceLocation(source_id, r[4], r[5], r[6], r[7], r[8], r[9], r[1], r[2], r[3])

    def read_wem(self, loc: SourceLocation) -> bytes:
        if loc.origin in ("archive", "loose"):
            data = self.reader.read(loc.asset_id)
        elif loc.origin == "embedded" and loc.bank_asset_id:
            chunks = self._banks.get(loc.bank_asset_id)
            if chunks is None:
                chunks = bnk.parse_chunks(self.reader.read(loc.bank_asset_id))
                self._banks = {loc.bank_asset_id: chunks}   # keep one bank: memory stays bounded
            data = bnk.media_data(chunks, loc.source_id)
            if data is None:
                raise CompileError("A soundbank does not contain the audio the Analyzer database lists.",
                                   details=f"{loc.vpath}")
        else:
            raise CompileError("The Analyzer database does not say where this game audio is stored.",
                               details=f"{loc.origin}: {loc.vpath}")
        _verify(data, loc)
        return data

    # ---------------------------------------------------------------- analyse
    def cached(self, source_id: int, listen_key: str = "") -> SourceResult:
        """The stored result for ``source_id`` without touching the game files."""

        loc = self.locate(source_id)
        result = SourceResult(source_id)
        if loc is None:
            result.status, result.error = "missing", "no audio file recorded for this source"
            return result
        result.origin = loc.origin
        key = loc.key
        m = self.cache.get_measure(key) if key else None
        if m:
            result.status, result.features, result.error, result.decoded_s = m["status"], m["features"], m["error"], m["decoded_s"]
            result.cached = True
        if key and listen_key:
            result.listening = self.cache.get_listening(key, listen_key) or {}
        return result

    def analyze(self, source_ids: Iterable[int], listener: Optional[Listener] = None, listen_key: str = "",
                progress: Optional[Callable[[str, int, int], None]] = None,
                cancel: Optional[Callable[[], bool]] = None, retry_errors: bool = False) -> Dict[int, SourceResult]:
        ids = list(dict.fromkeys(int(s) for s in source_ids))
        results: Dict[int, SourceResult] = {}
        work = self.paths.temp / "gameaudio"
        work.mkdir(parents=True, exist_ok=True)
        label = "Listening to the game's music" if listener else "Analysing the game's music"
        for index, sid in enumerate(ids):
            if cancel and cancel():
                raise OperationCancelled()
            if progress:
                progress(label, index, len(ids))
            res = self.cached(sid, listen_key if listener else "")
            need_measure = res.status == "pending" or (retry_errors and res.status == "error")
            need_listen = listener is not None and res.status != "missing" and not res.listening \
                and not (res.status == "error" and not retry_errors)
            if not need_measure and not need_listen:
                results[sid] = res
                continue
            results[sid] = self._process(sid, res, listener if need_listen else None, listen_key, work, cancel)
        if progress:
            progress(label, len(ids), len(ids))
        return results

    def _process(self, sid: int, res: SourceResult, listener: Optional[Listener], listen_key: str, work: Path,
                 cancel) -> SourceResult:
        loc = self.locate(sid)
        if loc is None:
            return res
        if self.vgmstream is None:
            res.status, res.error = "error", "the audio decoder (vgmstream) is not installed"
            return res           # not cached: installing the decoder fixes it
        wem, wav = work / f"{sid}.wem", work / f"{sid}.wav"
        try:
            data = self.read_wem(loc)
            key = loc.key or f"sha1:{hashlib.sha1(data).hexdigest()}:{len(data)}"
            wem.write_bytes(data)
            del data
            decode(self.vgmstream, wem, wav)
            wem.unlink(missing_ok=True)
            if res.status != "ok":
                features = self._measure(wav, cancel)
                res.features = features
                res.decoded_s = features.get("decoded_duration_s")
                res.status, res.error = "ok", ""
                self.cache.put_measure(key, "ok", features, decoded_s=res.decoded_s)
            if listener is not None:
                try:
                    res.listening = listener(wav)
                    self.cache.put_listening(key, listen_key, res.listening)
                except StudioError as exc:
                    log.warning("Listening failed for game source %s: %s %s", sid, exc.message, exc.details)
        except OperationCancelled:
            raise
        except (DecodeError, CompileError) as exc:
            res.status, res.error = "error", f"{exc.message} {exc.details}".strip()[:500]
            if loc.key:
                self.cache.put_measure(loc.key, "error", None, res.error)
            log.warning("Game audio %s: %s", sid, res.error)
        except (OSError, RuntimeError, ValueError) as exc:
            res.status, res.error = "error", f"{type(exc).__name__}: {exc}"[:500]
            if loc.key:
                self.cache.put_measure(loc.key, "error", None, res.error)
            log.warning("Game audio %s: %s", sid, res.error)
        finally:
            wem.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)
        return res

    @staticmethod
    def _measure(wav: Path, cancel) -> Dict[str, Any]:
        import soundfile as sf

        with sf.SoundFile(os_path(wav), mode="r") as f:
            rate = f.samplerate
            blocks = f.blocks(blocksize=rate * BLOCK_SECONDS // HOP * HOP, overlap=FRAME - HOP, dtype="float32",
                              always_2d=True)
            features = analyze_blocks(blocks, rate, cancel=cancel)
        return features.to_dict()

    # ------------------------------------------------------------ decode check
    def decode_check(self, source_ids: List[int], count: int = 6) -> Dict[str, Any]:
        """Decode a few sources (both storage kinds when available) without caching; report what happened."""

        work = self.paths.temp / "gameaudio"
        work.mkdir(parents=True, exist_ok=True)
        picked: List[SourceLocation] = []
        seen_origins: Dict[str, int] = {}
        for sid in source_ids:
            loc = self.locate(sid)
            if loc is None:
                continue
            if seen_origins.get(loc.origin, 0) < max(1, count // 2):
                picked.append(loc)
                seen_origins[loc.origin] = seen_origins.get(loc.origin, 0) + 1
            if len(picked) >= count:
                break
        items = []
        for loc in picked:
            item: Dict[str, Any] = {"source_id": loc.source_id, "stored": loc.origin, "codec": loc.codec,
                                    "expected_s": loc.duration_s}
            wem, wav = work / f"check_{loc.source_id}.wem", work / f"check_{loc.source_id}.wav"
            try:
                if self.vgmstream is None:
                    raise DecodeError("The audio decoder (vgmstream) is not installed.")
                wem.write_bytes(self.read_wem(loc))
                decode(self.vgmstream, wem, wav)
                import soundfile as sf

                info = sf.info(os_path(wav))
                item.update(ok=True, decoded_s=round(info.frames / info.samplerate, 3), channels=info.channels,
                            sample_rate=info.samplerate)
                if loc.duration_s and abs(item["decoded_s"] - loc.duration_s) > 0.5:
                    item["note"] = "decoded length differs from the Analyzer's"
            except (StudioError, OSError, RuntimeError) as exc:
                item.update(ok=False, error=f"{getattr(exc, 'message', str(exc))} {getattr(exc, 'details', '')}".strip()[:400])
            finally:
                wem.unlink(missing_ok=True)
                wav.unlink(missing_ok=True)
            items.append(item)
        ok = sum(1 for i in items if i.get("ok"))
        return {"checked": len(items), "decoded": ok, "items": items,
                "passed": bool(items) and ok == len(items)}
