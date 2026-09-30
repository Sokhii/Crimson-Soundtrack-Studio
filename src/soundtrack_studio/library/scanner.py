"""Recursive music library scanning into a project.

User files are only ever opened for reading. Unchanged files (same size and
modification time, current analysis version) are skipped; changed files are
re-probed, and their signal analysis is reused when the audio itself is
unchanged (same audio identity).
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..app_paths import os_path
from ..errors import LibraryError, OperationCancelled
from ..project.store import Project, now_iso
from .cache import AnalysisCache
from .features import ANALYSIS_VERSION, FRAME, HOP, AudioFeatures, analyze_blocks
from .formats import AudioReadError, ProbeResult, handler_for, supported_extensions

log = logging.getLogger(__name__)

BLOCK_SECONDS = 20


@dataclass
class ScanProgress:
    phase: str
    done: int = 0
    total: int = 0
    current: str = ""


@dataclass
class ScanStats:
    files_found: int = 0
    unchanged: int = 0
    analyzed: int = 0
    features_from_cache: int = 0
    probes_from_cache: int = 0
    errors: int = 0
    missing: int = 0
    duplicates: int = 0
    skipped_system_files: int = 0
    unreadable_folders: int = 0
    elapsed_s: float = 0.0
    error_samples: List[str] = field(default_factory=list)


@dataclass
class FileResult:
    rel_path: str
    size: int
    mtime_ns: int
    format: str
    probe: Optional[ProbeResult] = None
    features: Optional[AudioFeatures] = None
    error: Optional[str] = None
    probe_cached: bool = False
    features_cached: bool = False


def walk_library(root: Path, stats: ScanStats, cancel: Optional[Callable[[], bool]] = None) -> List[Tuple[str, Path, os.stat_result]]:
    extensions = set(supported_extensions())
    base = os_path(root.resolve(), force=True)  # children of a short base may exceed MAX_PATH
    found: List[Tuple[str, Path, os.stat_result]] = []

    def onerror(exc: OSError) -> None:
        stats.unreadable_folders += 1
        log.warning("Folder could not be read during library scan: %s", exc)

    for dirpath, dirnames, filenames in os.walk(base, onerror=onerror, followlinks=False):
        if cancel and cancel():
            raise OperationCancelled()
        dirnames.sort()
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() not in extensions:
                continue
            if name.startswith("._"):  # macOS resource-fork companions share the extension but are not audio
                stats.skipped_system_files += 1
                continue
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(full)
            except OSError:
                stats.errors += 1
                continue
            rel = os.path.relpath(full, base).replace("\\", "/")
            found.append((rel, Path(full), st))
    return found


def process_file(rel: str, path: Path, st: os.stat_result, cache: AnalysisCache,
                 cancel: Optional[Callable[[], bool]] = None) -> FileResult:
    handler = handler_for(path)
    result = FileResult(rel_path=rel, size=st.st_size, mtime_ns=st.st_mtime_ns, format=handler.name if handler else "?")
    if handler is None:
        result.error = "unsupported format"
        return result
    key = str(path)
    try:
        probe = cache.get_probe(key, st.st_size, st.st_mtime_ns)
        if probe is None:
            probe = handler.probe(path)
            cache.put_probe(key, st.st_size, st.st_mtime_ns, probe)
        else:
            result.probe_cached = True
        result.probe = probe
        features = cache.get_features(probe.identity, ANALYSIS_VERSION)
        if features is None:
            blocks = handler.blocks(path, probe.sample_rate * BLOCK_SECONDS // HOP * HOP, FRAME - HOP)
            features = analyze_blocks(blocks, probe.sample_rate, cancel=cancel)
            if probe.duration_s and features.decoded_duration_s and abs(features.decoded_duration_s - probe.duration_s) > 0.5:
                features.notes.append(
                    f"decoded length {features.decoded_duration_s:.2f}s differs from header {probe.duration_s:.2f}s; "
                    "the file may be truncated")
            cache.put_features(probe.identity, features)
        else:
            result.features_cached = True
        result.features = features
    except AudioReadError as exc:
        result.error = str(exc)
    return result


class LibraryScanner:
    def __init__(self, project: Project, cache: AnalysisCache, workers: Optional[int] = None) -> None:
        self.project = project
        self.cache = cache
        self.workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))

    def scan(self, root: Path, progress: Optional[Callable[[ScanProgress], None]] = None,
             cancel: Optional[Callable[[], bool]] = None) -> ScanStats:
        root = Path(root)
        if not root.is_dir():
            raise LibraryError("The music folder does not exist or cannot be opened.",
                               hint="Choose the folder again.", details=str(root))
        started = time.monotonic()
        stats = ScanStats()
        report = progress or (lambda p: None)
        root_id = self._root_id(root)
        scan_id = self.project.execute(
            "INSERT INTO library_scan(root_id, started_at, status) VALUES (?,?, 'running')", (root_id, now_iso())).lastrowid
        status = "failed"
        try:
            report(ScanProgress("Finding music files"))
            files = walk_library(root, stats, cancel)
            stats.files_found = len(files)
            existing = {r["rel_path"]: r for r in self.project.query(
                "SELECT t.id, t.rel_path, t.size, t.mtime_ns, t.status, f.analysis_version FROM track t"
                " LEFT JOIN track_features f ON f.track_id=t.id WHERE t.root_id=?", (root_id,))}
            todo = []
            unchanged_ids = []
            for rel, path, st in files:
                row = existing.get(rel)
                if (row is not None and row["size"] == st.st_size and row["mtime_ns"] == st.st_mtime_ns
                        and row["status"] == "ok" and row["analysis_version"] == ANALYSIS_VERSION):
                    unchanged_ids.append(row["id"])
                else:
                    todo.append((rel, path, st))
            stats.unchanged = len(unchanged_ids)
            with self.project.transaction() as c:
                c.executemany("UPDATE track SET last_seen_scan=? WHERE id=?", [(scan_id, i) for i in unchanged_ids])

            total = len(todo)
            report(ScanProgress("Analysing music", 0, total))
            done = 0
            with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="library") as pool:
                pending: Dict[Future, str] = {}
                queue = iter(todo)
                exhausted = False
                while pending or not exhausted:
                    while not exhausted and len(pending) < self.workers * 2:
                        item = next(queue, None)
                        if item is None:
                            exhausted = True
                            break
                        pending[pool.submit(process_file, *item, self.cache, cancel)] = item[0]
                    if not pending:
                        break
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        rel = pending.pop(fut)
                        result = fut.result()  # OperationCancelled propagates
                        self._store(root_id, scan_id, result, stats)
                        done += 1
                        report(ScanProgress("Analysing music", done, total, rel))
                    if cancel and cancel():
                        for fut in pending:
                            fut.cancel()
                        raise OperationCancelled()

            seen = {rel for rel, _p, _s in files}
            missing = [r["id"] for rel, r in existing.items() if rel not in seen and r["status"] != "missing"]
            with self.project.transaction() as c:
                c.executemany("UPDATE track SET status='missing', error=NULL WHERE id=?", [(i,) for i in missing])
            stats.missing = len(missing) + sum(1 for rel, r in existing.items() if rel not in seen and r["status"] == "missing")
            stats.duplicates = self._mark_duplicates()
            status = "completed"
            return stats
        except OperationCancelled:
            status = "cancelled"
            raise
        finally:
            stats.elapsed_s = round(time.monotonic() - started, 2)
            with self.project.transaction() as c:
                c.execute("UPDATE library_scan SET finished_at=?, status=?, stats_json=? WHERE id=?",
                          (now_iso(), status, json.dumps(asdict(stats)), scan_id))
                if status == "completed":
                    c.execute("UPDATE library_root SET last_scan_at=? WHERE id=?", (now_iso(), root_id))
            log.info("Library scan %s: found=%d unchanged=%d analyzed=%d cached_features=%d errors=%d missing=%d "
                     "duplicates=%d in %.1fs", status, stats.files_found, stats.unchanged, stats.analyzed,
                     stats.features_from_cache, stats.errors, stats.missing, stats.duplicates, stats.elapsed_s)

    # -------------------------------------------------------------- helpers
    def _root_id(self, root: Path) -> int:
        stored = str(root.resolve())
        row = self.project.query_one("SELECT id FROM library_root WHERE path=?", (stored,))
        if row:
            return int(row[0])
        return int(self.project.execute("INSERT INTO library_root(path, added_at) VALUES (?,?)",
                                        (stored, now_iso())).lastrowid)

    def _store(self, root_id: int, scan_id: int, r: FileResult, stats: ScanStats) -> None:
        status = "error" if r.error else "ok"
        if r.error:
            stats.errors += 1
            if len(stats.error_samples) < 20:
                stats.error_samples.append(f"{r.rel_path}: {r.error}")
        else:
            stats.analyzed += 1
            stats.features_from_cache += int(r.features_cached)
            stats.probes_from_cache += int(r.probe_cached)
        now = now_iso()
        with self.project.transaction() as c:
            c.execute(
                "INSERT INTO track(root_id, rel_path, format, size, mtime_ns, identity, status, error, first_seen, last_seen_scan)"
                " VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(root_id, rel_path) DO UPDATE SET format=excluded.format,"
                " size=excluded.size, mtime_ns=excluded.mtime_ns, identity=excluded.identity, status=excluded.status,"
                " error=excluded.error, last_seen_scan=excluded.last_seen_scan",
                (root_id, r.rel_path, r.format, r.size, r.mtime_ns, r.probe.identity if r.probe else None, status,
                 r.error, now, scan_id))
            track_id = c.execute("SELECT id FROM track WHERE root_id=? AND rel_path=?", (root_id, r.rel_path)).fetchone()[0]
            if r.probe:
                p, t = r.probe, r.probe.tags
                c.execute(
                    "INSERT OR REPLACE INTO track_metadata(track_id, codec, container, duration_s, sample_rate, channels,"
                    " bit_depth, total_samples, title, artist, album, album_artist, composer, genre, year, track_number,"
                    " disc_number, has_cover, tags_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (track_id, p.codec, p.container,
                     p.duration_s if p.duration_s is not None else (r.features.decoded_duration_s if r.features else None),
                     p.sample_rate, p.channels, p.bit_depth, p.total_samples, t.get("title"), t.get("artist"),
                     t.get("album"), t.get("album_artist"), t.get("composer"), t.get("genre"), t.get("year"),
                     t.get("track_number"), t.get("disc_number"), int(p.has_cover),
                     json.dumps({"raw": p.raw_tags, "warnings": p.warnings}, ensure_ascii=False)))
            else:
                c.execute("DELETE FROM track_metadata WHERE track_id=?", (track_id,))
            if r.features:
                c.execute("INSERT OR REPLACE INTO track_features(track_id, analysis_version, features_json, computed_at)"
                          " VALUES (?,?,?,?)", (track_id, r.features.analysis_version, json.dumps(r.features.to_dict()), now))
            else:
                c.execute("DELETE FROM track_features WHERE track_id=?", (track_id,))

    def _mark_duplicates(self) -> int:
        """Link tracks with identical audio (across all library roots of the project) to the first one."""

        # canonical copy = first by path, so the choice is stable whatever order worker threads finish in
        rows = self.project.query(
            "SELECT t.id, t.identity FROM track t JOIN library_root r ON r.id=t.root_id"
            " WHERE t.status='ok' AND t.identity IS NOT NULL ORDER BY r.path, t.rel_path")
        first: Dict[str, int] = {}
        updates = []
        for r in rows:
            canonical = first.setdefault(r["identity"], r["id"])
            updates.append((canonical if canonical != r["id"] else None, r["id"]))
        with self.project.transaction() as c:
            c.executemany("UPDATE track SET duplicate_of=? WHERE id=?", updates)
        return sum(1 for d, _ in updates if d is not None)
