"""Safe import of a user-supplied Analyzer database.

Why a snapshot instead of opening the file in place: the Analyzer keeps its
database in WAL journal mode, and SQLite creates ``-wal``/``-shm`` side files
beside a WAL database *even for a read-only connection*. Opening the user's
file directly would therefore change the folder it lives in. The Studio never
does that. Instead it:

1. reads the database (and its ``-wal`` file if one exists) byte-for-byte
   with plain file reads, hashing as it copies, into ``temp/``;
2. consolidates the copy into a single rollback-journal file with SQLite's
   backup API;
3. validates the consolidated copy;
4. stores it under ``data/databases/<hash>/analyzer.sqlite3`` with an
   ``import.json`` manifest.

The snapshot is then opened ``mode=ro&immutable=1`` and never written. The
original file is only ever opened with ``"rb"``. Because the snapshot lives
inside the portable folder, projects keep working if the original moves.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from ..app_paths import AppPaths, os_path
from ..errors import AnalyzerDbError, OperationCancelled
from . import contract
from .validator import ValidationReport, validate

log = logging.getLogger(__name__)

SNAPSHOT_NAME = "analyzer.sqlite3"
MANIFEST_NAME = "import.json"
# Where the Analyzer keeps its database relative to its own folder.
ANALYZER_DB_RELATIVE = Path("data") / "database" / "studio.sqlite3"
COPY_CHUNK = 4 * 1024 * 1024

Progress = Optional[Callable[[str, int, int], None]]


@dataclass
class ImportedDatabase:
    sha256: str
    snapshot_path: Path
    manifest: dict
    report: ValidationReport
    reused: bool = False

    @property
    def schema_version(self) -> Optional[int]:
        return self.report.schema_version


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_source(path: Path) -> Path:
    """Accept a database file or an Analyzer folder (drag/drop of the whole folder)."""

    path = Path(path)
    if path.is_dir():
        for candidate in (path / ANALYZER_DB_RELATIVE, path / "studio.sqlite3"):
            if candidate.is_file():
                return candidate
        raise AnalyzerDbError(
            "This folder does not contain a Crimson Desert Analyzer database.",
            hint=f"Select the database file itself, usually '{ANALYZER_DB_RELATIVE.as_posix()}' inside the Analyzer folder.",
            details=str(path))
    name = path.name.lower()
    for suffix in ("-wal", "-shm", "-journal"):
        if name.endswith(suffix):
            sibling = path.with_name(path.name[: -len(suffix)])
            if sibling.is_file():
                return sibling
            raise AnalyzerDbError("This is a SQLite side file, not the database itself.",
                                  hint="Select the main database file instead.", details=str(path))
    return path


def _check_header(path: Path) -> None:
    if not path.is_file():
        raise AnalyzerDbError("The selected Analyzer database file does not exist.", details=str(path))
    try:
        with open(os_path(path), "rb") as handle:
            header = handle.read(100)
    except OSError as exc:
        raise AnalyzerDbError("The selected Analyzer database could not be read.",
                              hint="Check that the file is not locked by another program and that you can read it.",
                              details=f"{path}: {exc}") from exc
    if len(header) < 100 or not header.startswith(contract.SQLITE_MAGIC):
        raise AnalyzerDbError(
            "The selected file is not a SQLite database.",
            hint="Select the database created by Crimson Desert Analyzer "
                 f"(usually '{ANALYZER_DB_RELATIVE.as_posix()}').",
            details=f"{path}: header {header[:16]!r}")


def _copy_hashing(src: Path, dst: Path, digest, progress: Progress, done: int, total: int,
                  cancel: Optional[Callable[[], bool]]) -> int:
    with open(os_path(src), "rb") as fin, open(dst, "wb") as fout:
        while True:
            if cancel and cancel():
                raise OperationCancelled()
            block = fin.read(COPY_CHUNK)
            if not block:
                break
            digest.update(block)
            fout.write(block)
            done += len(block)
            if progress:
                progress("Copying Analyzer database", done, total)
    return done


def import_database(source: Path, paths: AppPaths, progress: Progress = None,
                    cancel: Optional[Callable[[], bool]] = None) -> ImportedDatabase:
    source = resolve_source(Path(source))
    _check_header(source)
    wal = source.with_name(source.name + "-wal")
    has_wal = wal.is_file() and wal.stat().st_size > 0
    total = source.stat().st_size + (wal.stat().st_size if has_wal else 0)
    stat = source.stat()

    work = paths.temp / f"import-{uuid.uuid4().hex[:12]}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        digest = hashlib.sha256()
        raw = work / "raw.sqlite3"
        done = _copy_hashing(source, raw, digest, progress, 0, total, cancel)
        if has_wal:
            digest.update(b"\x00--wal--\x00")
            _copy_hashing(wal, work / "raw.sqlite3-wal", digest, progress, done, total, cancel)
        sha = digest.hexdigest()

        target_dir = paths.databases / sha[:32]
        target = target_dir / SNAPSHOT_NAME
        manifest_path = target_dir / MANIFEST_NAME
        if target.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("sha256") == sha:
                log.info("Analyzer database already imported (sha256 %s); reusing snapshot", sha[:12])
                manifest.update(_source_info(source, stat))
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                return ImportedDatabase(sha, target, manifest, ValidationReport.from_dict(manifest["validation"]), True)

        if progress:
            progress("Checking Analyzer database", 0, 0)
        snapshot = work / SNAPSHOT_NAME
        try:
            src_conn = sqlite3.connect(str(raw))
            try:
                dst_conn = sqlite3.connect(str(snapshot))
                try:
                    src_conn.backup(dst_conn)
                    dst_conn.execute("PRAGMA journal_mode=DELETE")
                finally:
                    dst_conn.close()
            finally:
                src_conn.close()
        except sqlite3.DatabaseError as exc:
            raise AnalyzerDbError(
                "The Analyzer database is damaged and cannot be read.",
                hint="If Crimson Desert Analyzer is still running, close it and try again. Otherwise generate a new database.",
                details=f"{source}: {exc}") from exc

        conn = open_snapshot(snapshot)
        try:
            report = validate(conn)
        finally:
            conn.close()
        if not report.ok:
            first = report.errors[0]
            raise AnalyzerDbError(first.message, hint=_hint_for(first.code), details=f"{source}: {first.details}")

        manifest = {
            "format": "css-analyzer-import",
            "format_version": 1,
            "sha256": sha,
            "imported_at": now_iso(),
            "included_wal": has_wal,
            "schema_version": report.schema_version,
            "validation": report.to_dict(),
            **_source_info(source, stat),
        }
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_target = target_dir / (SNAPSHOT_NAME + ".partial")
        shutil.move(str(snapshot), str(tmp_target))
        tmp_target.replace(target)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.info("Imported Analyzer database schema=%s sha256=%s banks=%s music_objects=%s",
                 report.schema_version, sha[:12], report.counts.get("banks"), report.counts.get("music_objects"))
        return ImportedDatabase(sha, target, manifest, report)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _source_info(source: Path, stat) -> dict:
    return {"source_path": str(source), "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}


def _hint_for(code: str) -> str:
    return {
        "corrupt": "Generate a new database with Crimson Desert Analyzer.",
        "not_analyzer": f"Select the database created by Crimson Desert Analyzer (usually '{ANALYZER_DB_RELATIVE.as_posix()}').",
        "no_installation": "Run 'Analyze Game' in Crimson Desert Analyzer, then select the database again.",
        "no_completed_scan": "Run the Analyzer scan again and let it finish, then select the database again.",
        "no_banks": "Run a full Analyzer scan of your Crimson Desert installation.",
    }.get(code, "Generate a new Analyzer database or select a compatible one.")


def open_snapshot(path: Path) -> sqlite3.Connection:
    """Open a Studio-owned snapshot strictly read-only (no journal/side files are created)."""

    uri = Path(path).resolve().as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_imported(paths: AppPaths, snapshot_path: Path) -> ImportedDatabase:
    manifest_path = snapshot_path.parent / MANIFEST_NAME
    if not snapshot_path.is_file() or not manifest_path.is_file():
        raise AnalyzerDbError("The imported Analyzer database is missing from the Studio folder.",
                              hint="Select the Analyzer database again.", details=str(snapshot_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = ValidationReport.from_dict(manifest["validation"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AnalyzerDbError("The imported Analyzer database record is damaged.",
                              hint="Select the Analyzer database again.", details=f"{manifest_path}: {exc}") from exc
    return ImportedDatabase(manifest["sha256"], snapshot_path, manifest, report, reused=True)


def source_changed(manifest: dict) -> Optional[bool]:
    """True if the original file changed since import, False if not, None if it is gone."""

    path = Path(manifest.get("source_path", ""))
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size != manifest.get("source_size") or stat.st_mtime_ns != manifest.get("source_mtime_ns")
