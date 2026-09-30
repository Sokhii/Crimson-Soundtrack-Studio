"""Studio project database (``projects/<slug>/project.sqlite3``).

A project is a folder under ``projects/`` with its own SQLite database. It
never shares tables with the Analyzer database. Paths inside the portable
folder are stored relative (``app:...``) so a moved Studio keeps its projects;
external paths (game, music) are stored absolute and checked on open.

Deterministic facts (file metadata, audio features) and - in later phases -
AI interpretations and user decisions live in separate tables so they can
never be confused.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .. import __version__
from ..app_paths import AppPaths
from ..errors import ProjectError

log = logging.getLogger(__name__)

PROJECT_DB_NAME = "project.sqlite3"
APPLICATION_ID = 0x43535331  # "CSS1" - identifies Studio project files
PROJECT_FORMAT = "crimson-soundtrack-studio-project"

MIGRATIONS: List[tuple[int, str]] = [
    (1, """
    CREATE TABLE project_meta (key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE project_setting (key TEXT PRIMARY KEY, value TEXT);

    -- Analyzer database references (the snapshot lives in data/databases/, read-only)
    CREATE TABLE analyzer_ref (
        id INTEGER PRIMARY KEY,
        sha256 TEXT NOT NULL,
        snapshot_path TEXT NOT NULL,         -- stored path (app:...)
        source_path TEXT,
        schema_version INTEGER,
        installation_id INTEGER,
        scan_id INTEGER,
        imported_at TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        validation_json TEXT
    );

    CREATE TABLE game_check (
        id INTEGER PRIMARY KEY,
        checked_at TEXT NOT NULL,
        game_path TEXT NOT NULL,
        analyzer_sha256 TEXT,
        status TEXT NOT NULL,
        details_json TEXT
    );

    CREATE TABLE library_root (
        id INTEGER PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        added_at TEXT NOT NULL,
        last_scan_at TEXT
    );

    CREATE TABLE library_scan (
        id INTEGER PRIMARY KEY,
        root_id INTEGER NOT NULL REFERENCES library_root(id) ON DELETE CASCADE,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL,
        stats_json TEXT
    );

    -- one row per audio file found in a library root (user files are never modified)
    CREATE TABLE track (
        id INTEGER PRIMARY KEY,
        root_id INTEGER NOT NULL REFERENCES library_root(id) ON DELETE CASCADE,
        rel_path TEXT NOT NULL,
        format TEXT NOT NULL,
        size INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL,
        identity TEXT,                       -- audio identity (FLAC MD5 or content hash)
        status TEXT NOT NULL,                -- ok | error | missing
        error TEXT,
        duplicate_of INTEGER REFERENCES track(id) ON DELETE SET NULL,
        first_seen TEXT NOT NULL,
        last_seen_scan INTEGER,
        UNIQUE (root_id, rel_path)
    );
    CREATE INDEX idx_track_identity ON track(identity);

    -- deterministic facts read from the file container/tags
    CREATE TABLE track_metadata (
        track_id INTEGER PRIMARY KEY REFERENCES track(id) ON DELETE CASCADE,
        codec TEXT, container TEXT,
        duration_s REAL, sample_rate INTEGER, channels INTEGER, bit_depth INTEGER, total_samples INTEGER,
        title TEXT, artist TEXT, album TEXT, album_artist TEXT, composer TEXT, genre TEXT,
        year INTEGER, track_number INTEGER, disc_number INTEGER,
        has_cover INTEGER NOT NULL DEFAULT 0,
        tags_json TEXT
    );

    -- deterministic signal measurements (never AI output)
    CREATE TABLE track_features (
        track_id INTEGER PRIMARY KEY REFERENCES track(id) ON DELETE CASCADE,
        analysis_version INTEGER NOT NULL,
        features_json TEXT NOT NULL,
        computed_at TEXT NOT NULL
    );

    -- user-visible warnings/errors attached to the project
    CREATE TABLE project_event (
        id INTEGER PRIMARY KEY,
        at TEXT NOT NULL,
        level TEXT NOT NULL,
        category TEXT NOT NULL,
        message TEXT NOT NULL,
        details TEXT
    );
    """),
    (2, """
    -- interpretations (never facts): one row per entity and source (rules | llm)
    CREATE TABLE semantic_profile (
        id INTEGER PRIMARY KEY,
        entity_type TEXT NOT NULL,           -- track | cue
        entity_key TEXT NOT NULL,            -- track id | Wwise segment id
        source TEXT NOT NULL,                -- rules | llm
        model_id TEXT NOT NULL DEFAULT '',
        prompt_version INTEGER NOT NULL DEFAULT 0,
        input_hash TEXT NOT NULL,            -- hash of the evidence document the profile was made from
        status TEXT NOT NULL,                -- ok | error
        profile_json TEXT,
        evidence_json TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (entity_type, entity_key, source)
    );
    -- the user's edits; always win over rules and AI
    CREATE TABLE semantic_override (
        entity_type TEXT NOT NULL,
        entity_key TEXT NOT NULL,
        override_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (entity_type, entity_key)
    );
    """),
    (3, """
    CREATE TABLE match_run (
        id INTEGER PRIMARY KEY,
        created_at TEXT NOT NULL,
        settings_json TEXT NOT NULL,
        model_id TEXT NOT NULL DEFAULT '',
        stats_json TEXT
    );
    -- machine proposals: recomputed by every matching run
    CREATE TABLE match_proposal (
        cue_key TEXT NOT NULL,               -- Wwise segment id
        rank INTEGER NOT NULL,               -- 0 = proposed replacement, 1.. = alternatives
        track_id INTEGER NOT NULL,
        score REAL NOT NULL,
        confidence REAL NOT NULL,
        components_json TEXT,
        reasons_json TEXT,
        warnings_json TEXT,
        ai_fit REAL,
        ai_reason TEXT,
        run_id INTEGER,
        PRIMARY KEY (cue_key, rank)
    );
    -- the user's decisions: never touched by matching runs and always win
    CREATE TABLE match_decision (
        cue_key TEXT PRIMARY KEY,
        action TEXT NOT NULL CHECK (action IN ('accept', 'manual', 'reject', 'keep_original')),
        track_id INTEGER,
        rejected_json TEXT,                  -- tracks the user rejected for this cue
        fit_mode TEXT NOT NULL DEFAULT 'auto',   -- auto | trim | loop | pad
        start_offset_s REAL NOT NULL DEFAULT 0,  -- where in the user's track playback starts
        note TEXT,
        updated_at TEXT NOT NULL
    );
    """),
]
LATEST_VERSION = MIGRATIONS[-1][0]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if text.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        text = f"_{text}"
    return text[:80] or "Project"


@dataclass
class ProjectInfo:
    name: str
    folder: Path
    created_at: str = ""
    modified_at: str = ""
    error: str = ""


class Project:
    """An open project. Thread-safe: one connection guarded by a lock."""

    def __init__(self, folder: Path, conn: sqlite3.Connection) -> None:
        self.folder = folder
        self.db_path = folder / PROJECT_DB_NAME
        self.conn = conn
        self.lock = threading.RLock()

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, paths: AppPaths, name: str) -> "Project":
        name = name.strip()
        if not name:
            raise ProjectError("Please enter a project name.")
        slug = slugify(name)
        folder = paths.projects / slug
        n = 2
        while folder.exists():
            folder = paths.projects / f"{slug} ({n})"
            n += 1
        folder.mkdir(parents=True)
        conn = _connect(folder / PROJECT_DB_NAME)
        project = cls(folder, conn)
        project._migrate()
        now = now_iso()
        with project.transaction() as c:
            for key, value in {"name": name, "format": PROJECT_FORMAT, "created_at": now, "modified_at": now,
                               "created_by": __version__}.items():
                c.execute("INSERT OR REPLACE INTO project_meta(key, value) VALUES (?, ?)", (key, value))
        log.info("Created project '%s' in %s", name, folder.name)
        return project

    @classmethod
    def open(cls, folder: Path) -> "Project":
        folder = Path(folder)
        db = folder / PROJECT_DB_NAME
        if not db.is_file():
            raise ProjectError("The project could not be found.", hint="It may have been moved or deleted.",
                               details=str(db))
        try:
            with open(db, "rb") as handle:
                header = handle.read(16)
        except OSError as exc:
            raise ProjectError("The project file could not be read.", details=f"{db}: {exc}") from exc
        if header != b"SQLite format 3\x00":
            raise ProjectError("The project file is damaged and cannot be opened.",
                               hint="Restore the project folder from a backup, or create a new project.", details=str(db))
        try:
            conn = _connect(db)
            app_id = conn.execute("PRAGMA application_id").fetchone()[0]
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            raise ProjectError("The project file is damaged and cannot be opened.",
                               hint="Restore the project folder from a backup, or create a new project.",
                               details=f"{db}: {exc}") from exc
        if app_id != APPLICATION_ID:
            conn.close()
            raise ProjectError("This file is not a Crimson Soundtrack Studio project.", details=str(db))
        if version > LATEST_VERSION:
            conn.close()
            raise ProjectError(
                "This project was saved by a newer version of Crimson Soundtrack Studio.\n"
                f"Project format: {version}\nSupported by this version: up to {LATEST_VERSION}",
                hint="Update Crimson Soundtrack Studio to open it. The project was not changed.", details=str(db))
        project = cls(folder, conn)
        try:
            check = conn.execute("PRAGMA quick_check(5)").fetchall()
            if [r[0] for r in check] != ["ok"]:
                raise sqlite3.DatabaseError("; ".join(str(r[0]) for r in check))
            project._migrate()
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise ProjectError("The project file is damaged and cannot be opened.",
                               hint="Restore the project folder from a backup, or create a new project.",
                               details=f"{db}: {exc}") from exc
        log.info("Opened project '%s' (format %s)", project.name, version)
        return project

    def close(self) -> None:
        with self.lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass

    def _migrate(self) -> None:
        with self.lock:
            current = self.conn.execute("PRAGMA user_version").fetchone()[0]
            for version, sql in MIGRATIONS:
                if version <= current:
                    continue
                self.conn.executescript(
                    f"BEGIN;\n{sql}\nPRAGMA application_id = {APPLICATION_ID};\nPRAGMA user_version = {version};\nCOMMIT;")
                log.info("Project database migrated to format %d", version)

    # --------------------------------------------------------------- access
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchone()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    @property
    def format_version(self) -> int:
        return int(self.query_one("PRAGMA user_version")[0])

    # ------------------------------------------------------ meta & settings
    def meta(self, key: str, default: str = "") -> str:
        row = self.query_one("SELECT value FROM project_meta WHERE key=?", (key,))
        return row[0] if row and row[0] is not None else default

    @property
    def name(self) -> str:
        return self.meta("name", self.folder.name)

    def touch(self) -> None:
        self.execute("INSERT OR REPLACE INTO project_meta(key, value) VALUES ('modified_at', ?)", (now_iso(),))

    def get(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM project_setting WHERE key=?", (key,))
        if row is None or row[0] is None:
            return default
        try:
            return json.loads(row[0])
        except ValueError:
            return default

    def set(self, key: str, value: Any) -> None:
        with self.transaction() as c:
            c.execute("INSERT OR REPLACE INTO project_setting(key, value) VALUES (?, ?)",
                      (key, json.dumps(value, ensure_ascii=False)))
            c.execute("INSERT OR REPLACE INTO project_meta(key, value) VALUES ('modified_at', ?)", (now_iso(),))

    # ------------------------------------------------------------ events
    def add_event(self, level: str, category: str, message: str, details: str = "") -> None:
        self.execute("INSERT INTO project_event(at, level, category, message, details) VALUES (?,?,?,?,?)",
                     (now_iso(), level, category, message, details))

    def recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.query("SELECT * FROM project_event ORDER BY id DESC LIMIT ?", (limit,))]

    # ----------------------------------------------------- analyzer refs
    def set_analyzer(self, *, sha256: str, snapshot_stored: str, source_path: str, schema_version: Optional[int],
                     installation_id: Optional[int], scan_id: Optional[int], validation: Dict[str, Any]) -> None:
        with self.transaction() as c:
            c.execute("UPDATE analyzer_ref SET active=0")
            c.execute(
                "INSERT INTO analyzer_ref(sha256, snapshot_path, source_path, schema_version, installation_id, scan_id,"
                " imported_at, active, validation_json) VALUES (?,?,?,?,?,?,?,1,?)",
                (sha256, snapshot_stored, source_path, schema_version, installation_id, scan_id, now_iso(),
                 json.dumps(validation, ensure_ascii=False)))

    def active_analyzer(self) -> Optional[Dict[str, Any]]:
        row = self.query_one("SELECT * FROM analyzer_ref WHERE active=1 ORDER BY id DESC LIMIT 1")
        if row is None:
            return None
        data = dict(row)
        data["validation"] = json.loads(data.pop("validation_json") or "{}")
        return data

    def record_game_check(self, game_path: str, analyzer_sha: Optional[str], status: str, details: Dict[str, Any]) -> None:
        self.execute("INSERT INTO game_check(checked_at, game_path, analyzer_sha256, status, details_json) VALUES (?,?,?,?,?)",
                     (now_iso(), game_path, analyzer_sha, status, json.dumps(details, ensure_ascii=False)))

    def last_game_check(self) -> Optional[Dict[str, Any]]:
        row = self.query_one("SELECT * FROM game_check ORDER BY id DESC LIMIT 1")
        if row is None:
            return None
        data = dict(row)
        data["details"] = json.loads(data.pop("details_json") or "{}")
        return data


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = DELETE")  # single file on disk: safe to copy/move the folder when closed
    except sqlite3.DatabaseError:
        conn.close()
        raise
    return conn


def list_projects(paths: AppPaths) -> List[ProjectInfo]:
    out: List[ProjectInfo] = []
    if not paths.projects.is_dir():
        return out
    for folder in sorted(p for p in paths.projects.iterdir() if p.is_dir()):
        db = folder / PROJECT_DB_NAME
        if not db.is_file():
            continue
        info = ProjectInfo(name=folder.name, folder=folder)
        try:
            uri = db.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                meta = dict(conn.execute("SELECT key, value FROM project_meta").fetchall())
            finally:
                conn.close()
            info.name = meta.get("name", folder.name)
            info.created_at = meta.get("created_at", "")
            info.modified_at = meta.get("modified_at", "")
        except sqlite3.Error as exc:
            info.error = f"unreadable ({exc})"
        out.append(info)
    out.sort(key=lambda p: p.modified_at, reverse=True)
    return out
