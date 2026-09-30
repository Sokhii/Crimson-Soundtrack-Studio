"""Shared cache of expensive audio work (``data/cache/audio_analysis.sqlite3``).

* probe cache - keyed by (path, size, mtime): container/tag metadata and the
  audio identity, so unchanged files are not re-opened;
* feature cache - keyed by (audio identity, analysis version): signal
  measurements, so the same audio is analysed once even if it is renamed,
  re-tagged, moved or used by several projects.

The cache is disposable: deleting it only costs time.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .features import AudioFeatures
from .formats import ProbeResult

CACHE_VERSION = 1


class AnalysisCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        try:
            self.conn = self._open(path)
        except sqlite3.DatabaseError:
            # a damaged cache is simply rebuilt
            path.unlink(missing_ok=True)
            self.conn = self._open(path)

    @staticmethod
    def _open(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.DatabaseError:
            conn.close()  # release the file so a damaged cache can be deleted (Windows locks open files)
            raise
        if version != CACHE_VERSION:
            conn.executescript("""
                DROP TABLE IF EXISTS probe; DROP TABLE IF EXISTS features;
                CREATE TABLE probe (path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                                    identity TEXT NOT NULL, probe_json TEXT NOT NULL);
                CREATE TABLE features (identity TEXT NOT NULL, analysis_version INTEGER NOT NULL,
                                       features_json TEXT NOT NULL, PRIMARY KEY (identity, analysis_version));
            """)
            conn.execute(f"PRAGMA user_version={CACHE_VERSION}")
            conn.commit()
        return conn

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def get_probe(self, path: str, size: int, mtime_ns: int) -> Optional[ProbeResult]:
        with self.lock:
            row = self.conn.execute("SELECT size, mtime_ns, probe_json FROM probe WHERE path=?", (path,)).fetchone()
        if not row or row[0] != size or row[1] != mtime_ns:
            return None
        try:
            return ProbeResult(**json.loads(row[2]))
        except (ValueError, TypeError):
            return None

    def put_probe(self, path: str, size: int, mtime_ns: int, probe: ProbeResult) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO probe(path, size, mtime_ns, identity, probe_json) VALUES (?,?,?,?,?)",
                              (path, size, mtime_ns, probe.identity, json.dumps(asdict(probe), ensure_ascii=False)))
            self.conn.commit()

    def get_features(self, identity: str, version: int) -> Optional[AudioFeatures]:
        with self.lock:
            row = self.conn.execute("SELECT features_json FROM features WHERE identity=? AND analysis_version=?",
                                    (identity, version)).fetchone()
        if not row:
            return None
        try:
            return AudioFeatures(**json.loads(row[0]))
        except (ValueError, TypeError):
            return None

    def put_features(self, identity: str, features: AudioFeatures) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO features(identity, analysis_version, features_json) VALUES (?,?,?)",
                              (identity, features.analysis_version, json.dumps(features.to_dict())))
            self.conn.commit()
