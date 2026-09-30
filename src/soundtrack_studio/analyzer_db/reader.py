"""Read-only query layer over an imported Analyzer snapshot.

This module only issues SELECT statements against a connection opened with
``mode=ro&immutable=1``; the Studio has no code path that writes to Analyzer
data. Higher layers (``game_model``) turn these rows into Studio concepts.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import contract
from .importer import open_snapshot

NAME_SOURCE_PRIORITY = {"bnk_stid": 5, "soundbanksinfo": 4, "file_name": 4, "knowledge": 3, "community": 2}


def loads(value: Optional[str], default: Any = None) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class AnalyzerReader:
    def __init__(self, snapshot_path: Path) -> None:
        self.path = Path(snapshot_path)
        self.conn: sqlite3.Connection = open_snapshot(self.path)
        self._tables = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "AnalyzerReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def has_table(self, name: str) -> bool:
        return name in self._tables

    def _q(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        return self.conn.execute(sql, tuple(params)).fetchall()

    # ------------------------------------------------------------ basics
    def schema_version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    def installation(self, inst_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM installation WHERE id=?", (inst_id,)).fetchone()
        return dict(row) if row else None

    def latest_completed_scan(self, inst_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT id, started_at, finished_at, status, mode, parser_version FROM scan"
            " WHERE installation_id=? AND status='completed' ORDER BY id DESC LIMIT 1", (inst_id,)).fetchone()
        return dict(row) if row else None

    def source_files(self, inst_id: int, kinds: Iterable[str] = ("pamt", "papgt", "paz", "loose")) -> List[Dict[str, Any]]:
        kinds = list(kinds)
        marks = ",".join("?" * len(kinds))
        return [dict(r) for r in self._q(
            f"SELECT rel_path, kind, size, mtime_ns FROM source_file WHERE installation_id=? AND kind IN ({marks})"
            " ORDER BY rel_path", [inst_id, *kinds])]

    # ------------------------------------------------------------- names
    def best_names(self, ids: Iterable[int]) -> Dict[int, str]:
        """Best display name per ID: hash-verified first, then by source priority (same rule as the Analyzer)."""

        ids = sorted({int(i) for i in ids if i is not None})
        best: Dict[int, tuple] = {}
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            marks = ",".join("?" * len(part))
            for r in self._q(f"SELECT id_value, name, source, hash_verified FROM name WHERE id_value IN ({marks})", part):
                rank = (int(r["hash_verified"]), NAME_SOURCE_PRIORITY.get(r["source"], 1))
                if r["id_value"] not in best or rank > best[r["id_value"]][0]:
                    best[r["id_value"]] = (rank, r["name"])
        return {k: v[1] for k, v in best.items()}

    def verified_names(self) -> Dict[int, str]:
        return {r["id_value"]: r["name"] for r in self._q("SELECT id_value, name FROM name WHERE hash_verified=1")}

    # ------------------------------------------------------------- banks
    def banks(self, inst_id: int) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._q(
            "SELECT b.asset_id, b.bank_id, b.version, b.object_count, b.media_count, a.vpath, a.size, a.content_hash"
            " FROM bnk b JOIN asset a ON a.id=b.asset_id WHERE a.installation_id=? ORDER BY a.vpath", (inst_id,))]

    # ------------------------------------------------------ music objects
    def music_objects(self, inst_id: int) -> List[Dict[str, Any]]:
        marks = ",".join(str(c) for c in contract.MUSIC_TYPE_CODES)
        out = []
        for r in self._q(
            f"SELECT o.bank_asset_id, o.object_id, o.type_code, o.type_name, o.parse_status, o.fields_json"
            f" FROM wwise_object o JOIN asset a ON a.id=o.bank_asset_id WHERE a.installation_id=?"
            f" AND o.type_code IN ({marks}) ORDER BY o.object_id, o.bank_asset_id", (inst_id,)):
            item = dict(r)
            item["fields"] = loads(item.pop("fields_json"), {}) or {}
            out.append(item)
        return out

    def refs(self, inst_id: int, kinds: Iterable[str], parsed_only: bool = True) -> List[Dict[str, Any]]:
        kinds = list(kinds)
        marks = ",".join("?" * len(kinds))
        sql = (f"SELECT r.bank_asset_id, r.from_object_id, r.to_id, r.kind FROM object_ref r"
               f" JOIN asset a ON a.id=r.bank_asset_id WHERE a.installation_id=? AND r.kind IN ({marks})")
        if parsed_only:
            sql += " AND r.confidence='parsed'"
        return [dict(r) for r in self._q(sql, [inst_id, *kinds])]

    def object_types(self, inst_id: int, object_ids: Iterable[int]) -> Dict[int, str]:
        ids = sorted({int(i) for i in object_ids})
        out: Dict[int, str] = {}
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            marks = ",".join("?" * len(part))
            for r in self._q(
                f"SELECT o.object_id, o.type_name FROM wwise_object o JOIN asset a ON a.id=o.bank_asset_id"
                f" WHERE a.installation_id=? AND o.object_id IN ({marks})", [inst_id, *part]):
                out.setdefault(r["object_id"], r["type_name"])
        return out

    # ------------------------------------------------------------- media
    def media(self, inst_id: int, source_ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
        ids = sorted({int(i) for i in source_ids})
        out: Dict[int, Dict[str, Any]] = {}
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            marks = ",".join("?" * len(part))
            for r in self._q(
                f"SELECT w.source_id, w.container, w.valid, w.codec, w.channels, w.sample_rate, w.duration_s,"
                f" w.loops_json, a.vpath, a.size FROM wem w JOIN asset a ON a.id=w.asset_id"
                f" WHERE a.installation_id=? AND w.source_id IN ({marks}) ORDER BY w.source_id, w.valid DESC", [inst_id, *part]):
                item = dict(r)
                item["loops"] = loads(item.pop("loops_json"), []) or []
                entry = out.setdefault(item["source_id"], {**item, "locations": []})
                entry["locations"].append({"container": item["container"], "path": item["vpath"], "size": item["size"]})
            for r in self._q(
                f"SELECT entity_key AS source_id, role, score, confidence FROM classification WHERE installation_id=?"
                f" AND entity_type='media' AND entity_key IN ({marks})", [inst_id, *part]):
                out.setdefault(r["source_id"], {"source_id": r["source_id"], "locations": []}).update(
                    {"role": r["role"], "role_score": r["score"], "role_confidence": r["confidence"]})
        return out

    def media_streaming(self, inst_id: int, source_ids: Iterable[int]) -> Dict[int, List[str]]:
        ids = sorted({int(i) for i in source_ids})
        out: Dict[int, List[str]] = {}
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            marks = ",".join("?" * len(part))
            for r in self._q(
                f"SELECT DISTINCT m.source_id, m.stream_type FROM media_source m JOIN asset a ON a.id=m.bank_asset_id"
                f" WHERE a.installation_id=? AND m.source_id IN ({marks})", [inst_id, *part]):
                if r["stream_type"]:
                    out.setdefault(r["source_id"], []).append(r["stream_type"])
        return out

    def classification_counts(self, inst_id: int) -> Dict[str, int]:
        return {r["role"]: r["c"] for r in self._q(
            "SELECT role, COUNT(*) c FROM classification WHERE installation_id=? AND entity_type='media' GROUP BY role",
            (inst_id,))}

    def open_unknowns(self, inst_id: int) -> int:
        if not self.has_table("unknown_structure"):
            return 0
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM unknown_structure WHERE installation_id=? AND status='open'", (inst_id,)).fetchone()[0])
