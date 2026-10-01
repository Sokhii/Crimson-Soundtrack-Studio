"""Semantic profile storage, caching and the resumable batch runner.

Effective profile precedence: user override > local AI (llm) > rules.

Resumability: every item is committed as soon as it is done. A rerun skips
items whose stored profile was made from the same evidence (``input_hash``)
by the same model and prompt version, so an interrupted analysis continues
where it stopped. AI answers are also kept in a shared cache
(``data/cache/ai_responses.sqlite3``), so the same track in another project,
or after a project reset, costs no model time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..ai.runtime import BackendError, InferenceBackend, ModelOutputError
from ..errors import OperationCancelled
from ..project.store import Project, now_iso
from . import llm, rules
from .context import input_hash
from .profile import ProfileParseError, SemanticProfile, merge

log = logging.getLogger(__name__)

ENTITY_TYPES = ("track", "cue")


@dataclass
class EffectiveProfile:
    profile: SemanticProfile
    rules: Optional[SemanticProfile] = None
    llm: Optional[SemanticProfile] = None
    override: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    llm_error: str = ""

    @property
    def source(self) -> str:
        return self.profile.source


@dataclass
class RunStats:
    total: int = 0
    rules_done: int = 0
    llm_done: int = 0
    llm_from_cache: int = 0
    skipped: int = 0
    llm_errors: int = 0
    errors: List[str] = field(default_factory=list)


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        try:
            self.conn = self._open(path)
        except sqlite3.DatabaseError:
            path.unlink(missing_ok=True)
            self.conn = self._open(path)

    @staticmethod
    def _open(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("CREATE TABLE IF NOT EXISTS response (key TEXT PRIMARY KEY, profile_json TEXT NOT NULL,"
                         " created_at TEXT NOT NULL)")
            conn.commit()
        except sqlite3.DatabaseError:
            conn.close()
            raise
        return conn

    @staticmethod
    def key(model_key: str, kind: str, ihash: str) -> str:
        return hashlib.sha256(f"{model_key}|{llm.PROMPT_VERSION}|{kind}|{ihash}".encode()).hexdigest()

    def get(self, key: str) -> Optional[SemanticProfile]:
        with self.lock:
            row = self.conn.execute("SELECT profile_json FROM response WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            return SemanticProfile.from_dict(json.loads(row[0]))
        except (ValueError, TypeError):
            return None

    def put(self, key: str, profile: SemanticProfile) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO response(key, profile_json, created_at) VALUES (?,?,?)",
                              (key, profile.to_json(), now_iso()))
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()


class SemanticStore:
    def __init__(self, project: Project, cache: Optional[ResponseCache] = None) -> None:
        self.project = project
        self.cache = cache

    # --------------------------------------------------------------- read
    def _rows(self, entity_type: str, keys: Optional[Iterable[str]] = None) -> Dict[Tuple[str, str], Dict[str, Any]]:
        rows = self.project.query("SELECT * FROM semantic_profile WHERE entity_type=?", (entity_type,))
        wanted = set(keys) if keys is not None else None
        return {(r["entity_key"], r["source"]): dict(r) for r in rows if wanted is None or r["entity_key"] in wanted}

    def overrides(self, entity_type: str) -> Dict[str, Dict[str, Any]]:
        return {r["entity_key"]: json.loads(r["override_json"]) for r in self.project.query(
            "SELECT entity_key, override_json FROM semantic_override WHERE entity_type=?", (entity_type,))}

    def effective(self, entity_type: str, key: str) -> Optional[EffectiveProfile]:
        return self.effective_all(entity_type, [str(key)]).get(str(key))

    def effective_all(self, entity_type: str, keys: Optional[Iterable[str]] = None) -> Dict[str, EffectiveProfile]:
        rows = self._rows(entity_type, [str(k) for k in keys] if keys is not None else None)
        overrides = self.overrides(entity_type)
        out: Dict[str, EffectiveProfile] = {}
        entity_keys = {k for k, _s in rows}
        for key in entity_keys:
            r = rows.get((key, "rules"))
            l_row = rows.get((key, "llm"))
            rules_p = SemanticProfile.from_dict(json.loads(r["profile_json"])) if r and r["profile_json"] else None
            if rules_p:
                rules_p.source = "rules"
            llm_p = None
            if l_row and l_row["status"] == "ok" and l_row["profile_json"]:
                llm_p = SemanticProfile.from_dict(json.loads(l_row["profile_json"]))
                llm_p.source, llm_p.model_id = "llm", l_row["model_id"]
            base = llm_p or rules_p
            if base is None:
                continue
            override = overrides.get(key, {})
            effective = merge(base, override) if override else base
            evidence = json.loads(r["evidence_json"]) if r and r["evidence_json"] else []
            out[key] = EffectiveProfile(effective, rules_p, llm_p, override, evidence,
                                        (l_row or {}).get("error") or "" if l_row and l_row["status"] != "ok" else "")
        return out

    # -------------------------------------------------------------- write
    def set_override(self, entity_type: str, key: str, values: Dict[str, Any]) -> None:
        self.project.execute("INSERT OR REPLACE INTO semantic_override(entity_type, entity_key, override_json, updated_at)"
                             " VALUES (?,?,?,?)", (entity_type, str(key), json.dumps(values, ensure_ascii=False), now_iso()))

    def clear_override(self, entity_type: str, key: str) -> None:
        self.project.execute("DELETE FROM semantic_override WHERE entity_type=? AND entity_key=?", (entity_type, str(key)))

    def _store(self, entity_type: str, key: str, source: str, ihash: str, profile: Optional[SemanticProfile],
               evidence: Optional[List[str]] = None, model_id: str = "", error: str = "") -> None:
        self.project.execute(
            "INSERT INTO semantic_profile(entity_type, entity_key, source, model_id, prompt_version, input_hash, status,"
            " profile_json, evidence_json, error, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(entity_type, entity_key, source) DO UPDATE SET model_id=excluded.model_id,"
            " prompt_version=excluded.prompt_version, input_hash=excluded.input_hash, status=excluded.status,"
            " profile_json=excluded.profile_json, evidence_json=excluded.evidence_json, error=excluded.error,"
            " created_at=excluded.created_at",
            (entity_type, key, source, model_id, llm.PROMPT_VERSION if source == "llm" else 0, ihash,
             "error" if error else "ok", profile.to_json() if profile else None,
             json.dumps(evidence or [], ensure_ascii=False), error or None, now_iso()))

    def run(self, entity_type: str, items: List[Tuple[str, Dict[str, Any]]], backend: Optional[InferenceBackend] = None,
            model_key: str = "", progress: Optional[Callable[[str, int, int], None]] = None,
            cancel: Optional[Callable[[], bool]] = None, max_llm_failures: int = 5,
            force: bool = False) -> RunStats:
        """Describe ``items`` = [(entity_key, evidence document)]; resumable, committed per item.

        ``force`` has the AI write every description again, ignoring saved and cached answers (the new answers replace
        them); the user's own edits are separate and untouched."""

        describe_rules = rules.describe_track if entity_type == "track" else rules.describe_cue
        stats = RunStats(total=len(items))
        existing = self._rows(entity_type, [k for k, _d in items])
        consecutive_failures = 0
        label = "Describing your music" if entity_type == "track" else "Describing the game's music"
        for index, (key, doc) in enumerate(items):
            if cancel and cancel():
                raise OperationCancelled()
            if progress:
                progress(label, index, len(items))
            ihash = input_hash(entity_type, doc)
            r = existing.get((key, "rules"))
            if not r or r["input_hash"] != ihash:
                profile, evidence = describe_rules(doc)
                self._store(entity_type, key, "rules", ihash, profile, evidence)
                stats.rules_done += 1
            if backend is None:
                continue
            l_row = existing.get((key, "llm"))
            if (not force and l_row and l_row["status"] == "ok" and l_row["input_hash"] == ihash and l_row["model_id"] == backend.model_id
                    and l_row["prompt_version"] == llm.PROMPT_VERSION):
                stats.skipped += 1
                continue
            cache_key = ResponseCache.key(model_key or backend.model_id, entity_type, ihash)
            cached = self.cache.get(cache_key) if self.cache and not force else None
            if cached is not None:
                cached.source, cached.model_id = "llm", backend.model_id
                self._store(entity_type, key, "llm", ihash, cached, model_id=backend.model_id)
                stats.llm_from_cache += 1
                continue
            try:
                profile = llm.describe(backend, entity_type, doc)
            except (ProfileParseError, BackendError) as exc:
                if isinstance(exc, BackendError):
                    message = f"{exc.message} {exc.details}".strip()[:500]
                else:
                    message = f"unusable model output: {exc}"
                self._store(entity_type, key, "llm", ihash, None, model_id=backend.model_id, error=message)
                stats.llm_errors += 1
                consecutive_failures += 1
                if len(stats.errors) < 10:
                    stats.errors.append(f"{entity_type} {key}: {message}")
                if isinstance(exc, BackendError) and not isinstance(exc, ModelOutputError) \
                        and consecutive_failures >= max_llm_failures:
                    raise
                continue
            consecutive_failures = 0
            self._store(entity_type, key, "llm", ihash, profile, model_id=backend.model_id)
            if self.cache:
                self.cache.put(cache_key, profile)
            stats.llm_done += 1
        if progress:
            progress(label, len(items), len(items))
        log.info("Semantic run (%s): %s", entity_type, {k: v for k, v in stats.__dict__.items() if k != "errors"})
        return stats

    def forget_missing(self, entity_type: str, keep_keys: Iterable[str]) -> int:
        keep = set(str(k) for k in keep_keys)
        stale = [r["entity_key"] for r in self.project.query(
            "SELECT DISTINCT entity_key FROM semantic_profile WHERE entity_type=?", (entity_type,)) if r["entity_key"] not in keep]
        with self.project.transaction() as c:
            c.executemany("DELETE FROM semantic_profile WHERE entity_type=? AND entity_key=?",
                          [(entity_type, k) for k in stale])
        return len(stale)
