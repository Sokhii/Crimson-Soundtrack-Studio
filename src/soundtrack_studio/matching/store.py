"""Proposals, user decisions and the final mapping handed to the compiler.

Decision precedence (highest first):
  keep_original  - the cue is not replaced
  manual         - the user picked this track
  accept         - the user accepted the proposal (pinned: reruns do not change it)
  reject         - the user rejected track(s); the cue is re-proposed without them
  (none)         - the machine proposal stands, but is only built after the user accepts it
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Optional, Set

from ..project.store import Project, now_iso
from .engine import CueResult, MatchSettings, confidence_label

ACTIONS = ("accept", "manual", "reject", "keep_original")
FIT_MODES = ("auto", "trim", "loop", "pad")


@dataclass
class Proposal:
    cue_key: str
    rank: int
    track_id: int
    score: float
    confidence: float
    components: Dict[str, float]
    reasons: List[str]
    warnings: List[str]
    ai_fit: Optional[float]
    ai_reason: str


@dataclass
class Decision:
    cue_key: str
    action: str
    track_id: Optional[int]
    rejected: List[int] = field(default_factory=list)
    fit_mode: str = "auto"
    start_offset_s: float = 0.0
    note: str = ""
    updated_at: str = ""


@dataclass
class MappingEntry:
    cue_key: str
    track_id: int
    fit_mode: str
    start_offset_s: float
    decided_by: str      # manual | accept


class MatchStore:
    def __init__(self, project: Project) -> None:
        self.project = project

    # ------------------------------------------------------------ proposals
    def save_run(self, results: Dict[str, CueResult], settings: MatchSettings, model_id: str,
                 stats: Dict[str, Any]) -> int:
        run_id = self.project.execute(
            "INSERT INTO match_run(created_at, settings_json, model_id, stats_json) VALUES (?,?,?,?)",
            (now_iso(), json.dumps(settings.to_dict()), model_id, json.dumps(stats))).lastrowid
        rows = []
        for key, result in results.items():
            for rank, c in enumerate(result.candidates):
                rows.append((key, rank, c.track_id, c.score, c.confidence, json.dumps(c.components),
                             json.dumps(c.reasons, ensure_ascii=False), json.dumps(c.warnings, ensure_ascii=False),
                             c.ai_fit, c.ai_reason or None, run_id))
        with self.project.transaction() as conn:
            conn.execute("DELETE FROM match_proposal")
            conn.executemany("INSERT INTO match_proposal VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.project.set("matching_skipped", {k: r.skipped_reason for k, r in results.items() if r.skipped_reason})
        return int(run_id)

    def proposals(self) -> Dict[str, List[Proposal]]:
        out: Dict[str, List[Proposal]] = {}
        for r in self.project.query("SELECT * FROM match_proposal ORDER BY cue_key, rank"):
            out.setdefault(r["cue_key"], []).append(Proposal(
                r["cue_key"], r["rank"], r["track_id"], r["score"], r["confidence"],
                json.loads(r["components_json"] or "{}"), json.loads(r["reasons_json"] or "[]"),
                json.loads(r["warnings_json"] or "[]"), r["ai_fit"], r["ai_reason"] or ""))
        return out

    def skipped(self) -> Dict[str, str]:
        return self.project.get("matching_skipped", {}) or {}

    def last_run(self) -> Optional[Dict[str, Any]]:
        row = self.project.query_one("SELECT * FROM match_run ORDER BY id DESC LIMIT 1")
        if row is None:
            return None
        data = dict(row)
        data["settings"] = json.loads(data.pop("settings_json"))
        data["stats"] = json.loads(data.pop("stats_json") or "{}")
        return data

    # ------------------------------------------------------------ decisions
    def decisions(self) -> Dict[str, Decision]:
        return {r["cue_key"]: Decision(r["cue_key"], r["action"], r["track_id"], json.loads(r["rejected_json"] or "[]"),
                                       r["fit_mode"], r["start_offset_s"], r["note"] or "", r["updated_at"])
                for r in self.project.query("SELECT * FROM match_decision")}

    def decision(self, cue_key: str) -> Optional[Decision]:
        return self.decisions().get(str(cue_key))

    def _write(self, d: Decision) -> None:
        if d.action not in ACTIONS:
            raise ValueError(f"unknown action {d.action}")
        if d.fit_mode not in FIT_MODES:
            raise ValueError(f"unknown fit mode {d.fit_mode}")
        self.project.execute(
            "INSERT OR REPLACE INTO match_decision(cue_key, action, track_id, rejected_json, fit_mode, start_offset_s,"
            " note, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (d.cue_key, d.action, d.track_id, json.dumps(sorted(set(d.rejected))), d.fit_mode, float(d.start_offset_s),
             d.note or None, now_iso()))

    def accept(self, cue_key: str, track_id: int) -> None:
        old = self.decision(cue_key)
        self._write(Decision(str(cue_key), "accept", int(track_id), [t for t in (old.rejected if old else []) if t != track_id],
                             old.fit_mode if old else "auto", old.start_offset_s if old else 0.0))

    def choose(self, cue_key: str, track_id: int) -> None:
        old = self.decision(cue_key)
        self._write(Decision(str(cue_key), "manual", int(track_id), [t for t in (old.rejected if old else []) if t != track_id],
                             old.fit_mode if old else "auto", old.start_offset_s if old else 0.0))

    def reject(self, cue_key: str, track_id: int) -> None:
        old = self.decision(cue_key)
        rejected = set(old.rejected if old else []) | {int(track_id)}
        self._write(Decision(str(cue_key), "reject", None, sorted(rejected), old.fit_mode if old else "auto",
                             old.start_offset_s if old else 0.0))

    def keep_original(self, cue_key: str) -> None:
        old = self.decision(cue_key)
        self._write(Decision(str(cue_key), "keep_original", None, old.rejected if old else []))

    def clear(self, cue_key: str) -> None:
        self.project.execute("DELETE FROM match_decision WHERE cue_key=?", (str(cue_key),))

    def set_fit(self, cue_key: str, fit_mode: str, start_offset_s: float = 0.0) -> None:
        old = self.decision(cue_key)
        if old is None:
            raise ValueError("choose or accept a track before changing how it fits")
        old.fit_mode, old.start_offset_s = fit_mode, max(0.0, float(start_offset_s))
        self._write(old)

    def accept_all(self, min_confidence: float = 0.0, levels: Optional[Collection[str]] = None) -> int:
        """Accept the top proposal of every undecided cue; ``levels`` ("high"/"medium"/"low") limits it to those
        confidence levels (a cue the user rejected earlier is proposed again, as before)."""

        decisions = self.decisions()
        count = 0
        for key, props in self.proposals().items():
            if key in decisions and decisions[key].action != "reject":
                continue
            if not props or props[0].rank != 0 or props[0].confidence < min_confidence:
                continue
            if levels is not None and confidence_label(props[0].confidence) not in levels:
                continue
            self.accept(key, props[0].track_id)
            count += 1
        return count

    def rejected_map(self) -> Dict[str, Set[int]]:
        return {k: set(d.rejected) for k, d in self.decisions().items() if d.rejected}

    def fixed_map(self) -> Dict[str, Optional[int]]:
        return {k: d.track_id for k, d in self.decisions().items() if d.action in ("accept", "manual", "keep_original")}

    # -------------------------------------------------------------- mapping
    def final_mapping(self) -> List[MappingEntry]:
        """Only cues the user accepted or chose are replaced."""

        return [MappingEntry(k, d.track_id, d.fit_mode, d.start_offset_s, d.action)
                for k, d in sorted(self.decisions().items()) if d.action in ("accept", "manual") and d.track_id]

    def status_counts(self) -> Dict[str, int]:
        decisions = self.decisions()
        proposals = self.proposals()
        counts = {"proposed": 0, "accepted": 0, "manual": 0, "rejected": 0, "keep_original": 0, "unmatched": 0}
        for key in set(proposals) | set(decisions):
            d = decisions.get(key)
            if d is None:
                counts["proposed" if proposals.get(key) else "unmatched"] += 1
            elif d.action == "accept":
                counts["accepted"] += 1
            elif d.action == "manual":
                counts["manual"] += 1
            elif d.action == "keep_original":
                counts["keep_original"] += 1
            else:
                counts["rejected"] += 1
        return counts
