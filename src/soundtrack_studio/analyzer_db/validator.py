"""Validation of an Analyzer database (run against the Studio's private snapshot)."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import contract


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    details: str = ""


@dataclass
class InstallationSummary:
    id: int
    root_path: str
    label: str
    last_scanned: Optional[str]
    latest_scan_id: Optional[int]
    latest_scan_status: Optional[str]
    latest_scan_finished: Optional[str]
    parser_version: Optional[int]


@dataclass
class ValidationReport:
    schema_version: Optional[int] = None
    issues: List[Issue] = field(default_factory=list)
    installations: List[InstallationSummary] = field(default_factory=list)
    selected_installation_id: Optional[int] = None
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def add(self, severity: str, code: str, message: str, details: str = "") -> None:
        self.issues.append(Issue(severity, code, message, details))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationReport":
        return cls(
            schema_version=data.get("schema_version"),
            issues=[Issue(**i) for i in data.get("issues", [])],
            installations=[InstallationSummary(**i) for i in data.get("installations", [])],
            selected_installation_id=data.get("selected_installation_id"),
            counts=dict(data.get("counts", {})),
        )


def _tables(conn: sqlite3.Connection) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        out[name] = {row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')}
    return out


def validate(conn: sqlite3.Connection, *, integrity: bool = True) -> ValidationReport:
    report = ValidationReport()
    try:
        if integrity:
            result = [r[0] for r in conn.execute("PRAGMA quick_check(20)")]
            if result != ["ok"]:
                report.add("error", "corrupt",
                           "The Analyzer database is damaged and cannot be read reliably.",
                           "; ".join(str(r) for r in result[:20]))
                return report
        tables = _tables(conn)
    except sqlite3.DatabaseError as exc:
        report.add("error", "corrupt", "The Analyzer database is damaged and cannot be read.", str(exc))
        return report

    if not all(t in tables for t in contract.SIGNATURE_TABLES):
        report.add("error", "not_analyzer",
                   "This file is a SQLite database, but it was not created by Crimson Desert Analyzer.",
                   f"missing identifying tables: {[t for t in contract.SIGNATURE_TABLES if t not in tables]}")
        return report

    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    report.schema_version = version
    if version not in contract.SUPPORTED_SCHEMA_VERSIONS:
        newer = version > contract.MAX_SCHEMA
        expected = ", ".join(str(v) for v in sorted(contract.SUPPORTED_SCHEMA_VERSIONS))
        report.add(
            "error", "schema_newer" if newer else "schema_older",
            "The selected Analyzer database is not compatible with this version of Crimson Soundtrack Studio.\n"
            f"Expected schema: {expected}\nFound schema: {version}",
            "Update Crimson Soundtrack Studio to read this database." if newer
            else "Generate a new database with a current version of Crimson Desert Analyzer.",
        )
        return report

    missing_tables = [t for t in contract.REQUIRED if t not in tables]
    missing_cols = {t: sorted(cols - tables[t]) for t, cols in contract.REQUIRED.items()
                    if t in tables and cols - tables[t]}
    if missing_tables or missing_cols:
        report.add("error", "schema_incomplete",
                   "The Analyzer database is missing information the Studio needs. It may be incomplete or "
                   "from a modified Analyzer build.",
                   f"missing tables: {missing_tables}; missing columns: {missing_cols}")
        return report
    for table, cols in contract.OPTIONAL.items():
        if table not in tables or cols - tables[table]:
            report.add("warning", "optional_missing",
                       f"Optional Analyzer data '{table}' is not available; some details will not be shown.")

    try:
        _check_content(conn, report)
    except sqlite3.DatabaseError as exc:
        report.add("error", "corrupt", "The Analyzer database could not be read.", str(exc))
    return report


def _check_content(conn: sqlite3.Connection, report: ValidationReport) -> None:
    rows = conn.execute(
        "SELECT i.id, i.root_path, i.label, i.last_scanned,"
        " (SELECT s.id FROM scan s WHERE s.installation_id=i.id ORDER BY s.id DESC LIMIT 1),"
        " (SELECT s.status FROM scan s WHERE s.installation_id=i.id ORDER BY s.id DESC LIMIT 1),"
        " (SELECT s.finished_at FROM scan s WHERE s.installation_id=i.id ORDER BY s.id DESC LIMIT 1),"
        " (SELECT s.parser_version FROM scan s WHERE s.installation_id=i.id ORDER BY s.id DESC LIMIT 1)"
        " FROM installation i ORDER BY i.id").fetchall()
    report.installations = [InstallationSummary(*r) for r in rows]
    if not report.installations:
        report.add("error", "no_installation",
                   "The Analyzer database does not contain a scanned Crimson Desert installation.",
                   "Run 'Analyze Game' in Crimson Desert Analyzer first.")
        return
    completed = [i for i in report.installations if i.latest_scan_status == "completed"]
    if not completed:
        any_completed = conn.execute("SELECT installation_id FROM scan WHERE status='completed' ORDER BY id DESC LIMIT 1").fetchone()
        if any_completed is None:
            report.add("error", "no_completed_scan",
                       "The Analyzer never finished scanning the game in this database.",
                       "Run the Analyzer scan again and let it complete.")
            return
        report.selected_installation_id = int(any_completed[0])
        report.add("warning", "latest_scan_incomplete",
                   "The most recent Analyzer scan did not complete; data from an earlier completed scan is used.")
    else:
        latest = max(completed, key=lambda i: (i.last_scanned or "", i.id))
        report.selected_installation_id = latest.id
    if len(report.installations) > 1:
        report.add("info", "multiple_installations",
                   f"The database contains {len(report.installations)} installations; the most recently scanned one is used.")

    inst = report.selected_installation_id
    counts = report.counts
    counts["banks"] = conn.execute(
        "SELECT COUNT(*) FROM bnk b JOIN asset a ON a.id=b.asset_id WHERE a.installation_id=?", (inst,)).fetchone()[0]
    marks = ",".join(str(c) for c in contract.MUSIC_TYPE_CODES)
    counts["music_objects"] = conn.execute(
        f"SELECT COUNT(*) FROM wwise_object o JOIN asset a ON a.id=o.bank_asset_id WHERE a.installation_id=? AND o.type_code IN ({marks})",
        (inst,)).fetchone()[0]
    counts["media"] = conn.execute(
        "SELECT COUNT(DISTINCT w.source_id) FROM wem w JOIN asset a ON a.id=w.asset_id WHERE a.installation_id=?", (inst,)).fetchone()[0]
    roles = ",".join(f"'{r}'" for r in contract.MUSIC_ROLES)
    counts["music_media"] = conn.execute(
        f"SELECT COUNT(*) FROM classification WHERE installation_id=? AND entity_type='media' AND role IN ({roles})",
        (inst,)).fetchone()[0]
    counts["source_files"] = conn.execute("SELECT COUNT(*) FROM source_file WHERE installation_id=?", (inst,)).fetchone()[0]
    counts["music_parse_problems"] = conn.execute(
        f"SELECT COUNT(*) FROM wwise_object o JOIN asset a ON a.id=o.bank_asset_id WHERE a.installation_id=?"
        f" AND o.type_code IN ({marks}) AND o.parse_status NOT IN ('parsed','shallow')", (inst,)).fetchone()[0]

    if counts["banks"] == 0:
        report.add("error", "no_banks", "The Analyzer database contains no Wwise soundbanks for this installation.")
    elif counts["music_objects"] == 0:
        report.add("warning", "no_music_structures",
                   "The Analyzer found no interactive music structures (segments, playlists, tracks). "
                   "Music replacement needs them; the database may come from an incomplete scan.")
    if counts["music_parse_problems"]:
        report.add("warning", "music_parse_problems",
                   f"{counts['music_parse_problems']} music objects were only partially decoded by the Analyzer; "
                   "they are shown but may be missing details.")
    if counts["source_files"] == 0:
        report.add("warning", "no_source_files",
                   "The database does not record the game's archive files, so it cannot be checked against "
                   "your Crimson Desert installation.")
