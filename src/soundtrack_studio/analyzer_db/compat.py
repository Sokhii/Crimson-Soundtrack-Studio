"""Checks that a Crimson Desert installation matches an Analyzer database.

The Analyzer (schema 1) records no game build number, but it does record the
size and modification time of every archive index (``.pamt``), archive
(``.paz``), the package table (``meta/0.papgt``) and loose files it saw.
Comparing those against the selected folder detects game updates and
wrong folders without re-scanning anything. Only ``stat`` calls are made;
no game file is opened.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..app_paths import os_path

MATCH = "match"                    # every recorded file present with identical size and mtime
PROBABLE_MATCH = "probable_match"  # sizes identical, some mtimes differ (copied/moved install)
MISMATCH = "mismatch"              # files missing or sizes differ (different version or folder)
NOT_GAME = "not_game_folder"
UNVERIFIABLE = "unverifiable"      # the database recorded nothing to compare


@dataclass
class CompatReport:
    status: str
    game_path: str
    checked_files: int = 0
    missing: List[str] = field(default_factory=list)
    size_mismatch: List[str] = field(default_factory=list)
    mtime_mismatch: List[str] = field(default_factory=list)
    recorded_root: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.status in (MATCH, PROBABLE_MATCH)

    def summary(self) -> str:
        return {
            MATCH: "The installation matches the Analyzer database.",
            PROBABLE_MATCH: "The installation matches the Analyzer database (file dates differ, which is normal "
                            "for a copied or moved installation).",
            MISMATCH: "The installation does not match the Analyzer database. The game was probably updated "
                      "after the database was created, or a different folder was selected. Re-run Crimson "
                      "Desert Analyzer on this installation.",
            NOT_GAME: "The selected folder does not look like a Crimson Desert installation.",
            UNVERIFIABLE: "The Analyzer database does not record enough information to verify this installation.",
        }[self.status]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["summary"] = self.summary()
        return data


def looks_like_game_folder(root: Path) -> bool:
    if (root / "meta" / "0.papgt").is_file() or (root / "bin64" / "CrimsonDesert.exe").is_file():
        return True
    try:
        return any(entry.is_dir() and entry.name.isdigit() and (Path(entry.path) / "0.pamt").is_file()
                   for entry in os.scandir(root))
    except OSError:
        return False


def check_installation(game_root: Path, recorded_files: List[Dict[str, Any]], recorded_root: str = "") -> CompatReport:
    root = Path(game_root)
    report = CompatReport(status=MATCH, game_path=str(root), recorded_root=recorded_root)
    if not root.is_dir() or not looks_like_game_folder(root):
        report.status = NOT_GAME
        return report
    # Loose files are compared too, except large media which the check does not need.
    relevant = [f for f in recorded_files if f["kind"] in ("pamt", "papgt", "paz") or
                (f["kind"] == "loose" and f["rel_path"].lower().endswith((".exe", ".papgt", ".pamt")))]
    if not relevant:
        report.status = UNVERIFIABLE
        return report
    for f in relevant:
        path = root / Path(f["rel_path"])
        report.checked_files += 1
        try:
            st = os.stat(os_path(path))
        except OSError:
            report.missing.append(f["rel_path"])
            continue
        if st.st_size != f["size"]:
            report.size_mismatch.append(f["rel_path"])
        elif f.get("mtime_ns") and st.st_mtime_ns != f["mtime_ns"]:
            report.mtime_mismatch.append(f["rel_path"])
    if report.missing or report.size_mismatch:
        report.status = MISMATCH
    elif report.mtime_mismatch:
        report.status = PROBABLE_MATCH
    if recorded_root and os.path.normcase(os.path.normpath(recorded_root)) != os.path.normcase(os.path.normpath(str(root))):
        report.notes.append("The Analyzer scanned the game at a different path; this is fine if the installation was moved.")
    return report
