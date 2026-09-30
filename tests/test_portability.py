"""Portability: everything the app writes stays under its root, and a copied folder keeps working.

The subprocess tests run the real entry point with a fake user profile (HOME, XDG
folders, TMPDIR) so any write outside the application folder is caught.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from soundtrack_studio.app_paths import AppPaths

ROOT = Path(__file__).resolve().parents[1]


def run_app(app_root: Path, fake_home: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XDG_", "CRIMSON_STUDIO"))}
    env.update({
        "HOME": str(fake_home), "USERPROFILE": str(fake_home),
        "TMPDIR": str(fake_home / "systemtmp"), "TMP": str(fake_home / "systemtmp"), "TEMP": str(fake_home / "systemtmp"),
        "CRIMSON_STUDIO_HOME": str(app_root), "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
    })
    return subprocess.run([sys.executable, "-m", "soundtrack_studio", *args], env=env, cwd=str(cwd),
                          capture_output=True, text=True, timeout=300)


def all_files(folder: Path):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


@pytest.mark.slow
def test_no_files_outside_app_folder_and_copy_keeps_state(tmp_path):
    fake_home = tmp_path / "home"
    (fake_home / "systemtmp").mkdir(parents=True)
    elsewhere = tmp_path / "some other working dir"
    elsewhere.mkdir()
    app_d = tmp_path / "D" / "CrimsonSoundtrackStudio"
    app_d.mkdir(parents=True)

    result = run_app(app_d, fake_home, "--selftest", "--keep", cwd=elsewhere)
    assert result.returncode == 0, result.stdout + result.stderr
    gui = run_app(app_d, fake_home, "--smoke-gui", cwd=elsewhere)
    assert gui.returncode == 0, gui.stdout + gui.stderr

    # nothing in the fake user profile or system temp, nothing in the working directory
    assert all_files(fake_home) == ["systemtmp"], all_files(fake_home)
    assert all_files(elsewhere) == []
    for rel in ("data/config/settings.json", "data/cache/audio_analysis.sqlite3", "logs/studio.log",
                "logs/selftest.json", "projects/__selftest__/project.sqlite3", "output/__selftest__/build-check.txt",
                "models/custom/.selftest-probe"):
        assert (app_d / rel).is_file(), rel
    assert any(app_d.glob("data/databases/*/analyzer.sqlite3"))
    assert any(app_d.glob("data/cache/game_model/*.json"))

    # copy the whole folder to another "drive" and run from there
    app_e = tmp_path / "E" / "Tools" / "CrimsonSoundtrackStudio"
    shutil.copytree(app_d, app_e)
    state = run_app(app_e, fake_home, "--print-state", cwd=elsewhere)
    assert state.returncode == 0, state.stderr
    data = json.loads(state.stdout)
    assert data["root"] == str(app_e.resolve())
    assert data["settings_file"] and data["last_project"] == "app:projects/__selftest_build__"
    assert data["last_project_resolves"]
    assert sorted(p["name"] for p in data["projects"]) == ["__selftest__", "__selftest_build__"]
    assert "models/custom/.selftest-probe" in data["models"]
    assert "output/__selftest__/build-check.txt" in data["output"]
    assert "output/__selftest_mod__/manifest.json" in data["output"]
    assert any(d.endswith("analyzer.sqlite3") for d in data["databases"])
    # the copied project opens with its imported database and caches from the new location
    from soundtrack_studio.services import Studio

    studio = Studio(AppPaths(app_e))
    assert studio.open_last_project() is not None
    assert studio.game_model().stats["cues"] == 4
    assert studio.library_counts()["ok"] == 4
    studio.shutdown()
    assert all_files(fake_home) == ["systemtmp"]


def test_portability_check_detects_outside_writes(tmp_path, monkeypatch):
    from soundtrack_studio import portability

    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    root = AppPaths(tmp_path / "app").ensure()
    before = portability.snapshot([home], [root.root])
    (home / ".config" / "CrimsonSoundtrackStudio").mkdir()
    (home / ".config" / "unrelated.txt").write_text("x")
    (root.logs / "inside.log").write_text("x")
    after = portability.snapshot([home], [root.root])
    created = set(after) - set(before)
    flagged = [p for p in created if portability._flag(p)]
    assert len(flagged) == 1 and flagged[0].endswith("CrimsonSoundtrackStudio")
    assert not any("inside.log" in p for p in after)


def test_child_environment_restores_user_environment(tmp_path, monkeypatch):
    from soundtrack_studio import environment, portability

    root = AppPaths(tmp_path / "app").ensure()
    monkeypatch.setenv("TMP", "C:/Users/x/AppData/Local/Temp")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(environment, "ORIGINAL_ENV", {})
    monkeypatch.setattr(portability, "ORIGINAL_ENV", environment.ORIGINAL_ENV)
    for key in environment.portable_environment(root):
        if key != "TMP":
            monkeypatch.delenv(key, raising=False)
    environment.configure_process_environment(root)
    env = portability.child_environment(root)
    assert env["TMP"] == "C:/Users/x/AppData/Local/Temp"
    assert "XDG_CACHE_HOME" not in env
