import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest

from soundtrack_studio import app_paths
from soundtrack_studio.app_paths import AppPaths, detect_app_root, os_path
from soundtrack_studio.config import Settings
from soundtrack_studio.environment import configure_process_environment, portable_environment
from soundtrack_studio.logging_setup import RedactingFilter, setup_logging, shutdown_logging


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(app_paths.ENV_HOME_OVERRIDE, str(tmp_path / "x"))
    assert detect_app_root() == (tmp_path / "x").resolve()


def test_frozen_root_is_executable_folder_not_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv(app_paths.ENV_HOME_OVERRIDE, raising=False)
    exe = tmp_path / "D" / "CrimsonSoundtrackStudio" / "CrimsonSoundtrackStudio.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    elsewhere = tmp_path / "somewhere else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert detect_app_root() == exe.parent.resolve()


def test_source_root_is_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv(app_paths.ENV_HOME_OVERRIDE, raising=False)
    monkeypatch.chdir(tmp_path)
    assert detect_app_root() == Path(app_paths.__file__).resolve().parents[2] / "dev_home"


def test_ensure_creates_portable_layout(tmp_path):
    p = AppPaths(tmp_path / "app").ensure()
    for rel in ("data/config", "data/cache", "data/databases", "models/low", "models/medium", "models/high",
                "models/custom", "projects", "output", "logs", "temp"):
        assert (p.root / rel).is_dir(), rel


def test_stored_paths_are_relative_inside_root_and_survive_a_move(tmp_path):
    a = AppPaths(tmp_path / "D" / "Studio").ensure()
    stored = a.to_stored(a.projects / "My Project")
    assert stored == "app:projects/My Project"
    external = tmp_path / "Music"
    assert a.to_stored(external) == str(external)
    b = AppPaths(tmp_path / "E" / "Tools" / "Studio")
    assert b.from_stored(stored) == b.root / "projects" / "My Project"
    assert b.from_stored(str(external)) == external
    assert a.from_stored("") is None and a.to_stored(None) == ""


def test_os_path_long_windows_paths():
    long = "C:\\" + "\\".join(["folder" * 5] * 10) + "\\track.flac"
    assert os_path(long, windows=True).startswith("\\\\?\\C:\\")
    assert os_path("C:\\short\\a.flac", windows=True) == "C:\\short\\a.flac"
    unc = "\\\\server\\share\\" + "x" * 250
    assert os_path(unc, windows=True).startswith("\\\\?\\UNC\\server\\share")
    assert os_path("/" + "a" * 300, windows=False) == "/" + "a" * 300


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs a non-root POSIX user for permission checks")
def test_check_writable_reports_readonly_folder(tmp_path):
    root = tmp_path / "ro"
    root.mkdir()
    os.chmod(root, 0o500)
    try:
        assert "cannot write" in AppPaths(root).check_writable()
    finally:
        os.chmod(root, 0o700)


def test_environment_redirects_into_app_folder(monkeypatch, tmp_path):
    p = AppPaths(tmp_path / "app").ensure()
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for key in portable_environment(p):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    env = configure_process_environment(p)
    for key, value in env.items():
        assert os.environ[key] == value and p.is_inside(value), key
    assert Path(tempfile.gettempdir()) == p.temp


def test_settings_roundtrip_and_unknown_keys(tmp_path):
    p = AppPaths(tmp_path / "app").ensure()
    s = Settings()
    s.remember_project("app:projects/A")
    s.remember_project("app:projects/B")
    s.remember_project("app:projects/A")
    s.save(p)
    raw = json.loads(p.settings_file.read_text(encoding="utf-8"))
    raw["future_option"] = 42
    p.settings_file.write_text(json.dumps(raw), encoding="utf-8")
    loaded = Settings.load(p)
    assert loaded.last_project == "app:projects/A"
    assert loaded.recent_projects == ["app:projects/A", "app:projects/B"]
    assert loaded.extra["_unknown.future_option"] == 42
    assert p.is_inside(p.settings_file)


def test_corrupt_settings_fall_back_to_defaults(tmp_path):
    p = AppPaths(tmp_path / "app").ensure()
    p.settings_file.write_text("{not json", encoding="utf-8")
    assert Settings.load(p).last_project == ""
    assert (p.config / "settings.corrupt.json").is_file()


def test_logging_goes_to_app_logs_and_redacts_home(tmp_path):
    p = AppPaths(tmp_path / "app").ensure()
    log_file = setup_logging(p)
    try:
        logging.getLogger("t").info("path is %s", str(Path.home() / "Music" / "secret"))
    finally:
        shutdown_logging()
    text = log_file.read_text(encoding="utf-8")
    assert p.is_inside(log_file)
    assert "<home>" in text and "Crimson Soundtrack Studio" in text
    assert str(Path.home()) + os.sep + "Music" not in text


def test_redacting_filter_handles_args():
    f = RedactingFilter()
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "%s/file", (str(Path.home()),), None)
    f.filter(record)
    assert record.getMessage().startswith("<home>")


def test_clean_temp_removes_only_old_entries(tmp_path):
    p = AppPaths(tmp_path / "app").ensure()
    old = p.temp / "import-old"
    old.mkdir()
    (old / "x").write_text("x")
    fresh = p.temp / "fresh.tmp"
    fresh.write_text("y")
    past = __import__("time").time() - 3 * 86400
    os.utime(old, (past, past))
    assert p.clean_temp() == 1
    assert not old.exists() and fresh.exists()


def test_is_file_goes_through_the_long_path_prefix(tmp_path, monkeypatch):
    from soundtrack_studio import app_paths

    real = tmp_path / "song.flac"
    real.write_bytes(b"x")
    seen = []
    monkeypatch.setattr(app_paths, "os_path", lambda p, **kw: seen.append(str(p)) or str(p))
    assert app_paths.is_file(real) and seen == [str(real)]          # the user's file is opened via os_path()
    assert not app_paths.is_file(tmp_path / "missing.flac")
    # the track in the bug report: 272 characters, which Windows only opens with the \\?\ prefix
    name = ("C:\\Users\\Matthew\\Desktop\\Crimson Desert Music Modding\\Music for Crimson Desert Modding\\Game of Thrones "
            "Music\\Ramin Djawadi\\Game Of Thrones Season 2 (Music From The HBO Series)\\Don't Die With A Clean Sword - "
            "From The Game Of Thrones Season 2 Soundtrack - Ramin Djawadi.flac")
    assert len(name) > 260 and os_path(name, windows=True).startswith("\\\\?\\C:\\Users\\Matthew")


def test_long_path_helper_follows_os_path(tmp_path):
    from soundtrack_studio import app_paths

    p = tmp_path / "x.gguf"
    assert app_paths.long_path(p) == Path(os_path(p))
