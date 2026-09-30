import sqlite3

import pytest

from soundtrack_studio.errors import ProjectError
from soundtrack_studio.project import store
from soundtrack_studio.project.store import Project, list_projects, slugify


def test_create_save_reopen(paths):
    p = Project.create(paths, "Mushoku Replacement")
    p.set("game_path", "D:/Games/Crimson Desert")
    p.set("nested", {"a": [1, 2]})
    p.add_event("warning", "library", "3 files unreadable")
    folder = p.folder
    p.close()
    again = Project.open(folder)
    assert again.name == "Mushoku Replacement"
    assert again.get("game_path") == "D:/Games/Crimson Desert"
    assert again.get("nested") == {"a": [1, 2]}
    assert again.get("missing", "default") == "default"
    assert again.recent_events()[0]["message"] == "3 files unreadable"
    assert again.format_version == store.LATEST_VERSION
    assert paths.is_inside(folder) and folder.parent == paths.projects
    again.close()
    assert sorted(f.name for f in folder.iterdir()) == ["project.sqlite3"]  # single file, safe to copy


def test_duplicate_and_unsafe_names(paths):
    a = Project.create(paths, "Same")
    b = Project.create(paths, "Same")
    assert a.folder != b.folder and b.folder.name == "Same (2)"
    assert slugify('a<b>c:"d/e\\f|g?h*') == "abcdefgh"
    assert slugify("CON") == "_CON"
    assert slugify("  trailing dots... ") == "trailing dots"
    assert slugify("音楽 ♪ Ünïcødé") == "音楽 ♪ Ünïcødé"
    with pytest.raises(ProjectError):
        Project.create(paths, "   ")
    a.close()
    b.close()


def test_list_projects(paths):
    for name in ("One", "Two"):
        Project.create(paths, name).close()
    (paths.projects / "not a project").mkdir()
    assert {p.name for p in list_projects(paths)} == {"One", "Two"}


def test_newer_project_is_refused_unchanged(paths):
    p = Project.create(paths, "Future")
    p.conn.execute(f"PRAGMA user_version = {store.LATEST_VERSION + 5}")
    folder = p.folder
    p.close()
    before = (folder / "project.sqlite3").read_bytes()
    with pytest.raises(ProjectError, match="newer version"):
        Project.open(folder)
    assert (folder / "project.sqlite3").read_bytes() == before


def test_corrupted_project(paths):
    p = Project.create(paths, "Broken")
    folder = p.folder
    p.close()
    db = folder / "project.sqlite3"
    data = bytearray(db.read_bytes())
    for i in range(1024, len(data)):
        data[i] = 0xAB
    db.write_bytes(bytes(data))
    with pytest.raises(ProjectError, match="damaged"):
        Project.open(folder)


def test_garbage_project_file(paths):
    folder = paths.projects / "Garbage"
    folder.mkdir()
    (folder / "project.sqlite3").write_bytes(b"hello")
    with pytest.raises(ProjectError, match="damaged"):
        Project.open(folder)


def test_foreign_sqlite_is_not_a_project(paths):
    folder = paths.projects / "Foreign"
    folder.mkdir()
    conn = sqlite3.connect(str(folder / "project.sqlite3"))
    conn.execute("CREATE TABLE x (y)")
    conn.commit()
    conn.close()
    with pytest.raises(ProjectError, match="not a Crimson Soundtrack Studio project"):
        Project.open(folder)


def test_missing_project(paths):
    with pytest.raises(ProjectError, match="could not be found"):
        Project.open(paths.projects / "nope")


def test_migration_to_newer_format_keeps_data(paths, monkeypatch):
    p = Project.create(paths, "Old")
    p.set("keep", "me")
    folder = p.folder
    p.close()
    extra = (store.LATEST_VERSION + 1, "CREATE TABLE future_feature (id INTEGER PRIMARY KEY);")
    monkeypatch.setattr(store, "MIGRATIONS", store.MIGRATIONS + [extra])
    monkeypatch.setattr(store, "LATEST_VERSION", extra[0])
    again = Project.open(folder)
    assert again.format_version == extra[0]
    assert again.get("keep") == "me"
    assert again.query("SELECT name FROM sqlite_master WHERE name='future_feature'")
    again.close()


def test_transaction_rolls_back(paths):
    p = Project.create(paths, "Tx")
    with pytest.raises(RuntimeError):
        with p.transaction() as c:
            c.execute("INSERT INTO project_setting(key, value) VALUES ('x', '1')")
            raise RuntimeError("boom")
    assert p.get("x") is None
    p.close()
