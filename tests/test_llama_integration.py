"""Real llama.cpp runtime end to end (process start, model load, schema-constrained answers, shutdown).

Needs a llama-server binary: set CSS_LLAMA_SERVER to its path (CI sets it to the bundled
Windows runtime; locally build llama.cpp). The model is a tiny generated GGUF (tools/make_tiny_gguf.py),
so this tests the plumbing, not description quality.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

SERVER = os.environ.get("CSS_LLAMA_SERVER")
pytestmark = pytest.mark.skipif(not SERVER or not Path(SERVER).is_file(), reason="set CSS_LLAMA_SERVER")
gguf = pytest.importorskip("gguf")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


@pytest.fixture
def studio_with_runtime(paths, tmp_path, game, analyzer_db):
    from make_tiny_gguf import build

    from soundtrack_studio.services import Studio
    from soundtrack_studio.testing.fixtures import write_test_flac

    runtime = paths.runtime / "llama"
    shutil.copytree(Path(SERVER).parent, runtime)
    model_file = build(paths.models / "custom" / "tiny.gguf")
    studio = Studio(paths)
    studio.create_project("Runtime")
    studio.import_analyzer(analyzer_db)
    write_test_flac(tmp_path / "Music" / "Dark Requiem.flac", seconds=3, tags={"TITLE": "Dark Requiem"})
    studio.set_library_path(tmp_path / "Music")
    studio.scan_library()
    studio.select_model(studio.add_custom_model(model_file).id)
    yield studio
    studio.shutdown()


def test_runtime_describes_music_and_shuts_down(studio_with_runtime):
    studio = studio_with_runtime
    result = studio.test_model(studio.settings.ai_model_id)
    assert result["ok"], result
    backend = studio.backend()
    results = studio.analyze_semantics()
    assert results["track"].llm_done == 1 and results["cue"].llm_done == 4
    assert all(p.source == "llm" for p in studio.profiles("cue").values())
    again = studio.analyze_semantics()
    assert again["track"].skipped == 1 and again["cue"].skipped == 4
    process = backend.process
    studio.stop_ai()
    assert process.poll() is not None  # the child process is gone
    log = (studio.paths.logs / "llama-server.log").read_text(errors="replace")
    assert "tiny.gguf" in log
