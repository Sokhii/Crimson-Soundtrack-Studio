import hashlib
import http.server
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from soundtrack_studio.ai import downloader, hardware
from soundtrack_studio.ai.catalog import (LocalModel, ModelCatalog, ModelError, ModelRegistry, ensure_catalog,
                                          model_status, register_custom_model)
from soundtrack_studio.ai.runtime import ModelOutputError, ScriptedBackend, find_llama_server, run_inference_check
from soundtrack_studio.errors import OperationCancelled


def fake_gguf(path: Path, size: int = 300_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + bytes((i * 7) & 0xFF for i in range(size - 4)))
    return path


# ---------------------------------------------------------------- catalog
def test_catalog_tiers_and_paths(paths):
    registry = ModelRegistry(paths)
    catalog = ModelCatalog.load(paths, registry)
    for tier in ("low", "medium", "high"):
        default = catalog.default_for_tier(tier)
        assert default is not None and default.tier == tier
        assert default.install_path(paths).parent == paths.models / tier / default.id
        assert default.approximate_size_gb > 0 and default.recommended_vram_gb > 0
    assert "custom" in catalog.tiers
    assert (paths.config / "model_catalog.json").is_file()


def test_catalog_upgrade_keeps_backup_and_user_edits_survive(paths):
    target = ensure_catalog(paths)
    data = json.loads(target.read_text())
    data["catalog_version"] = 0
    target.write_text(json.dumps(data))
    ensure_catalog(paths)
    assert (paths.config / "model_catalog.v0.bak.json").is_file()
    data = json.loads(target.read_text())
    data["models"].append({"id": "my-model", "display_name": "Mine", "tier": "low", "filename": "mine.gguf"})
    data["models"].append({"id": "broken", "tier": "nope"})
    target.write_text(json.dumps(data))
    catalog = ModelCatalog.load(paths)
    assert catalog.get("my-model") is not None and catalog.get("broken") is None


def test_custom_model_registration(paths, tmp_path):
    registry = ModelRegistry(paths)
    gguf = fake_gguf(tmp_path / "Some Model Q4.gguf")
    model = register_custom_model(paths, registry, gguf)
    assert model.tier == "custom" and model.install_path(paths) == gguf.resolve()
    assert ModelCatalog.load(paths, ModelRegistry(paths)).get(model.id) is not None  # persisted
    (tmp_path / "bad.gguf").write_bytes(b"nope")
    with pytest.raises(ModelError, match="not a GGUF"):
        register_custom_model(paths, registry, tmp_path / "bad.gguf")
    inside = fake_gguf(paths.models / "custom" / "x.gguf")
    assert register_custom_model(paths, registry, inside).local_path == "app:models/custom/x.gguf"


def test_model_status_partial_and_missing(paths):
    registry = ModelRegistry(paths)
    model = ModelCatalog.load(paths).default_for_tier("low")
    assert model_status(paths, registry, model)["status"] == "not_downloaded"
    target = model.install_path(paths)
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".part").write_bytes(b"GGUF123")
    assert model_status(paths, registry, model)["status"] == "partial"
    registry.update(model.id, status="verified")
    target.with_name(target.name + ".part").unlink()
    assert model_status(paths, registry, model)["status"] == "missing"


def test_hardware_detection_and_recommendation():
    info = hardware.detect()
    assert info.cpu_threads >= 1
    assert hardware.recommend_tier(hardware.HardwareInfo(8, 16, 8, [hardware.GpuInfo("x", 24)])) == "high"
    assert hardware.recommend_tier(hardware.HardwareInfo(8, 16, 8, [hardware.GpuInfo("x", 8)])) == "medium"
    assert hardware.recommend_tier(hardware.HardwareInfo(4, 8, 4, [])) == "low"
    assert hardware.fits(hardware.HardwareInfo(4, 16, 8, [hardware.GpuInfo("g", 6)]), 5, 8) == "cpu"
    assert hardware.fits(hardware.HardwareInfo(4, 4, 2, []), 12, 16) == "no"


# -------------------------------------------------------------- downloads
def local_open(url, headers=None, timeout=60):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(urllib.request.Request(url, headers=headers or {}), timeout=timeout)


class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    fail_after = None  # simulate a dropped connection after N bytes

    def do_GET(self):  # noqa: N802
        data = self.payload
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = data[start:]
        if self.fail_after is not None and not rng:
            body = body[:self.fail_after]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    payload = b"GGUF" + bytes((i * 13) & 0xFF for i in range(3_000_000))
    RangeHandler.payload = payload
    RangeHandler.fail_after = None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, payload
    httpd.shutdown()


def test_download_verifies_hash_and_stays_in_models(paths, server, monkeypatch):
    httpd, payload = server
    monkeypatch.setattr(downloader, "_open", local_open)
    model = LocalModel(id="t", display_name="T", tier="low", filename="t.gguf")
    url = f"http://127.0.0.1:{httpd.server_port}/t.gguf"
    source = {"url": url, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    result = downloader.download_model(model, paths, source=source)
    assert result["hash_checked_against_source"] and Path(result["path"]) == paths.models / "low" / "t" / "t.gguf"
    assert (paths.models / "low" / "t" / "t.gguf").read_bytes() == payload


def test_download_resumes_after_interruption_and_cancel(paths, server, monkeypatch):
    httpd, payload = server
    monkeypatch.setattr(downloader, "_open", local_open)
    model = LocalModel(id="r", display_name="R", tier="medium", filename="r.gguf")
    source = {"url": f"http://127.0.0.1:{httpd.server_port}/r.gguf", "sha256": hashlib.sha256(payload).hexdigest(),
              "size": len(payload)}
    with pytest.raises(OperationCancelled):
        downloader.download_model(model, paths, cancel=lambda: True, source=source)
    RangeHandler.fail_after = 1_000_000  # first request ends early
    with pytest.raises(ModelError, match="incomplete"):
        downloader.download_model(model, paths, source=source)
    part = model.install_path(paths).with_name("r.gguf.part")
    assert part.stat().st_size == 1_000_000
    RangeHandler.fail_after = None
    progress = []
    result = downloader.download_model(model, paths, progress=lambda *a: progress.append(a), source=source)
    assert result["size"] == len(payload) and not part.exists()


def test_corrupt_download_is_removed(paths, server, monkeypatch):
    httpd, payload = server
    monkeypatch.setattr(downloader, "_open", local_open)
    model = LocalModel(id="c", display_name="C", tier="high", filename="c.gguf")
    source = {"url": f"http://127.0.0.1:{httpd.server_port}/c.gguf", "sha256": "0" * 64, "size": len(payload)}
    with pytest.raises(ModelError, match="corrupt"):
        downloader.download_model(model, paths, source=source)
    assert not any(p.suffix == ".gguf" for p in (paths.models / "high" / "c").glob("*"))


def test_pick_gguf():
    files = [{"path": "Model-Q8_0.gguf"}, {"path": "model-q4_k_m.gguf"}, {"path": "mmproj-Q4_K_M.gguf"},
             {"path": "big-Q4_K_M-00001-of-00002.gguf"}, {"path": "README.md"}]
    assert downloader.pick_gguf(files, "Model-Q4_K_M.gguf", "Q4_K_M")["path"] == "model-q4_k_m.gguf"
    assert downloader.pick_gguf(files, "Model-Q8_0.gguf", "Q8_0")["path"] == "Model-Q8_0.gguf"
    assert downloader.pick_gguf(files, "x.gguf", "IQ2_XS") is None


def test_runtime_lookup(paths):
    assert find_llama_server(paths) is None
    exe = paths.runtime / "llama" / ("llama-server.exe" if __import__("os").name == "nt" else "llama-server")
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert find_llama_server(paths) == exe


def test_inference_check_handles_incomplete_output():
    class Broken(ScriptedBackend):
        def chat(self, *a, **k):
            raise ModelOutputError("incomplete", details="HTTP 500: does not match")

    assert run_inference_check(Broken([]))["ok"] is False
    assert run_inference_check(ScriptedBackend(['{"ok": true, "word": "x"}']))["ok"] is True
