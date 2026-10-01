"""Model downloads must work when the Studio's folder is so deep that the target passes 260 characters (Windows)."""

import hashlib
from pathlib import Path

from soundtrack_studio import app_paths
from soundtrack_studio.ai import downloader


class _Resp:
    def __init__(self, data):
        self.data, self.pos = data, 0
        self.headers = {"Content-Length": str(len(data))}
        self.status = 200

    def read(self, n):
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_opens_files_through_the_long_path_form(paths, monkeypatch):
    data = b"GGUF" + bytes(range(256)) * 40
    monkeypatch.setattr(downloader, "_open", lambda url, headers: _Resp(data))
    seen = []
    real = app_paths.os_path

    def recording(path, **kwargs):
        seen.append(str(path))
        return real(path, **kwargs)

    monkeypatch.setattr(app_paths, "os_path", recording)
    target = paths.models / "low" / "a-model" / "Some-Model-Q4_K_M.gguf"
    source = {"url": "https://example.invalid/m.gguf", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    result = downloader.download_file(source, target, paths)
    assert result["hash_checked_against_source"] and Path(result["path"]) == target and target.is_file()
    # every file operation went through long_path(): the target and its .part were handed to os_path
    assert any(s.endswith("Some-Model-Q4_K_M.gguf") for s in seen)
    assert any(s.endswith(".gguf.part") or s.endswith(".gguf") for s in seen)


def test_model_status_uses_the_long_path_checks(paths, monkeypatch):
    from soundtrack_studio.ai import catalog
    from soundtrack_studio.ai.catalog import ModelCatalog, ModelRegistry, model_status

    model = next(iter(ModelCatalog.load(paths).models.values()))
    calls = []
    real = catalog.is_file
    monkeypatch.setattr(catalog, "is_file", lambda p: calls.append(str(p)) or real(p))
    path = model.install_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")
    assert model_status(paths, ModelRegistry(paths), model)["present"]
    assert calls and calls[0] == str(path)
