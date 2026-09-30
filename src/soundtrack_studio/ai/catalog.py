"""Local model catalog, tiers and the installed-model registry.

* The catalog is data: ``data/config/model_catalog.json`` (copied from the
  bundled default on first run, upgraded when a newer build ships a newer
  catalog; the user's copy is kept as a backup) can be edited without a new build.
* Catalog models install to ``models/<tier>/<model id>/<file>.gguf``.
* Custom models are any local GGUF file the user adds; they are registered in
  ``data/config/models.json`` together with the download/verification state of
  every model. Nothing is stored outside the application folder.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..app_paths import MODEL_TIERS, AppPaths
from ..errors import StudioError

log = logging.getLogger(__name__)

BUNDLED_CATALOG = Path(__file__).resolve().parents[1] / "resources" / "model_catalog.json"
CATALOG_TIERS = ("low", "medium", "high")


class ModelError(StudioError):
    title = "AI model problem"


@dataclass
class LocalModel:
    id: str
    display_name: str
    tier: str
    backend: str = "llama.cpp"
    source: str = "huggingface"
    repository: str = ""
    revision: str = "main"
    filename: str = ""
    quantization: str = ""
    download_url: str = ""
    approximate_size_gb: float = 0.0
    approximate_ram_gb: float = 0.0
    recommended_vram_gb: float = 0.0
    context_size: int = 8192
    json_mode: str = "schema"
    license: str = ""
    license_url: str = ""
    alternate_repositories: List[str] = field(default_factory=list)
    local_path: str = ""          # stored path for custom models (app:... when inside the root)

    def install_dir(self, paths: AppPaths) -> Path:
        return paths.model_tier_dir(self.tier) / self.id

    def install_path(self, paths: AppPaths) -> Path:
        if self.local_path:
            return paths.from_stored(self.local_path) or Path(self.local_path)
        return self.install_dir(paths) / self.filename

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TierInfo:
    key: str
    label: str
    best_for: str
    recommended_vram: str
    recommended_ram_gb: float
    default_model: str


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_catalog(paths: AppPaths) -> Path:
    target = paths.config / "model_catalog.json"
    bundled_text = BUNDLED_CATALOG.read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(bundled_text, encoding="utf-8")
        return target
    try:
        current = int(json.loads(target.read_text(encoding="utf-8")).get("catalog_version", 0))
    except (OSError, ValueError, TypeError):
        current = -1
    bundled = int(json.loads(bundled_text).get("catalog_version", 0))
    if current < bundled:
        backup = target.with_name(f"model_catalog.v{max(current, 0)}.bak.json")
        target.replace(backup)
        target.write_text(bundled_text, encoding="utf-8")
        log.info("Model catalog upgraded to version %d (previous copy kept as %s)", bundled, backup.name)
    return target


class ModelRegistry:
    """Download/verification state of models + custom model definitions (``data/config/models.json``)."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.file = paths.config / "models.json"
        self._lock = threading.RLock()
        self.data: Dict[str, Any] = {"version": 1, "models": {}, "custom": []}
        if self.file.is_file():
            try:
                loaded = json.loads(self.file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except (OSError, ValueError):
                log.warning("models.json unreadable; model states reset (files in models/ are kept)")

    def save(self) -> None:
        with self._lock:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.file)

    def state(self, model_id: str) -> Dict[str, Any]:
        return dict(self.data["models"].get(model_id, {}))

    def update(self, model_id: str, **values: Any) -> None:
        with self._lock:
            self.data["models"].setdefault(model_id, {}).update(values)
            self.save()

    def forget(self, model_id: str) -> None:
        with self._lock:
            self.data["models"].pop(model_id, None)
            self.data["custom"] = [c for c in self.data["custom"] if c.get("id") != model_id]
            self.save()

    def custom_models(self) -> List[LocalModel]:
        known = set(LocalModel.__dataclass_fields__)
        return [LocalModel(**{k: v for k, v in c.items() if k in known}) for c in self.data.get("custom", [])]

    def add_custom(self, model: LocalModel) -> None:
        with self._lock:
            self.data["custom"] = [c for c in self.data["custom"] if c.get("id") != model.id] + [model.to_dict()]
            self.save()


class ModelCatalog:
    def __init__(self, models: List[LocalModel], tiers: Dict[str, TierInfo], notes: str = "") -> None:
        self.models = {m.id: m for m in models}
        self.tiers = tiers
        self.notes = notes

    @classmethod
    def load(cls, paths: AppPaths, registry: Optional[ModelRegistry] = None) -> "ModelCatalog":
        file = ensure_catalog(paths)
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("model_catalog.json unreadable (%s); using the bundled catalog", exc)
            data = json.loads(BUNDLED_CATALOG.read_text(encoding="utf-8"))
        known = set(LocalModel.__dataclass_fields__)
        models = []
        for raw in data.get("models", []):
            item = {k: v for k, v in raw.items() if k in known}
            if item.get("tier") not in CATALOG_TIERS or not item.get("id") or not item.get("filename"):
                log.warning("Skipping invalid catalog entry %r", raw.get("id"))
                continue
            models.append(LocalModel(**item))
        tiers = {key: TierInfo(key, t.get("label", key.title()), t.get("best_for", ""), t.get("recommended_vram", ""),
                               float(t.get("recommended_ram_gb", 0) or 0), t.get("default_model", ""))
                 for key, t in data.get("tiers", {}).items() if key in CATALOG_TIERS}
        tiers["custom"] = TierInfo("custom", "Custom", "Any compatible GGUF model file you already have.",
                                   "depends on the model", 0, "")
        catalog = cls(models, tiers, data.get("notes", ""))
        for model in (registry.custom_models() if registry else []):
            catalog.models[model.id] = model
        return catalog

    def for_tier(self, tier: str) -> List[LocalModel]:
        return [m for m in self.models.values() if m.tier == tier]

    def default_for_tier(self, tier: str) -> Optional[LocalModel]:
        info = self.tiers.get(tier)
        if info and info.default_model in self.models:
            return self.models[info.default_model]
        options = self.for_tier(tier)
        return options[0] if options else None

    def get(self, model_id: str) -> Optional[LocalModel]:
        return self.models.get(model_id)


def is_gguf(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def register_custom_model(paths: AppPaths, registry: ModelRegistry, gguf_path: Path, display_name: str = "",
                          context_size: int = 8192) -> LocalModel:
    gguf_path = Path(gguf_path)
    if not gguf_path.is_file():
        raise ModelError("The selected model file does not exist.", details=str(gguf_path))
    if not is_gguf(gguf_path):
        raise ModelError(f"{gguf_path.name} is not a GGUF model file.",
                         hint="Choose a model in GGUF format (the format used by llama.cpp).")
    slug = re.sub(r"[^a-z0-9._-]+", "-", gguf_path.stem.lower()).strip("-")[:60] or "model"
    model = LocalModel(id=f"custom-{slug}", display_name=display_name or gguf_path.stem, tier="custom",
                       source="local", filename=gguf_path.name, local_path=paths.to_stored(gguf_path.resolve()),
                       approximate_size_gb=round(gguf_path.stat().st_size / 1e9, 2), context_size=context_size,
                       license="user supplied")
    registry.add_custom(model)
    registry.update(model.id, status="available", size=gguf_path.stat().st_size, added_at=now_iso())
    return model


def model_status(paths: AppPaths, registry: ModelRegistry, model: LocalModel) -> Dict[str, Any]:
    state = registry.state(model.id)
    path = model.install_path(paths)
    present = path.is_file()
    partial = path.with_name(path.name + ".part")
    status = state.get("status") or ("available" if present else "not_downloaded")
    if not present and status in ("available", "verified"):
        status = "missing"
    if not present and partial.is_file():
        status = "partial"
    return {"id": model.id, "present": present, "path": str(path), "status": status,
            "partial_bytes": partial.stat().st_size if partial.is_file() else 0,
            "sha256": state.get("sha256"), "verified_at": state.get("verified_at"),
            "hash_checked_against_source": state.get("hash_checked_against_source"),
            "inference_ok": state.get("inference_ok"), "tier": model.tier}


def all_tiers() -> tuple:
    return MODEL_TIERS
