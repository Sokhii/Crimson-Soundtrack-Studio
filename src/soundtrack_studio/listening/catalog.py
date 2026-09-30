"""The optional listening model(s): files, pinned hashes, install location, status.

The files are pinned to a repository revision and verified by SHA-256 (ONNX
graphs) after download; the small JSON/tokenizer files are validated by
parsing. They live in ``models/listening/<id>/`` like every other model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..app_paths import AppPaths


@dataclass(frozen=True)
class ModelFile:
    role: str                      # audio | text | tokenizer | preprocessor | config
    path: str                      # path inside the repository (and inside the install folder)
    sha256: Optional[str] = None
    size: Optional[int] = None


@dataclass(frozen=True)
class ListeningModel:
    id: str
    display_name: str
    description: str
    repository: str
    revision: str
    files: tuple
    license: str
    license_url: str
    upstream: str
    approximate_size_mb: int

    def install_dir(self, paths: AppPaths) -> Path:
        return paths.models / "listening" / self.id

    def file_map(self) -> Dict[str, str]:
        return {f.role: f.path for f in self.files}

    def url(self, f: ModelFile) -> str:
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{f.path}"


CLAP_MUSIC_SPEECH = ListeningModel(
    id="clap-larger-music-speech",
    display_name="CLAP (LAION larger_clap_music_and_speech)",
    description=("Listens to the audio itself and recognises instruments, vocals, mood and style, and how much two "
                 "pieces sound alike. It works alongside the description model above; it does not replace it."),
    repository="Xenova/larger_clap_music_and_speech",       # ONNX export of laion/larger_clap_music_and_speech
    revision="e9fd5ac1dbf3280936a7fc3ec8a020453ff184db",
    files=(
        ModelFile("audio", "onnx/audio_model.onnx",
                  "3ecc72d27740e2a09ced20cf22fd6244122e5e506008763a0f368b3b4ff6eac8", 281749092),
        # fp16 text tower: identical to PyTorch in CI (cosine 1.0000); the int8 one averaged only 0.978
        ModelFile("text", "onnx/text_model_fp16.onnx",
                  "79da753839b7fd22afc3fa2076cb75ffb5729c24b3249042cf3ec6a4e2c924b3", 251029088),
        ModelFile("tokenizer", "tokenizer.json", None, 2108774),
        ModelFile("preprocessor", "preprocessor_config.json", None, 541),
        ModelFile("config", "config.json", None, 596),
    ),
    license="Apache-2.0",
    license_url="https://huggingface.co/laion/larger_clap_music_and_speech",
    upstream="https://github.com/LAION-AI/CLAP",
    approximate_size_mb=535,
)

LISTENING_MODELS: List[ListeningModel] = [CLAP_MUSIC_SPEECH]


def get_listening_model(model_id: str) -> Optional[ListeningModel]:
    return next((m for m in LISTENING_MODELS if m.id == model_id), None)


def validate_small_file(path: Path, role: str) -> None:
    """JSON/tokenizer files have no pinned hash: they must parse and look right."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if role == "preprocessor" and data.get("feature_extractor_type") != "ClapFeatureExtractor":
        raise ValueError("not a CLAP preprocessor configuration")
    if role == "tokenizer" and "model" not in data:
        raise ValueError("not a tokenizer definition")
    if role == "config" and data.get("model_type") != "clap":
        raise ValueError("not a CLAP model configuration")


def listening_status(paths: AppPaths, model: ListeningModel, state: Dict[str, Any]) -> Dict[str, Any]:
    folder = model.install_dir(paths)
    present = {f.role: (folder / f.path).is_file() for f in model.files}
    installed = all(present.values())
    size = sum((folder / f.path).stat().st_size for f in model.files if present[f.role])
    return {"installed": installed, "partial": any(present.values()) and not installed,
            "size_mb": round(size / 1e6), "verified": bool(state.get("verified")) and installed,
            "folder": str(folder), "last_test": state.get("last_test")}
