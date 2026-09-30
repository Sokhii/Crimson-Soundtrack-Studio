"""CLAP inference with ONNX Runtime.

Two ONNX graphs from the model folder: the audio tower (``input_features`` ->
``audio_embeds``) and the text tower (``input_ids``/``attention_mask`` ->
``text_embeds``). Embeddings are L2-normalised, so a dot product is the cosine
similarity CLAP was trained with. On Windows the DirectML provider (any DirectX
12 GPU: AMD, NVIDIA, Intel) is used when available, with the CPU as fallback.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..app_paths import os_path
from ..errors import StudioError
from .features import ClapFrontEnd, FrontEndConfig

log = logging.getLogger(__name__)


class ListeningError(StudioError):
    title = "Listening model problem"


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def available_providers() -> List[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


class ClapModel:
    def __init__(self, model_dir: Path, files: Dict[str, str], device: str = "auto", threads: int = 0) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - bundled in releases
            raise ListeningError("The listening runtime (ONNX Runtime) is not installed.") from exc
        self.model_dir = Path(model_dir)
        self.files = files
        try:
            cfg = json.loads((self.model_dir / files["preprocessor"]).read_text(encoding="utf-8"))
            self.front = ClapFrontEnd(FrontEndConfig.from_preprocessor_config(cfg))
        except (OSError, ValueError, KeyError) as exc:
            raise ListeningError("The listening model's settings file is missing or unreadable.",
                                 hint="Verify or re-download the listening model.", details=str(exc)) from exc
        options = ort.SessionOptions()
        options.log_severity_level = 3
        if threads:
            options.intra_op_num_threads = threads
        wanted = ["CPUExecutionProvider"]
        if device != "cpu" and "DmlExecutionProvider" in ort.get_available_providers():
            wanted = ["DmlExecutionProvider", "CPUExecutionProvider"]
            options.enable_mem_pattern = False     # required by DirectML
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.audio = self._session(ort, files["audio"], options, wanted)
        self.provider = self.audio.get_providers()[0]
        self._text = None
        self._ort, self._options = ort, options
        self._audio_inputs = {i.name: i for i in self.audio.get_inputs()}
        if "input_features" not in self._audio_inputs:
            raise ListeningError("The listening model has an unexpected format.",
                                 details=f"audio inputs: {list(self._audio_inputs)}")

    def _session(self, ort, rel: str, options, providers):
        path = self.model_dir / rel
        if not path.is_file():
            raise ListeningError("A listening model file is missing.", hint="Download the listening model again.",
                                 details=str(path))
        try:
            return ort.InferenceSession(os_path(path), sess_options=options, providers=providers)
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises its own exception types
            if providers[0] != "CPUExecutionProvider":
                log.warning("Listening model: %s failed (%s); using the CPU", providers[0], exc)
                return ort.InferenceSession(os_path(path), sess_options=options, providers=["CPUExecutionProvider"])
            raise ListeningError("The listening model could not be loaded.", hint="Verify or re-download it.",
                                 details=f"{path.name}: {exc}") from exc

    # ------------------------------------------------------------------ audio
    def embed_audio(self, excerpts: Sequence[np.ndarray], batch_size: int = 4) -> np.ndarray:
        """Mono 48 kHz excerpts (<= 10 s each) -> normalised embeddings, shape (n, dim)."""

        out = []
        for start in range(0, len(excerpts), batch_size):
            feats = self.front.batch(excerpts[start:start + batch_size])
            feed = {"input_features": feats}
            if "is_longer" in self._audio_inputs:
                feed["is_longer"] = np.zeros((len(feats), 1), dtype=bool)
            try:
                result = self.audio.run(None, feed)
            except Exception as exc:  # noqa: BLE001
                if self.provider == "CPUExecutionProvider":
                    raise ListeningError("The listening model failed on a piece of audio.", details=str(exc)) from exc
                log.warning("Listening model: %s failed while running (%s); switching to the CPU", self.provider, exc)
                self.audio = self._session(self._ort, self.files["audio"], self._ort.SessionOptions(),
                                           ["CPUExecutionProvider"])
                self.provider = "CPUExecutionProvider"
                result = self.audio.run(None, feed)
            names = [o.name for o in self.audio.get_outputs()]
            out.append(result[names.index("audio_embeds")] if "audio_embeds" in names else result[0])
        return _normalise(np.concatenate(out, axis=0))

    # ------------------------------------------------------------------- text
    def _text_session(self):
        if self._text is None:
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:  # pragma: no cover
                raise ListeningError("The listening runtime (tokenizers) is not installed.") from exc
            # the text tower runs once per prompt list; the CPU is plenty and avoids GPU memory
            self._text = self._session(self._ort, self.files["text"], self._ort.SessionOptions(), ["CPUExecutionProvider"])
            tok_path = self.model_dir / self.files["tokenizer"]
            if not tok_path.is_file():
                raise ListeningError("A listening model file is missing.", details=str(tok_path))
            self._tokenizer = Tokenizer.from_file(os_path(tok_path))
            self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            self._tokenizer.enable_truncation(max_length=512)
        return self._text

    def embed_text(self, texts: Sequence[str], batch_size: int = 1) -> np.ndarray:
        """Normalised text embeddings. One prompt per run by default: the ONNX text tower mishandles padding
        (measured against PyTorch in CI: batched min cosine 0.63, one at a time 0.92, mean 0.98)."""

        session = self._text_session()
        names = {i.name for i in session.get_inputs()}
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self._tokenizer.encode_batch(list(texts[start:start + batch_size]))
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64)}
            if "attention_mask" in names:
                feed["attention_mask"] = np.array([e.attention_mask for e in enc], dtype=np.int64)
            result = session.run(None, feed)
            outputs = [o.name for o in session.get_outputs()]
            out.append(result[outputs.index("text_embeds")] if "text_embeds" in outputs else result[0])
        return _normalise(np.concatenate(out, axis=0))


def default_threads() -> int:
    return max(1, (os.cpu_count() or 4) - 1)


def load_model(model_dir: Path, files: Dict[str, str], device: str = "auto") -> ClapModel:
    return ClapModel(model_dir, files, device, threads=default_threads() if device == "cpu" else 0)


def version_info() -> Dict[str, Optional[str]]:
    try:
        import onnxruntime as ort
        return {"onnxruntime": ort.__version__, "providers": ",".join(ort.get_available_providers())}
    except ImportError:
        return {"onnxruntime": None, "providers": None}
