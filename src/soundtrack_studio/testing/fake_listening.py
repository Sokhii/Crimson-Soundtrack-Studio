"""A tiny stand-in for the CLAP listening model (same ONNX inputs/outputs and file layout), for tests.

The "audio tower" averages the log-mel frames and projects them; the "text tower" averages word embeddings.
The numbers mean nothing musically; the point is to exercise the real loading, preprocessing, caching and
integration code paths without a 400 MB download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIM = 16


def write_fake_listening_model(folder: Path, files: dict, words) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(7)
    folder.mkdir(parents=True, exist_ok=True)
    # audio: input_features [B,1,T,64] -> mean over (1,2) -> [B,64] @ W -> audio_embeds [B,DIM]
    w = numpy_helper.from_array(rng.standard_normal((64, DIM)).astype(np.float32), "W")
    axes = numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "axes")
    audio = helper.make_graph(
        [helper.make_node("ReduceMean", ["input_features", "axes"], ["pooled"], keepdims=0),
         helper.make_node("MatMul", ["pooled", "W"], ["audio_embeds"])],
        "fake_clap_audio",
        [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, ["batch", 1, "frames", 64])],
        [helper.make_tensor_value_info("audio_embeds", TensorProto.FLOAT, ["batch", DIM])], [w, axes])
    _save(onnx, helper.make_model(audio, opset_imports=[helper.make_opsetid("", 18)]), folder / files["audio"])
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for word in sorted({w for text in words for w in text.lower().split()}):
        vocab.setdefault(word, len(vocab))
    e = numpy_helper.from_array(rng.standard_normal((len(vocab), DIM)).astype(np.float32), "E")
    axis1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1")
    text = helper.make_graph(
        [helper.make_node("Gather", ["E", "input_ids"], ["tok"]),
         helper.make_node("ReduceMean", ["tok", "axis1"], ["text_embeds"], keepdims=0)],
        "fake_clap_text",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "len"]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "len"])],
        [helper.make_tensor_value_info("text_embeds", TensorProto.FLOAT, ["batch", DIM])], [e, axis1])
    _save(onnx, helper.make_model(text, opset_imports=[helper.make_opsetid("", 18)]), folder / files["text"])
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    (folder / files["tokenizer"]).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(folder / files["tokenizer"]))
    (folder / files["preprocessor"]).write_text(json.dumps({
        "feature_extractor_type": "ClapFeatureExtractor", "feature_size": 64, "fft_window_size": 1024,
        "frequency_max": 14000, "frequency_min": 50, "hop_length": 480, "max_length_s": 10, "padding": "repeatpad",
        "sampling_rate": 48000, "truncation": "rand_trunc"}), encoding="utf-8")
    (folder / files["config"]).write_text(json.dumps({"model_type": "clap", "projection_dim": DIM}), encoding="utf-8")


def _save(onnx, model, path: Path) -> None:
    model.ir_version = 9
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def install_fake_listening_model(studio) -> None:
    """Put the stand-in model where the real one would be installed and mark it downloaded."""

    from ..listening.catalog import CLAP_MUSIC_SPEECH as model
    from ..listening.listen import all_prompts

    write_fake_listening_model(model.install_dir(studio.paths), model.file_map(), all_prompts())
    studio.registry.update(model.id, verified=True, files={f.path: "fake" for f in model.files})
