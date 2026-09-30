"""The listening pipeline against the original implementation (runs in the `listening-reference` CI job).

Requires the downloaded model (``CSS_CLAP_HOME`` = a portable root where tools/fetch_listening_model.py put it),
plus PyTorch and transformers (CI only; neither is part of the application).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

HOME = os.environ.get("CSS_CLAP_HOME")
pytestmark = pytest.mark.skipif(not HOME, reason="set CSS_CLAP_HOME (see tools/fetch_listening_model.py)")
REFERENCE_REPO = "laion/larger_clap_music_and_speech"


def _signals():
    rng = np.random.default_rng(3)
    t = np.arange(480000) / 48000
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220, 277.2, 329.6)) * 0.15 * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    beat = np.zeros_like(t)
    for k in range(0, 480000, 12000):
        beat[k:k + 2000] += rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300) * 0.8
    return {"chord_10s": chord.astype(np.float32), "beat_10s": (beat + 0.02 * chord).astype(np.float32),
            "chord_3s": chord[:144000].astype(np.float32), "noise_7s": (rng.standard_normal(336000) * 0.1).astype(np.float32)}


@pytest.fixture(scope="module")
def model_dir():
    from soundtrack_studio.app_paths import AppPaths
    from soundtrack_studio.listening.catalog import CLAP_MUSIC_SPEECH

    return CLAP_MUSIC_SPEECH.install_dir(AppPaths(Path(HOME))), CLAP_MUSIC_SPEECH.file_map()


@pytest.fixture(scope="module")
def ours(model_dir):
    from soundtrack_studio.listening.clap import ClapModel

    return ClapModel(model_dir[0], model_dir[1], device="cpu")


@pytest.fixture(scope="module")
def reference():
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    model = transformers.ClapModel.from_pretrained(REFERENCE_REPO).eval()
    processor = transformers.ClapProcessor.from_pretrained(REFERENCE_REPO)
    return torch, model, processor


def _projected(torch, out, projection):
    """transformers 4.x returns the projected tensor; 5.x a model output whose pooler_output holds it."""

    arr = out if isinstance(out, torch.Tensor) else out.pooler_output
    if arr.shape[-1] != 512:
        arr = projection(arr)
    return arr.numpy()


def test_front_end_matches_transformers(ours, reference):
    _torch, _model, processor = reference
    for name, x in _signals().items():
        theirs = processor.feature_extractor(x, sampling_rate=48000, return_tensors="np")["input_features"]
        mine = ours.front.batch([x])
        assert mine.shape == theirs.shape, name
        diff = float(np.max(np.abs(mine - theirs)))
        print(f"front end {name}: max |diff| = {diff:.2e} dB")
        assert diff < 1e-2, name


def test_audio_embeddings_match_pytorch(ours, reference):
    torch, model, processor = reference
    signals = _signals()
    mine = ours.embed_audio(list(signals.values()))
    for i, (name, x) in enumerate(signals.items()):
        feats = processor.feature_extractor(x, sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            ref = _projected(torch, model.get_audio_features(**feats), model.audio_projection)[0]
        ref = ref / np.linalg.norm(ref)
        cos = float(np.dot(mine[i], ref))
        print(f"audio embedding {name}: cosine(ONNX, PyTorch) = {cos:.5f}")
        assert cos > 0.999, name


def test_text_embeddings_and_tokens_match(ours, reference):
    from soundtrack_studio.listening.listen import all_prompts

    torch, model, processor = reference
    prompts = all_prompts()[:24] + ["instrumental music without vocals", "a song with a singer singing lyrics"]
    mine = ours.embed_text(prompts)
    tok = processor.tokenizer(prompts, padding=True, return_tensors="pt")
    ours._text_session()
    enc = ours._tokenizer.encode_batch(prompts)
    assert [e.ids[:sum(e.attention_mask)] for e in enc] == [
        ids[:int(m.sum())].tolist() for ids, m in zip(tok["input_ids"], tok["attention_mask"])]
    with torch.no_grad():
        ref = _projected(torch, model.get_text_features(**tok), model.text_projection)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    cos = np.sum(mine * ref, axis=1)
    print(f"text embeddings (as used by the app vs PyTorch): min cosine {cos.min():.4f}, mean {cos.mean():.4f}")
    assert cos.min() > 0.999


def test_public_domain_recordings_report(ours, tmp_path):
    """Diagnostics only: vocal margins and heard tags on real recordings listed in CSS_CLAP_SAMPLES (JSON)."""

    from soundtrack_studio.listening.listen import PromptBank, all_prompts, listen_file

    samples = json.loads(os.environ.get("CSS_CLAP_SAMPLES", "[]"))
    if not samples:
        pytest.skip("no sample recordings downloaded")
    prompts = all_prompts()
    bank = PromptBank(dict(zip(prompts, ours.embed_text(prompts))))
    for item in samples:
        result = listen_file(ours, Path(item["path"]), "reference")
        summary = bank.summary(result)
        print(f"[{item['label']}] {Path(item['path']).name}: vocals={summary.get('vocals')} "
              f"margin={summary.get('vocals_margin')} instruments={summary.get('instrumentation')} "
              f"mood={summary.get('mood')} style={summary.get('style')}")


def test_text_variants_diagnostics(model_dir, reference):
    """Which text tower / batching matches PyTorch best (prints only; CSS_CLAP_EXTRA holds extra ONNX files)."""

    from soundtrack_studio.listening.clap import ClapModel
    from soundtrack_studio.listening.listen import all_prompts

    torch, model, processor = reference
    prompts = all_prompts()[:40]
    tok = processor.tokenizer(prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        ref = _projected(torch, model.get_text_features(**tok), model.text_projection)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    folder, files = model_dir
    variants = {"quantized": files["text"]}
    extra = os.environ.get("CSS_CLAP_EXTRA")
    for name in ("text_model.onnx", "text_model_fp16.onnx"):
        if extra and (Path(extra) / name).is_file():
            variants[name] = str((Path(extra) / name).resolve())
    for name, path in variants.items():
        m = ClapModel(folder, {**files, "text": path}, device="cpu")
        batch = m.embed_text(prompts)
        single = np.concatenate([m.embed_text([p]) for p in prompts])
        for how, emb in (("batched", batch), ("one at a time", single)):
            cos = np.sum(emb * ref, axis=1)
            print(f"text {name} {how}: min {cos.min():.4f} mean {cos.mean():.4f}")
