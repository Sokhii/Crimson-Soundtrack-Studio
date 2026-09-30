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


GENRE_PAIRS = [
    ("a pop song with a singer", "an instrumental pop song"),
    ("a rock song with a singer", "an instrumental rock song"),
    ("electronic dance music with a singer", "instrumental electronic dance music"),
    ("an anime song with a female singer", "instrumental anime music"),
    ("a ballad sung by a singer", "an instrumental ballad"),
    ("orchestral music with a singer", "orchestral music without singing"),
    ("a folk song with a singer", "instrumental folk music"),
    ("a jazz song with a singer", "instrumental jazz"),
    ("a song sung by a synthesized singing voice", "synthesizer music without vocals"),
    ("a hip hop song with a rapper", "an instrumental hip hop beat"),
]
SIMPLE_PAIRS = [("a song with vocals", "a song without vocals"), ("singing", "no singing"),
                ("music with a singer", "music without a singer")]


def test_public_domain_recordings_report(ours, tmp_path):
    """Diagnostics: compares vocal-detection strategies per excerpt on real recordings (CSS_CLAP_SAMPLES, JSON)
    and on spliced 'instrumental intro, then singing' files like a typical song. Prints only."""

    import soundfile as sf

    from soundtrack_studio.compiler.audio import resample
    from soundtrack_studio.listening.listen import PromptBank, all_prompts, excerpt_embeddings, listen_file

    samples = json.loads(os.environ.get("CSS_CLAP_SAMPLES", "[]"))
    if not samples:
        pytest.skip("no sample recordings downloaded")
    extra = [p for pair in GENRE_PAIRS + SIMPLE_PAIRS for p in pair]
    prompts = list(dict.fromkeys(all_prompts() + extra))
    vectors = dict(zip(prompts, ours.embed_text(prompts)))
    bank = PromptBank(vectors)

    def load48(path, seconds=None, start_frac=0.0):
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = resample(data.mean(axis=1, keepdims=True), rate, 48000)[:, 0]
        a = int(len(mono) * start_frac)
        return mono[a:a + int(seconds * 48000)] if seconds else mono

    vocal_files = [x for x in samples if x["label"] == "vocals"]
    inst_files = [x for x in samples if x["label"] == "instrumental"]
    for i, (v, n) in enumerate(zip(vocal_files[:4], inst_files[:4])):
        try:
            mix = np.concatenate([load48(n["path"], 40), load48(v["path"], 20, 1 / 3)])
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            print(f"splice {i}: {exc}")
            continue
        path = tmp_path / f"spliced_{i}.wav"
        sf.write(str(path), mix, 48000)
        samples.append({"label": "vocals", "title": f"SPLICE 40 s {n['title']} + 20 s {v['title']}", "path": str(path)})

    def s1(e):
        return bank.vocal_margin(e)

    def s2(e):
        pairs = [(float(e @ vectors[a]), float(e @ vectors[b])) for a, b in GENRE_PAIRS]
        best = sorted(pairs, key=lambda p: -max(p))[:3]
        return float(np.mean([a - b for a, b in best]))

    def s3(e):
        return float(np.mean([float(e @ vectors[a]) - float(e @ vectors[b]) for a, b in SIMPLE_PAIRS]))

    strategies = {"S1 current": s1, "S2 genre-matched": s2, "S3 simple pairs": s3}
    rows = []
    for item in samples:
        try:
            result = listen_file(ours, Path(item["path"]), "reference")
        except Exception as exc:  # noqa: BLE001
            print(f"[{item['label']}] {item.get('title')}: unreadable ({exc})")
            continue
        exc_emb = excerpt_embeddings(result)
        margins = {name: [f(e) for e in exc_emb] for name, f in strategies.items()}
        rows.append((item, margins))
        print(f"[{item['label']}] {item.get('title')} ({len(exc_emb)} excerpts): " + " | ".join(
            f"{name}: {[round(m, 3) for m in ms]}" for name, ms in margins.items()))
    for name in strategies:
        for tv in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03):
            vok = vtot = iok = itot = jok = jtot = 0
            for item, margins in rows:
                ms = margins[name]
                if not ms:
                    continue
                sung = sum(1 for m in ms if m > tv) >= max(1, int(np.ceil(len(ms) / 3 - 1e-9)))
                if item["label"] == "vocals":
                    vtot += 1
                    vok += sung
                elif item["label"] == "japanese":      # Japanese recordings, mostly sung (labels unverified)
                    jtot += 1
                    jok += sung
                elif item["label"] == "instrumental":
                    itot += 1
                    iok += not sung
            print(f"TALLY {name} threshold {tv:+.2f}: vocals found {vok}/{vtot}, japanese marked sung {jok}/{jtot}, "
                  f"instrumentals kept {iok}/{itot}")


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
