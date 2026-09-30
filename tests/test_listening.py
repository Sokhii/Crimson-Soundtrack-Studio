"""Optional listening model: front end, excerpt reading, caching and integration (with a stand-in model)."""

from __future__ import annotations

import numpy as np
import pytest

from soundtrack_studio.listening.catalog import CLAP_MUSIC_SPEECH, listening_status
from soundtrack_studio.listening.features import ClapFrontEnd, FrontEndConfig, mel_filter_bank
from soundtrack_studio.listening.listen import (PromptBank, VOCAL_PROMPTS, INSTRUMENTAL_PROMPTS, all_prompts,
                                                read_excerpts, sounds_alike)
from soundtrack_studio.testing.fake_listening import install_fake_listening_model
from soundtrack_studio.testing.fixtures import ANALYZER_FAKE_INSTALL_DB, extract_analyzer_fake_install, write_test_flac

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def test_mel_filters_and_frontend_shapes():
    filters = mel_filter_bank(513, 64, 50, 14000, 48000)
    assert filters.shape == (513, 64) and (filters.sum(axis=0) > 0).all()
    fe = ClapFrontEnd(FrontEndConfig())
    ten = np.random.default_rng(0).standard_normal(480000).astype(np.float32) * 0.1
    assert fe.prepare(ten).shape == (1, 1001, 64)
    short = ten[:48000 * 3]
    mel = fe.prepare(short)
    assert mel.shape == (1, 1001, 64)
    # repeatpad: 3 copies of the 3 s excerpt, then 1 s of silence at the end
    assert mel[0, -40:].max() < -90 and mel[0, :20].max() > -60
    with pytest.raises(ValueError):
        fe.prepare(np.zeros(480001, np.float32))
    assert fe.batch([ten, short]).shape == (2, 1, 1001, 64)


def test_read_excerpts_seeks_and_skips_silence(tmp_path):
    long = write_test_flac(tmp_path / "long.flac", seconds=75, sample_rate=44100)
    ex = read_excerpts(long)
    assert len(ex.audio) == 6 and all(len(a) == 480000 for a in ex.audio)
    assert 0 < ex.starts_s[0] < ex.starts_s[-1] < 75 - 10
    short = write_test_flac(tmp_path / "short.flac", seconds=4, sample_rate=48000)
    ex = read_excerpts(short)
    assert len(ex.audio) == 1 and len(ex.audio[0]) == 4 * 48000
    import soundfile as sf

    quiet = tmp_path / "quiet.flac"
    data = np.zeros((48000 * 60, 2), np.float32)
    data[48000 * 20:48000 * 40] = 0.2 * np.sin(np.arange(48000 * 20) / 48000 * 2 * np.pi * 330)[:, None]
    sf.write(str(quiet), data, 48000)
    ex = read_excerpts(quiet)
    assert 1 <= len(ex.audio) < 6            # silent excerpts are skipped


def test_prompt_bank_summary_format():
    prompts = all_prompts()
    rng = np.random.default_rng(1)
    vectors = {p: v / np.linalg.norm(v) for p, v in zip(prompts, rng.standard_normal((len(prompts), 8)).astype(np.float32))}
    bank = PromptBank(vectors)
    target = np.mean([vectors[p] for p in INSTRUMENTAL_PROMPTS], axis=0) - np.mean([vectors[p] for p in VOCAL_PROMPTS], axis=0)
    target += vectors["music featuring strings"] * 2
    summary = bank.summary({"embedding": (target / np.linalg.norm(target)).tolist()})
    assert summary["vocals"] == "instrumental" and summary["vocals_margin"] < 0
    assert "strings" in summary.get("instrumentation", {})
    assert bank.summary({"embedding": []}) is None
    assert PromptBank({}).summary({"embedding": [1.0] * 8}) is None
    assert sounds_alike({"embedding": [1, 0]}, {"embedding": [0, 1]}) == pytest.approx(0.0)
    assert sounds_alike({"embedding": [1, 0]}, None) is None


@pytest.fixture
def listening_studio(studio, tmp_path):
    install_fake_listening_model(studio)
    music = tmp_path / "Music"
    write_test_flac(music / "Calm.flac", seconds=40, bpm=None, tone_hz=220, tags={"TITLE": "Calm", "GENRE": "Vocal"})
    write_test_flac(music / "Drive.flac", seconds=30, bpm=128, tone_hz=440, tags={"TITLE": "Drive"})
    studio.set_library_path(music)
    studio.scan_library()
    return studio


def test_status_selection_and_test_button(listening_studio):
    s = listening_studio
    entry = s.listening_models()[0]
    assert entry["installed"] and not entry["active"] and entry["model"].id == CLAP_MUSIC_SPEECH.id
    assert s.active_listening_model() is None and s.listening_key() == ""
    s.select_listening_model(CLAP_MUSIC_SPEECH.id)
    assert s.active_listening_model() is CLAP_MUSIC_SPEECH and s.listening_key().startswith(CLAP_MUSIC_SPEECH.id)
    result = s.test_listening_model(CLAP_MUSIC_SPEECH.id)
    assert result["ok"] and result["provider"] and result["dimension"] == 16
    assert (s.paths.logs / "listening_check.json").is_file()


def test_listening_is_optional(listening_studio):
    s = listening_studio
    assert s.listen_to_library()["tracks"] == 0          # no listening model selected: nothing happens
    docs = dict(s.semantic_items("track"))
    assert all("heard" not in d for d in docs.values())


def test_listen_to_library_caches_and_feeds_descriptions(listening_studio):
    s = listening_studio
    s.select_listening_model(CLAP_MUSIC_SPEECH.id)
    stats = s.listen_to_library()
    assert stats == {"tracks": 2, "listened": 2, "cached": 0, "errors": 0}
    assert s.listen_to_library()["cached"] == 2            # second run: from the cache
    heard = s.track_listening()
    assert len(heard) == 2 and all(len(r["embedding"]) == 16 for r in heard.values())
    summary = s.heard_summary(next(iter(heard.values())))
    assert summary["source"].startswith("listening model") and summary["vocals"] in ("sung vocals", "instrumental", "unclear")
    docs = dict(s.semantic_items("track"))
    assert all("heard" in d for d in docs.values())
    results = s.analyze_semantics(use_ai=False)
    assert results["track"].total == 2
    # the cache survives restarting the model and is shared by projects
    s.release_listening_model()
    assert s.heard_summary(next(iter(heard.values()))) == summary


def test_game_audio_is_listened_to(listening_studio, tmp_path, monkeypatch):
    from test_game_audio import _fake_decoder

    s = listening_studio
    s.select_listening_model(CLAP_MUSIC_SPEECH.id)
    game = extract_analyzer_fake_install(tmp_path / "Crimson Desert")
    s.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    s.set_game_path(game)
    monkeypatch.setenv("CSS_VGMSTREAM", str(_fake_decoder(tmp_path)))
    summary = s.analyze_game_audio()
    assert summary["ok"] == 4 and summary["listened"] == 4
    results = s.game_audio_results()
    model = s.game_model()
    measured, heard = s.cue_audio(model.cues[0], results)
    assert measured and heard and len(heard["embedding"]) == 16
    docs = dict(s.semantic_items("cue", include_short_cues=True))
    assert "heard" in docs[str(model.cues[0].segment_id)]
    # tracks and cues are comparable
    s.listen_to_library()
    track = next(iter(s.track_listening().values()))
    assert -1.0 <= sounds_alike(track, heard) <= 1.0


def test_delete_listening_model(listening_studio):
    s = listening_studio
    s.select_listening_model(CLAP_MUSIC_SPEECH.id)
    s.delete_listening_model(CLAP_MUSIC_SPEECH.id)
    assert s.settings.listening_model_id == "" and not CLAP_MUSIC_SPEECH.install_dir(s.paths).exists()
    assert not listening_status(s.paths, CLAP_MUSIC_SPEECH, {})["installed"]
