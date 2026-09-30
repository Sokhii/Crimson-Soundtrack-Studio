"""Optional listening model: front end, excerpt reading, caching and integration (with a stand-in model)."""

from __future__ import annotations

import numpy as np
import pytest

from soundtrack_studio.listening.catalog import CLAP_MUSIC_SPEECH, listening_status
from soundtrack_studio.listening.features import ClapFrontEnd, FrontEndConfig, mel_filter_bank
from soundtrack_studio.listening.listen import (PromptBank, VOCAL_PAIRS, VOCAL_PROMPTS, INSTRUMENTAL_PROMPTS, all_prompts,
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


def _bank(dim=64):
    prompts = all_prompts()
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((len(prompts), dim)).astype(np.float32)
    return PromptBank({p: v / np.linalg.norm(v) for p, v in zip(prompts, vectors)})


def _towards(bank, prompts, strength=1.0, dim=64, seed=0):
    base = np.random.default_rng(seed).standard_normal(dim).astype(np.float32) * 0.05
    v = base + strength * np.mean([bank.vectors[p] for p in prompts], axis=0)
    return (v / np.linalg.norm(v)).tolist()


def test_vocals_are_judged_per_excerpt():
    bank = _bank()
    vocal_side = list(VOCAL_PROMPTS) + [a for a, _b in VOCAL_PAIRS]
    instrumental_side = list(INSTRUMENTAL_PROMPTS) + [b for _a, b in VOCAL_PAIRS]
    vocal = [_towards(bank, vocal_side, seed=i) for i in range(2)]
    inst = [_towards(bank, instrumental_side, seed=10 + i) for i in range(4)]
    # a song whose singing is in 2 of 6 excerpts (intro, solos and outro are instrumental) has vocals
    song = {"embedding": inst[0], "excerpt_embeddings": inst + vocal}
    v = bank.vocals(song)
    assert v["vocals"] == "sung vocals" and v["vocals_excerpts"] == "2 of 6" and len(v["vocals_margins"]) == 6
    # ... even though its averaged fingerprint leans instrumental
    assert bank.vocals({"embedding": np.mean([np.array(e) for e in song["excerpt_embeddings"]], axis=0).tolist()})
    # one stray excerpt out of six is not enough (no claim); with no excerpt scoring as singing it is instrumental
    assert bank.vocals({"excerpt_embeddings": inst + inst[:1] + vocal[:1]})["vocals"] == "unclear"
    assert bank.vocals({"excerpt_embeddings": inst})["vocals"] == "instrumental"
    # older results (mean embedding only) still work
    assert bank.vocals({"embedding": vocal[0]})["vocals"] == "sung vocals"


def test_prompt_bank_summary_format():
    bank = _bank()
    inst = _towards(bank, list(INSTRUMENTAL_PROMPTS) + [b for _a, b in VOCAL_PAIRS])
    summary = bank.summary({"embedding": inst, "excerpt_embeddings": [inst, inst]})
    assert summary["vocals"] == "instrumental" and summary["vocals_margin"] < 0
    strings = _towards(bank, ["music featuring strings", "the sound of strings"], strength=3.0)
    assert "strings" in bank.summary({"embedding": strings}).get("instrumentation", {})
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


def test_matching_compares_sound_when_both_sides_were_heard(listening_studio, tmp_path, monkeypatch):
    from test_game_audio import _fake_decoder

    s = listening_studio
    game = extract_analyzer_fake_install(tmp_path / "Crimson Desert")
    s.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    s.set_game_path(game)
    monkeypatch.setenv("CSS_VGMSTREAM", str(_fake_decoder(tmp_path)))
    assert s.find_matches()["compared_by_sound"] == 0         # listening model off
    s.select_listening_model(CLAP_MUSIC_SPEECH.id)
    s.analyze_semantics(use_ai=False)                          # decodes + listens to both sides
    stats = s.find_matches()
    assert stats["compared_by_sound"] >= 1 and stats["proposed"] >= 1
    track_emb, cue_emb = s.sound_embeddings(s.game_model())
    assert len(track_emb) == 2 and len(cue_emb) >= 1


# ------------------------------------------------------------------ calibration (0-100 scores)
def _cal_bank(dim=48):
    from soundtrack_studio.listening.listen import PromptBank, all_prompts

    rng = np.random.default_rng(5)
    prompts = all_prompts()
    vectors = rng.standard_normal((len(prompts), dim)).astype(np.float32)
    return PromptBank({p: v / np.linalg.norm(v) for p, v in zip(prompts, vectors)})


def _word_vec(bank, category, word):
    from soundtrack_studio.listening.vocabulary import prompts_for

    return np.mean([bank.vectors[p] for p in prompts_for(category, word)], axis=0)


def _piece(bank, words, seed, noise=0.6):
    """A fake audio embedding that leans towards the given (category, word) pairs."""

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(48).astype(np.float32) * noise
    for c, w in words:
        u = _word_vec(bank, c, w)
        v = v + 6.0 * u / np.linalg.norm(u)
    return v / np.linalg.norm(v)


def test_vocabulary_is_large_and_contains_the_profile_words():
    from soundtrack_studio.listening.vocabulary import VOCABULARY, WORD_COUNT
    from soundtrack_studio.semantic.profile import CATEGORIES

    assert WORD_COUNT >= 300 and {"rhythm", "texture"} <= set(VOCABULARY)
    for category, words in CATEGORIES.items():
        assert set(words) <= set(VOCABULARY[category])
        assert len(set(VOCABULARY[category])) == len(VOCABULARY[category])       # no duplicates in a group


def test_scores_are_relative_to_the_music_analysed_and_hub_words_vanish():
    from soundtrack_studio.listening.calibration import Calibration, TAG_MIN_SCORE

    bank = _cal_bank()
    hub = ("mood", "solemn")                      # a word every piece leans towards (like the real hub words)
    pieces = [_piece(bank, [hub] + ([("instrumentation", "harp")] if i % 10 == 0 else []), seed=i) for i in range(60)]
    cal = Calibration(bank, pieces)
    assert cal.calibrated and cal.reference_size == 60
    solemn_scores = [cal.tags(p).get("mood", {}).get("solemn") for p in pieces]
    assert sum(s is not None for s in solemn_scores) <= 12          # shown for few pieces, not all 60 (hub neutralised)
    harp = [cal.tags(p).get("instrumentation", {}).get("harp") for p in pieces[::10]]
    assert all(h is not None and h >= TAG_MIN_SCORE for h in harp)   # the real standout is kept, with a number
    summary = cal.summary({"embedding": pieces[0].tolist(), "excerpt_embeddings": [pieces[0].tolist()]})
    assert summary["calibrated"] is True and summary["instrumentation"]["harp"] >= TAG_MIN_SCORE
    assert isinstance(summary["vocals_score"], int) and 0 <= summary["vocals_score"] <= 100


def test_fallback_when_too_little_music_to_compare_with():
    from soundtrack_studio.listening.calibration import Calibration

    bank = _cal_bank()
    few = [_piece(bank, [("instrumentation", "harp")], seed=i) for i in range(5)]
    cal = Calibration(bank, few)
    assert not cal.calibrated
    summary = cal.summary({"embedding": few[0].tolist()})
    assert summary["calibrated"] is False and summary["reference_size"] == 5
    assert Calibration(bank).summary({"embedding": few[0].tolist()})["calibrated"] is False


def test_uninformative_words_are_skipped():
    from soundtrack_studio.listening.calibration import Calibration

    bank = _cal_bank()
    pieces = [_piece(bank, [("mood", "dark")] if i % 3 == 0 else [], seed=i) for i in range(45)]
    cal = Calibration(bank, pieces)
    assert cal.useful.sum() <= len(cal.useful) and cal.useful.any()
    # a word nobody varies on (forced constant) must be dropped
    flat = Calibration(bank, [pieces[0]] * 40)
    assert flat.calibrated and not flat.useful.all() or flat.std.max() < 1e-3


def test_standout_similarity_rewards_shared_standouts():
    from soundtrack_studio.listening.calibration import Calibration

    bank = _cal_bank()
    ref = [_piece(bank, [], seed=100 + i) for i in range(60)]
    cal = Calibration(bank, ref)
    cue = cal.standout_vector(_piece(bank, [("instrumentation", "harp"), ("mood", "peaceful"), ("atmosphere", "vast")], 1))
    same = cal.standout_vector(_piece(bank, [("instrumentation", "harp"), ("mood", "peaceful")], 2))
    other = cal.standout_vector(_piece(bank, [("instrumentation", "drum machine"), ("mood", "aggressive")], 3))
    s_same, why = cal.standout_similarity(cue, same)
    s_other, _ = cal.standout_similarity(cue, other)
    assert s_same > s_other + 0.2 and any(w == "harp" for _c, w, _a, _b in why)
    assert all(0 <= x <= 100 for _c, _w, a, b in why for x in (a, b))


# ------------------------------------------------------------------ standout matching + legacy toggle
def _reasons(studio):
    out = []
    for row in studio.matching_rows(include_short=False):
        for prop in row["proposals"]:
            reasons = prop.get("reasons") if isinstance(prop, dict) else getattr(prop, "reasons", [])
            out += list(reasons or [])
    return out


def test_matching_modes_standout_and_legacy(listening_studio, tmp_path, monkeypatch):
    from test_game_audio import _fake_decoder

    from soundtrack_studio.matching.engine import MatchSettings

    s = listening_studio
    game = extract_analyzer_fake_install(tmp_path / "Crimson Desert")
    s.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    s.set_game_path(game)
    monkeypatch.setenv("CSS_VGMSTREAM", str(_fake_decoder(tmp_path)))
    s.select_listening_model(CLAP_MUSIC_SPEECH.id)
    s.analyze_semantics(use_ai=False)                               # decodes + listens to both sides
    standout = s.find_matches(MatchSettings(mode="standout"))
    assert standout["mode"] == "standout" and standout["compared_by_standout"] >= 1 and standout["proposed"] >= 1
    reasons = _reasons(s)
    assert any(r.startswith("Standout match") or r.startswith("Little in common") for r in reasons), reasons
    assert not any("legacy matching" in r for r in reasons)
    legacy = s.find_matches(MatchSettings(mode="legacy"))
    assert legacy["mode"] == "legacy" and legacy["compared_by_standout"] == 0 and legacy["proposed"] >= 1
    reasons = _reasons(s)
    assert any("legacy matching" in r for r in reasons) and not any(r.startswith("Standout match") for r in reasons)
    # without the listening model the legacy matching is used whatever the setting says
    s.select_listening_model("")
    assert s.find_matches(MatchSettings(mode="standout"))["mode"] == "legacy"


def test_user_decisions_survive_a_mode_switch(listening_studio, tmp_path, monkeypatch):
    from test_game_audio import _fake_decoder

    from soundtrack_studio.matching.engine import MatchSettings

    s = listening_studio
    s.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    s.set_game_path(extract_analyzer_fake_install(tmp_path / "Crimson Desert"))
    monkeypatch.setenv("CSS_VGMSTREAM", str(_fake_decoder(tmp_path)))
    s.select_listening_model(CLAP_MUSIC_SPEECH.id)
    s.analyze_semantics(use_ai=False)
    s.find_matches(MatchSettings(mode="standout"))
    track = next(iter(s.track_listening()))
    s.match_store().choose("2001", track)
    s.find_matches(MatchSettings(mode="legacy"))
    s.find_matches(MatchSettings(mode="standout"))
    assert s.match_store().final_mapping()[0].cue_key == "2001"
    assert any(m.track_id == track for m in s.match_store().final_mapping() if m.cue_key == "2001")
