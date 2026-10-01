"""Loudness (BS.1770 / LUFS), true peak, and levelling without distortion."""

import json
import math

import numpy as np
import pytest

from soundtrack_studio.compiler import loudness as L

RATE = 48000


def sine(freq, amp, seconds=10, rate=RATE, stereo=True):
    x = (amp * np.sin(2 * np.pi * freq * np.arange(int(rate * seconds)) / rate)).astype(np.float32)
    return np.stack([x, x], axis=1) if stereo else x[:, None]


def test_k_weighting_coefficients_match_the_standard_at_48k():
    shelf, hp = L.k_weighting(48000)
    assert shelf[0] == pytest.approx([1.53512485958697, -2.69169618940638, 1.19839281085285], abs=1e-9)
    assert shelf[1] == pytest.approx([1.0, -1.69065929318241, 0.73248077421585], abs=1e-9)
    assert hp[1] == pytest.approx([1.0, -1.99004745483398, 0.99007225036621], abs=1e-9)


@pytest.mark.parametrize("freq,amp,expected", [
    (1000, 0.1, -20.0),      # 1 kHz stereo sine: loudness = amplitude level (BS.1770 calibration)
    (1000, 1.0, 0.0),
    (60, 0.3, -14.08),       # the high-pass lowers bass (reference implementation: -14.08)
])
def test_integrated_loudness_reference_values(freq, amp, expected):
    assert L.integrated_lufs(sine(freq, amp), RATE) == pytest.approx(expected, abs=0.1)


def test_gating_ignores_silence_and_streaming_equals_one_shot():
    rng = np.random.default_rng(3)
    burst = (rng.standard_normal((RATE * 10, 2)) * 0.1).astype(np.float32)
    padded = np.concatenate([np.zeros((RATE * 5, 2), np.float32), burst, np.zeros((RATE * 5, 2), np.float32)])
    assert L.integrated_lufs(padded, RATE) == pytest.approx(L.integrated_lufs(burst, RATE), abs=0.2)
    meter = L.LoudnessMeter(RATE)
    for i in range(0, len(padded), 31337):
        meter.add(padded[i:i + 31337])
    assert meter.lufs() == pytest.approx(L.integrated_lufs(padded, RATE), abs=0.01)
    assert L.integrated_lufs(np.zeros((RATE * 3, 2), np.float32), RATE) is None


def test_true_peak_sees_peaks_between_samples():
    n = np.arange(RATE)
    x = (0.5 * np.sin(2 * np.pi * (RATE / 4) * n / RATE + np.pi / 4)).astype(np.float32)   # samples at 0.354
    x2 = np.stack([x, x], axis=1)
    assert 20 * math.log10(float(np.abs(x).max())) == pytest.approx(-9.03, abs=0.05)
    assert L.true_peak_dbtp(x2) == pytest.approx(-6.02, abs=0.1)


def test_level_reaches_target_when_peaks_allow():
    x = sine(440, 0.05)
    y, res = L.level(x, RATE, -16.0)
    assert L.integrated_lufs(y, RATE) == pytest.approx(-16.0, abs=0.1)
    assert res.short_by_db == 0 and res.true_peak_after <= -1.0 + 1e-6


def test_level_never_pushes_peaks_over_the_ceiling():
    """A very dynamic track (quiet bed, short loud hit) cannot reach a high target without its peaks distorting:
    it is raised only as far as the ceiling allows, nothing is limited or clipped, and the shortfall is reported."""

    x = sine(220, 0.02, seconds=20)
    x[RATE * 10:RATE * 10 + 2400] = sine(220, 0.6, seconds=0.05)
    y, res = L.level(x, RATE, -10.0, ceiling_dbtp=-1.0)
    assert L.true_peak_dbtp(y) <= -1.0 + 0.01
    assert res.short_by_db > 0 and res.gain_db == pytest.approx(-1.0 - L.true_peak_dbtp(x), abs=0.01)
    ratio = y[x != 0] / x[x != 0]
    assert np.allclose(ratio, ratio[0], rtol=1e-4)                     # one gain for the whole piece: no limiting


def test_own_level_only_lowers_too_hot_tracks():
    quiet = sine(440, 0.1)
    y, res = L.level(quiet, RATE, None)
    assert res.gain_db == 0 and np.array_equal(y, quiet)
    hot = sine(440, 1.0)
    y, res = L.level(hot, RATE, None)
    assert res.gain_db < 0 and L.true_peak_dbtp(y) <= -1.0 + 0.01


# ----------------------------------------------------------------- builds
@pytest.fixture
def loud_built(studio, tmp_path, monkeypatch):
    from test_game_audio import _fake_decoder
    from soundtrack_studio.testing.fixtures import ANALYZER_FAKE_INSTALL_DB, extract_analyzer_fake_install, write_test_flac

    game = extract_analyzer_fake_install(tmp_path / "Crimson Desert")
    studio.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    studio.set_game_path(game)
    monkeypatch.setenv("CSS_VGMSTREAM", str(_fake_decoder(tmp_path)))
    music = tmp_path / "Music"
    write_test_flac(music / "Long Theme.flac", seconds=200, bpm=120, tone_hz=330, tags={"TITLE": "Long Theme"})
    studio.set_library_path(music)
    studio.scan_library()
    studio.analyze_game_audio()
    track = studio.library_tracks()[0]["id"]
    studio.match_store().choose("2001", track)
    return studio


def test_game_audio_measurements_include_loudness(loud_built):
    studio = loud_built
    measured = [r.features.get("loudness_lufs") for r in studio.game_audio_results().values() if r.status == "ok"]
    assert measured and all(m is not None and -40 < m < 0 for m in measured)


def test_build_matches_the_originals_loudness(loud_built):
    from soundtrack_studio.compiler import wem
    from soundtrack_studio.compiler.build import BuildSettings

    studio = loud_built
    refs = studio.loudness_references(studio.game_model(), studio.match_store().final_mapping())
    original = refs["2001"]["lufs"]
    result = studio.build_mod(BuildSettings(mod_name="Match", loudness_mode="match", make_zip=False))
    cue = result.report["cues"]["2001"]
    assert cue["loudness_reference"] == "original (measured loudness)" and cue["original_lufs"] == original
    audio = wem.read_pcm((result.output_dir / "files/0004/sound/433831842.wem").read_bytes())
    reached = L.integrated_lufs(audio, RATE)
    assert reached == pytest.approx(original - cue["short_of_target_db"], abs=0.3)
    assert L.true_peak_dbtp(audio) <= -1.0 + 0.05
    summary = result.report["loudness"]
    assert summary["mode"] == "match" and summary["cues"] == 1
    assert json.loads(result.report_path.read_text())["loudness"]["cues"] == 1


def test_fixed_and_own_level_modes(loud_built):
    from soundtrack_studio.compiler import wem
    from soundtrack_studio.compiler.build import BuildSettings

    studio = loud_built
    fixed = studio.build_mod(BuildSettings(mod_name="Fixed", loudness_mode="fixed", target_lufs=-20.0, make_zip=False))
    cue = fixed.report["cues"]["2001"]
    assert cue["loudness_reference"] == "fixed target" and cue["loudness_target_lufs"] == -20.0
    audio = wem.read_pcm((fixed.output_dir / "files/0004/sound/433831842.wem").read_bytes())
    assert L.integrated_lufs(audio, RATE) == pytest.approx(-20.0 - cue["short_of_target_db"], abs=0.3)
    own = studio.build_mod(BuildSettings(mod_name="Own", loudness_mode="off", make_zip=False))
    assert own.report["cues"]["2001"]["loudness_target_lufs"] is None


def test_match_without_measurements_falls_back_to_the_target(studio, tmp_path):
    from soundtrack_studio.compiler.build import BuildSettings
    from soundtrack_studio.testing.fixtures import ANALYZER_FAKE_INSTALL_DB, extract_analyzer_fake_install, write_test_flac

    game = extract_analyzer_fake_install(tmp_path / "Crimson Desert")
    studio.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    studio.set_game_path(game)
    write_test_flac(tmp_path / "Music" / "a.flac", seconds=200, tags={"TITLE": "A"})
    studio.set_library_path(tmp_path / "Music")
    studio.scan_library()
    studio.match_store().choose("2001", studio.library_tracks()[0]["id"])
    result = studio.build_mod(BuildSettings(mod_name="Fallback", loudness_mode="match", target_lufs=-18.0,
                                            make_zip=False))
    cue = result.report["cues"]["2001"]
    assert cue["loudness_reference"] == "fixed target (the original was not measured)"
    assert cue["loudness_target_lufs"] == -18.0


def test_projects_saved_before_loudness_modes(studio, tmp_path):
    studio.create_project("Old")
    studio.require_project().set("build_settings", {"mod_name": "X", "normalize": False, "target_rms_dbfs": -18.0})
    assert studio.build_settings().loudness_mode == "off"
    studio.require_project().set("build_settings", {"mod_name": "X", "normalize": True})
    assert studio.build_settings().loudness_mode == "match"
