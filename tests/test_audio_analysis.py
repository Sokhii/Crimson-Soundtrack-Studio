import io

import numpy as np
import pytest

from soundtrack_studio.library import features as F
from soundtrack_studio.library.flac_meta import FlacFormatError, read_flac_info
from soundtrack_studio.library.formats import AudioReadError, FlacHandler
from soundtrack_studio.testing.fixtures import write_test_flac

SR = 44100


def blocks(x, block=SR * 20):
    x = x if x.ndim == 2 else x[:, None]
    overlap = F.FRAME - F.HOP
    start = 0
    while start < len(x):
        yield x[start:start + block].astype(np.float32)
        if start + block >= len(x):
            break
        start += block - overlap


def clicks(bpm, seconds, freq=1000.0):
    x = np.zeros(int(SR * seconds))
    t = np.arange(int(0.03 * SR)) / SR
    burst = np.sin(2 * np.pi * freq * t) * np.exp(-t * 80)
    period = 60.0 / bpm
    for k in range(int(seconds / period)):
        s = int(k * period * SR)
        x[s:s + burst.size] += burst * 0.8
    return x


@pytest.mark.parametrize("bpm", [90, 120, 128, 150])
def test_tempo_of_click_tracks(bpm):
    f = F.analyze_blocks(blocks(clicks(bpm, 60)), SR)
    assert f.tempo_bpm == pytest.approx(bpm, abs=1.5)
    assert f.tempo_confidence > 0.5


def test_stationary_and_beating_sounds_get_no_tempo():
    t = np.arange(SR * 40) / SR
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t)
    signals = {
        "sine": 0.5 * np.sin(2 * np.pi * 440 * t),
        # two partials beat against each other: a smooth periodic fluctuation, not a rhythm
        "two-note pad": 0.3 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t)) * am,
        "three-note pad": 0.3 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t) + np.sin(2 * np.pi * 330 * t)),
    }
    for name, signal in signals.items():
        f = F.analyze_blocks(blocks(signal), SR)
        assert f.tempo_bpm is None and f.onset_rate == 0.0, name
        assert any("tempo left empty" in n for n in f.notes), name


def test_noise_has_no_tempo():
    f = F.analyze_blocks(blocks(np.random.default_rng(0).normal(0, 0.1, SR * 40)), SR)
    assert f.tempo_bpm is None and f.tempo_confidence < F.TEMPO_MIN_CONFIDENCE


def test_tempo_of_noise_bursts_and_plucked_notes():
    rng = np.random.default_rng(2)
    n = int(0.02 * SR)
    bursts = np.zeros(SR * 40)
    for k in range(int(40 * 110 / 60)):
        s = int(k * 60 / 110 * SR)
        bursts[s:s + n] += rng.normal(0, 0.5, n) * np.exp(-np.arange(n) / SR * 150)
    assert F.analyze_blocks(blocks(bursts), SR).tempo_bpm == pytest.approx(110, abs=1.5)
    plucks = np.zeros(SR * 40)
    tt = np.arange(int(0.5 * SR)) / SR
    for k in range(int(40 * 100 / 60)):
        f0 = [220, 247, 262, 294, 330][k % 5]
        tone = sum(np.sin(2 * np.pi * f0 * h * tt) / h for h in range(1, 8)) * np.exp(-tt * 6) * 0.2
        s = int(k * 60 / 100 * SR)
        plucks[s:s + tone.size] += tone[:plucks.size - s]
    assert F.analyze_blocks(blocks(plucks), SR).tempo_bpm == pytest.approx(100, abs=1.5)


def test_music_like_signal():
    rng = np.random.default_rng(1)
    t = np.arange(SR * 60) / SR
    pad = 0.15 * np.sin(2 * np.pi * 220 * t)
    song = pad + clicks(100, 60, 80) * 0.9 + clicks(200, 60, 6000) * 0.2 + rng.normal(0, 0.01, t.size)
    f = F.analyze_blocks(blocks(song), SR)
    assert f.tempo_bpm == pytest.approx(100, abs=1.5)


def test_levels_and_spectrum_of_a_sine():
    t = np.arange(SR * 10) / SR
    f = F.analyze_blocks(blocks(0.5 * np.sin(2 * np.pi * 1000 * t)), SR)
    assert f.peak_dbfs == pytest.approx(-6.02, abs=0.05)
    assert f.rms_dbfs == pytest.approx(-9.03, abs=0.05)
    assert f.crest_db == pytest.approx(3.01, abs=0.1)
    assert f.spectral_centroid_hz == pytest.approx(1000, rel=0.05)
    assert f.spectral_flatness < 0.01
    assert f.band_energy["mid"] > 0.95
    assert f.decoded_duration_s == pytest.approx(10.0, abs=0.001)
    assert f.stereo_width is None


def test_stereo_width():
    t = np.arange(SR * 5) / SR
    s = np.sin(2 * np.pi * 300 * t)
    same = F.analyze_blocks(blocks(np.stack([s, s], 1)), SR)
    opposite = F.analyze_blocks(blocks(np.stack([s, -s], 1)), SR)
    assert same.stereo_width == pytest.approx(0.0, abs=1e-6)
    assert opposite.stereo_width == pytest.approx(1.0, abs=1e-6)


def test_silence():
    f = F.analyze_blocks(blocks(np.zeros(SR * 5)), SR)
    assert f.rms_dbfs is None and f.silence_ratio == 1.0 and "the file is silent" in f.notes


def test_block_size_does_not_change_results():
    x = clicks(120, 45)
    a = F.analyze_blocks(blocks(x, SR * 20), SR)
    b = F.analyze_blocks(blocks(x, SR * 7 // F.HOP * F.HOP), SR)
    assert a.decoded_duration_s == b.decoded_duration_s
    assert a.rms_dbfs == b.rms_dbfs and a.peak_dbfs == b.peak_dbfs
    assert a.tempo_bpm == pytest.approx(b.tempo_bpm, abs=0.2)


def test_no_samples():
    f = F.analyze_blocks(iter([]), SR)
    assert f.decoded_duration_s == 0 and f.notes


# ------------------------------------------------------------------ FLAC
def test_flac_probe_reads_streaminfo_and_tags(tmp_path):
    path = write_test_flac(tmp_path / "a.flac", seconds=2, sample_rate=48000, channels=2, subtype="PCM_24",
                           tags={"TITLE": "Theme", "ARTIST": "A", "ALBUMARTIST": "Various", "COMPOSER": "C",
                                 "GENRE": "Orchestral", "DATE": "1999-01-01", "TRACKNUMBER": "03/12",
                                 "DISCNUMBER": "1/2", "ARTIST ": "B"})
    probe = FlacHandler().probe(path)
    assert (probe.sample_rate, probe.channels, probe.bit_depth, probe.total_samples) == (48000, 2, 24, 96000)
    assert probe.duration_s == pytest.approx(2.0)
    assert probe.tags["title"] == "Theme" and probe.tags["album_artist"] == "Various"
    assert probe.tags["artist"] == "A; B"  # multiple values are kept
    assert probe.tags["year"] == 1999 and probe.tags["track_number"] == 3 and probe.tags["disc_number"] == 1
    assert probe.identity.startswith("flac-md5:")


def test_identity_ignores_tag_edits(tmp_path):
    from soundtrack_studio.testing.fixtures import set_flac_tags

    path = write_test_flac(tmp_path / "b.flac", seconds=1, tags={"TITLE": "Old"})
    before = FlacHandler().probe(path).identity
    set_flac_tags(path, {"TITLE": "New", "ARTIST": "Someone"})
    after = FlacHandler().probe(path)
    assert after.identity == before and after.tags["title"] == "New"


def test_missing_metadata(tmp_path):
    path = write_test_flac(tmp_path / "c.flac", seconds=1)
    probe = FlacHandler().probe(path)
    assert all(v is None for v in probe.tags.values())


def test_flac_with_id3v2_prefix(tmp_path):
    path = write_test_flac(tmp_path / "d.flac", seconds=1, tags={"TITLE": "X"})
    payload = b"\x00" * 20
    id3 = b"ID3\x04\x00\x00" + bytes([0, 0, 0, len(payload)]) + payload
    data = id3 + path.read_bytes()
    info = read_flac_info(io.BytesIO(data), len(data))
    assert info.tag("TITLE") == "X"


@pytest.mark.parametrize("data,message", [
    (b"RIFF" + b"\x00" * 60, "missing 'fLaC'"),
    (b"fLaC" + bytes([0x84, 0, 0, 4]) + b"\x00" * 4, "not STREAMINFO"),
    (b"fLaC" + bytes([0x80, 0, 0, 34]) + b"\x00" * 10, "past the end"),
    (b"fLaC" + bytes([0x80, 0, 0, 20]) + b"\x00" * 20, "wrong size"),
    (b"fLaC\x00", "truncated"),
])
def test_invalid_flac_headers(data, message):
    with pytest.raises(FlacFormatError, match=message):
        read_flac_info(io.BytesIO(data), len(data))


def test_zero_md5_falls_back_to_content_hash(tmp_path):
    path = write_test_flac(tmp_path / "e.flac", seconds=1)
    data = bytearray(path.read_bytes())
    data[8 + 18:8 + 34] = b"\x00" * 16  # STREAMINFO MD5 field
    path.write_bytes(bytes(data))
    probe = FlacHandler().probe(path)
    assert probe.identity.startswith("sha256:") and probe.warnings


def test_undecodable_audio_is_a_readable_error(tmp_path):
    path = write_test_flac(tmp_path / "f.flac", seconds=2)
    data = path.read_bytes()
    info = read_flac_info(io.BytesIO(data), len(data))
    path.write_bytes(data[:info.metadata_end] + b"\xff" * 5000)
    handler = FlacHandler()
    handler.probe(path)  # metadata is fine
    with pytest.raises(AudioReadError):
        for _ in handler.blocks(path, 4096, 1024):
            pass
