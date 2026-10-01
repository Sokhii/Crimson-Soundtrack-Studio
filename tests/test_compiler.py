import json
import os
import sqlite3
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import fingerprint

from soundtrack_studio.compiler import audio, bnk, wem
from soundtrack_studio.compiler.archive import (CompileError, GameFileReader, chacha20_key_nonce, decode_entry,
                                                hashlittle)
from soundtrack_studio.compiler.build import BuildSettings
from soundtrack_studio.testing.fixtures import (ANALYZER_FAKE_INSTALL_DB, extract_analyzer_fake_install,
                                                write_test_flac)

VGMSTREAM = os.environ.get("CSS_VGMSTREAM")


# ------------------------------------------------------------------ archive
def test_lookup3_and_chacha_key_test_vector():
    assert hashlittle(b"rendererconfigurationmaterial.xml", 0xC5EDE) == 0xAF3DCEF3
    key, nonce = chacha20_key_nonce("some/dir/RendererConfigurationMaterial.xml")
    assert key.hex() == "90ac5ccf9aa656c59ca050c396aa5ac99ea252c19aa656c596aa5ac992ae5ecd"
    assert nonce == struct.pack("<I", 0xAF3DCEF3) * 4


def test_decode_entry_roundtrip_chacha_and_lz4():
    import lz4.block
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

    payload = b"BKHD" + bytes(range(256)) * 40
    packed = lz4.block.compress(payload, store_size=False)
    key, nonce = chacha20_key_nonce("x.bnk")
    encrypted = Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor().update(packed)
    assert decode_entry(encrypted, 0x32, len(payload), "sound/x.bnk") == payload
    with pytest.raises(CompileError, match="encryption"):
        decode_entry(packed, 0x12, len(payload), "x")
    with pytest.raises(CompileError, match="size"):
        decode_entry(payload, 0, len(payload) + 1, "x")


@pytest.fixture
def fake_game(tmp_path):
    return extract_analyzer_fake_install(tmp_path / "Crimson Desert")


def analyzer_conn():
    conn = sqlite3.connect(f"file:{ANALYZER_FAKE_INSTALL_DB.as_posix()}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def test_reader_verifies_against_analyzer_hash(fake_game):
    conn = analyzer_conn()
    reader = GameFileReader(fake_game, conn)
    data = reader.read(2)  # sound/bgm.bnk, LZ4-compressed in the Analyzer-built archive
    assert data[:4] == b"BKHD"
    paz = fake_game / "0004" / "0.paz"
    blob = bytearray(paz.read_bytes())
    blob[10] ^= 0xFF  # corrupt the stored bank
    paz.write_bytes(bytes(blob))
    with pytest.raises(CompileError):
        reader.read(2)
    conn.close()


# ---------------------------------------------------------------------- bnk
def bank_bytes(fake_game):
    conn = analyzer_conn()
    try:
        return GameFileReader(fake_game, conn).read(2)
    finally:
        conn.close()


def test_find_sources_matches_analyzer(fake_game):
    data = bank_bytes(fake_game)
    sources = bnk.describe_sources(data)
    assert set(sources) == {433831842, 353733717, 480286974, 558103}
    assert sources[558103][0].stream_type == 0 and sources[433831842][0].stream_type == 2
    assert all(r.plugin_id == bnk.PLUGIN_VORBIS for refs in sources.values() for r in refs)


def test_patch_streamed_and_embedded(fake_game):
    data = bank_bytes(fake_game)
    new_media = wem.build_pcm_wem(np.zeros((480, 2), np.float32), 48000)
    patched, _notes = bnk.patch_bank(data, [
        bnk.SourcePatch(1001, 433831842, bnk.PLUGIN_VORBIS, 2, 0),
        bnk.SourcePatch(1004, 558103, bnk.PLUGIN_VORBIS, 0, None, embedded_data=new_media)])
    sources = bnk.describe_sources(patched)
    s = sources[433831842][0]
    assert (s.plugin_id, s.stream_type, s.in_memory_size) == (bnk.PLUGIN_PCM, 2, 0)
    e = sources[558103][0]
    assert (e.plugin_id, e.stream_type, e.in_memory_size) == (bnk.PLUGIN_PCM, 0, len(new_media))
    chunks = bnk.parse_chunks(patched)
    assert bnk.media_data(chunks, 558103) == new_media
    assert sources[353733717][0].plugin_id == bnk.PLUGIN_VORBIS  # untouched sources stay as they were
    # HIRC unchanged apart from the patched fields: same length and object layout
    assert [c.tag for c in chunks] == [c.tag for c in bnk.parse_chunks(data)]
    old_hirc = next(c for c in bnk.parse_chunks(data) if c.tag == b"HIRC").payload
    new_hirc = next(c for c in chunks if c.tag == b"HIRC").payload
    assert len(old_hirc) == len(new_hirc) and sum(a != b for a, b in zip(old_hirc, new_hirc)) <= 12


def test_patch_refuses_mismatching_database(fake_game):
    data = bank_bytes(fake_game)
    with pytest.raises(CompileError, match="codec differs"):
        bnk.patch_bank(data, [bnk.SourcePatch(1001, 433831842, bnk.PLUGIN_ADPCM, 2, 0)])
    with pytest.raises(CompileError, match="expected layout"):
        bnk.patch_bank(data, [bnk.SourcePatch(1001, 999, None, None, None)])
    bad = bytearray(data)
    struct.pack_into("<I", bad, 8, 140)  # BKHD version
    with pytest.raises(CompileError, match="version"):
        bnk.patch_bank(bytes(bad), [])


def test_rebuild_media_keeps_alignment_and_order():
    didx = struct.pack("<III", 1, 0, 5) + struct.pack("<III", 2, 16, 3) + struct.pack("<III", 3, 32, 4)
    data = b"AAAAA" + b"\0" * 11 + b"BBB" + b"\0" * 13 + b"CCCC"
    chunks = [bnk.Chunk(b"BKHD", struct.pack("<I", 150) + b"\0" * 4), bnk.Chunk(b"DIDX", didx), bnk.Chunk(b"DATA", data)]
    out = bnk.rebuild_media(chunks, {2: None, 3: b"XXXXXXXXXXXXXXXXXX"})
    entries = bnk.didx_entries(out)
    assert [e[0] for e in entries] == [1, 3] and all(off % 16 == 0 for _i, off, _s in entries)
    assert bnk.media_data(out, 3) == b"X" * 18 and bnk.media_data(out, 1) == b"AAAAA"
    assert bnk.didx_entries(bnk.rebuild_media(chunks, {1: None, 2: None, 3: None})) == []


def test_damaged_banks_are_rejected():
    with pytest.raises(CompileError):
        bnk.parse_chunks(b"NOPE" + b"\0" * 12)
    with pytest.raises(CompileError):
        bnk.parse_chunks(b"BKHD" + struct.pack("<I", 100) + b"\0" * 4)


# --------------------------------------------------------------- wem/audio
def test_pcm_wem_roundtrip_and_layout():
    x = (np.random.default_rng(0).standard_normal((1000, 2)) * 0.2).astype(np.float32)
    data = wem.build_pcm_wem(x, 48000)
    info = wem.read_wem_info(data)
    assert (info.format_tag, info.channels, info.sample_rate, info.bits_per_sample, info.fmt_size, info.frames) == \
        (0xFFFE, 2, 48000, 16, 0x18, 1000)
    config = struct.unpack_from("<I", data, 12 + 8 + 0x14)[0]
    assert config & 0xFF == 2 and (config >> 8) & 0xF == 1 and config >> 12 == 0x3
    assert np.max(np.abs(wem.read_pcm(data) - x)) < 1e-4
    with pytest.raises(CompileError):
        wem.read_wem_info(data[:-10])


def test_fit_modes():
    x = np.ones((48000 * 10, 2), np.float32) * 0.5
    out, how = audio.fit_to_length(x, 48000 * 4, "auto")
    assert len(out) == 48000 * 4 and "trimmed" in how and out[-1, 0] == 0.0
    out, how = audio.fit_to_length(x, 48000 * 25, "auto")
    assert "looped" in how and np.allclose(out[48000:48000 * 18], 0.5, atol=1e-3)
    out, how = audio.fit_to_length(x, 48000 * 25, "pad")
    assert "once" in how and np.all(out[48000 * 11:] == 0)
    out, how = audio.fit_to_length(x[:0], 480, "auto")
    assert not out.any()


def test_resample_quality():
    t = np.arange(44100 * 2) / 44100
    x = (0.5 * np.sin(2 * np.pi * 1000 * t))[:, None].astype(np.float32)
    y = audio.resample(x, 44100, 48000)
    assert len(y) == 96000
    spectrum = np.abs(np.fft.rfft(y[10000:58000, 0] * np.hanning(48000)))
    assert abs(np.argmax(spectrum) * 48000 / 48000 - 1000) <= 1
    hf = np.sin(2 * np.pi * 30000 * np.arange(96000 * 2) / 96000)[:, None].astype(np.float32)
    assert np.sqrt(np.mean(audio.resample(hf, 96000, 48000)[2000:-2000] ** 2)) < 0.01  # no aliasing


def test_normalize_and_channels():
    x = (np.sin(np.linspace(0, 400, 48000)) * 0.9)[:, None].astype(np.float32)
    y, gain = audio.normalize(x, -18.0, -1.0)
    assert abs(20 * np.log10(np.sqrt(np.mean(y ** 2))) + 18) < 0.5 and gain < 0
    assert audio.map_channels(x, 2).shape == (48000, 2)
    assert audio.map_channels(np.ones((5, 2), np.float32), 1).shape == (5, 1)
    assert audio.map_channels(np.ones((5, 2), np.float32), 4).shape == (5, 4)


# ------------------------------------------------------------------ build
@pytest.fixture
def built(studio, tmp_path, fake_game):
    studio.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    studio.set_game_path(fake_game)
    music = tmp_path / "Music"
    write_test_flac(music / "Long Theme.flac", seconds=200, bpm=120, tone_hz=330, tags={"TITLE": "Long Theme"})
    write_test_flac(music / "Short Motif.flac", seconds=60, bpm=None, tone_hz=110, tags={"TITLE": "Short Motif"})
    studio.set_library_path(music)
    studio.scan_library()
    studio.find_matches()
    return studio, fake_game, music


def test_end_to_end_build(built):
    studio, fake_game, music = built
    tracks = {t["title"]: t["id"] for t in studio.library_tracks()}
    store = studio.match_store()
    store.choose("2001", tracks["Long Theme"])
    store.choose("2003", tracks["Short Motif"])
    store.set_fit("2003", "loop")
    store.choose("2004", tracks["Short Motif"])  # the embedded transition segment
    game_before, music_before = fingerprint(fake_game), fingerprint(music)
    result = studio.build_mod(BuildSettings(mod_name="Test Mod", author="me"))
    assert result.validation.ok and result.validation.checked_banks == 1 and result.validation.checked_wems == 2
    out = result.output_dir
    assert out == studio.paths.output / "Test Mod" and result.zip_path.is_file()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["format"] == "crimson_browser_mod_v1" and manifest["files_dir"] == "files"
    files = sorted(p.relative_to(out / "files").as_posix() for p in (out / "files").rglob("*") if p.is_file())
    assert files == ["0004/sound/433831842.wem", "0004/sound/480286974.wem", "0004/sound/bgm.bnk"]
    report = json.loads((out / "build_report.json").read_text())
    assert report["cues"]["2003"]["fit"].startswith("looped") and report["validation"]["ok"]
    long_wem = wem.read_wem_info((out / "files/0004/sound/433831842.wem").read_bytes())
    assert long_wem.frames == 180 * 48000 and long_wem.channels == 2  # exact segment length
    bank = (out / "files/0004/sound/bgm.bnk").read_bytes()
    srcs = bnk.describe_sources(bank)
    assert srcs[433831842][0].plugin_id == bnk.PLUGIN_PCM and srcs[353733717][0].plugin_id == bnk.PLUGIN_VORBIS
    assert wem.read_wem_info(bnk.media_data(bnk.parse_chunks(bank), 558103)).frames == 4 * 48000
    assert fingerprint(fake_game) == game_before and fingerprint(music) == music_before
    assert not list(studio.paths.temp.glob("build-*"))
    status = {s.number: s.state for s in studio.project_status().steps}
    assert status[8] == status[9] == status[10] == "ok"
    history = studio.builds()
    assert history[0]["status"] == "completed" and history[0]["summary"]["cues"] == 3


def test_package_folder_layout_and_rebuild_keeps_previous(built):
    studio, _game, _music = built
    studio.match_store().accept_all()
    studio.build_mod(BuildSettings(mod_name="Layout", layout="package_folders", make_zip=False))
    out = studio.paths.output / "Layout"
    assert (out / "0004" / "sound" / "bgm.bnk").is_file() and not (out / "manifest.json").exists()
    studio.build_mod(BuildSettings(mod_name="Layout", layout="package_folders", make_zip=False))
    assert (studio.paths.output / "Layout.previous" / "0004").is_dir()


def test_build_refusals(built):
    studio, fake_game, music = built
    from soundtrack_studio.errors import GameInstallError

    with pytest.raises(CompileError, match="No replacements"):
        studio.build_mod()
    tracks = {t["title"]: t["id"] for t in studio.library_tracks()}
    studio.match_store().choose("2001", tracks["Long Theme"])
    with open(fake_game / "0004" / "0.pamt", "ab") as handle:  # simulate a game update
        handle.write(b"x")
    with pytest.raises(GameInstallError, match="does not match"):
        studio.build_mod()
    assert studio.builds() == [] or studio.builds()[0]["status"] != "completed"


def test_missing_music_file_is_reported(built):
    studio, _game, music = built
    tracks = {t["title"]: t["id"] for t in studio.library_tracks()}
    studio.match_store().choose("2001", tracks["Long Theme"])
    (music / "Long Theme.flac").unlink()
    with pytest.raises(CompileError, match="music file is missing"):
        studio.build_mod()
    assert studio.builds()[0]["status"] == "failed"


@pytest.mark.skipif(not VGMSTREAM or not Path(VGMSTREAM or "").is_file(), reason="set CSS_VGMSTREAM to vgmstream-cli")
def test_output_decodes_with_vgmstream(built, tmp_path):
    studio, _game, _music = built
    tracks = {t["title"]: t["id"] for t in studio.library_tracks()}
    studio.match_store().choose("2001", tracks["Long Theme"])
    studio.match_store().choose("2004", tracks["Short Motif"])
    out = studio.build_mod(BuildSettings(mod_name="Vgm", make_zip=False)).output_dir / "files" / "0004" / "sound"
    for target, seconds in ((out / "433831842.wem", 180.0), (out / "bgm.bnk", 4.0)):
        meta = subprocess.run([VGMSTREAM, "-m", str(target)], capture_output=True, text=True).stdout
        assert "Audiokinetic Wwise" in meta and "16-bit Little Endian PCM" in meta, meta
        assert f"stream total samples: {int(seconds * 48000)} " in meta, meta


def test_slice_timeline_handles_clips_that_start_before_or_run_past_the_segment():
    timeline = np.arange(1, 101, dtype=np.float32).reshape(-1, 1).repeat(2, axis=1)
    inside = audio.slice_timeline(timeline, 10, 20)
    assert inside.shape == (20, 2) and inside[0, 0] == 11 and inside[-1, 0] == 30
    # a lead-in before the entry point (negative start): silence first, then the timeline from its beginning
    early = audio.slice_timeline(timeline, -30, 80)
    assert early.shape == (80, 2) and not early[:30].any() and early[30, 0] == 1 and early[-1, 0] == 50
    # the same shape as the failure seen in a real build: a long clip with a 2.25 s lead-in
    long_clip = audio.slice_timeline(timeline, -108, 5000)
    assert long_clip.shape == (5000, 2) and not long_clip[:108].any() and long_clip[108, 0] == 1
    assert long_clip[108 + 99, 0] == 100 and not long_clip[108 + 100:].any()
    # running past the end, starting past the end, and entirely before the start
    assert audio.slice_timeline(timeline, 90, 30)[9, 0] == 100 and not audio.slice_timeline(timeline, 90, 30)[10:].any()
    assert not audio.slice_timeline(timeline, 500, 10).any()
    assert not audio.slice_timeline(timeline, -50, 20).any()
