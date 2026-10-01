"""Wwise Vorbis builds: header reading, prefetch data, bank patching, the WwiseConsole wrapper and a full build.

Wwise cannot run in CI; ``testing.fake_wwise`` stands in for WwiseConsole.exe with the real header layout.
"""

import json
import struct

import numpy as np
import pytest

from soundtrack_studio.compiler import bnk, wem
from soundtrack_studio.compiler.archive import CompileError
from soundtrack_studio.compiler.build import BuildSettings
from soundtrack_studio.compiler.wwise import WwiseEncoder, WwiseError, find_console, install_name
from soundtrack_studio.testing.fake_wwise import install_fake_wwise
from soundtrack_studio.testing.fixtures import ANALYZER_FAKE_INSTALL_DB, extract_analyzer_fake_install, write_test_flac
from conftest import fingerprint


def vorbis_wem(samples=96000, channels=2, seek=64, setup=217, audio=4000, avg=16000) -> bytes:
    data = bytes(seek) + bytes(range(256))[:setup % 256].ljust(setup, b"s") + bytes((i * 7) % 251 for i in range(audio))
    config = channels | (1 << 8) | (3 << 12)
    fmt = struct.pack("<HHIIHHHHI", 0xFFFF, channels, 48000, avg, 0, 0, 0x30, 0, config)
    fmt += struct.pack("<IIIIII", samples, 0xD9, len(data) - seek, 0, seek, seek + setup) + bytes(0x42 - 0x30)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


# ----------------------------------------------------------------- headers
def test_vorbis_header_and_prefetch_prefix():
    data = vorbis_wem(samples=278710)
    info = wem.read_wem_info(data)
    assert info.is_vorbis and info.frames == 278710 and info.channels == 2 and info.fmt_size == 0x42
    setup, audio = wem.vorbis_offsets(data)
    assert (setup - info.data_offset, audio - info.data_offset) == (64, 64 + 217)
    pre = wem.prefetch_prefix(data, seconds=0.1)
    assert data.startswith(pre) and len(pre) == audio + 1600      # header + seek table + setup + 0.1 s
    assert wem.prefetch_prefix(data, seconds=60) == data           # never more than the file
    # the prefix alone is readable (as it sits in a bank)
    assert wem.read_wem_info(pre, allow_truncated=True).frames == 278710
    with pytest.raises(CompileError):
        wem.read_wem_info(pre)


def test_write_wav_roundtrip(tmp_path):
    x = np.stack([np.linspace(-0.5, 0.5, 1000), np.zeros(1000)], axis=1).astype(np.float32)
    wem.write_wav(tmp_path / "a.wav", x, 48000)
    import soundfile as sf

    y, rate = sf.read(str(tmp_path / "a.wav"), dtype="float32")
    assert rate == 48000 and y.shape == (1000, 2) and np.allclose(y, x, atol=1e-4)


# -------------------------------------------------------------- bank patch
def _bank(sources, media):
    """A minimal v150 bank: one MusicTrack per source (plugin, stream, id, in-memory size) and DIDX/DATA media."""

    hirc = b""
    for i, (plugin, stream, sid, inmem) in enumerate(sources):
        body = struct.pack("<I", 100 + i) + bytes(3) + struct.pack("<IBIIB", plugin, stream, sid, inmem, 0) + bytes(6)
        hirc += bytes([bnk.HIRC_MUSIC_TRACK]) + struct.pack("<I", len(body)) + body
    didx, data = b"", b""
    for mid, blob in sorted(media.items()):
        data += bytes((-len(data)) % 16)
        didx += struct.pack("<III", mid, len(data), len(blob))
        data += blob
    return bnk.write_chunks([bnk.Chunk(b"BKHD", struct.pack("<I", 150) + bytes(12)), bnk.Chunk(b"DIDX", didx),
                             bnk.Chunk(b"DATA", data), bnk.Chunk(b"HIRC", struct.pack("<I", len(sources)) + hirc)])


def test_vorbis_patch_keeps_codec_and_storage_and_swaps_media():
    V = bnk.PLUGIN_VORBIS
    old_pre, old_inbank, other = b"P" * 1500, b"I" * 900, b"O" * 700
    bank = _bank([(V, 1, 111, 1500), (V, 0, 222, 900), (V, 2, 333, 0), (V, 1, 444, 700)],
                 {111: old_pre, 222: old_inbank, 444: other})
    new_file = vorbis_wem(audio=9000)
    new_pre = wem.prefetch_prefix(new_file)
    new_inbank = vorbis_wem(samples=4800, audio=600)
    patched, notes = bnk.patch_bank(bank, [
        bnk.SourcePatch(100, 111, V, 1, 1500, codec="vorbis", prefetch_data=new_pre),
        bnk.SourcePatch(101, 222, V, 0, None, embedded_data=new_inbank, codec="vorbis"),
        bnk.SourcePatch(102, 333, V, 2, 0, codec="vorbis")])
    src = bnk.describe_sources(patched)
    assert (src[111][0].plugin_id, src[111][0].stream_type, src[111][0].in_memory_size) == (V, 1, len(new_pre))
    assert (src[222][0].plugin_id, src[222][0].stream_type, src[222][0].in_memory_size) == (V, 0, len(new_inbank))
    assert (src[333][0].plugin_id, src[333][0].stream_type, src[333][0].in_memory_size) == (V, 2, 0)
    chunks = bnk.parse_chunks(patched)
    assert bnk.media_data(chunks, 111) == new_pre and bnk.media_data(chunks, 222) == new_inbank
    assert bnk.media_data(chunks, 444) == other                      # untouched sources keep their data
    assert [m for m, _o, _s in bnk.didx_entries(chunks)] == [111, 222, 444]
    assert any("prefetch" in n for n in notes)
    with pytest.raises(CompileError, match="prefetch data"):
        bnk.patch_bank(bank, [bnk.SourcePatch(100, 111, V, 1, 1500, codec="vorbis")])
    pcm_bank = _bank([(bnk.PLUGIN_PCM, 2, 555, 0)], {})
    with pytest.raises(CompileError, match="not Vorbis"):
        bnk.patch_bank(pcm_bank, [bnk.SourcePatch(100, 555, None, None, None, codec="vorbis")])


def test_rebuild_media_slots_in_a_new_media_id():
    bank = _bank([(bnk.PLUGIN_VORBIS, 1, 50, 0)], {10: b"a" * 20, 90: b"b" * 20})
    chunks = bnk.rebuild_media(bnk.parse_chunks(bank), {50: b"c" * 33})
    assert [(m, s) for m, _o, s in bnk.didx_entries(chunks)] == [(10, 20), (50, 33), (90, 20)]
    assert bnk.media_data(chunks, 50) == b"c" * 33 and bnk.media_data(chunks, 90) == b"b" * 20


# ----------------------------------------------------------- WwiseConsole
def test_find_console_and_version(tmp_path, monkeypatch):
    monkeypatch.delenv("CSS_WWISE_CONSOLE", raising=False)
    root = tmp_path / "Audiokinetic" / "Wwise2023.1.4.8496"
    exe = root / "Authoring" / "x64" / "Release" / "bin" / "WwiseConsole.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("WWISEROOT", str(root))
    assert find_console() == exe and install_name(exe) == "Wwise2023.1.4.8496"
    chosen = tmp_path / "elsewhere" / "WwiseConsole.exe"
    chosen.parent.mkdir()
    chosen.write_bytes(b"MZ")
    assert find_console(str(chosen)) == chosen


def test_encoder_converts_and_checks(paths, tmp_path, monkeypatch):
    console = install_fake_wwise(tmp_path / "wwise")
    enc = WwiseEncoder(paths, console)
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    for name, frames in (("1", 48000), ("2", 9600)):
        wem.write_wav(wav_dir / f"{name}.wav", np.zeros((frames, 2), np.float32), 48000)
    out = enc.convert([wav_dir / "1.wav", wav_dir / "2.wav"], tmp_path / "work")
    assert set(out) == {"1", "2"} and wem.read_wem_info(out["1"]).frames == 48000
    assert enc.project.is_file() and paths.is_inside(enc.project)
    assert (paths.logs / "wwise_console.log").is_file()
    monkeypatch.setenv("FAKE_WWISE_PCM", "1")
    with pytest.raises(WwiseError, match="not to Vorbis"):
        enc.convert([wav_dir / "1.wav"], tmp_path / "work")
    monkeypatch.delenv("FAKE_WWISE_PCM")
    monkeypatch.setenv("FAKE_WWISE_FAIL", "1")
    with pytest.raises(WwiseError, match="reported an error"):
        enc.convert([wav_dir / "1.wav"], tmp_path / "work")


def test_studio_status_and_test_button(studio, tmp_path, monkeypatch):
    monkeypatch.delenv("CSS_WWISE_CONSOLE", raising=False)
    monkeypatch.delenv("WWISEROOT", raising=False)
    console = install_fake_wwise(tmp_path / "wwise")
    studio.set_wwise_console(str(console))
    status = studio.wwise_status()
    assert status["found"] and status["console"] == str(console) and status["recommended"] == "2023.1"
    result = studio.test_wwise()
    assert result["ok"] and result["samples"] == 96000
    assert json.loads((studio.paths.logs / "wwise_check.json").read_text())["ok"]
    assert not list(studio.paths.temp.glob("wwise-test-*"))
    with pytest.raises(WwiseError):
        studio.set_wwise_console(str(tmp_path / "missing.exe"))


# -------------------------------------------------------------- full build
@pytest.fixture
def vorbis_built(studio, tmp_path, monkeypatch):
    game = extract_analyzer_fake_install(tmp_path / "Crimson Desert")
    studio.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    studio.set_game_path(game)
    music = tmp_path / "Music"
    write_test_flac(music / "Long Theme.flac", seconds=200, bpm=120, tone_hz=330, tags={"TITLE": "Long Theme"})
    write_test_flac(music / "Short Motif.flac", seconds=60, bpm=None, tone_hz=110, tags={"TITLE": "Short Motif"})
    studio.set_library_path(music)
    studio.scan_library()
    studio.find_matches()
    monkeypatch.setenv("CSS_WWISE_CONSOLE", str(install_fake_wwise(tmp_path / "wwise")))
    return studio, game, music


def test_vorbis_build_end_to_end(vorbis_built):
    studio, game, music = vorbis_built
    tracks = {t["title"]: t["id"] for t in studio.library_tracks()}
    store = studio.match_store()
    store.choose("2001", tracks["Long Theme"])
    store.choose("2004", tracks["Short Motif"])            # the in-bank transition segment
    before = fingerprint(game), fingerprint(music)
    assert studio.build_settings().encoder == "wwise_vorbis"    # the app's default format
    result = studio.build_mod(BuildSettings(mod_name="Vorbis Mod", encoder="wwise_vorbis"))
    assert result.validation.ok, result.validation.errors
    out = result.output_dir / "files" / "0004" / "sound"
    info = wem.read_wem_info((out / "433831842.wem").read_bytes())
    assert info.is_vorbis and info.frames == 180 * 48000 and info.channels == 2
    bank = (out / "bgm.bnk").read_bytes()
    src = bnk.describe_sources(bank)
    assert all(r.plugin_id == bnk.PLUGIN_VORBIS for refs in src.values() for r in refs)   # codec unchanged
    assert src[433831842][0].stream_type == 2 and src[558103][0].stream_type == 0
    embedded = bnk.media_data(bnk.parse_chunks(bank), 558103)
    assert wem.read_wem_info(embedded).is_vorbis and src[558103][0].in_memory_size == len(embedded)
    report = json.loads((result.output_dir / "css_build_report.json").read_text())
    assert report["codec"] == "vorbis" and report["sources"]["433831842"]["codec"] == "vorbis"
    assert (fingerprint(game), fingerprint(music)) == before
    assert not list(studio.paths.temp.glob("build-*"))          # no WAVs or Wwise output left behind


def test_vorbis_build_needs_wwise(vorbis_built, monkeypatch):
    studio, _game, _music = vorbis_built
    monkeypatch.delenv("CSS_WWISE_CONSOLE")
    monkeypatch.delenv("WWISEROOT", raising=False)
    monkeypatch.setattr("soundtrack_studio.compiler.wwise._candidates", lambda: iter(()))
    tracks = {t["title"]: t["id"] for t in studio.library_tracks()}
    studio.match_store().choose("2001", tracks["Long Theme"])
    with pytest.raises(CompileError, match="Wwise is needed"):
        studio.build_mod(BuildSettings(mod_name="X", encoder="wwise_vorbis"))
    # PCM still builds without Wwise
    assert studio.build_mod(BuildSettings(mod_name="Y", encoder="pcm")).validation.ok


def test_long_path_prefix_is_removed_for_wwise():
    from soundtrack_studio.compiler.wwise import _plain

    assert _plain("\\\\?\\C:\\Studio\\temp\\build-1\\wav") == "C:\\Studio\\temp\\build-1\\wav"
    assert _plain("\\\\?\\UNC\\server\\share\\x") == "\\\\server\\share\\x"
    assert _plain("C:\\plain") == "C:\\plain"


def test_find_console_in_launcher_default_and_other_drives(tmp_path, monkeypatch):
    from soundtrack_studio.compiler import wwise

    monkeypatch.delenv("CSS_WWISE_CONSOLE", raising=False)
    monkeypatch.delenv("WWISEROOT", raising=False)
    # the Launcher's default install folder is <drive>:\Audiokinetic\Wwise<version> (no space), older ones used a space
    for name in ("Wwise 2022.1.0.1", "Wwise2023.1.4.8496"):
        exe = tmp_path / "D" / "Audiokinetic" / name / "Authoring" / "x64" / "Release" / "bin" / "WwiseConsole.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"MZ")
    monkeypatch.setattr(wwise, "search_roots", lambda: [tmp_path / "C" / "Audiokinetic", tmp_path / "D" / "Audiokinetic"])
    found = wwise.find_console()
    assert found is not None and "Wwise2023.1.4.8496" in str(found)          # the game's version is preferred
    assert wwise.install_name(found) == "Wwise2023.1.4.8496"
    win32 = tmp_path / "E" / "Audiokinetic" / "Wwise2023.1.1.1" / "Authoring" / "Win32" / "Release" / "bin" / "WwiseConsole.exe"
    win32.parent.mkdir(parents=True)
    win32.write_bytes(b"MZ")
    monkeypatch.setattr(wwise, "search_roots", lambda: [tmp_path / "E" / "Audiokinetic"])
    assert wwise.find_console() == win32
    assert any("Audiokinetic" in p for p in wwise.searched_places())


def test_warnings_exit_code_is_not_an_error_but_missing_output_is(paths, tmp_path, monkeypatch):
    """WwiseConsole exits with 2 when it finished with warnings; the log in the bug report was exit 2 with no output."""

    enc = WwiseEncoder(paths, install_fake_wwise(tmp_path / "wwise"))
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wem.write_wav(wav_dir / "1.wav", np.zeros((4800, 2), np.float32), 48000)
    monkeypatch.setenv("FAKE_WWISE_WARN", "1")
    assert set(enc.convert([wav_dir / "1.wav"], tmp_path / "work")) == {"1"}          # warnings, but a result
    monkeypatch.delenv("FAKE_WWISE_WARN")
    monkeypatch.setenv("FAKE_WWISE_MISSING", "1")
    with pytest.raises(WwiseError, match="did not produce") as err:
        enc.convert([wav_dir / "1.wav"], tmp_path / "work")
    assert "Can't open source or output file" in err.value.details              # the reason is shown, not hidden


def test_too_deep_folder_gets_a_clear_message(paths, tmp_path, monkeypatch):
    from soundtrack_studio.compiler import wwise

    enc = WwiseEncoder(paths, install_fake_wwise(tmp_path / "wwise"))
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wem.write_wav(wav_dir / "1.wav", np.zeros((4800, 2), np.float32), 48000)
    monkeypatch.setattr(wwise, "MAX_SAFE_PATH", 40)
    with pytest.raises(WwiseError, match="too deep") as err:
        enc.convert([wav_dir / "1.wav"], tmp_path / "work")
    assert "shorter location" in err.value.hint


@pytest.mark.skipif(__import__("os").name != "nt", reason="8.3 short names are a Windows feature")
def test_short_path_shortens_deep_folders_on_windows(tmp_path):
    from soundtrack_studio.compiler.wwise import short_path

    deep = tmp_path / ("a long folder name number one" * 1) / ("a long folder name number two") / "data"
    deep.mkdir(parents=True)
    short = short_path(deep / "not yet created" / "file.wproj")
    assert short.endswith("not yet created\\file.wproj") and len(short) <= len(str(deep / "not yet created" / "file.wproj"))
    assert short_path(deep) and __import__("pathlib").Path(short_path(deep)).is_dir()


# ----------------------------------------- the conversion project is set to Vorbis by the Studio
PCM_WORK_UNIT = """<?xml version="1.0" encoding="utf-8"?>
<WwiseDocument Type="WorkUnit" ID="{289DFBFC-2CF4-4F01-972E-9BA48FE725AE}" SchemaVersion="119">
\t<Conversions>
\t\t<WorkUnit Name="Default Work Unit" ID="{289DFBFC-2CF4-4F01-972E-9BA48FE725AE}" PersistMode="Standalone">
\t\t\t<ChildrenList>
\t\t\t\t<Conversion Name="Default Conversion Settings" ID="{6D1B890C-9826-4384-BF07-C15223E9FB56}">
\t\t\t\t\t<PropertyList>
\t\t\t\t\t\t<Property Name="SampleRate" Type="int32">
\t\t\t\t\t\t\t<ValueList>
\t\t\t\t\t\t\t\t<Value Platform="Windows">0</Value>
\t\t\t\t\t\t\t</ValueList>
\t\t\t\t\t\t</Property>
\t\t\t\t\t</PropertyList>
\t\t\t\t\t<ConversionPluginInfoList>
\t\t\t\t\t\t<ConversionPluginInfo Platform="Windows">
\t\t\t\t\t\t\t<ConversionPlugin Name="" ID="{11111111-2222-3333-4444-555555555555}" PluginName="PCM" CompanyID="0" PluginID="1"/>
\t\t\t\t\t\t</ConversionPluginInfo>
\t\t\t\t\t</ConversionPluginInfoList>
\t\t\t\t</Conversion>
\t\t\t</ChildrenList>
\t\t</WorkUnit>
\t</Conversions>
</WwiseDocument>
"""


def _conversions(xml: str):
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml.split("?>", 1)[1])
    return {c.get("Name"): c.find("./ConversionPluginInfoList/ConversionPluginInfo[@Platform='Windows']/ConversionPlugin")
            for c in root.iter("Conversion")}


def test_work_unit_is_patched_to_vorbis_with_our_conversion():
    from soundtrack_studio.compiler.wwise import CONVERSION_NAME, vorbis_conversion_work_unit

    out = vorbis_conversion_work_unit(PCM_WORK_UNIT)
    conversions = _conversions(out)
    assert set(conversions) == {"Default Conversion Settings", CONVERSION_NAME}
    for plugin in conversions.values():
        assert (plugin.get("PluginName"), plugin.get("CompanyID"), plugin.get("PluginID")) == ("Vorbis", "0", "4")
    assert 'ID="{6D1B890C-9826-4384-BF07-C15223E9FB56}"' in out       # the project's own ids stay valid
    assert out.count('ID="{11111111-2222-3333-4444-555555555555}"') == 1  # ours gets a different plug-in id
    assert vorbis_conversion_work_unit(out) == out                      # idempotent


def test_unexpected_work_unit_falls_back_to_what_wwise_wrote():
    from soundtrack_studio.compiler.wwise import CONVERSION_NAME, vorbis_conversion_work_unit

    for broken in ("", "<WwiseDocument/>", "garbage"):
        out = vorbis_conversion_work_unit(broken)
        conversions = _conversions(out)
        assert set(conversions) == {"Default Conversion Settings", CONVERSION_NAME}
        assert all(p.get("PluginName") == "Vorbis" for p in conversions.values())


def test_encoder_sets_up_the_project_and_requests_our_conversion(paths, tmp_path):
    from soundtrack_studio.compiler.wwise import CONVERSION_NAME

    enc = WwiseEncoder(paths, install_fake_wwise(tmp_path / "wwise"))
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wem.write_wav(wav_dir / "1.wav", np.zeros((4800, 2), np.float32), 48000)
    (enc.project.parent / "Conversion Settings").mkdir(parents=True)
    (enc.project.parent / "Conversion Settings" / "Default Work Unit.wwu").write_text(PCM_WORK_UNIT, encoding="utf-8")
    enc.project.write_text("<WwiseDocument/>", encoding="utf-8")            # a project created earlier, still PCM
    enc.convert([wav_dir / "1.wav"], tmp_path / "work")
    unit = (enc.project.parent / "Conversion Settings" / "Default Work Unit.wwu").read_text(encoding="utf-8")
    assert all(p.get("PluginName") == "Vorbis" for p in _conversions(unit).values())
    assert enc.conversion == CONVERSION_NAME
