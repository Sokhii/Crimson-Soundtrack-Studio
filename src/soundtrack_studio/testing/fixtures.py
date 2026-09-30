"""Synthetic, legally redistributable test data.

* :func:`make_fake_game` - a tiny folder shaped like a Crimson Desert install
  (package directories with ``0.pamt``/``0.paz``, ``meta/0.papgt``, exe);
  contents are filler bytes, only sizes/paths matter.
* :func:`write_analyzer_db` - a database with the Analyzer's exact schema v1
  (``analyzer_schema_v1.sql``) holding a small interactive-music hierarchy
  modelled on real Analyzer output: switch -> playlists -> segments -> tracks
  -> WEM media, a transition segment, events, names and classifications.
* :func:`write_test_flac` - original generated audio (tones/click tracks)
  with Vorbis comments, for library tests.

Used by the automated tests and by ``--selftest`` in the frozen build.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
from pathlib import Path
from typing import Dict, List, Optional

SCHEMA_V1 = Path(__file__).with_name("analyzer_schema_v1.sql")

BANK_BGM = 412724365        # FNV-1("bgm")
BANK_ENV = 616116187        # FNV-1("env_region_desert")
STATE_GROUP = 3263969508    # BGM_Region
STATE_DESERT = 1850388778
STATE_FOREST = 491961918
EVENT_PLAY = 60970509       # Play_BGM_World
MUSIC_MEDIA = {1001: 433831842, 1002: 353733717, 1003: 480286974, 1004: 558103}
SEGMENT_MS = {2001: 180000.0, 2002: 150000.0, 2003: 200000.0, 2004: 4000.0}


def make_fake_game(root: Path) -> List[Dict]:
    files = {
        "0000/0.pamt": 97, "0000/0.paz": 5, "0004/0.pamt": 452, "0004/0.paz": 40960,
        "meta/0.papgt": 64, "bin64/CrimsonDesert.exe": 7,
    }
    out = []
    for rel, size in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes((i * 31 + len(rel)) & 0xFF for i in range(size)))
        st = path.stat()
        kind = "pamt" if rel.endswith(".pamt") else "paz" if rel.endswith(".paz") else "papgt" if rel.endswith(".papgt") else "loose"
        out.append({"rel_path": rel, "kind": kind, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return out


def write_analyzer_db(path: Path, *, game_root: Optional[Path] = None, game_files: Optional[List[Dict]] = None,
                      schema_version: int = 1, scan_status: str = "completed", with_music: bool = True,
                      wal: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")  # the Analyzer keeps its database in WAL mode
    conn.executescript(SCHEMA_V1.read_text(encoding="utf-8"))
    conn.execute(f"PRAGMA user_version={schema_version}")
    root = str(game_root or "C:/Games/Crimson Desert")
    now = "2026-09-30T12:00:00+00:00"
    conn.execute("INSERT INTO installation(id, root_path, label, first_seen, last_scanned) VALUES (1,?,?,?,?)",
                 (root, "Crimson Desert", now, now))
    conn.execute("INSERT INTO scan(id, installation_id, started_at, finished_at, status, mode, parser_version, stats_json)"
                 " VALUES (1,1,?,?,?,'analyze',1,'{}')", (now, now if scan_status == "completed" else None, scan_status))
    for f in game_files or []:
        conn.execute("INSERT INTO source_file(installation_id, rel_path, kind, size, mtime_ns, last_seen_scan) VALUES (1,?,?,?,?,1)",
                     (f["rel_path"], f["kind"], f["size"], f["mtime_ns"]))

    def asset(aid: int, origin: str, vpath: str, ext: str, size: int, parent: Optional[int] = None) -> None:
        conn.execute("INSERT INTO asset(id, installation_id, origin, locator, vpath, ext, size, parent_asset_id, fingerprint,"
                     " content_hash, hash_kind, parser_version, analyzed_scan_id, last_seen_scan, status)"
                     " VALUES (?,1,?,?,?,?,?,?,'fp',?,'sha1',1,1,1,'ok')",
                     (aid, origin, f"{origin}:{vpath}", vpath, ext, size, parent, f"{aid:040x}"))

    asset(1, "archive", f"sound/windows/{BANK_BGM}.bnk", ".bnk", 20000)
    asset(2, "archive", f"sound/windows/{BANK_ENV}.bnk", ".bnk", 3000)
    conn.execute("INSERT INTO bnk VALUES (1,?,150,0,0,0,?,1,1,'[]','{}','[]')", (BANK_BGM, 16 if with_music else 3))
    conn.execute("INSERT INTO bnk VALUES (2,?,150,0,0,0,3,0,1,'[]','{}','[]')", (BANK_ENV,))

    def obj(bank: int, oid: int, code: int, name: str, fields: dict, status: str = "parsed") -> None:
        conn.execute("INSERT INTO wwise_object(bank_asset_id, object_id, type_code, type_name, offset, size, parse_status, fields_json)"
                     " VALUES (?,?,?,?,0,64,?,?)", (bank, oid, code, name, status, json.dumps(fields)))

    def ref(bank: int, src: int, dst: int, kind: str) -> None:
        conn.execute("INSERT INTO object_ref(bank_asset_id, from_object_id, to_id, kind, confidence) VALUES (?,?,?,?, 'parsed')",
                     (bank, src, dst, kind))

    meter = {"grid_offset_ms": 0.0, "grid_period_ms": 2000.0, "override_parent": True, "tempo_bpm": 120.0, "time_signature": "4/4"}
    # events: one music event, one ambience event
    obj(1, EVENT_PLAY, 4, "Event", {"actions": [5001]})
    obj(1, 5001, 3, "Action", {"action_type": "0x0403", "target_id": 4001})
    ref(1, EVENT_PLAY, 5001, "event_action")
    ref(1, 5001, 4001, "action_target")
    obj(2, 2811450822, 4, "Event", {"actions": [5002]})
    obj(2, 5002, 3, "Action", {"action_type": "0x0403", "target_id": 6002})
    ref(2, 2811450822, 5002, "event_action")
    ref(2, 5002, 6002, "action_target")
    obj(2, 6002, 5, "RandomSequenceContainer", {"children": [6001]})
    obj(2, 6001, 2, "Sound", {"parent_id": 6002, "source_id": 16721128})
    ref(2, 6002, 6001, "child")
    ref(2, 6001, 16721128, "source")

    if with_music:
        obj(1, 4001, 12, "MusicSwitchContainer", {
            "arguments": [{"group_id": STATE_GROUP, "group_type": "state"}], "children": [3001, 3002],
            "decision_tree_leaves": [{"audio_node_id": 3001, "path": [0, STATE_DESERT]},
                                     {"audio_node_id": 3002, "path": [0, STATE_FOREST]}],
            "meter": meter, "parent_id": 0, "transition_rules": 1,
            "transition_rules_detail": [{"dst": [4294967295], "src": [4294967295], "transition_segment": 2004}]})
        for pl, segs in ((3001, [2001, 2002]), (3002, [2003])):
            obj(1, pl, 13, "MusicRandomSequenceContainer", {
                "children": segs, "meter": meter, "parent_id": 4001,
                "playlist": [{"item_id": pl * 10, "segment_id": 0, "type": "continuous_sequence", "loop": 0, "children": len(segs)}]
                + [{"item_id": pl * 10 + i + 1, "segment_id": s, "loop": 1, "children": 0} for i, s in enumerate(segs)]})
            ref(1, 4001, pl, "child")
            ref(1, pl, 4001, "parent")
            for s in segs:
                ref(1, pl, s, "child")
                ref(1, pl, s, "playlist_segment")
        ref(1, 4001, 2004, "transition_segment")
        parents = {2001: 3001, 2002: 3001, 2003: 3002, 2004: 4001}
        for seg, track in ((2001, 1001), (2002, 1002), (2003, 1003), (2004, 1004)):
            ms = SEGMENT_MS[seg]
            obj(1, seg, 10, "MusicSegment", {"children": [track], "duration_ms": ms, "meter": meter, "parent_id": parents[seg],
                                             "markers": [{"id": 0, "name": "Entry", "position_ms": 0.0},
                                                         {"id": 1, "name": "Exit", "position_ms": ms - 2000.0}]})
            ref(1, seg, track, "child")
            ref(1, track, seg, "parent")
            ref(1, seg, parents[seg], "parent")
            sid = MUSIC_MEDIA[track]
            stream = "embedded" if track == 1004 else "prefetch_streaming"
            source = {"source_id": sid, "stream_type": stream, "plugin_id": 262145, "codec_plugin": "0x00040001"}
            clip = {"source_id": sid, "play_at_ms": 0.0, "begin_trim_ms": 0.0, "end_trim_ms": 0.0,
                    "source_duration_ms": ms, "track_index": 0}
            obj(1, track, 11, "MusicTrack", {"parent_id": seg, "sources": [source], "playlist": [clip],
                                             "subtracks": 1, "track_type": "normal"})
            ref(1, track, sid, "track_source")
            aid = 100 + track
            if track == 1004:
                asset(aid, "embedded", f"sound/windows/{BANK_BGM}.bnk#{sid}.wem", ".wem", 4000, parent=1)
                container, bank_asset = "embedded", 1
            else:
                asset(aid, "archive", f"sound/windows/media/{sid}.wem", ".wem", 400000)
                container, bank_asset = "loose", None
            conn.execute("INSERT INTO wem(asset_id, source_id, container, bank_asset_id, valid, codec, format_tag, channels,"
                         " sample_rate, duration_s, duration_method, sample_count, data_size, loops_json, cues_json)"
                         " VALUES (?,?,?,?,1,'Wwise Vorbis',65535,2,48000,?,'vorbis_sample_count',?,1000,'[]','{}')",
                         (aid, sid, container, bank_asset, ms / 1000.0, int(ms * 48)))
            conn.execute("INSERT INTO media_source(bank_asset_id, object_id, owner_type, source_id, stream_type, in_memory_size,"
                         " plugin_id, details_json) VALUES (1,?, 'MusicTrack', ?, ?, 0, 262145, ?)",
                         (track, sid, stream, json.dumps({**source, "clip": clip})))
            conn.execute("INSERT INTO classification(installation_id, entity_type, entity_key, role, score, confidence, evidence_json, scan_id)"
                         " VALUES (1,'media',?,'music',0.9,'high','{}',1)", (sid,))
            conn.execute("INSERT INTO media_context(installation_id, source_id, owner_object_id, owner_type, container_ids_json,"
                         " container_types_json, event_ids_json, bank_ids_json) VALUES (1,?,?,'MusicTrack','[]','[]',?,?)",
                         (sid, track, json.dumps([EVENT_PLAY]), json.dumps([BANK_BGM])))
    asset(200, "archive", "sound/windows/media/16721128.wem", ".wem", 90000)
    conn.execute("INSERT INTO wem(asset_id, source_id, container, valid, codec, channels, sample_rate, duration_s, loops_json, cues_json)"
                 " VALUES (200,16721128,'loose',1,'PCM',1,48000,30.0,'[]','{}')")
    conn.execute("INSERT INTO classification(installation_id, entity_type, entity_key, role, score, confidence, evidence_json, scan_id)"
                 " VALUES (1,'media',16721128,'ambience',0.4,'medium','{}',1)")
    names = [(BANK_BGM, "bgm", "bank"), (BANK_ENV, "env_region_desert", "bank"), (EVENT_PLAY, "Play_BGM_World", "event"),
             (2811450822, "Play_env_region_desert", "event"), (STATE_GROUP, "BGM_Region", "stategroup"),
             (STATE_DESERT, "Desert", "state")]
    for value, name, kind in names:
        conn.execute("INSERT INTO name(id_value, name, kind, source, hash_verified) VALUES (?,?,?,'soundbanksinfo',1)",
                     (value, name, kind))
    conn.commit()
    conn.close()
    return path


# ------------------------------------------------------------------ FLAC
def write_test_flac(path: Path, *, seconds: float = 3.0, sample_rate: int = 44100, channels: int = 2,
                    bpm: Optional[float] = 120.0, tone_hz: float = 220.0, subtype: str = "PCM_16",
                    tags: Optional[Dict[str, str]] = None) -> Path:
    """Write original generated audio (a tone plus an optional click track) as FLAC with Vorbis comments."""

    import numpy as np
    import soundfile as sf

    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    signal = 0.2 * np.sin(2 * np.pi * tone_hz * t)
    if bpm:
        period = int(sample_rate * 60.0 / bpm)
        burst_t = np.arange(int(0.03 * sample_rate)) / sample_rate
        burst = 0.7 * np.sin(2 * np.pi * 1500 * burst_t) * np.exp(-burst_t * 80)
        for start in range(0, n - burst.size, period):
            signal[start:start + burst.size] += burst
    data = np.stack([signal * (1.0 - 0.3 * c) for c in range(channels)], axis=1).astype(np.float32)
    Path(_native(path.parent)).mkdir(parents=True, exist_ok=True)
    sf.write(_native(path), data, sample_rate, format="FLAC", subtype=subtype)
    if tags is not None:
        set_flac_tags(path, tags)
    return path


def _native(path: Path) -> str:
    from ..app_paths import os_path
    return os_path(path)


def set_flac_tags(path: Path, tags: Dict[str, str], vendor: str = "css-test") -> None:
    """Replace the Vorbis comment block of a FLAC file (test helper; never used on user files)."""

    data = Path(_native(path)).read_bytes()
    assert data[:4] == b"fLaC"
    pos, blocks = 4, []
    while True:
        header = data[pos]
        length = int.from_bytes(data[pos + 1:pos + 4], "big")
        blocks.append((header & 0x7F, data[pos + 4:pos + 4 + length]))
        pos += 4 + length
        if header & 0x80:
            break
    audio = data[pos:]
    comments = [f"{k}={v}".encode("utf-8") for k, v in tags.items()]
    vc = struct.pack("<I", len(vendor)) + vendor.encode() + struct.pack("<I", len(comments))
    vc += b"".join(struct.pack("<I", len(c)) + c for c in comments)
    blocks = [b for b in blocks if b[0] not in (1, 4)] + [(4, vc)]
    out = bytearray(b"fLaC")
    for i, (btype, payload) in enumerate(blocks):
        last = 0x80 if i == len(blocks) - 1 else 0
        out += bytes([last | btype]) + len(payload).to_bytes(3, "big") + payload
    out += audio
    tmp = Path(_native(path)).with_name(Path(path).name + ".tmp")
    tmp.write_bytes(bytes(out))
    os.replace(tmp, _native(path))
