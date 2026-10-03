"""Builds a :class:`GameMusicModel` from an Analyzer snapshot, with caching.

The interpretation is cached per (snapshot hash, installation, builder
version) under ``data/cache/game_model/``; an unchanged Analyzer database is
never re-interpreted.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from ..analyzer_db import contract
from ..analyzer_db.reader import AnalyzerReader
from ..app_paths import AppPaths
from .model import (KIND_BY_TYPE, BankInfo, GameMusicModel, MediaInfo, MusicCue, MusicNode, TrackClip)

log = logging.getLogger(__name__)

BUILDER_VERSION = 3
# reference kinds from a container to something it plays (same set the Analyzer walks)
CHILD_KINDS = ("child", "playlist_segment", "switch_assoc")
PARSE_RANK = {"parsed": 3, "shallow": 2, "partial": 1}


def cache_path(paths: AppPaths, sha: str, inst_id: int) -> Path:
    return paths.cache / "game_model" / f"{sha[:32]}_i{inst_id}_v{BUILDER_VERSION}.json"


def load_or_build(paths: AppPaths, snapshot: Path, sha: str, inst_id: int,
                  progress: Optional[Callable[[str], None]] = None) -> GameMusicModel:
    file = cache_path(paths, sha, inst_id)
    if file.is_file():
        try:
            model = GameMusicModel.from_dict(json.loads(file.read_text(encoding="utf-8")))
            if model.builder_version == BUILDER_VERSION and model.analyzer_sha256 == sha:
                log.info("Game music model loaded from cache (%d nodes)", len(model.nodes))
                return model
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("Ignoring unreadable game model cache %s: %s", file.name, exc)
    with AnalyzerReader(snapshot) as reader:
        model = build(reader, sha, inst_id, progress)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(".tmp")
    tmp.write_text(json.dumps(model.to_dict(), ensure_ascii=False), encoding="utf-8")
    tmp.replace(file)
    return model


def build(reader: AnalyzerReader, sha: str, inst_id: int,
          progress: Optional[Callable[[str], None]] = None) -> GameMusicModel:
    def step(text: str) -> None:
        log.debug(text)
        if progress:
            progress(text)

    warnings: List[str] = []
    installation = reader.installation(inst_id) or {"id": inst_id}
    scan = reader.latest_completed_scan(inst_id) or {}

    step("Reading soundbanks")
    bank_rows = reader.banks(inst_id)
    bank_of_asset = {b["asset_id"]: b["bank_id"] for b in bank_rows}

    step("Reading music structures")
    nodes: Dict[int, MusicNode] = {}
    for row in reader.music_objects(inst_id):
        oid = row["object_id"]
        bank_id = bank_of_asset.get(row["bank_asset_id"])
        existing = nodes.get(oid)
        if existing is not None:
            if bank_id is not None and bank_id not in existing.banks:
                existing.banks.append(bank_id)
            if PARSE_RANK.get(row["parse_status"], 0) <= PARSE_RANK.get(existing.parse_status, 0):
                continue
        node = _node_from_row(row)
        if existing is not None:
            node.banks = existing.banks
        elif bank_id is not None:
            node.banks = [bank_id]
        nodes[oid] = node

    step("Linking hierarchy")
    for row in reader.refs(inst_id, CHILD_KINDS + ("parent", "transition_segment", "stinger_segment")):
        f, t, kind = row["from_object_id"], row["to_id"], row["kind"]
        if kind == "parent":
            parent, child = t, f
        elif kind in ("transition_segment", "stinger_segment"):
            if f in nodes and t in nodes and t not in nodes[f].transition_segments:
                nodes[f].transition_segments.append(t)
            continue
        else:
            parent, child = f, t
        if parent in nodes and child in nodes:
            _link(nodes, parent, child)
    for node in list(nodes.values()):
        for child in list(node.fields.get("children") or []):
            if child in nodes:
                _link(nodes, node.object_id, child)
        parent_id = node.fields.get("parent_id")
        if parent_id and parent_id in nodes:
            _link(nodes, parent_id, node.object_id)
    roots = sorted(oid for oid, n in nodes.items() if not n.parent_ids)

    step("Reading events")
    event_actions: Dict[int, Set[int]] = defaultdict(set)
    for r in reader.refs(inst_id, ("event_action",)):
        event_actions[r["from_object_id"]].add(r["to_id"])
    action_to_events: Dict[int, Set[int]] = defaultdict(set)
    for ev, actions in event_actions.items():
        for a in actions:
            action_to_events[a].add(ev)
    for r in reader.refs(inst_id, ("action_target",)):
        target = r["to_id"]
        if target in nodes:
            for ev in action_to_events.get(r["from_object_id"], ()):
                if ev not in nodes[target].events:
                    nodes[target].events.append(ev)

    step("Resolving names")
    name_ids: Set[int] = set(nodes) | {b["bank_id"] for b in bank_rows}
    for n in nodes.values():
        name_ids.update(n.events)
        name_ids.update(a["group_id"] for a in n.arguments if a.get("group_id"))
        for leaf in n.fields.get("decision_tree_leaves") or []:
            name_ids.update(v for v in leaf.get("path", []) if v)
    names = reader.best_names(name_ids)
    for n in nodes.values():
        n.name = names.get(n.object_id)
        for a in n.arguments:
            a["name"] = names.get(a.get("group_id"))
        _resolve_switch_states(n, names)

    step("Reading media")
    all_sources = sorted({s for n in nodes.values() for s in n.source_ids})
    media_rows = reader.media(inst_id, all_sources)
    streaming = reader.media_streaming(inst_id, all_sources)
    media_names = reader.best_names(all_sources)
    community = reader.community_notes(all_sources)
    media: Dict[int, MediaInfo] = {}
    missing_media = 0
    for sid in all_sources:
        row = media_rows.get(sid)
        if not row or "codec" not in row:
            missing_media += 1
            media[sid] = MediaInfo(source_id=sid, found=False, streaming=sorted(set(streaming.get(sid, []))),
                                   role=(row or {}).get("role"), name=media_names.get(sid),
                                   community=community.get(sid, []))
            continue
        media[sid] = MediaInfo(
            source_id=sid, codec=row.get("codec"), channels=row.get("channels"), sample_rate=row.get("sample_rate"),
            duration_s=row.get("duration_s"),
            containers=sorted({loc["container"] for loc in row["locations"]}),
            paths=sorted({loc["path"] for loc in row["locations"]}),
            streaming=sorted(set(streaming.get(sid, []))), has_loop_points=bool(row.get("loops")),
            role=row.get("role"), role_confidence=row.get("role_confidence"), name=media_names.get(sid),
            community=community.get(sid, []),
            stream_bytes=max((loc.get("size") or 0 for loc in row["locations"]
                              if loc.get("container") != "embedded"), default=0) or None)
    if missing_media:
        warnings.append(f"{missing_media} media IDs referenced by music tracks have no WEM record in the Analyzer "
                        "database (possibly cut content or files outside the scanned archives).")

    banks: Dict[int, BankInfo] = {}
    music_per_bank: Dict[int, int] = defaultdict(int)
    for n in nodes.values():
        for b in n.banks:
            music_per_bank[b] += 1
    for b in bank_rows:
        banks.setdefault(b["bank_id"], BankInfo(
            bank_id=b["bank_id"], name=names.get(b["bank_id"]), path=b["vpath"], version=b["version"],
            object_count=b["object_count"], media_count=b["media_count"], music_objects=music_per_bank.get(b["bank_id"], 0)))

    step("Building cues")
    event_names = {e: names[e] for n in nodes.values() for e in n.events if e in names}
    cues = _build_cues(nodes, media, event_names)

    classification = reader.classification_counts(inst_id)
    music_media_total = sum(classification.get(r, 0) for r in contract.MUSIC_ROLES)
    outside = _music_outside_hierarchy(reader, inst_id, set(all_sources))
    if outside:
        warnings.append(f"{outside} media items the Analyzer classifies as music are not part of the interactive "
                        "music hierarchy (played by plain Sound objects). They are not shown as cues yet.")
    partial = sum(1 for n in nodes.values() if n.parse_status not in ("parsed", "shallow"))
    if partial:
        warnings.append(f"{partial} music objects were only partially decoded by the Analyzer.")

    kinds = defaultdict(int)
    for n in nodes.values():
        kinds[n.kind] += 1
    stats = {
        "nodes": len(nodes), "roots": len(roots), "cues": len(cues), "media": len(media),
        "switches": kinds["switch"], "playlists": kinds["playlist"], "segments": kinds["segment"],
        "tracks": kinds["track"], "banks": len(banks), "music_banks": sum(1 for b in banks.values() if b.music_objects),
        "analyzer_music_media": music_media_total, "music_outside_hierarchy": outside,
        "open_unknowns": reader.open_unknowns(inst_id),
    }
    log.info("Built game music model: %s", stats)
    return GameMusicModel(
        builder_version=BUILDER_VERSION, analyzer_sha256=sha, schema_version=reader.schema_version(),
        installation=installation, scan=scan, nodes=nodes, roots=roots, media=media, banks=banks, cues=cues,
        event_names=event_names, stats=stats, warnings=warnings)


# ---------------------------------------------------------------- helpers
def _node_from_row(row: dict) -> MusicNode:
    fields = row["fields"]
    kind = KIND_BY_TYPE[row["type_code"]]
    node = MusicNode(object_id=row["object_id"], kind=kind, type_name=row["type_name"],
                     parse_status=row["parse_status"], fields=fields)
    meter = fields.get("meter") or {}
    if meter.get("tempo_bpm") and meter.get("override_parent", True):
        node.tempo_bpm = float(meter["tempo_bpm"])
        node.time_signature = meter.get("time_signature")
    if kind == "segment":
        node.duration_ms = fields.get("duration_ms")
        node.markers = list(fields.get("markers") or [])
    elif kind == "track":
        node.track_type = fields.get("track_type")
        node.subtracks = fields.get("subtracks")
        for src in fields.get("sources") or []:
            sid = src.get("source_id")
            if sid is not None and sid not in node.source_ids:
                node.source_ids.append(sid)
        for clip in fields.get("playlist") or []:
            sid = clip.get("source_id")
            if sid is None:
                continue
            node.clips.append(TrackClip(
                source_id=sid, play_at_ms=clip.get("play_at_ms") or 0.0, begin_trim_ms=clip.get("begin_trim_ms") or 0.0,
                end_trim_ms=clip.get("end_trim_ms") or 0.0, source_duration_ms=clip.get("source_duration_ms"),
                track_index=clip.get("track_index") or 0))
            if sid not in node.source_ids:
                node.source_ids.append(sid)
    elif kind == "playlist":
        node.playlist = list(fields.get("playlist") or [])
    elif kind == "switch":
        node.arguments = [dict(a) for a in fields.get("arguments") or []]
    return node


def _link(nodes: Dict[int, MusicNode], parent: int, child: int) -> None:
    if parent == child:
        return
    if child not in nodes[parent].children:
        nodes[parent].children.append(child)
    if parent not in nodes[child].parent_ids:
        nodes[child].parent_ids.append(parent)


def _resolve_switch_states(node: MusicNode, names: Dict[int, str]) -> None:
    if node.kind != "switch":
        return
    groups = [a.get("name") or str(a.get("group_id")) for a in node.arguments]
    for leaf in node.fields.get("decision_tree_leaves") or []:
        child = leaf.get("audio_node_id")
        path = list(leaf.get("path") or [])
        # the first path element is the tree root (0); the rest are one value per argument
        values = path[1:] if len(path) == len(groups) + 1 else path
        labels = []
        for i, value in enumerate(values):
            group = groups[i] if i < len(groups) else "?"
            if value == 0:
                labels.append(f"{group}=*")
            else:
                labels.append(f"{group}={names.get(value, value)}")
        if child is not None:
            node.child_states.setdefault(str(child), []).extend(labels)


def _ancestor_path(nodes: Dict[int, MusicNode], oid: int) -> List[int]:
    path = [oid]
    seen = {oid}
    current = nodes[oid]
    while current.parent_ids:
        parent = current.parent_ids[0]
        if parent in seen or parent not in nodes:
            break
        seen.add(parent)
        path.append(parent)
        current = nodes[parent]
    return list(reversed(path))


def _build_cues(nodes: Dict[int, MusicNode], media: Dict[int, MediaInfo], event_names: Dict[int, str]) -> List[MusicCue]:
    transitions = {t for n in nodes.values() for t in n.transition_segments}
    cues: List[MusicCue] = []
    for oid, node in nodes.items():
        if node.kind != "segment":
            continue
        path = _ancestor_path(nodes, oid)
        states: List[str] = []
        events: List[int] = []
        tempo, signature = node.tempo_bpm, node.time_signature
        for i, pid in enumerate(path):
            pnode = nodes[pid]
            events.extend(e for e in pnode.events if e not in events)
            if i + 1 < len(path):
                states.extend(pnode.child_states.get(str(path[i + 1]), []))
            if tempo is None and pnode.tempo_bpm:
                tempo, signature = pnode.tempo_bpm, pnode.time_signature
        tracks = [c for c in node.children if c in nodes and nodes[c].kind == "track"]
        sources: List[int] = []
        for t in tracks:
            sources.extend(s for s in nodes[t].source_ids if s not in sources)
        infos = [media[s] for s in sources if s in media]
        banks = sorted({b for b in node.banks} | {b for t in tracks for b in nodes[t].banks})
        cues.append(MusicCue(
            segment_id=oid, label=node.label, path=path, path_labels=[nodes[p].label for p in path],
            duration_ms=node.duration_ms, tempo_bpm=tempo, time_signature=signature, markers=node.markers,
            track_ids=tracks, source_ids=sources, states=states, events=events,
            event_names=[event_names[e] for e in events if e in event_names], banks=banks,
            codecs=sorted({m.codec for m in infos if m.codec}), channels=sorted({m.channels for m in infos if m.channels}),
            streaming=sorted({s for m in infos for s in m.streaming}), is_transition=oid in transitions,
            parent_count=max(1, len(node.parent_ids)),
            silent=bool(infos) and len(infos) == len(sources) and all(m.is_silent for m in infos)))
    cues.sort(key=lambda c: (c.path_labels, c.segment_id))
    return cues


def _music_outside_hierarchy(reader: AnalyzerReader, inst_id: int, hierarchy_sources: Set[int]) -> int:
    roles = ",".join(f"'{r}'" for r in ("music", "likely_music"))
    rows = reader.conn.execute(
        f"SELECT entity_key FROM classification WHERE installation_id=? AND entity_type='media' AND role IN ({roles})",
        (inst_id,)).fetchall()
    return sum(1 for (key,) in rows if key not in hierarchy_sources)
