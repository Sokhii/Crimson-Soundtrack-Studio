"""Build planning: accepted mappings -> per-source audio jobs and per-bank patches.

For every replaced segment (see docs/research/modding_format.md, section 3):
* the *primary* music track (largest clip coverage of the segment timeline)
  gets the user's music; each of its clip sources receives the slice of the
  rendered timeline it plays;
* every other track of the segment (simultaneous layers/stems) gets silence;
* every bank that contains a patched source is patched (the Analyzer lists them
  in ``media_source``; the ``bgm`` bank has a twin with the same objects).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..game_model.model import GameMusicModel, MusicNode, TrackClip
from ..matching.store import MappingEntry
from .archive import CompileError

STREAM_CODES = {"embedded": 0, "prefetch_streaming": 1, "streaming": 2}


@dataclass
class BankUse:
    bank_asset_id: int
    bank_vpath: str
    object_id: int
    stream_type: int
    in_memory_size: Optional[int]
    plugin_id: Optional[int]


@dataclass
class SourceJob:
    source_id: int
    cue_key: str
    track_id: Optional[int]          # None = silence (a non-primary layer)
    role: str                        # music | silence
    play_at_s: float
    source_duration_s: float
    channels: int
    uses: List[BankUse] = field(default_factory=list)
    stream_type: int = 2             # 0 = in bank; 1/2 = a streamed .wem file
    wem_vpath: Optional[str] = None  # where the streamed file lives in the game archives


@dataclass
class BuildPlan:
    jobs: Dict[int, SourceJob] = field(default_factory=dict)
    cues: Dict[str, MappingEntry] = field(default_factory=dict)
    cue_durations: Dict[str, float] = field(default_factory=dict)
    cue_channels: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def banks(self) -> Dict[int, str]:
        return {u.bank_asset_id: u.bank_vpath for j in self.jobs.values() for u in j.uses}


def _coverage(node: MusicNode) -> float:
    return sum(c.source_duration_ms or 0 for c in node.clips)


def primary_track(model: GameMusicModel, segment: MusicNode) -> Optional[MusicNode]:
    tracks = [model.nodes[c] for c in segment.children if c in model.nodes and model.nodes[c].kind == "track"]
    if not tracks:
        return None
    return sorted(tracks, key=lambda t: (-_coverage(t), segment.children.index(t.object_id)))[0]


def build_plan(model: GameMusicModel, conn, installation_id: int, mapping: List[MappingEntry]) -> BuildPlan:
    plan = BuildPlan()
    mapped_segments = {int(m.cue_key) for m in mapping}
    for entry in mapping:
        seg_id = int(entry.cue_key)
        segment = model.nodes.get(seg_id)
        if segment is None or segment.kind != "segment":
            plan.warnings.append(f"Cue {entry.cue_key} no longer exists in the game data and was skipped.")
            continue
        if not segment.duration_ms:
            plan.warnings.append(f"{segment.label} has no known duration and was skipped.")
            continue
        primary = primary_track(model, segment)
        if primary is None:
            plan.warnings.append(f"{segment.label} has no music tracks and was skipped.")
            continue
        plan.cues[entry.cue_key] = entry
        plan.cue_durations[entry.cue_key] = segment.duration_ms / 1000
        tracks = [model.nodes[c] for c in segment.children if c in model.nodes and model.nodes[c].kind == "track"]
        media = [model.media[s] for s in primary.source_ids if s in model.media]
        plan.cue_channels[entry.cue_key] = max([m.channels or 2 for m in media] or [2])
        for track in tracks:
            role = "music" if track is primary else "silence"
            clips = list(track.clips)
            if not clips:  # a source without a playlist entry: assume it spans the segment
                clips = [TrackClip(source_id=s, play_at_ms=0.0,
                                   source_duration_ms=(model.media[s].duration_s * 1000
                                                       if s in model.media and model.media[s].duration_s
                                                       else segment.duration_ms))
                         for s in track.source_ids]
            for clip in clips:
                sid = clip.source_id
                info = model.media.get(sid)
                duration = (clip.source_duration_ms or 0) / 1000 or (info.duration_s if info and info.duration_s else 0)
                if not duration:
                    raise CompileError("A music clip has no known length.", details=f"source {sid}")
                if sid in plan.jobs:
                    existing = plan.jobs[sid]
                    if existing.cue_key != entry.cue_key:
                        plan.warnings.append(
                            f"Audio {sid} is shared by {existing.cue_key} and {entry.cue_key}; the replacement for "
                            f"{existing.cue_key} is used for both.")
                    continue
                job = SourceJob(sid, entry.cue_key, entry.track_id if role == "music" else None, role,
                                (clip.play_at_ms or 0) / 1000, duration,
                                (info.channels if info and info.channels else plan.cue_channels[entry.cue_key]))
                _attach_bank_uses(conn, installation_id, job)
                plan.jobs[sid] = job
                other_segments = _segments_using(model, sid) - mapped_segments
                if other_segments:
                    names = ", ".join(model.nodes[s].label for s in sorted(other_segments)[:3])
                    plan.warnings.append(f"Audio {sid} of {segment.label} is also used by {names}, which will change too.")
    return plan


def _segments_using(model: GameMusicModel, source_id: int) -> set:
    out = set()
    for node in model.nodes.values():
        if node.kind == "track" and source_id in node.source_ids:
            out.update(p for p in node.parent_ids if p in model.nodes and model.nodes[p].kind == "segment")
    return out


def _attach_bank_uses(conn, installation_id: int, job: SourceJob) -> None:
    rows = conn.execute(
        "SELECT m.bank_asset_id, m.object_id, m.stream_type, m.in_memory_size, m.plugin_id, a.vpath FROM media_source m"
        " JOIN asset a ON a.id=m.bank_asset_id WHERE a.installation_id=? AND m.source_id=? AND m.owner_type='MusicTrack'"
        " ORDER BY m.bank_asset_id", (installation_id, job.source_id)).fetchall()
    if not rows:
        raise CompileError("The Analyzer database does not say which soundbank holds a music source.",
                           details=f"source {job.source_id}")
    stream_types = set()
    for r in rows:
        stream = STREAM_CODES.get(r[2], None) if isinstance(r[2], str) else r[2]
        if stream is None:
            raise CompileError("A music source has an unknown storage type.", details=f"{job.source_id}: {r[2]}")
        stream_types.add(stream)
        job.uses.append(BankUse(r[0], r[5], r[1], stream, r[3], r[4]))
    if len(stream_types) > 1 and 0 in stream_types:
        raise CompileError("A music source is stored inside one soundbank but streamed in another; this layout is "
                           "not supported.", details=f"source {job.source_id}")
    job.stream_type = 0 if stream_types == {0} else 2
    if job.stream_type != 0:
        row = conn.execute(
            "SELECT a.vpath FROM wem w JOIN asset a ON a.id=w.asset_id WHERE a.installation_id=? AND w.source_id=?"
            " AND a.origin IN ('archive', 'loose') ORDER BY a.origin='archive' DESC LIMIT 1",
            (installation_id, job.source_id)).fetchone()
        if row is None:
            raise CompileError("The Analyzer database does not record where a streamed music file lives.",
                               details=f"source {job.source_id}")
        job.wem_vpath = row[0]


def timeline_slice_frames(job: SourceJob, rate: int) -> Tuple[int, int]:
    return int(round(job.play_at_s * rate)), int(round(job.source_duration_s * rate))
