"""Studio-side model of the game's interactive music, derived from Analyzer data.

The model deliberately keeps Wwise's structure instead of flattening it to
"one WEM = one song":

    MusicSwitchContainer  (chooses by state/switch, e.g. region)
      └─ MusicRandomSequenceContainer  (playlist: order, loops, weights)
           └─ MusicSegment  (timeline: duration, entry/exit markers, tempo)
                └─ MusicTrack  (clips on the timeline; stems/sub-tracks)
                     └─ media (WEM source): codec, channels, rate, streaming

A :class:`MusicCue` is a *view* over one segment with its context (ancestor
chain, state paths, events, tracks and media). It is what the browser lists
and what later phases will describe semantically. The replacement granularity
(segment vs. playlist) is intentionally not fixed yet; see docs/architecture.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

KIND_BY_TYPE = {10: "segment", 11: "track", 12: "switch", 13: "playlist"}
KIND_LABEL = {"segment": "Segment", "track": "Track", "switch": "Switch", "playlist": "Playlist"}


@dataclass
class MediaInfo:
    source_id: int
    codec: Optional[str] = None
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    duration_s: Optional[float] = None
    containers: List[str] = field(default_factory=list)     # embedded | loose | ...
    paths: List[str] = field(default_factory=list)          # virtual paths inside the game archives
    streaming: List[str] = field(default_factory=list)      # streaming | prefetch_streaming | embedded ...
    has_loop_points: bool = False
    role: Optional[str] = None                              # Analyzer classification (music/likely_music/...)
    role_confidence: Optional[str] = None
    name: Optional[str] = None
    community: List[Dict[str, Any]] = field(default_factory=list)   # public research notes (via the Analyzer)
    found: bool = True                                      # False when no WEM record exists for the ID


@dataclass
class TrackClip:
    source_id: int
    play_at_ms: float = 0.0
    begin_trim_ms: float = 0.0
    end_trim_ms: float = 0.0
    source_duration_ms: Optional[float] = None
    track_index: int = 0


@dataclass
class MusicNode:
    object_id: int
    kind: str
    type_name: str
    name: Optional[str] = None
    banks: List[int] = field(default_factory=list)
    parse_status: str = "parsed"
    parent_ids: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    transition_segments: List[int] = field(default_factory=list)
    events: List[int] = field(default_factory=list)          # events whose actions target this node directly
    # segment
    duration_ms: Optional[float] = None
    markers: List[Dict[str, Any]] = field(default_factory=list)
    tempo_bpm: Optional[float] = None
    time_signature: Optional[str] = None
    # track
    track_type: Optional[str] = None
    subtracks: Optional[int] = None
    source_ids: List[int] = field(default_factory=list)
    clips: List[TrackClip] = field(default_factory=list)
    # playlist
    playlist: List[Dict[str, Any]] = field(default_factory=list)
    # switch
    arguments: List[Dict[str, Any]] = field(default_factory=list)   # [{group_id, group_type, name}]
    child_states: Dict[str, List[str]] = field(default_factory=dict)  # child id (str) -> state names on the path
    fields: Dict[str, Any] = field(default_factory=dict)             # raw decoded fields (advanced view)

    @property
    def label(self) -> str:
        return self.name or f"{KIND_LABEL.get(self.kind, self.kind)} {self.object_id}"


@dataclass
class MusicCue:
    segment_id: int
    label: str
    path: List[int]                       # root ... segment (object ids)
    path_labels: List[str]
    duration_ms: Optional[float]
    tempo_bpm: Optional[float]
    time_signature: Optional[str]
    markers: List[Dict[str, Any]]
    track_ids: List[int]
    source_ids: List[int]
    states: List[str]                     # state/switch names along the path, e.g. ["BGM_Region=Desert"]
    events: List[int]
    event_names: List[str]
    banks: List[int]
    codecs: List[str]
    channels: List[int]
    streaming: List[str]
    is_transition: bool = False
    parent_count: int = 1                 # >1 when the segment is reused by several containers


@dataclass
class BankInfo:
    bank_id: int
    name: Optional[str]
    path: str
    version: int
    object_count: int
    media_count: int
    music_objects: int = 0


@dataclass
class GameMusicModel:
    builder_version: int
    analyzer_sha256: str
    schema_version: int
    installation: Dict[str, Any]
    scan: Dict[str, Any]
    nodes: Dict[int, MusicNode]
    roots: List[int]
    media: Dict[int, MediaInfo]
    banks: Dict[int, BankInfo]
    cues: List[MusicCue]
    event_names: Dict[int, str]
    stats: Dict[str, Any]
    warnings: List[str]

    # ------------------------------------------------------ serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {
            "builder_version": self.builder_version,
            "analyzer_sha256": self.analyzer_sha256,
            "schema_version": self.schema_version,
            "installation": self.installation,
            "scan": self.scan,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "roots": self.roots,
            "media": [asdict(m) for m in self.media.values()],
            "banks": [asdict(b) for b in self.banks.values()],
            "cues": [asdict(c) for c in self.cues],
            "event_names": {str(k): v for k, v in self.event_names.items()},
            "stats": self.stats,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameMusicModel":
        nodes = {}
        for n in data["nodes"]:
            clips = [TrackClip(**c) for c in n.pop("clips", [])]
            node = MusicNode(**n)
            node.clips = clips
            nodes[node.object_id] = node
        return cls(
            builder_version=data["builder_version"],
            analyzer_sha256=data["analyzer_sha256"],
            schema_version=data["schema_version"],
            installation=data["installation"],
            scan=data["scan"],
            nodes=nodes,
            roots=list(data["roots"]),
            media={m["source_id"]: MediaInfo(**m) for m in data["media"]},
            banks={b["bank_id"]: BankInfo(**b) for b in data["banks"]},
            cues=[MusicCue(**c) for c in data["cues"]],
            event_names={int(k): v for k, v in data["event_names"].items()},
            stats=data["stats"],
            warnings=list(data["warnings"]),
        )

    # ----------------------------------------------------------- queries
    def children_of(self, object_id: int) -> List[MusicNode]:
        node = self.nodes.get(object_id)
        return [self.nodes[c] for c in node.children if c in self.nodes] if node else []

    def media_for_node(self, object_id: int) -> List[MediaInfo]:
        seen: List[int] = []
        stack = [object_id]
        visited = set()
        while stack:
            current = stack.pop()
            if current in visited or current not in self.nodes:
                continue
            visited.add(current)
            node = self.nodes[current]
            seen.extend(s for s in node.source_ids if s not in seen)
            stack.extend(reversed(node.children))
        return [self.media[s] for s in seen if s in self.media]
