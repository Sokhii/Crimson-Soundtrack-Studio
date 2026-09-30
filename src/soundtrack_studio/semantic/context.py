"""Evidence documents: the facts a description is based on.

Each track or game cue is turned into a small, deterministic JSON document of
*facts* (tags, measurements, names, structure). The document is the input to
both the rule-based describer and the local AI, and its hash identifies the
input for caching: if the facts do not change, nothing is recomputed.

Game cues cannot be listened to (the Studio does not decode game audio), so
their evidence is the Wwise structure and names recorded by the Analyzer, plus
public community notes where the Analyzer imported them.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List

from ..game_model.model import GameMusicModel, MusicCue

CONTEXT_VERSION = 1


def input_hash(kind: str, doc: Dict[str, Any]) -> str:
    text = json.dumps({"kind": kind, "v": CONTEXT_VERSION, "doc": doc}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _r(value, digits=2):
    return None if value is None else round(float(value), digits)


def track_document(track: Dict[str, Any]) -> Dict[str, Any]:
    """``track`` is a row from ``Studio.library_tracks()``."""

    f = track.get("features") or {}
    raw_tags = {}
    try:
        raw_tags = (json.loads(track.get("tags_json") or "{}").get("raw") or {})
    except (TypeError, ValueError):
        pass
    comment = "; ".join(raw_tags.get("COMMENT", [])[:1])[:200] if isinstance(raw_tags, dict) else ""
    doc = {
        "file_name": (track.get("rel_path") or "").rsplit("/", 1)[-1],
        "folder": "/".join((track.get("rel_path") or "").split("/")[:-1])[-120:],
        "title": track.get("title"), "artist": track.get("artist"), "album": track.get("album"),
        "album_artist": track.get("album_artist"), "composer": track.get("composer"), "genre": track.get("genre"),
        "year": track.get("year"), "comment": comment or None,
        "duration_s": _r(track.get("duration_s"), 1), "channels": track.get("channels"),
        "measurements": {
            "tempo_bpm_estimate": f.get("tempo_bpm"), "tempo_confidence": _r(f.get("tempo_confidence")),
            "energy_index": f.get("energy_index"), "brightness_index": f.get("brightness_index"),
            "average_level_dbfs": f.get("rms_dbfs"), "level_spread_db": f.get("level_spread_db"),
            "onsets_per_second": f.get("onset_rate"), "attack_ratio": f.get("broadband_onset_ratio"),
            "stereo_width": f.get("stereo_width"), "band_energy": f.get("band_energy") or None,
            "spectral_flatness": f.get("spectral_flatness"), "silence_ratio": f.get("silence_ratio"),
        },
    }
    return _prune(doc)


def cue_document(model: GameMusicModel, cue: MusicCue) -> Dict[str, Any]:
    media = [model.media[s] for s in cue.source_ids if s in model.media]
    names: List[str] = []
    for label in cue.path_labels:
        if not re.match(r"^(Switch|Playlist|Segment|Track) \d+$", label):
            names.append(label)
    media_names = [m.name for m in media if m.name]
    community = []
    for m in media:
        for note in m.community[:2]:
            community.append({k: note[k] for k in ("description", "context", "category", "original_name") if note.get(k)})
    bank_names = [model.banks[b].name for b in cue.banks if b in model.banks and model.banks[b].name]
    tracks = [model.nodes[t] for t in cue.track_ids if t in model.nodes]
    doc = {
        "structure_names": names, "media_names": media_names[:6], "bank_names": bank_names[:4],
        "plays_when": cue.states[:6], "started_by_events": cue.event_names[:6],
        "community_notes": community[:4],
        "duration_s": _r(cue.duration_ms / 1000 if cue.duration_ms else None, 1),
        "tempo_bpm": cue.tempo_bpm, "time_signature": cue.time_signature,
        "track_layers": len(tracks), "track_types": sorted({t.track_type or "normal" for t in tracks}),
        "is_transition": cue.is_transition, "markers": len(cue.markers),
        "channels": cue.channels, "reused_by_containers": cue.parent_count,
    }
    return _prune(doc)


def _prune(value):
    if isinstance(value, dict):
        out = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [_prune(v) for v in value if v not in (None, "", [], {})]
    return value


def document_text(doc: Dict[str, Any]) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=1)
