"""Thematic matching of the user's tracks to the game's music cues.

Pipeline (deterministic first, AI last):

    game cue ──► deterministic eligibility (not a transition/short segment unless asked)
             ──► candidate filtering (readable, not a duplicate, described, duration usable)
             ──► semantic similarity (profiles) + sounds alike (listening model, when used)
                 + duration fit + tempo agreement
             ──► optional local-AI judgement of the shortlist only
             ──► diversity-aware assignment (limits how often one track is reused)
             ──► proposals with confidence, reasons and warnings

The matcher never decides for the user: it writes *proposals*. User decisions
(accept / choose / reject / keep original) live in a separate table and are
applied on top; a rerun respects rejections and never changes decisions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from ..ai.runtime import BackendError, InferenceBackend
from ..errors import OperationCancelled
from ..game_model.model import GameMusicModel, MusicCue
from ..semantic import llm
from ..semantic.profile import CATEGORIES, SemanticProfile, similarity
from ..semantic.store import EffectiveProfile

log = logging.getLogger(__name__)

SHORT_CUE_MS = 15000
MIN_DURATION_RATIO = 0.2       # tracks shorter than 20 % of the cue would loop 5+ times: excluded
SHORTLIST = 5


@dataclass
class MatchSettings:
    include_short_cues: bool = False
    allow_reuse: bool = True
    max_uses_per_track: int = 0          # 0 = automatic (enough for every cue to get a proposal)
    reuse_penalty: float = 0.06          # score subtracted per earlier use of the same track
    use_ai: bool = False
    use_sound: bool = True               # compare the audio itself when the listening model heard both sides
    mode: str = "standout"               # "standout" = compare standout scores (new) | "legacy" = tag overlap
    alternatives: int = SHORTLIST - 1

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TrackInfo:
    id: int
    label: str
    duration_s: float
    channels: int
    tempo_bpm: Optional[float]
    profile: SemanticProfile
    embedding: Any = None                # listening-model audio embedding (normalised numpy vector) or None
    standout: Any = None                 # standout vector (``Calibration.standout_vector``) or None


STANDOUT_FULL = 0.6     # cosine of two standout vectors that counts as a complete match


@dataclass
class Candidate:
    track_id: int
    score: float
    semantic: float
    duration_fit: float
    tempo_fit: Optional[float]
    components: Dict[str, float]
    reasons: List[str]
    warnings: List[str]
    confidence: float
    ai_fit: Optional[float] = None
    ai_reason: str = ""


@dataclass
class CueResult:
    cue_key: str
    candidates: List[Candidate] = field(default_factory=list)   # [0] = proposal
    skipped_reason: str = ""


def is_short(cue: MusicCue) -> bool:
    return cue.is_transition or (cue.duration_ms is not None and cue.duration_ms < SHORT_CUE_MS)


def duration_fit(track_s: float, cue_s: float) -> Tuple[float, str]:
    """1.0 = ideal. Longer tracks are trimmed (mild penalty when much longer), shorter ones loop."""

    if not cue_s or not track_s:
        return 0.5, "unknown length"
    ratio = track_s / cue_s
    if ratio >= 1.0:
        score = 1.0 - min(0.3, 0.1 * math.log2(ratio))
        how = "fits" if ratio < 1.15 else f"trimmed (uses {1 / ratio:.0%} of the track, faded out)"
    else:
        score = max(0.0, (ratio - MIN_DURATION_RATIO) / (1 - MIN_DURATION_RATIO)) ** 0.7
        how = f"loops about {1 / ratio:.1f}× to fill the cue"
    return round(score, 3), how


def tempo_fit(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if not a or not b:
        return None
    ratio = max(a, b) / min(a, b)
    ratio = min(ratio, abs(ratio - 2) + 1, abs(ratio - 0.5) + 1)  # half/double tempo is compatible
    return round(max(0.0, 1.0 - (ratio - 1.0) * 2.5), 3)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def explain(cue_p: SemanticProfile, track_p: SemanticProfile, parts: Dict[str, float]) -> List[str]:
    reasons = []
    for category in CATEGORIES:
        score = parts.get(category)
        shared = [t for t in getattr(cue_p, category) if t in getattr(track_p, category)]
        if shared and score is not None and score >= 0.4:
            reasons.append(f"Shared {category}: {', '.join(shared)}")
        elif score is not None and score >= 0.5:
            reasons.append(f"Related {category}: {', '.join(getattr(cue_p, category)[:3])} ↔ "
                           f"{', '.join(getattr(track_p, category)[:3])}")
    for scale in ("darkness", "energy", "tension"):
        score = parts.get(scale)
        if score is not None:
            if score >= 0.85:
                reasons.append(f"Similar {scale} ({getattr(cue_p, scale):.2f} vs {getattr(track_p, scale):.2f})")
            elif score < 0.5:
                reasons.append(f"Different {scale} ({getattr(cue_p, scale):.2f} vs {getattr(track_p, scale):.2f})")
    if parts.get("vocal_presence") == 0.0:
        reasons.append("Vocals differ (one has vocals, the other is instrumental)")
    return reasons


def explain_standouts(calibration: Any, cue_vec: Any, track_vec: Any, components: Dict[str, float]) -> List[str]:
    """Reasons for a standout-score match: the words that stand out for both, and strong cue words the track lacks."""

    import numpy as np

    _sim, shared = calibration.standout_similarity(cue_vec, track_vec)
    reasons = [f"Standout match {components.get('standout_cosine', 0):.2f}: " + ", ".join(
        f"{word} ({a} vs {b})" for _c, word, a, b in shared[:5])] if shared else [
        f"Little in common that stands out (standout match {components.get('standout_cosine', 0):.2f})"]
    from ..listening.calibration import phi_score, STANDOUT_Z_FLOOR

    missing = [i for i in np.argsort(-cue_vec)[:6] if cue_vec[i] > 1.0 and track_vec[i] <= 0.0]
    if missing:
        scores = phi_score(cue_vec + STANDOUT_Z_FLOOR)
        reasons.append("Stands out in the game music but not in the track: " + ", ".join(
            f"{calibration.words[i][1]} ({int(round(scores[i]))})" for i in missing[:3]))
    return reasons


def confidence(score: float, cue_p: SemanticProfile, track_p: SemanticProfile, margin: float,
               coverage: float = 1.0) -> float:
    """How much to trust a proposal: match quality × evidence quality × comparability × clear winner."""

    evidence = 0.4 + 0.6 * min(cue_p.confidence, track_p.confidence)
    comparable = 0.5 + 0.5 * coverage
    clarity = 0.7 + 0.3 * min(1.0, margin / 0.1)
    return round(max(0.0, min(1.0, score * evidence * comparable * clarity * 1.25)), 3)


def confidence_label(value: float) -> str:
    return "high" if value >= 0.6 else "medium" if value >= 0.35 else "low"


def score_candidate(cue: MusicCue, cue_p: SemanticProfile, track: TrackInfo, explain_now: bool = True,
                    alike: Optional[Tuple[float, float]] = None, standout_cos: Optional[float] = None,
                    standout_ctx: Any = None) -> Optional[Candidate]:
    """``alike`` = (audio cosine similarity, its rank among the library for this cue: 1.0 = most alike);
    ``standout_cos`` = cosine of the cue's and the track's standout vectors (None = legacy tag matching)."""

    cue_s = (cue.duration_ms or 0) / 1000
    if cue_s and track.duration_s < cue_s * MIN_DURATION_RATIO:
        return None
    standout = None if standout_cos is None else max(0.0, min(1.0, standout_cos / STANDOUT_FULL))
    raw_sem, parts = similarity(cue_p, track.profile, standout)
    if standout_cos is not None:
        parts["standout_cosine"] = round(standout_cos, 3)
    coverage = parts.get("coverage", 0.0)
    # agreement on one attribute is weak evidence: shrink towards neutral when little could be compared
    sem = round(0.5 + (raw_sem - 0.5) * (0.35 + 0.65 * coverage), 4) if parts else 0.0
    dfit, _how = duration_fit(track.duration_s, cue_s)
    tfit = tempo_fit(cue.tempo_bpm, track.tempo_bpm)
    tempo_part = 0.05 * tfit if tfit is not None else 0.05 * 0.5
    components = {**parts, "duration": dfit}
    if alike is not None:
        # heard on both sides: the sound itself counts alongside the described character
        score = 0.55 * sem + 0.25 * alike[1] + 0.15 * dfit + tempo_part
        components.update(sounds_alike=round(alike[0], 4), sounds_alike_rank=round(alike[1], 3))
    else:
        score = 0.78 * sem + 0.17 * dfit + tempo_part
    cand = Candidate(track.id, round(score, 4), sem, dfit, tfit, components, [], [], 0.0)
    if explain_now:
        add_explanations(cand, cue, cue_p, track, standout_ctx)
    return cand


def add_explanations(cand: Candidate, cue: MusicCue, cue_p: SemanticProfile, track: TrackInfo,
                     standout_ctx: Any = None) -> None:
    """Reasons and warnings (only built for candidates that will be shown). ``standout_ctx`` =
    (Calibration, cue standout vector) when the candidate was scored by standout scores."""

    if cand.reasons or cand.warnings:
        return
    cue_s = (cue.duration_ms or 0) / 1000
    _dfit, how = duration_fit(track.duration_s, cue_s)
    if standout_ctx is not None and track.standout is not None and "standout" in cand.components:
        reasons = explain_standouts(standout_ctx[0], standout_ctx[1], track.standout, cand.components)
        reasons += [r for r in explain(cue_p, track.profile, {k: v for k, v in cand.components.items()
                                                              if k not in CATEGORIES}) if r.startswith(("Similar", "Different", "Vocals"))]
    else:
        reasons = explain(cue_p, track.profile, cand.components)
        if standout_ctx is None and "standout" not in cand.components:
            reasons.insert(0, "Matched by tags (legacy matching)")
    rank = cand.components.get("sounds_alike_rank")
    if rank is not None:
        if rank >= 0.8:
            reasons.insert(0, f"Sounds similar to the original (among the closest {max(1, round((1 - rank) * 100))}% "
                              f"of your library; audio similarity {cand.components['sounds_alike']:.2f})")
        elif rank <= 0.2:
            reasons.append(f"Sounds quite different from the original (audio similarity "
                           f"{cand.components['sounds_alike']:.2f})")
    reasons.append(f"Length: track {_fmt(track.duration_s)}, cue {_fmt(cue_s)} → {how}")
    if cand.tempo_fit is not None and cand.tempo_fit >= 0.8:
        reasons.append(f"Compatible tempo ({track.tempo_bpm:g} vs {cue.tempo_bpm:g} BPM)")
    warnings = []
    if cue_s and track.duration_s < cue_s:
        warnings.append("The track is shorter than the cue and will loop.")
    if track.channels == 1 and cue.channels and max(cue.channels) >= 2:
        warnings.append("Mono track: it will play equally on both channels.")
    if cue_p.source == "rules" and cue_p.confidence < 0.3:
        warnings.append("The game cue's character was guessed from internal names only; please listen and check.")
    if track.profile.source == "rules":
        warnings.append("Your track was described by rules only (no AI model); the match is approximate.")
    cand.reasons, cand.warnings = reasons, warnings


class Matcher:
    def __init__(self, model: GameMusicModel, cue_profiles: Dict[str, EffectiveProfile], tracks: List[TrackInfo],
                 settings: MatchSettings, *, rejected: Optional[Dict[str, Set[int]]] = None,
                 fixed: Optional[Dict[str, Optional[int]]] = None, backend: Optional[InferenceBackend] = None,
                 cue_docs: Optional[Dict[str, Dict[str, Any]]] = None,
                 cue_embeddings: Optional[Dict[str, Any]] = None,
                 cue_standouts: Optional[Dict[str, Any]] = None, calibration: Any = None) -> None:
        self.model = model
        self.cue_embeddings = cue_embeddings or {}
        self.cue_standouts = (cue_standouts or {}) if settings.mode == "standout" else {}
        self.calibration = calibration
        self.cue_profiles = cue_profiles
        self.tracks = {t.id: t for t in tracks}
        self.settings = settings
        self.rejected = rejected or {}
        self.fixed = fixed or {}           # cue -> track chosen by the user (None = keep original)
        self.backend = backend
        self.cue_docs = cue_docs or {}
        self.ai_errors: List[str] = []

    def eligible_cues(self) -> List[MusicCue]:
        return [c for c in self.model.cues if self.settings.include_short_cues or not is_short(c)]

    def run(self, progress: Optional[Callable[[str, int, int], None]] = None,
            cancel: Optional[Callable[[], bool]] = None) -> Dict[str, CueResult]:
        cues = self.eligible_cues()
        results: Dict[str, CueResult] = {}
        ranked: Dict[str, List[Candidate]] = {}
        for i, cue in enumerate(cues):
            if cancel and cancel():
                raise OperationCancelled()
            if progress and i % 25 == 0:
                progress("Comparing music", i, len(cues))
            key = str(cue.segment_id)
            eff = self.cue_profiles.get(key)
            if eff is None:
                results[key] = CueResult(key, skipped_reason="The game cue has not been described yet.")
                continue
            candidates = []
            alike = self._alike(key)
            standout = self._standout(key)
            for track in self.tracks.values():
                if track.id in self.rejected.get(key, ()):
                    continue
                cand = score_candidate(cue, eff.profile, track, explain_now=False, alike=alike.get(track.id),
                                       standout_cos=standout.get(track.id))
                if cand is not None:
                    candidates.append(cand)
            candidates.sort(key=lambda c: (-c.score, c.track_id))
            ranked[key] = candidates[: max(SHORTLIST, self.settings.alternatives + 1) * 3]
            ctx = (self.calibration, self.cue_standouts[key]) if key in self.cue_standouts and self.calibration else None
            for cand in ranked[key]:
                add_explanations(cand, cue, eff.profile, self.tracks[cand.track_id], ctx)
            results[key] = CueResult(key)
            if not candidates:
                results[key].skipped_reason = "No track in your library is long enough or described."

        if self.backend is not None:
            self._ai_judge(cues, ranked, progress, cancel)
        self._assign(cues, ranked, results)
        if progress:
            progress("Comparing music", len(cues), len(cues))
        return results

    def _standout(self, key: str) -> Dict[int, float]:
        """Cosine of the cue's standout vector with every track's (empty = legacy tag matching for this cue)."""

        import numpy as np

        cue_vec = self.cue_standouts.get(key)
        tracks = [t for t in self.tracks.values() if t.standout is not None]
        if cue_vec is None or not tracks or float(np.linalg.norm(cue_vec)) < 1e-6:
            return {}
        matrix = np.stack([t.standout for t in tracks])
        if matrix.shape[1] != len(cue_vec):
            return {}
        norms = np.maximum(np.linalg.norm(matrix, axis=1), 1e-6)
        cos = (matrix @ cue_vec) / (norms * float(np.linalg.norm(cue_vec)))
        return {t.id: float(cos[i]) for i, t in enumerate(tracks) if norms[i] > 1e-5}

    def _alike(self, key: str) -> Dict[int, Tuple[float, float]]:
        """Audio similarity of every heard track to this cue: {track id: (cosine, rank 0..1)}."""

        import numpy as np

        cue_emb = self.cue_embeddings.get(key) if self.settings.use_sound else None
        heard = [t for t in self.tracks.values() if t.embedding is not None]
        if cue_emb is None or len(heard) < 2:
            return {}
        matrix = np.stack([t.embedding for t in heard])
        if matrix.shape[1] != len(cue_emb):
            return {}
        sims = matrix @ cue_emb
        order = np.argsort(np.argsort(sims))              # 0 = least alike
        n = len(heard)
        return {t.id: (float(sims[i]), float(order[i]) / (n - 1)) for i, t in enumerate(heard)}

    # ------------------------------------------------------------ AI shortlist
    def _ai_judge(self, cues: List[MusicCue], ranked: Dict[str, List[Candidate]], progress, cancel) -> None:
        todo = [c for c in cues if ranked.get(str(c.segment_id))]
        for i, cue in enumerate(todo):
            if cancel and cancel():
                raise OperationCancelled()
            if progress:
                progress("Asking the local AI to judge the best candidates", i, len(todo))
            key = str(cue.segment_id)
            shortlist = ranked[key][:SHORTLIST]
            items = [{"label": self.tracks[c.track_id].label, "profile": self.tracks[c.track_id].profile} for c in shortlist]
            try:
                judged = llm.rerank(self.backend, self.cue_profiles[key].profile, self.cue_docs.get(key, {}), items)
            except BackendError as exc:
                self.ai_errors.append(f"cue {key}: {exc.message}")
                if len(self.ai_errors) >= 5 and len(self.ai_errors) > i // 2:
                    log.warning("Local AI judging stopped after repeated failures")
                    return
                continue
            for idx, verdict in judged.items():
                cand = shortlist[idx]
                cand.ai_fit, cand.ai_reason = verdict["fit"], verdict["reason"]
                cand.score = round(0.55 * cand.score + 0.45 * verdict["fit"], 4)
            ranked[key].sort(key=lambda c: (-c.score, c.track_id))

    # --------------------------------------------------------------- assignment
    def _assign(self, cues: List[MusicCue], ranked: Dict[str, List[Candidate]], results: Dict[str, CueResult]) -> None:
        uses: Dict[int, int] = {}
        for key, track_id in self.fixed.items():
            if track_id is not None:
                uses[track_id] = uses.get(track_id, 0) + 1
        free_cues = [c for c in cues if ranked.get(str(c.segment_id)) and str(c.segment_id) not in self.fixed]
        limit = self.settings.max_uses_per_track
        if limit <= 0:
            limit = max(1, math.ceil(len(free_cues) / max(1, len(self.tracks))) + 1)
        if not self.settings.allow_reuse:
            limit = 1
        # most clear-cut cues pick first, so contested tracks go where they fit best
        order = sorted(free_cues, key=lambda c: -ranked[str(c.segment_id)][0].score)
        for cue in order:
            key = str(cue.segment_id)
            options = []
            for cand in ranked[key]:
                used = uses.get(cand.track_id, 0)
                if used >= limit:
                    continue
                adjusted = cand.score - self.settings.reuse_penalty * used
                options.append((adjusted, cand))
            if not options:
                results[key].skipped_reason = "Every suitable track is already used as often as allowed."
                continue
            options.sort(key=lambda x: (-x[0], x[1].track_id))
            chosen = options[0][1]
            uses[chosen.track_id] = uses.get(chosen.track_id, 0) + 1
            margin = options[0][0] - options[1][0] if len(options) > 1 else 0.2
            if uses[chosen.track_id] > 1:
                chosen.warnings.append(f"This track is also proposed for {uses[chosen.track_id] - 1} other cue(s).")
            cue_p = self.cue_profiles[key].profile
            for cand in [chosen] + [c for _s, c in options[1:]]:
                cand.confidence = confidence(cand.score, cue_p, self.tracks[cand.track_id].profile,
                                             margin if cand is chosen else 0.0, cand.components.get("coverage", 1.0))
            alternatives = [c for _s, c in options[1:1 + self.settings.alternatives]]
            results[key].candidates = [chosen] + alternatives
        for key in self.fixed:
            if key in results and key in ranked:
                results[key].candidates = ranked[key][: 1 + self.settings.alternatives]
                cue_p = self.cue_profiles[key].profile
                for cand in results[key].candidates:
                    cand.confidence = confidence(cand.score, cue_p, self.tracks[cand.track_id].profile, 0.0,
                                                 cand.components.get("coverage", 1.0))


def track_infos(tracks: Iterable[Dict[str, Any]], profiles: Dict[str, EffectiveProfile],
                embeddings: Optional[Dict[int, Any]] = None, standouts: Optional[Dict[int, Any]] = None) -> List[TrackInfo]:
    embeddings = embeddings or {}
    standouts = standouts or {}
    out = []
    for t in tracks:
        if t["status"] != "ok" or t.get("duplicate_of"):
            continue
        eff = profiles.get(str(t["id"]))
        if eff is None or not t.get("duration_s"):
            continue
        label = " – ".join(x for x in (t.get("artist"), t.get("title") or t["rel_path"].rsplit("/", 1)[-1]) if x)
        out.append(TrackInfo(t["id"], label, float(t["duration_s"]), int(t.get("channels") or 2),
                             (t.get("features") or {}).get("tempo_bpm"), eff.profile, embeddings.get(t["id"]),
                             standouts.get(t["id"])))
    return out
