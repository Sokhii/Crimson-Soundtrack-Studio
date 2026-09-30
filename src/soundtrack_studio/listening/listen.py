"""Listening: excerpt selection, audio embeddings, and "what was heard".

``listen_file`` reads up to ``MAX_EXCERPTS`` ten-second excerpts spread over a
file (seeking, never decoding the whole file), skips near-silent ones, and
stores the mean CLAP audio embedding. Everything else is derived from that
embedding at read time:

* **heard summary** - zero-shot scores of the profile vocabulary (instruments,
  mood, style, ...) and of vocals vs instrumental, from prompt embeddings
  computed once per model and cached;
* **sounds alike** - the cosine similarity of two embeddings (matching).

So thresholds or prompts can change without listening to anything again.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..app_paths import os_path
from ..compiler.audio import resample
from ..semantic.profile import CATEGORIES
from .clap import ClapModel, ListeningError

LISTEN_VERSION = 2   # 2: per-excerpt embeddings (vocals are judged per excerpt)
EXCERPT_S = 10.0
MAX_EXCERPTS = 6
SILENT_DBFS = -50.0

# prompt templates per vocabulary category; each term's score is the mean over its templates
TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "mood": ("{} music", "a {} piece of music"),
    "emotion": ("music that expresses {}", "music full of {}"),
    "atmosphere": ("{} music", "music with a {} atmosphere"),
    "instrumentation": ("music featuring {}", "the sound of {}"),
    "style": ("{} music", "a piece of {} music"),
}
SUMMARY_FIELDS = ("instrumentation", "mood", "atmosphere", "style", "emotion")
VOCAL_PROMPTS = ("a song with a singer singing lyrics", "music with sung vocals", "a vocalist singing a melody",
                 "a woman singing", "a man singing", "a pop song with a lead singer")
INSTRUMENTAL_PROMPTS = ("instrumental music without vocals", "an instrumental piece with no singing",
                        "instrumental music")
# Simple contrasting pairs: they catch modern and synthesized voices (Vocaloid, anime and J-pop vocals) that the
# descriptive prompts above miss, but on their own they also fire on piano, jazz and old orchestral recordings.
VOCAL_PAIRS = (("a song with vocals", "a song without vocals"), ("singing", "no singing"),
               ("music with a singer", "music without a singer"))
# Vocals are judged per ten-second excerpt with a combined score
#     c = (mean cosine to VOCAL_PROMPTS - mean cosine to INSTRUMENTAL_PROMPTS)
#         + PAIR_WEIGHT * mean over VOCAL_PAIRS of (cosine to the 'with' prompt - cosine to the 'without' prompt).
# The two parts make different mistakes, so together they separate far better than either alone (fitted on 33
# freely licensed recordings in CI, see docs/research/listening.md; 23/24 sung found, 8/9 instrumentals kept).
# Averaging a whole song first hides singing behind intros, solos and dense arrangements, hence per excerpt.
# A track has sung vocals when enough excerpts score above VOCAL_MARGIN; it is instrumental only when none do;
# anything in between is "unclear" (no claim). Only embeddings are stored, so all of this can be re-tuned
# without listening again.
PAIR_WEIGHT = 3.0
VOCAL_MARGIN = 0.05                # an excerpt with singing
VOCAL_SHARE = 1 / 3              # share of excerpts that must have singing (at least one): 2 of 6
STRONG_Z, MODERATE_Z, MAX_PER_FIELD = 1.6, 1.0, 3
PROMPT_EMBED_VERSION = 4   # 4: vocal pair prompts; 3: fp16 text tower (identical to PyTorch); 2: int8 tower one at a time


def all_prompts() -> List[str]:
    prompts: List[str] = []
    for category, terms in CATEGORIES.items():
        for term in terms:
            prompts += [t.format(term) for t in TEMPLATES[category]]
    pair_prompts = [p for pair in VOCAL_PAIRS for p in pair]
    return list(dict.fromkeys(prompts + list(VOCAL_PROMPTS) + list(INSTRUMENTAL_PROMPTS) + pair_prompts))


@dataclass
class Excerpts:
    audio: List[np.ndarray]
    starts_s: List[float]
    duration_s: float


def read_excerpts(path: Path, max_excerpts: int = MAX_EXCERPTS, excerpt_s: float = EXCERPT_S,
                  start_offset_s: float = 0.0) -> Excerpts:
    """Mono 48 kHz excerpts spread over the file, read by seeking (the whole file is never decoded)."""

    import soundfile as sf

    try:
        with sf.SoundFile(os_path(path), mode="r") as f:
            rate, frames = f.samplerate, f.frames
            duration = frames / rate if rate else 0.0
            length = int(excerpt_s * rate)
            if frames <= length:
                starts = [0]
            else:
                lo, hi = int(frames * 0.05), int(frames * 0.95) - length
                if hi <= lo:
                    lo, hi = 0, frames - length
                n = max(1, min(max_excerpts, int((hi - lo) / length) + 1))
                starts = [int(lo + (hi - lo) * i / max(1, n - 1)) for i in range(n)] if n > 1 else [lo + (hi - lo) // 2]
            chunks, used = [], []
            for s in starts:
                f.seek(s)
                data = f.read(min(length, frames - s), dtype="float32", always_2d=True)
                if len(data) == 0:
                    continue
                mono = data.mean(axis=1, keepdims=True)
                rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-12)
                if 20 * np.log10(rms) < SILENT_DBFS and len(starts) > 1:
                    continue
                chunks.append(resample(mono, rate, 48000)[:, 0])
                used.append(start_offset_s + s / rate)
    except (RuntimeError, OSError, getattr(sf, "SoundFileError", RuntimeError)) as exc:
        raise ListeningError("The audio could not be read for listening.", details=f"{Path(path).name}: {exc}") from exc
    return Excerpts(chunks, used, duration)


def listen_file(model: ClapModel, path: Path, model_key: str) -> Dict[str, Any]:
    ex = read_excerpts(path)
    if not ex.audio:
        return {"version": LISTEN_VERSION, "model": model_key, "silent": True, "excerpts": 0,
                "duration_s": round(ex.duration_s, 2)}
    emb = model.embed_audio(ex.audio)
    mean = emb.mean(axis=0)
    mean = mean / max(float(np.linalg.norm(mean)), 1e-12)
    return {"version": LISTEN_VERSION, "model": model_key, "excerpts": len(ex.audio),
            "excerpt_starts_s": [round(s, 1) for s in ex.starts_s], "duration_s": round(ex.duration_s, 2),
            "embedding": [round(float(v), 5) for v in mean],
            "excerpt_embeddings": [[round(float(v), 4) for v in e] for e in emb]}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def excerpt_embeddings(result: Optional[Dict[str, Any]]) -> List[np.ndarray]:
    return [_unit(e) for e in (result or {}).get("excerpt_embeddings") or []]


def embedding_of(result: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not result or not result.get("embedding"):
        return None
    v = np.asarray(result["embedding"], dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def sounds_alike(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[float]:
    ea, eb = embedding_of(a), embedding_of(b)
    if ea is None or eb is None or ea.shape != eb.shape:
        return None
    return float(np.dot(ea, eb))


class PromptBank:
    """Text embeddings of every prompt for one model, cached in ``data/cache/listening.sqlite3``."""

    def __init__(self, vectors: Dict[str, np.ndarray]) -> None:
        self.vectors = vectors

    def complete(self) -> bool:
        return all(p in self.vectors for p in all_prompts())

    def _score(self, emb: np.ndarray, prompts: Sequence[str]) -> float:
        return float(np.mean([float(np.dot(emb, self.vectors[p])) for p in prompts]))

    def vocal_margin(self, emb: np.ndarray) -> float:
        """Combined singing score of one excerpt (see VOCAL_PAIRS)."""

        descriptive = self._score(emb, VOCAL_PROMPTS) - self._score(emb, INSTRUMENTAL_PROMPTS)
        pairs = float(np.mean([float(np.dot(emb, self.vectors[a])) - float(np.dot(emb, self.vectors[b]))
                               for a, b in VOCAL_PAIRS]))
        return descriptive + PAIR_WEIGHT * pairs

    def vocals(self, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Vocals judged per excerpt (older results without excerpts: from the mean embedding)."""

        parts = excerpt_embeddings(result) or [e for e in [embedding_of(result)] if e is not None]
        return _vocal_decision([self.vocal_margin(e) for e in parts])

    def summary(self, result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """What the listening model heard, in the profile vocabulary (``semantic.rules._apply_heard`` format)."""

        emb = embedding_of(result)
        if emb is None or not self.complete():
            return None
        out: Dict[str, Any] = {"source": "listening model (CLAP)"}
        out.update(self.vocals(result))
        for category in SUMMARY_FIELDS:
            terms = CATEGORIES[category]
            scores = np.array([self._score(emb, [t.format(term) for t in TEMPLATES[category]]) for term in terms])
            sd = float(scores.std()) or 1.0
            z = (scores - float(scores.mean())) / sd
            picked = {}
            for i in np.argsort(-z)[:MAX_PER_FIELD]:
                if z[i] >= STRONG_Z:
                    picked[terms[i]] = "strong"
                elif z[i] >= MODERATE_Z:
                    picked[terms[i]] = "moderate"
            if category == "instrumentation" and out["vocals"] != "sung vocals":
                picked.pop("solo voice", None)        # a voice-like timbre is not evidence of vocals by itself
            if picked:
                out[category] = picked
        return out


def _vocal_decision(margins: List[float]) -> Dict[str, Any]:
    n = len(margins)
    sung = sum(1 for m in margins if m > VOCAL_MARGIN)
    needed = max(1, int(np.ceil(VOCAL_SHARE * n - 1e-9)))
    if sung >= needed:
        verdict = "sung vocals"
    elif sung == 0:
        verdict = "instrumental"
    else:
        verdict = "unclear"
    return {"vocals": verdict, "vocals_excerpts": f"{sung} of {n}", "vocals_margin": round(float(np.mean(margins)), 3),
            "vocals_margins": [round(float(m), 3) for m in margins]}


class ListeningCache:
    """Listening results for user tracks (by audio identity) and prompt embeddings (by model)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(os_path(path), check_same_thread=False)
        self.conn.executescript(
            "CREATE TABLE IF NOT EXISTS track (identity TEXT NOT NULL, model_key TEXT NOT NULL, result_json TEXT NOT NULL,"
            " PRIMARY KEY (identity, model_key));"
            "CREATE TABLE IF NOT EXISTS prompt (model_key TEXT NOT NULL, prompt TEXT NOT NULL, vector BLOB NOT NULL,"
            " PRIMARY KEY (model_key, prompt));")
        self.conn.commit()

    def get_track(self, identity: str, model_key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            row = self.conn.execute("SELECT result_json FROM track WHERE identity=? AND model_key=?",
                                    (identity, model_key)).fetchone()
        return json.loads(row[0]) if row else None

    def put_track(self, identity: str, model_key: str, result: Dict[str, Any]) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO track VALUES (?,?,?)", (identity, model_key, json.dumps(result)))
            self.conn.commit()

    def prompt_bank(self, model_key: str) -> PromptBank:
        model_key = f"{model_key}:p{PROMPT_EMBED_VERSION}"
        with self.lock:
            rows = self.conn.execute("SELECT prompt, vector FROM prompt WHERE model_key=?", (model_key,)).fetchall()
        return PromptBank({p: np.frombuffer(v, dtype=np.float32) for p, v in rows})

    def ensure_prompts(self, model_key: str, model: ClapModel) -> PromptBank:
        bank = self.prompt_bank(model_key)
        missing = [p for p in all_prompts() if p not in bank.vectors]
        if missing:
            vectors = model.embed_text(missing)
            stored_key = f"{model_key}:p{PROMPT_EMBED_VERSION}"
            with self.lock:
                self.conn.executemany("INSERT OR REPLACE INTO prompt VALUES (?,?,?)",
                                      [(stored_key, p, v.astype(np.float32).tobytes()) for p, v in zip(missing, vectors)])
                self.conn.commit()
            bank = self.prompt_bank(model_key)
        return bank

    def close(self) -> None:
        with self.lock:
            self.conn.close()
