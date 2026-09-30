"""Calibrated 0-100 scores for what the listening model hears.

Raw CLAP similarities are not comparable between words: some words ("solemn", "whimsical", "atmospheric")
score high for almost every piece, which is why the same tags kept showing up. So each word is judged
against *all the music analysed* (the user's tracks plus the game's music):

    z = (this piece's similarity to the word - the word's average over all music) / the word's spread
    score = 100 x Phi(z)        (Phi = the normal cumulative distribution)

A score of 92 for "melancholic" means this piece is more melancholic-sounding than about 92% of the music
analysed; a word that scores high for everything averages out to 50 for every piece and stops appearing.
Only clear standouts (``TAG_MIN_SCORE`` and above) are shown as tags; the full z-vector is what
standout-score matching compares (``z_vector`` / ``standout_vector``).

With fewer than ``MIN_REFERENCE`` pieces there is no stable baseline; the score is then taken relative to the
other words of the same piece (the earlier ranking) and the summary says ``calibrated: False``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .listen import PromptBank, embedding_of
from .vocabulary import VOCABULARY, prompts_for

MIN_REFERENCE = 30
TAG_MIN_SCORE = 85            # a tag is shown only from here (z >= 1.04)
MAX_TAGS_PER_GROUP = 6
# a word whose spread over the reference is below this share of the median spread tells pieces apart
# too little to be useful and is skipped
MIN_SPREAD_RATIO = 0.4
STANDOUT_Z_FLOOR = 0.5        # below this a word does not count as standing out (matching)
STANDOUT_Z_CAP = 3.0
_SQRT2 = math.sqrt(2.0)


def phi_score(z: np.ndarray) -> np.ndarray:
    """100 x the normal CDF, vectorised."""

    erf = np.vectorize(math.erf, otypes=[float])
    return 100.0 * 0.5 * (1.0 + erf(np.asarray(z, dtype=np.float64) / _SQRT2))


class Calibration:
    def __init__(self, bank: PromptBank, reference: Sequence[np.ndarray] = ()) -> None:
        self.bank = bank
        self.words: List[Tuple[str, str]] = [(c, w) for c, ws in VOCABULARY.items() for w in ws]
        self.complete = bank.complete()
        if self.complete:
            self.matrix = np.stack([np.mean([bank.vectors[p] for p in prompts_for(c, w)], axis=0)
                                    for c, w in self.words]).astype(np.float32)        # (W, dim)
        else:
            self.matrix = np.zeros((len(self.words), 1), np.float32)
        ref = [np.asarray(r, np.float32) for r in reference if r is not None]
        self.reference_size = len(ref)
        self.calibrated = self.complete and len(ref) >= MIN_REFERENCE and ref[0].shape[0] == self.matrix.shape[1]
        if self.calibrated:
            sims = np.stack(ref) @ self.matrix.T                                        # (N, W)
            self.mean = sims.mean(axis=0)
            self.std = np.maximum(sims.std(axis=0), 1e-6)
            self.useful = self.std >= MIN_SPREAD_RATIO * float(np.median(self.std))
        else:
            self.mean = self.std = None
            self.useful = np.ones(len(self.words), bool)

    # ---------------------------------------------------------------- scores
    def z_vector(self, emb: np.ndarray) -> np.ndarray:
        """Per-word z-scores of one piece (over the reference; or within the piece when not calibrated)."""

        sims = self.matrix @ np.asarray(emb, np.float32)
        if self.calibrated:
            return (sims - self.mean) / self.std
        sd = float(sims.std()) or 1.0
        return (sims - float(sims.mean())) / sd

    def standout_vector(self, emb: np.ndarray) -> np.ndarray:
        """What stands out about a piece: z above STANDOUT_Z_FLOOR (capped), zero elsewhere; uninformative words 0."""

        z = self.z_vector(emb)
        out = np.clip(z - STANDOUT_Z_FLOOR, 0.0, STANDOUT_Z_CAP - STANDOUT_Z_FLOOR)
        out[~self.useful] = 0.0
        return out.astype(np.float32)

    def tags(self, emb: np.ndarray, vocals: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        """Standout words per group with their 0-100 scores, strongest first."""

        scores = phi_score(self.z_vector(emb))
        per_group: Dict[str, List[Tuple[int, str]]] = {}
        for i, (category, word) in enumerate(self.words):
            if not self.useful[i] or scores[i] < TAG_MIN_SCORE:
                continue
            if category == "instrumentation" and word in ("solo voice", "male vocals", "female vocals") \
                    and vocals not in ("sung vocals", None):
                continue                      # a voice-like timbre is not evidence of singing by itself
            per_group.setdefault(category, []).append((int(round(scores[i])), word))
        return {c: {w: s for s, w in sorted(items, key=lambda t: (-t[0], t[1]))[:MAX_TAGS_PER_GROUP]}
                for c, items in per_group.items()}

    def summary(self, result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """What the listening model heard: vocals (verdict + 0-100 score) and standout tags with 0-100 scores."""

        emb = embedding_of(result)
        if emb is None or not self.complete:
            return None
        out: Dict[str, Any] = {"source": "listening model (CLAP)", "calibrated": self.calibrated,
                               "reference_size": self.reference_size}
        out.update(self.bank.vocals(result))
        out.update(self.tags(emb, out["vocals"]))
        return out

    # -------------------------------------------------------------- matching
    def standout_similarity(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, List[Tuple[str, str, int, int]]]:
        """Cosine of two standout vectors (0..1) and the shared standouts as (category, word, score a, score b)."""

        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return 0.0, []
        sim = float(np.dot(a, b) / (na * nb))
        shared = np.minimum(a, b)
        order = np.argsort(-shared)[:8]
        scores_a, scores_b = phi_score(a + STANDOUT_Z_FLOOR), phi_score(b + STANDOUT_Z_FLOOR)
        reasons = [(self.words[i][0], self.words[i][1], int(round(scores_a[i])), int(round(scores_b[i])))
                   for i in order if shared[i] > 0.4]
        return max(0.0, min(1.0, sim)), reasons
