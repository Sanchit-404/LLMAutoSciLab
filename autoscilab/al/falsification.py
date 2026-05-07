"""
FalsificationSelector: selects experiment points that maximally discriminate
between competing hypotheses in a HypothesisDistribution.

This is the acquisition function for MEI (Mechanistic Ensemble Inference).
Instead of optimising expected improvement over a GP surrogate, it targets
regions where the ensemble of hypotheses disagrees most — operationalising
Popperian falsification: find the experiment most likely to prove at least
one hypothesis wrong.

Algorithm
---------
1. Sample a dense candidate grid within parameter bounds.
2. Score each candidate by ensemble disagreement (std of log10 predictions).
3. Select the top-k diverse candidates (greedy max-distance in param space).
4. Return as experiment proposals.

Usage
-----
    selector = FalsificationSelector(oracle, distribution)
    points = selector.select(n=8)
    # → list of dicts {param: value, ...}
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from autoscilab.llm.hypothesis_dist import HypothesisDistribution
    from autoscilab.oracle.base import BaseOracle


class FalsificationSelector:
    """
    Selects the most discriminating experiment points given a
    HypothesisDistribution.

    Parameters
    ----------
    oracle : BaseOracle
        Provides parameter names and bounds.
    distribution : HypothesisDistribution
        Current ensemble distribution with K competing hypotheses.
    n_candidates : int
        Number of random candidates to evaluate (higher = better coverage,
        slower scoring). 2000 is a good default.
    diversity_weight : float
        Weight for diversity in the greedy selection step.
        0.0 = pure disagreement; 1.0 = pure space-filling.
        0.3 works well in practice.
    """

    def __init__(
        self,
        oracle: "BaseOracle",
        distribution: "HypothesisDistribution",
        n_candidates: int = 2000,
        diversity_weight: float = 0.3,
    ):
        self._oracle           = oracle
        self._distribution     = distribution
        self._n_candidates     = n_candidates
        self._diversity_weight = diversity_weight

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def select(
        self,
        n: int = 8,
        rng: np.random.Generator | None = None,
    ) -> list[dict]:
        """
        Select n parameter combinations that maximally discriminate between
        the hypotheses in the distribution.

        Parameters
        ----------
        n : int
            Number of points to select.
        rng : np.random.Generator | None
            Optional RNG for reproducibility.

        Returns
        -------
        list[dict]
            Each dict maps parameter names to float values.
        """
        if rng is None:
            rng = np.random.default_rng()

        bounds = self._oracle.parameter_bounds   # dict[str, (lo, hi)]
        param_names = self._oracle.parameter_names

        lo = np.array([bounds[p][0] for p in param_names], dtype=float)
        hi = np.array([bounds[p][1] for p in param_names], dtype=float)

        # ── 1. Sample candidate grid ──────────────────────────────────────
        candidates = rng.uniform(lo, hi, size=(self._n_candidates, len(param_names)))

        # ── 2. Score by ensemble disagreement ────────────────────────────
        disagreement = self._distribution.compute_disagreement(candidates, param_names)

        # ── 3. Greedy diverse selection ───────────────────────────────────
        selected_idx = _greedy_diverse_select(
            candidates, disagreement,
            n=n,
            diversity_weight=self._diversity_weight,
        )

        # ── 4. Convert to list of dicts ───────────────────────────────────
        points = []
        for idx in selected_idx:
            point = {p: float(candidates[idx, j]) for j, p in enumerate(param_names)}
            points.append(point)

        return points

    def disagreement_stats(self, rng: np.random.Generator | None = None) -> dict:
        """
        Return summary statistics on ensemble disagreement across parameter space.
        Useful for logging and deciding whether MEI is needed.
        """
        if rng is None:
            rng = np.random.default_rng()

        bounds = self._oracle.parameter_bounds
        param_names = self._oracle.parameter_names
        lo = np.array([bounds[p][0] for p in param_names], dtype=float)
        hi = np.array([bounds[p][1] for p in param_names], dtype=float)

        candidates = rng.uniform(lo, hi, size=(500, len(param_names)))
        disagreement = self._distribution.compute_disagreement(candidates, param_names)

        return {
            "mean":   float(np.mean(disagreement)),
            "median": float(np.median(disagreement)),
            "max":    float(np.max(disagreement)),
            "p90":    float(np.percentile(disagreement, 90)),
        }


# ──────────────────────────────────────────────────────────────────────────── #
#  Greedy diverse selection                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

def _greedy_diverse_select(
    X: np.ndarray,
    scores: np.ndarray,
    n: int,
    diversity_weight: float = 0.3,
) -> list[int]:
    """
    Greedy selection balancing high disagreement score and spatial diversity.

    At each step, the combined score is:
        combined = (1 - w) * normalised_disagreement + w * normalised_min_dist

    where min_dist is the minimum distance to any already-selected point.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Candidate parameter points.
    scores : np.ndarray, shape (N,)
        Disagreement score per candidate (higher = more discriminating).
    n : int
        Number of points to select.
    diversity_weight : float
        Weight for the diversity term (0 = pure score, 1 = pure diversity).

    Returns
    -------
    list[int]
        Indices of selected candidates.
    """
    N = len(X)
    n = min(n, N)

    # Normalise scores to [0, 1]
    score_range = scores.max() - scores.min()
    norm_scores = (scores - scores.min()) / (score_range + 1e-12)

    # Normalise X columns for distance computation
    x_range = X.max(axis=0) - X.min(axis=0)
    x_range = np.where(x_range > 0, x_range, 1.0)
    X_norm = (X - X.min(axis=0)) / x_range

    selected: list[int] = []
    min_dists = np.full(N, np.inf)

    for _ in range(n):
        if not selected:
            # First point: just pick the highest disagreement
            idx = int(np.argmax(norm_scores))
        else:
            # Update min distances to selected set
            last = X_norm[selected[-1]]
            dists_to_last = np.linalg.norm(X_norm - last, axis=1)
            min_dists = np.minimum(min_dists, dists_to_last)

            # Normalise distances
            dist_range = min_dists.max() - min_dists.min()
            norm_dists = (min_dists - min_dists.min()) / (dist_range + 1e-12)

            combined = (1 - diversity_weight) * norm_scores + diversity_weight * norm_dists
            # Zero out already-selected
            for s in selected:
                combined[s] = -np.inf
            idx = int(np.argmax(combined))

        selected.append(idx)

    return selected
