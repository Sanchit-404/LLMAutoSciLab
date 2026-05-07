"""
HypothesisDistribution: stores and analyses K hypotheses from the small LLM ensemble.

Each hypothesis is a Python expression string. This module:
  - Clusters hypotheses by structural skeleton (functional form)
  - Computes ensemble agreement score
  - Scores parameter-space points by ensemble disagreement (for falsification)
  - Builds a synthesis prompt for the large LLM

The disagreement score is the key signal for the FalsificationSelector:
high disagreement at a point X means the K hypotheses predict very different
values there → measuring X will maximally discriminate between hypotheses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────── #
#  Data structures                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class HypothesisEntry:
    """One hypothesis from the ensemble."""
    expression: str    # Python eval-able expression string
    skeleton: str      # Structural skeleton (constants replaced with C)
    source_idx: int    # Which ensemble member produced this (0-indexed)


@dataclass
class HypothesisDistribution:
    """
    Distribution over K hypotheses sampled from the small LLM ensemble.

    Attributes
    ----------
    hypotheses : list[HypothesisEntry]
        All K raw hypotheses (may include duplicates).
    unique_hypotheses : list[HypothesisEntry]
        Deduplicated by skeleton (one representative per structural form).
    clusters : list[list[int]]
        Indices into `hypotheses`, grouped by matching skeleton.
        Sorted descending by cluster size.
    majority_cluster : int
        Index of the largest cluster (most common functional form).
    agreement_score : float
        Fraction of hypotheses in the majority cluster ∈ [0, 1].
        1.0 means all K samples agree on the same functional form.
        <0.4 means high structural uncertainty — need more experiments.
    synthesis_prompt : str
        Human-readable description of the ensemble for the large LLM.
    """
    hypotheses: list[HypothesisEntry] = field(default_factory=list)
    unique_hypotheses: list[HypothesisEntry] = field(default_factory=list)
    clusters: list[list[int]] = field(default_factory=list)
    majority_cluster: int = 0
    agreement_score: float = 0.0
    synthesis_prompt: str = ""

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_expressions(cls, expressions: list[str]) -> "HypothesisDistribution":
        """
        Build a HypothesisDistribution from a list of expression strings.

        Parameters
        ----------
        expressions : list[str]
            Python expression strings from K ensemble calls.

        Returns
        -------
        HypothesisDistribution
        """
        entries = [
            HypothesisEntry(
                expression=expr.strip(),
                skeleton=_extract_skeleton(expr),
                source_idx=i,
            )
            for i, expr in enumerate(expressions)
            if expr and expr.strip()
        ]

        clusters = _cluster_by_skeleton([e.skeleton for e in entries])
        majority = (
            max(range(len(clusters)), key=lambda i: len(clusters[i]))
            if clusters else 0
        )
        agreement = (
            len(clusters[majority]) / len(entries)
            if entries else 0.0
        )

        # One representative per cluster (first member)
        unique = [entries[c[0]] for c in clusters] if clusters else []

        dist = cls(
            hypotheses=entries,
            unique_hypotheses=unique,
            clusters=clusters,
            majority_cluster=majority,
            agreement_score=agreement,
        )
        dist.synthesis_prompt = dist._build_synthesis_prompt()
        return dist

    # ------------------------------------------------------------------ #
    #  Accessors                                                           #
    # ------------------------------------------------------------------ #

    def get_majority_hypothesis(self) -> Optional[str]:
        """Return the most commonly sampled hypothesis expression."""
        if not self.clusters or not self.hypotheses:
            return None
        majority_idx = self.clusters[self.majority_cluster][0]
        return self.hypotheses[majority_idx].expression

    def get_all_expressions(self) -> list[str]:
        """Return all K expression strings."""
        return [h.expression for h in self.hypotheses]

    def get_unique_expressions(self) -> list[str]:
        """Return one expression per structural cluster."""
        return [h.expression for h in self.unique_hypotheses]

    def n_unique_structures(self) -> int:
        """Number of structurally distinct hypotheses."""
        return len(self.clusters)

    def entropy(self) -> float:
        """
        Shannon entropy (bits) of the cluster distribution.
        0.0 = all samples agree on one structure.
        High = mass spread across many structures — still uncertain.
        """
        if not self.clusters or not self.hypotheses:
            return 0.0
        total = len(self.hypotheses)
        probs = [len(c) / total for c in self.clusters]
        return float(-sum(p * np.log2(p) for p in probs if p > 0))

    # ------------------------------------------------------------------ #
    #  Disagreement scoring (core of falsification acquisition)            #
    # ------------------------------------------------------------------ #

    def compute_disagreement(
        self,
        X: np.ndarray,
        param_names: list[str],
    ) -> np.ndarray:
        """
        Score each row of X by ensemble disagreement.

        Disagreement = std(log10(predictions)) across the K hypotheses.
        High disagreement → hypotheses make very different predictions here
        → measuring this point will best discriminate between them.

        Points where all hypotheses predict the same value have zero
        disagreement and are useless for falsification.

        NaN predictions (e.g. evaluation error, log of negative) are treated
        as maximum disagreement — the hypothesis is undefined there, which is
        itself informative.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Parameter grid to evaluate.
        param_names : list[str]
            Names matching columns of X (used to bind eval namespace).

        Returns
        -------
        np.ndarray, shape (N,)
            Disagreement score per row. Always finite.
        """
        import math

        N = len(X)
        _ns_base = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "sqrt": math.sqrt, "asin": math.asin, "acos": math.acos,
            "atan": math.atan, "pi": math.pi, "e": math.e,
            "abs": abs, "sinh": math.sinh, "cosh": math.cosh,
            "tanh": math.tanh, "__builtins__": {},
        }

        # Build prediction matrix: shape (K, N)
        all_preds: list[list[float]] = []
        for hyp in self.hypotheses:
            preds: list[float] = []
            try:
                code = compile(hyp.expression, "<hyp>", "eval")
                for xi in X:
                    try:
                        ns = {
                            **_ns_base,
                            **{p: float(xi[j]) for j, p in enumerate(param_names)},
                        }
                        val = float(eval(code, ns))  # noqa: S307
                        preds.append(val if np.isfinite(val) and val > 0 else np.nan)
                    except Exception:
                        preds.append(np.nan)
            except Exception:
                preds = [np.nan] * N
            all_preds.append(preds)

        arr = np.array(all_preds, dtype=float)  # (K, N)

        with np.errstate(invalid="ignore", divide="ignore"):
            log_arr = np.log10(np.where(arr > 0, arr, np.nan))

        # Disagreement = std across K samples (axis=0)
        disagreement = np.nanstd(log_arr, axis=0)

        # Fill NaN (all hypotheses failed) with the max observed disagreement
        max_d = float(np.nanmax(disagreement)) if np.any(np.isfinite(disagreement)) else 1.0
        disagreement = np.where(np.isfinite(disagreement), disagreement, max_d)

        return disagreement

    # ------------------------------------------------------------------ #
    #  Synthesis prompt for the large LLM                                  #
    # ------------------------------------------------------------------ #

    def _build_synthesis_prompt(self) -> str:
        """Build a compact description of the ensemble for the large LLM."""
        lines = [
            f"Small-LLM ensemble: {len(self.hypotheses)} samples, "
            f"{self.n_unique_structures()} unique structures, "
            f"agreement={self.agreement_score:.0%}",
        ]

        for i, cluster_idxs in enumerate(self.clusters):
            label = " ← MAJORITY" if i == self.majority_cluster else ""
            lines.append(
                f"\n  Structure {i + 1} ({len(cluster_idxs)} votes){label}:"
            )
            # Show up to 2 representative expressions per cluster
            for idx in cluster_idxs[:2]:
                lines.append(f"    {self.hypotheses[idx].expression}")
            if len(cluster_idxs) > 2:
                lines.append(f"    ... ({len(cluster_idxs) - 2} more similar)")

        if self.agreement_score < 0.5:
            lines.append(
                "\n⚠ LOW AGREEMENT — the ensemble is split across multiple structures. "
                "Prioritise experiments that discriminate between the top clusters."
            )
        elif self.agreement_score >= 0.8:
            lines.append(
                "\n✓ HIGH AGREEMENT — most samples converge on the same structure. "
                "Focus on refining constants rather than changing functional form."
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HypothesisDistribution(K={len(self.hypotheses)}, "
            f"structures={self.n_unique_structures()}, "
            f"agreement={self.agreement_score:.0%})"
        )


# ──────────────────────────────────────────────────────────────────────────── #
#  Clustering helpers                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def _extract_skeleton(expr: str) -> str:
    """
    Reduce an expression to its structural skeleton by replacing all
    numeric literals with the placeholder 'C' and removing whitespace.

    Example:
        "1.5 * C_A**2 / (0.3 + C_A**2)"  →  "C*C_A**C/(C+C_A**C)"

    This is intentionally coarse — two expressions with the same skeleton
    share the same functional form and differ only in their constants.
    """
    # Replace numeric literals (int, float, scientific notation)
    skeleton = re.sub(r"(?<![a-zA-Z_])\d+\.?\d*(?:[eE][+-]?\d+)?", "C", expr)
    skeleton = re.sub(r"\s+", "", skeleton)
    return skeleton


def _cluster_by_skeleton(skeletons: list[str]) -> list[list[int]]:
    """
    Group indices by exact skeleton match.

    Returns a list of clusters sorted by descending size.
    Each cluster is a list of indices into the original sequence.
    """
    groups: dict[str, list[int]] = {}
    for i, sk in enumerate(skeletons):
        groups.setdefault(sk, []).append(i)
    return sorted(groups.values(), key=len, reverse=True)
