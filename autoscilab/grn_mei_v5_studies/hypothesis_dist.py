from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autoscilab.grn_mei_v5_studies.graph_search import canonicalize_edges


@dataclass
class GraphHypothesisEntry:
    hypothesis_text: str
    rationale: str
    edges: list[tuple[str, str, int]]
    source_idx: int

    @property
    def skeleton(self) -> tuple[tuple[str, str, int], ...]:
        return tuple(self.edges)

    @property
    def edge_summary(self) -> str:
        return ", ".join(f"{src}->{dst}({sign:+d})" for src, dst, sign in self.edges)


@dataclass
class GraphHypothesisDistribution:
    hypotheses: list[GraphHypothesisEntry] = field(default_factory=list)
    unique_hypotheses: list[GraphHypothesisEntry] = field(default_factory=list)
    clusters: list[list[int]] = field(default_factory=list)
    majority_cluster: int = 0
    agreement_score: float = 0.0
    synthesis_prompt: str = ""

    @classmethod
    def from_samples(cls, samples: list[dict[str, Any]]) -> "GraphHypothesisDistribution":
        entries = [
            GraphHypothesisEntry(
                hypothesis_text=str(sample["hypothesis_text"]).strip(),
                rationale=str(sample.get("rationale", "")).strip(),
                edges=canonicalize_edges(sample["edges"]),
                source_idx=int(sample.get("source_idx", i)),
            )
            for i, sample in enumerate(samples)
            if sample.get("hypothesis_text") and sample.get("edges")
        ]
        clusters = _cluster_by_skeletons([entry.skeleton for entry in entries])
        majority = max(range(len(clusters)), key=lambda i: len(clusters[i])) if clusters else 0
        agreement = len(clusters[majority]) / len(entries) if entries else 0.0
        unique = [entries[cluster[0]] for cluster in clusters] if clusters else []
        dist = cls(
            hypotheses=entries,
            unique_hypotheses=unique,
            clusters=clusters,
            majority_cluster=majority,
            agreement_score=agreement,
        )
        dist.synthesis_prompt = dist._build_synthesis_prompt()
        return dist

    def n_unique_structures(self) -> int:
        return len(self.clusters)

    def entropy(self) -> float:
        if not self.clusters or not self.hypotheses:
            return 0.0
        total = len(self.hypotheses)
        probs = [len(cluster) / total for cluster in self.clusters]
        return float(-sum(p * np.log2(p) for p in probs if p > 0))

    def _build_synthesis_prompt(self) -> str:
        lines = [
            f"Small-model graph ensemble: {len(self.hypotheses)} samples, "
            f"{self.n_unique_structures()} unique graph families, "
            f"agreement={self.agreement_score:.0%}",
        ]
        for idx, cluster in enumerate(self.clusters):
            label = " <- MAJORITY" if idx == self.majority_cluster else ""
            rep = self.hypotheses[cluster[0]]
            lines.append(f"\nStructure {idx + 1} ({len(cluster)} votes){label}:")
            lines.append(f"  edges: {rep.edge_summary}")
            lines.append(f"  representative mechanism: {rep.hypothesis_text}")
            if rep.rationale:
                lines.append(f"  rationale: {rep.rationale}")
        if self.agreement_score < 0.5:
            lines.append(
                "\nLOW AGREEMENT: graph-family support is spread across multiple competing structures. "
                "Prioritize hypotheses and experiments that explicitly separate the top graph families."
            )
        elif self.agreement_score >= 0.8:
            lines.append(
                "\nHIGH AGREEMENT: the ensemble concentrates on one graph family. "
                "Prefer refinement and validation unless the observed data contradict it."
            )
        return "\n".join(lines)


def _cluster_by_skeletons(
    skeletons: list[tuple[tuple[str, str, int], ...]],
) -> list[list[int]]:
    cluster_map: dict[tuple[tuple[str, str, int], ...], list[int]] = {}
    for idx, skeleton in enumerate(skeletons):
        cluster_map.setdefault(skeleton, []).append(idx)
    clusters = list(cluster_map.values())
    clusters.sort(key=len, reverse=True)
    return clusters
