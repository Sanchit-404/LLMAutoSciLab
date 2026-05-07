from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

_GRN_VARS = {"signal", "pert_A", "pert_B", "pert_C", "pert_R"}
_GRN_NODES = {"signal", "A", "B", "C", "R"}


class GRNSearchRegion(BaseModel):
    bounds: dict[str, list[float]] = Field(
        description=(
            "Complete parameter bounds with ALL 5 keys exactly once: "
            "signal, pert_A, pert_B, pert_C, pert_R. "
            "Each value must be [min, max] within the physical domain."
        )
    )
    n_experiments: int = Field(ge=1, le=20)
    priority: Literal["high", "medium", "low"] = "medium"
    rationale: str

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, value: dict[str, list[float]]) -> dict[str, list[float]]:
        keys = set(value.keys())
        if keys != _GRN_VARS:
            missing = sorted(_GRN_VARS - keys)
            extra = sorted(keys - _GRN_VARS)
            raise ValueError(
                f"bounds must contain exactly {_GRN_VARS}; missing={missing} extra={extra}"
            )
        for key, item in value.items():
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"bounds[{key}] must be a 2-item list [min, max]")
        return value


class NaturalLanguageHypothesis(BaseModel):
    hypothesis_id: str = Field(description="Stable identifier like h1, h2, h3.")
    text: str = Field(description="Concise natural-language mechanism hypothesis.")
    confidence: float = Field(ge=0.0, le=1.0)


class SampledGraphHypothesis(BaseModel):
    text: str = Field(description="One concise natural-language graph mechanism hypothesis.")
    rationale: str = Field(description="Why this mechanism is plausible from the current data.")
    confidence: float = Field(ge=0.0, le=1.0)


class HypothesisReasoningProposal(BaseModel):
    reasoning: str = Field(description="Reason about the current data and why the hypotheses are plausible.")
    primary_hypothesis_id: str = Field(description="hypothesis_id of the currently preferred hypothesis.")
    hypotheses: list[NaturalLanguageHypothesis] = Field(min_length=2, max_length=5)
    search_regions: list[GRNSearchRegion] = Field(min_length=1, max_length=8)
    phase: Literal["discovery", "validation", "contradiction"] = "discovery"
    confidence: float = Field(ge=0.0, le=1.0)
    done: bool = False


class GraphEdge(BaseModel):
    src: Literal["signal", "A", "B", "C", "R"]
    dst: Literal["A", "B", "C", "R"]
    sign: Literal[-1, 1]

    @field_validator("dst")
    @classmethod
    def validate_dst(cls, value: str) -> str:
        if value not in _GRN_NODES - {"signal"}:
            raise ValueError("dst must be one of A, B, C, R")
        return value


class GraphTranslation(BaseModel):
    hypothesis_id: str = Field(description="Must match one of the reasoning-stage hypothesis ids.")
    rationale: str = Field(description="How the natural-language hypothesis was mapped to graph structure.")
    assumptions: list[str] = Field(default_factory=list, max_length=6)
    edges: list[GraphEdge] = Field(
        default_factory=list,
        max_length=6,
        description="Signed directed edges over nodes signal, A, B, C, R.",
    )

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, value: list[GraphEdge]) -> list[GraphEdge]:
        seen: set[tuple[str, str]] = set()
        graph: dict[str, set[str]] = {}
        for edge in value:
            if edge.src == edge.dst:
                raise ValueError("self-loops are not allowed")
            key = (edge.src, edge.dst)
            if key in seen:
                raise ValueError("duplicate directed edges are not allowed")
            seen.add(key)
            graph.setdefault(edge.src, set()).add(edge.dst)
        frontier = ["signal"]
        reached = {"signal"}
        while frontier:
            cur = frontier.pop()
            if cur == "C":
                return value
            for nxt in graph.get(cur, set()):
                if nxt in reached:
                    continue
                reached.add(nxt)
                frontier.append(nxt)
        raise ValueError("graph must contain a directed path from signal to C")


class GraphTranslationProposal(BaseModel):
    translations: list[GraphTranslation] = Field(min_length=2, max_length=5)


class SingleGraphTranslationProposal(BaseModel):
    translation: GraphTranslation
