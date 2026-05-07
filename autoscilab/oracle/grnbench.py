"""
GRNBench oracle - perturbation-driven gene-regulatory motif discovery benchmark.

This benchmark is intentionally structured closer to ChemBench than NewtonBench:
- fixed input schema across all domains
- hidden motif family per domain
- scalar reporter measurement for the current GP/AL loop
- full steady-state node activities exposed in metadata for future graph-level evaluation

Current objective:
    Discover a law mapping perturbation settings -> reporter expression.

Future extension:
    Score recovered motif/topology directly using metadata-driven graph metrics.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from autoscilab.oracle.base import BaseOracle, OracleResult

GRN_GRAPH_NODES = ["signal", "A", "B", "C", "R"]
GRN_INPUT_VARS = ["signal", "pert_A", "pert_B", "pert_C", "pert_R"]

GRN_INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "signal": (0.05, 10.0),
    "pert_A": (0.10, 4.0),
    "pert_B": (0.10, 4.0),
    "pert_C": (0.10, 4.0),
    "pert_R": (0.10, 4.0),
}

GRN_LOG_VARS = set(GRN_INPUT_VARS)
GRN_RMSLE_THRESHOLD = 0.08
GRN_DIFFICULTIES = ["easy", "medium", "hard"]
GRN_LAW_VERSIONS = ["v0", "v1", "v2"]
GRN_FUNCTION_SIGNATURE = (
    "def discovered_law(signal, pert_A, pert_B, pert_C, pert_R):"
)

GRN_UNIVERSAL_GRAMMAR = (
    "Candidate GRN motif families:\n"
    "  1. Activation chain: signal -> A -> B -> C\n"
    "  2. Coherent feedforward: signal -> A, A -> B, and A together with B activates C\n"
    "  3. Incoherent feedforward: signal -> A, A activates B, A activates C, but B represses C\n"
    "  4. Negative feedback: signal -> A -> B -> C and C induces a repressor R that suppresses C\n"
    "  5. Toggle / decision circuit: signal activates A, A and R mutually repress, B follows A, C reports the chosen state\n"
    "\n"
    "Experimental motifs to discriminate:\n"
    "  - Source sweep: vary signal and pert_A while keeping others near 1.0\n"
    "  - Mediator intervention: sweep pert_B at fixed signal to test whether B is required\n"
    "  - Direct-path test: compare response to pert_A versus pert_B to distinguish chain vs feedforward\n"
    "  - Repressor break test: vary pert_R to see whether high reporter collapses under induced repression\n"
    "  - Saturation test: use both low and high signal because some motifs only separate near saturation"
)


def _hill_act(x: float, k: float, n: float) -> float:
    x = max(float(x), 1e-12)
    k = max(float(k), 1e-12)
    n = max(float(n), 1.0)
    xn = x ** n
    kn = k ** n
    return xn / (kn + xn)


def _hill_rep(x: float, k: float, n: float) -> float:
    x = max(float(x), 0.0)
    k = max(float(k), 1e-12)
    n = max(float(n), 1.0)
    xn = x ** n
    kn = k ** n
    return kn / (kn + xn)


def _deterministic_noise(
    domain: str,
    difficulty: str,
    law_version: str,
    inputs: dict[str, float],
    noise_level: float,
) -> float:
    if noise_level == 0.0:
        return 0.0
    seed_str = (
        f"{domain}|{difficulty}|{law_version}|"
        + "|".join(f"{k}={v:.8g}" for k, v in sorted(inputs.items()))
    )
    digest = hashlib.sha256(seed_str.encode()).digest()
    seed_int = struct.unpack("<Q", digest[:8])[0] % (2 ** 32)
    return float(np.random.default_rng(seed_int).normal(0.0, noise_level))


def _merge_params(*parts: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for part in parts:
        merged.update(part)
    return merged


_COMMON_BASE = {
    "basal_a": 0.04,
    "basal_b": 0.03,
    "basal_c": 0.02,
    "basal_r": 0.02,
    "alpha_a": 0.95,
    "alpha_b": 0.92,
    "alpha_c": 0.98,
    "alpha_r": 0.90,
    "reporter_scale": 100.0,
    "K_s": 1.00,
    "K_ab": 0.60,
    "K_ac": 0.65,
    "K_bc": 0.55,
    "K_cr": 0.45,
    "K_rc": 0.40,
    "K_ar": 0.50,
    "K_ra": 0.55,
    "n_s": 2.0,
    "n_ab": 2.0,
    "n_ac": 2.0,
    "n_bc": 2.0,
    "n_cr": 2.0,
    "n_rc": 2.0,
    "n_ar": 2.5,
    "n_ra": 2.5,
}

_DIFFICULTY_OVERRIDES: dict[str, dict[str, float]] = {
    "easy": {
        "K_s": 1.20,
        "K_ab": 0.80,
        "K_ac": 0.80,
        "K_bc": 0.75,
        "n_s": 1.6,
        "n_ab": 1.6,
        "n_ac": 1.6,
        "n_bc": 1.6,
    },
    "medium": {},
    "hard": {
        "K_s": 0.45,
        "K_ab": 0.30,
        "K_ac": 0.35,
        "K_bc": 0.28,
        "K_cr": 0.25,
        "K_rc": 0.25,
        "K_ar": 0.30,
        "K_ra": 0.30,
        "n_s": 3.2,
        "n_ab": 3.0,
        "n_ac": 3.0,
        "n_bc": 3.0,
        "n_cr": 3.2,
        "n_rc": 3.2,
        "n_ar": 4.0,
        "n_ra": 4.0,
    },
}

_VERSION_OVERRIDES: dict[str, dict[str, float]] = {
    "v0": {},
    "v1": {
        "K_s": 0.70,
        "K_ab": 0.45,
        "K_ac": 0.45,
        "K_bc": 0.40,
        "alpha_b": 1.00,
        "alpha_c": 1.05,
    },
    "v2": {
        "K_s": 1.60,
        "K_ab": 1.10,
        "K_ac": 1.10,
        "K_bc": 0.95,
        "alpha_b": 0.82,
        "alpha_c": 0.88,
        "alpha_r": 1.00,
    },
}

_DOMAIN_BASES: dict[str, dict[str, float]] = {
    "g0_activation_chain": {},
    "g1_coherent_feedforward": {
        "alpha_c": 1.10,
    },
    "g2_incoherent_feedforward": {
        "alpha_c": 1.12,
        "K_bc": 0.40,
        "n_bc": 2.6,
    },
    "g3_negative_feedback": {
        "alpha_r": 1.10,
        "K_cr": 0.30,
        "K_rc": 0.32,
        "n_cr": 2.8,
        "n_rc": 2.8,
    },
    "g4_toggle_switch": {
        "alpha_a": 1.05,
        "alpha_b": 0.85,
        "alpha_c": 1.12,
        "alpha_r": 1.05,
        "K_ar": 0.28,
        "K_ra": 0.28,
        "n_ar": 4.4,
        "n_ra": 4.4,
    },
}


GRN_DOMAIN_REGISTRY: dict[str, dict[str, Any]] = {
    "g0_activation_chain": {
        "equation_hint": "reporter ~ pert_C * hill(pert_B * hill(pert_A * hill(signal)))",
        "relevant_vars": ["signal", "pert_A", "pert_B", "pert_C"],
        "tags": ["activation_chain", "hierarchical_activation"],
        "motif_id": "activation_chain",
        "reaction_schema": "signal activates A; A activates B; B activates reporter gene C",
        "edges": [
            ("signal", "A", 1),
            ("A", "B", 1),
            ("B", "C", 1),
        ],
        "hypothesis_grammar": (
            "Primary motif: activation chain. "
            "Perturbing B should matter only through the A->B->C path; pert_R is irrelevant."
        ),
    },
    "g1_coherent_feedforward": {
        "equation_hint": "reporter ~ pert_C * hill(pert_A * hill(signal)) * hill(pert_B * hill(A))",
        "relevant_vars": ["signal", "pert_A", "pert_B", "pert_C"],
        "tags": ["coherent_feedforward", "and_gate"],
        "motif_id": "coherent_feedforward",
        "reaction_schema": "signal activates A; A activates B; A and B jointly activate reporter C",
        "edges": [
            ("signal", "A", 1),
            ("A", "B", 1),
            ("A", "C", 1),
            ("B", "C", 1),
        ],
        "hypothesis_grammar": (
            "Primary motif: coherent feedforward. "
            "Reporter requires both the direct A->C path and the mediated A->B->C path."
        ),
    },
    "g2_incoherent_feedforward": {
        "equation_hint": "reporter ~ activation(A) * repression(B(A))",
        "relevant_vars": ["signal", "pert_A", "pert_B", "pert_C"],
        "tags": ["incoherent_feedforward", "activation_repression"],
        "motif_id": "incoherent_feedforward",
        "reaction_schema": "signal activates A; A activates both B and reporter C; B represses C",
        "edges": [
            ("signal", "A", 1),
            ("A", "B", 1),
            ("A", "C", 1),
            ("B", "C", -1),
        ],
        "hypothesis_grammar": (
            "Primary motif: incoherent feedforward. "
            "Increasing A can increase the reporter at low signal but eventually induce its own repressor through B."
        ),
    },
    "g3_negative_feedback": {
        "equation_hint": "reporter ~ activation(B(A(signal))) * repression(R(C))",
        "relevant_vars": ["signal", "pert_A", "pert_B", "pert_C", "pert_R"],
        "tags": ["negative_feedback", "autoregulatory_repressor"],
        "motif_id": "negative_feedback",
        "reaction_schema": "signal activates A; A activates B; B activates C; C induces repressor R that suppresses C",
        "edges": [
            ("signal", "A", 1),
            ("A", "B", 1),
            ("B", "C", 1),
            ("C", "R", 1),
            ("R", "C", -1),
        ],
        "hypothesis_grammar": (
            "Primary motif: negative feedback. "
            "pert_R should directly modulate the reporter only when the feedback branch is active."
        ),
    },
    "g4_toggle_switch": {
        "equation_hint": "reporter ~ state-selection between A-high and R-high branch",
        "relevant_vars": ["signal", "pert_A", "pert_B", "pert_C", "pert_R"],
        "tags": ["toggle", "decision_circuit", "mutual_repression"],
        "motif_id": "toggle_switch",
        "reaction_schema": "signal biases A; A and R mutually repress; B follows A; reporter C marks the chosen state",
        "edges": [
            ("signal", "A", 1),
            ("A", "R", -1),
            ("R", "A", -1),
            ("A", "B", 1),
            ("A", "C", 1),
            ("B", "C", 1),
        ],
        "hypothesis_grammar": (
            "Primary motif: toggle / bistable decision circuit. "
            "A and R form competing branches; signal and perturbations determine which branch dominates."
        ),
    },
}

GRN_BENCHMARK_DOMAINS = sorted(GRN_DOMAIN_REGISTRY)


@dataclass(frozen=True)
class GRNBenchTask:
    domain: str
    difficulty: str
    law_version: str
    seed: int = 0


def iter_grnbench_tasks(
    domains: list[str] | None = None,
    difficulties: list[str] | None = None,
    law_versions: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[GRNBenchTask]:
    selected_domains = domains or GRN_BENCHMARK_DOMAINS
    selected_difficulties = difficulties or GRN_DIFFICULTIES
    selected_versions = law_versions or GRN_LAW_VERSIONS
    selected_seeds = seeds or [0]
    return [
        GRNBenchTask(
            domain=domain,
            difficulty=difficulty,
            law_version=law_version,
            seed=seed,
        )
        for domain in selected_domains
        for difficulty in selected_difficulties
        for law_version in selected_versions
        for seed in selected_seeds
    ]


def _build_params(domain_id: str, difficulty: str, law_version: str) -> dict[str, float]:
    if difficulty not in _DIFFICULTY_OVERRIDES:
        raise ValueError(f"Unknown GRNBench difficulty: {difficulty!r}")
    if law_version not in _VERSION_OVERRIDES:
        raise ValueError(f"Unknown GRNBench law_version: {law_version!r}")
    return _merge_params(
        _COMMON_BASE,
        _DOMAIN_BASES[domain_id],
        _DIFFICULTY_OVERRIDES[difficulty],
        _VERSION_OVERRIDES[law_version],
    )


def _graph_adjacency_from_edges(edges: list[tuple[str, str, int]]) -> np.ndarray:
    node_to_idx = {n: i for i, n in enumerate(GRN_GRAPH_NODES)}
    adj = np.zeros((len(GRN_GRAPH_NODES), len(GRN_GRAPH_NODES)), dtype=int)
    for src, dst, sign in edges:
        adj[node_to_idx[src], node_to_idx[dst]] = int(np.sign(sign))
    return adj


def _adjacency_to_edges(adj: np.ndarray) -> list[tuple[str, str, int]]:
    edges: list[tuple[str, str, int]] = []
    for i, src in enumerate(GRN_GRAPH_NODES):
        for j, dst in enumerate(GRN_GRAPH_NODES):
            sign = int(np.sign(adj[i, j]))
            if sign != 0:
                edges.append((src, dst, sign))
    return edges


def _normalize_predicted_graph(graph: Any) -> tuple[np.ndarray | None, str | None]:
    motif_id: str | None = None
    if isinstance(graph, str):
        motif_id = graph
        if graph in GRN_DOMAIN_REGISTRY:
            graph = {"motif_id": GRN_DOMAIN_REGISTRY[graph]["motif_id"]}
        else:
            graph = {"motif_id": graph}

    if isinstance(graph, dict):
        motif_id = graph.get("motif_id") or graph.get("motif") or graph.get("domain_id")
        if motif_id in GRN_DOMAIN_REGISTRY:
            motif_id = GRN_DOMAIN_REGISTRY[motif_id]["motif_id"]

        if "adjacency" in graph:
            arr = np.array(graph["adjacency"], dtype=int)
            return arr, motif_id
        if "edges" in graph:
            edges = []
            for item in graph["edges"]:
                if isinstance(item, dict):
                    edges.append((item["src"], item["dst"], int(item["sign"])))
                else:
                    src, dst, sign = item
                    edges.append((src, dst, int(sign)))
            return _graph_adjacency_from_edges(edges), motif_id
        if motif_id:
            for cfg in GRN_DOMAIN_REGISTRY.values():
                if cfg["motif_id"] == motif_id:
                    return _graph_adjacency_from_edges(cfg["edges"]), motif_id
    return None, motif_id


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _simulate_chain(p: dict[str, float], q: dict[str, float]) -> dict[str, float]:
    a = q["pert_A"] * (p["basal_a"] + p["alpha_a"] * _hill_act(q["signal"], p["K_s"], p["n_s"]))
    b = q["pert_B"] * (p["basal_b"] + p["alpha_b"] * _hill_act(a, p["K_ab"], p["n_ab"]))
    c = q["pert_C"] * (p["basal_c"] + p["alpha_c"] * _hill_act(b, p["K_bc"], p["n_bc"]))
    r = q["pert_R"] * p["basal_r"]
    return {"A": a, "B": b, "C": c, "R": r}


def _simulate_coherent_ffl(p: dict[str, float], q: dict[str, float]) -> dict[str, float]:
    a = q["pert_A"] * (p["basal_a"] + p["alpha_a"] * _hill_act(q["signal"], p["K_s"], p["n_s"]))
    b = q["pert_B"] * (p["basal_b"] + p["alpha_b"] * _hill_act(a, p["K_ab"], p["n_ab"]))
    direct = _hill_act(a, p["K_ac"], p["n_ac"])
    mediated = _hill_act(b, p["K_bc"], p["n_bc"])
    c = q["pert_C"] * (p["basal_c"] + p["alpha_c"] * direct * mediated)
    r = q["pert_R"] * p["basal_r"]
    return {"A": a, "B": b, "C": c, "R": r}


def _simulate_incoherent_ffl(p: dict[str, float], q: dict[str, float]) -> dict[str, float]:
    a = q["pert_A"] * (p["basal_a"] + p["alpha_a"] * _hill_act(q["signal"], p["K_s"], p["n_s"]))
    b = q["pert_B"] * (p["basal_b"] + p["alpha_b"] * _hill_act(a, p["K_ab"], p["n_ab"]))
    direct = _hill_act(a, p["K_ac"], p["n_ac"])
    repression = _hill_rep(b, p["K_bc"], p["n_bc"])
    c = q["pert_C"] * (p["basal_c"] + p["alpha_c"] * direct * repression)
    r = q["pert_R"] * p["basal_r"]
    return {"A": a, "B": b, "C": c, "R": r}


def _simulate_negative_feedback(p: dict[str, float], q: dict[str, float]) -> dict[str, float]:
    a = q["pert_A"] * (p["basal_a"] + p["alpha_a"] * _hill_act(q["signal"], p["K_s"], p["n_s"]))
    b = q["pert_B"] * (p["basal_b"] + p["alpha_b"] * _hill_act(a, p["K_ab"], p["n_ab"]))
    c = q["pert_C"] * (p["basal_c"] + p["alpha_c"] * _hill_act(b, p["K_bc"], p["n_bc"]))
    r = q["pert_R"] * p["basal_r"]
    for _ in range(18):
        c = q["pert_C"] * (
            p["basal_c"]
            + p["alpha_c"] * _hill_act(b, p["K_bc"], p["n_bc"]) * _hill_rep(r, p["K_cr"], p["n_cr"])
        )
        r = q["pert_R"] * (
            p["basal_r"]
            + p["alpha_r"] * _hill_act(c, p["K_rc"], p["n_rc"])
        )
    return {"A": a, "B": b, "C": c, "R": r}


def _simulate_toggle(p: dict[str, float], q: dict[str, float]) -> dict[str, float]:
    a = q["pert_A"] * (p["basal_a"] + 0.6 * p["alpha_a"] * _hill_act(q["signal"], p["K_s"], p["n_s"]))
    r = q["pert_R"] * (p["basal_r"] + 0.6 * p["alpha_r"])
    b = q["pert_B"] * p["basal_b"]
    c = q["pert_C"] * p["basal_c"]
    for _ in range(28):
        a = q["pert_A"] * (
            p["basal_a"]
            + p["alpha_a"] * _hill_act(q["signal"], p["K_s"], p["n_s"]) * _hill_rep(r, p["K_ar"], p["n_ar"])
        )
        r = q["pert_R"] * (
            p["basal_r"]
            + p["alpha_r"] * _hill_rep(a, p["K_ra"], p["n_ra"])
        )
        b = q["pert_B"] * (p["basal_b"] + p["alpha_b"] * _hill_act(a, p["K_ab"], p["n_ab"]))
        c = q["pert_C"] * (
            p["basal_c"]
            + p["alpha_c"] * _hill_act(a, p["K_ac"], p["n_ac"]) * _hill_act(b, p["K_bc"], p["n_bc"])
        )
    return {"A": a, "B": b, "C": c, "R": r}


_SIMULATORS = {
    "g0_activation_chain": _simulate_chain,
    "g1_coherent_feedforward": _simulate_coherent_ffl,
    "g2_incoherent_feedforward": _simulate_incoherent_ffl,
    "g3_negative_feedback": _simulate_negative_feedback,
    "g4_toggle_switch": _simulate_toggle,
}


class GRNBenchOracle(BaseOracle):
    def __init__(
        self,
        domain_id: str,
        difficulty: str = "easy",
        noise_level: float = 0.01,
        law_version: str | None = None,
        grammar_mode: str = "universal",
    ):
        if domain_id not in GRN_DOMAIN_REGISTRY:
            raise ValueError(
                f"Unknown GRNBench domain: {domain_id!r}. "
                f"Choose from: {sorted(GRN_DOMAIN_REGISTRY)}"
            )
        if difficulty not in GRN_DIFFICULTIES:
            raise ValueError(f"difficulty must be 'easy'|'medium'|'hard', got {difficulty!r}")

        self._domain_id = domain_id
        self._difficulty = difficulty
        self._noise_level = noise_level
        self._cfg = GRN_DOMAIN_REGISTRY[domain_id]
        self._sim_fn = _SIMULATORS[domain_id]
        self._grammar_mode = grammar_mode

        if law_version is None:
            import random
            self._law_version = random.choice(GRN_LAW_VERSIONS)
        elif law_version in GRN_LAW_VERSIONS:
            self._law_version = law_version
        else:
            raise ValueError(f"law_version must be v0/v1/v2, got {law_version!r}")

        self._params = _build_params(domain_id, difficulty, self._law_version)
        print(
            f"[GRNBenchOracle] {domain_id} | {difficulty} | {self._law_version} | noise={noise_level}"
        )

    @property
    def domain(self) -> str:
        return self._domain_id

    @property
    def parameter_names(self) -> list[str]:
        return GRN_INPUT_VARS

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(GRN_INPUT_BOUNDS)

    @property
    def domain_tags(self) -> list[str]:
        return list(self._cfg.get("tags", []))

    @property
    def graph_nodes(self) -> list[str]:
        return list(GRN_GRAPH_NODES)

    @property
    def motif_id(self) -> str:
        return str(self._cfg["motif_id"])

    @property
    def ground_truth_edges(self) -> list[tuple[str, str, int]]:
        return list(self._cfg["edges"])

    @property
    def ground_truth_adjacency(self) -> np.ndarray:
        return _graph_adjacency_from_edges(self.ground_truth_edges)

    @property
    def function_signature(self) -> str:
        return GRN_FUNCTION_SIGNATURE

    @property
    def param_description(self) -> str:
        if self._grammar_mode == "universal":
            grammar_text: str | None = GRN_UNIVERSAL_GRAMMAR
        elif self._grammar_mode == "none":
            grammar_text = None
        else:
            grammar_text = self._cfg["hypothesis_grammar"]

        if self._grammar_mode in ("none", "universal"):
            motif_line = "Motif family: hidden GRN motif (unknown)"
        else:
            motif_line = f"Motif family: {self._cfg['reaction_schema']}"

        lines = [
            motif_line,
            "Inputs: signal [0.05-10], pert_A [0.1-4], pert_B [0.1-4], pert_C [0.1-4], pert_R [0.1-4]",
            "Interpretation: perturbation multipliers where 1.0 is baseline, <1 is knockdown, >1 is overexpression",
            "Observables: reporter intensity plus marker panel A, B, C, R for each perturbation",
        ]
        if grammar_text is not None:
            lines.append(f"Hypothesis grammar: {grammar_text}")
        return "\n".join(lines)

    @property
    def objective_profile(self) -> dict[str, Any]:
        return {
            "name": "grn_mechanism_discovery",
            "direction": "minimize_error",
            "type": "law_discovery",
            "difficulty": self._difficulty,
            "law_version": self._law_version,
            "motif_id": self.motif_id,
        }

    def _clean_reporter(self, params: dict[str, float]) -> tuple[float, dict[str, float]]:
        states = self._sim_fn(self._params, params)
        reporter = max(1e-6, self._params["reporter_scale"] * states["C"])
        return float(reporter), states

    def run(self, params: dict[str, float]) -> OracleResult:
        q = {name: float(params[name]) for name in GRN_INPUT_VARS}
        reporter_true, states = self._clean_reporter(q)
        noise = _deterministic_noise(
            self._domain_id,
            self._difficulty,
            self._law_version,
            q,
            self._noise_level,
        )
        reporter_obs = max(reporter_true * (1.0 + noise), 1e-6)
        return OracleResult(
            params=q,
            measurement=reporter_obs,
            domain=self._domain_id,
            noise_level=self._noise_level,
            metadata={
                "difficulty": self._difficulty,
                "law_version": self._law_version,
                "states": states,
                "marker_panel": {
                    "A": float(states["A"]),
                    "B": float(states["B"]),
                    "C": float(states["C"]),
                    "R": float(states["R"]),
                    "reporter": float(reporter_obs),
                    "reporter_true": float(reporter_true),
                },
                "reporter_true": reporter_true,
                "motif_id": self.motif_id,
                "graph_edges": self.ground_truth_edges,
            },
        )

    def evaluate_graph(self, graph: Any) -> dict[str, Any]:
        pred_adj, pred_motif = _normalize_predicted_graph(graph)
        true_adj = self.ground_truth_adjacency
        true_edges = set(_adjacency_to_edges(true_adj))

        if pred_adj is None:
            return {
                "motif_accuracy": 1.0 if pred_motif == self.motif_id else 0.0,
                "edge_precision": 0.0,
                "edge_recall": 0.0,
                "edge_f1": 0.0,
                "sign_accuracy": 0.0,
                "exact_graph_accuracy": 0.0,
                "error": "Could not parse predicted graph representation",
            }

        pred_adj = np.array(pred_adj, dtype=int)
        if pred_adj.shape != true_adj.shape:
            return {
                "motif_accuracy": 1.0 if pred_motif == self.motif_id else 0.0,
                "edge_precision": 0.0,
                "edge_recall": 0.0,
                "edge_f1": 0.0,
                "sign_accuracy": 0.0,
                "exact_graph_accuracy": 0.0,
                "error": f"Adjacency shape mismatch: expected {true_adj.shape}, got {pred_adj.shape}",
            }

        pred_edges = set(_adjacency_to_edges(pred_adj))
        tp = len(pred_edges & true_edges)
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

        pred_nonzero = pred_adj != 0
        true_nonzero = true_adj != 0
        overlap = pred_nonzero & true_nonzero
        sign_accuracy = float(np.mean(pred_adj[overlap] == true_adj[overlap])) if np.any(overlap) else 0.0

        return {
            "motif_accuracy": 1.0 if pred_motif == self.motif_id else 0.0,
            "edge_precision": float(precision),
            "edge_recall": float(recall),
            "edge_f1": float(f1),
            "sign_accuracy": float(sign_accuracy),
            "exact_graph_accuracy": 1.0 if np.array_equal(pred_adj, true_adj) else 0.0,
            "n_true_edges": len(true_edges),
            "n_pred_edges": len(pred_edges),
        }

    def evaluate_law(self, law_str: str) -> dict[str, Any]:
        rng = np.random.default_rng(42)
        n_test = 1000
        test_samples = {
            name: np.exp(rng.uniform(np.log(lo), np.log(hi), n_test))
            for name, (lo, hi) in GRN_INPUT_BOUNDS.items()
        }

        y_true = np.array([
            self._clean_reporter({name: float(test_samples[name][i]) for name in GRN_INPUT_VARS})[0]
            for i in range(n_test)
        ])

        import math as _math
        exec_globals: dict[str, Any] = {"math": _math, "np": np, "__builtins__": __builtins__}
        try:
            exec(law_str, exec_globals)
            discovered_law = exec_globals.get("discovered_law")
            if discovered_law is None:
                return {
                    "rmsle": float("nan"),
                    "exact_accuracy": 0.0,
                    "symbolic_equivalent": None,
                    "error": "No 'discovered_law' function found in law_str",
                }
        except Exception as exc:
            return {
                "rmsle": float("nan"),
                "exact_accuracy": 0.0,
                "symbolic_equivalent": None,
                "error": f"Failed to compile discovered law: {exc}",
            }

        try:
            y_pred = np.array([
                discovered_law(*[float(test_samples[name][i]) for name in GRN_INPUT_VARS])
                for i in range(n_test)
            ], dtype=float)
        except Exception as exc:
            return {
                "rmsle": float("nan"),
                "exact_accuracy": 0.0,
                "symbolic_equivalent": None,
                "error": f"Failed to evaluate discovered law: {exc}",
            }

        y_t = np.abs(y_true)
        y_p = np.abs(y_pred)
        mask = np.isfinite(y_t) & np.isfinite(y_p) & (y_t > 0) & (y_p > 0)
        if not mask.any():
            return {
                "rmsle": float("nan"),
                "exact_accuracy": 0.0,
                "symbolic_equivalent": None,
                "error": "All values invalid",
            }

        rmsle = float(np.sqrt(np.mean((np.log1p(y_p[mask]) - np.log1p(y_t[mask])) ** 2)))
        return {
            "rmsle": rmsle,
            "exact_accuracy": 1.0 if rmsle < GRN_RMSLE_THRESHOLD else 0.0,
            "symbolic_equivalent": None,
        }
