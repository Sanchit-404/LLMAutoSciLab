from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

from autoscilab.data.store import ExperimentStore
from autoscilab.grn.graph_search import GraphHypothesisFit, search_graph_hypotheses
from autoscilab.oracle.grnbench import (
    GRNBenchOracle,
    GRN_INPUT_BOUNDS,
    GRN_INPUT_VARS,
)


@dataclass
class GraphDiscoveryConfig:
    domain_id: str
    difficulty: str = "easy"
    law_version: str = "v0"
    noise_level: float = 0.01
    budget: int = 20
    init_points: int = 4
    batch_size: int = 4
    min_data_for_graph_fit: int = 5
    candidate_pool_size: int = 512
    beam_size: int = 8
    max_edges: int = 6
    top_k_graphs: int = 5
    restarts: int = 3
    max_indegree: int = 3
    per_depth_candidate_cap: int | None = 48
    diversity_weight: float = 0.15
    state_weight: float = 0.5
    seed: int = 0
    results_dir: Path = Path("results/")


@dataclass
class GraphDiscoveryResult:
    status: str
    n_oracle_calls: int
    duration_s: float
    best_graph: dict | None
    final_graph_eval: dict | None
    history: list[dict] = field(default_factory=list)
    result_dir: str | None = None


def _sample_points(n: int, rng: np.random.Generator) -> list[dict[str, float]]:
    sampler = LatinHypercube(d=len(GRN_INPUT_VARS), seed=int(rng.integers(0, 2**31 - 1)))
    lhs = sampler.random(n=n)
    points: list[dict[str, float]] = []
    for i in range(n):
        p: dict[str, float] = {}
        for j, var in enumerate(GRN_INPUT_VARS):
            lo, hi = GRN_INPUT_BOUNDS[var]
            p[var] = float(np.exp(np.log(lo) + lhs[i, j] * (np.log(hi) - np.log(lo))))
        points.append(p)
    return points


def _top_diverse_indices(
    points: np.ndarray,
    scores: np.ndarray,
    k: int,
    diversity_weight: float,
) -> list[int]:
    if len(points) == 0 or k <= 0:
        return []
    order = np.argsort(scores)[::-1]
    chosen = [int(order[0])]
    if k == 1:
        return chosen
    pmin = points.min(axis=0)
    pmax = points.max(axis=0) + 1e-12
    normed = (points - pmin) / (pmax - pmin)
    while len(chosen) < min(k, len(points)):
        best_idx = None
        best_score = -1e18
        for idx in order:
            idx = int(idx)
            if idx in chosen:
                continue
            min_dist = min(float(np.linalg.norm(normed[idx] - normed[c])) for c in chosen)
            score = float(scores[idx]) + diversity_weight * min_dist
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        chosen.append(best_idx)
    return chosen


def _select_by_graph_disagreement(
    fits: list[GraphHypothesisFit],
    n_points: int,
    rng: np.random.Generator,
    candidate_pool_size: int,
    diversity_weight: float,
) -> list[dict[str, float]]:
    candidates = _sample_points(candidate_pool_size, rng)
    X = np.array([[p[v] for v in GRN_INPUT_VARS] for p in candidates], dtype=float)
    pred_logs = []
    for fit in fits:
        y_hat = np.maximum(fit.predict(X), 1e-9)
        pred_logs.append(np.log(y_hat))
    disagree = np.var(np.vstack(pred_logs), axis=0)
    picked = _top_diverse_indices(X, disagree, n_points, diversity_weight)
    return [candidates[i] for i in picked]


def _fits_to_payload(oracle: GRNBenchOracle, fits: list[GraphHypothesisFit]) -> list[dict]:
    payload = []
    for fit in fits:
        payload.append(
            {
                "edges": fit.graph["edges"],
                "summary": fit.summary,
                "train_rmsle": fit.train_rmsle,
                "reporter_rmsle": fit.reporter_rmsle,
                "state_rmsle": fit.state_rmsle,
                "score": fit.score,
                "complexity": fit.complexity,
                "graph_eval": oracle.evaluate_graph(fit.graph),
            }
        )
    return payload


class GraphDiscoveryLoop:
    def __init__(self, config: GraphDiscoveryConfig, oracle: GRNBenchOracle):
        self._cfg = config
        self._oracle = oracle
        self._rng = np.random.default_rng(config.seed)
        self._store = ExperimentStore(oracle.domain, GRN_INPUT_VARS)

    def _fit_graphs(self) -> list[GraphHypothesisFit]:
        return search_graph_hypotheses(
            self._store,
            beam_size=self._cfg.beam_size,
            max_edges=self._cfg.max_edges,
            top_k=self._cfg.top_k_graphs,
            restarts=self._cfg.restarts,
            max_indegree=self._cfg.max_indegree,
            per_depth_candidate_cap=self._cfg.per_depth_candidate_cap,
            state_weight=self._cfg.state_weight,
        )

    def run(self) -> GraphDiscoveryResult:
        out_dir = Path(self._cfg.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        history: list[dict] = []
        history_path = out_dir / "history.json"

        for p in _sample_points(min(self._cfg.init_points, self._cfg.budget), self._rng):
            self._store.add(self._oracle.run(p))
        print(
            f"[GRN-Graph] {self._cfg.domain_id} | init={len(self._store)} "
            f"budget={self._cfg.budget} noise={self._cfg.noise_level}"
        )

        iteration = 0
        last_fits: list[GraphHypothesisFit] = []
        while len(self._store) < self._cfg.budget:
            iteration += 1
            print(f"[GRN-Graph] iter={iteration} n={len(self._store)}/{self._cfg.budget}")
            fits: list[GraphHypothesisFit] = []
            if len(self._store) >= self._cfg.min_data_for_graph_fit:
                fits = self._fit_graphs()
                last_fits = fits

            if len(fits) >= 2:
                selected = _select_by_graph_disagreement(
                    fits=fits,
                    n_points=min(self._cfg.batch_size, self._cfg.budget - len(self._store)),
                    rng=self._rng,
                    candidate_pool_size=self._cfg.candidate_pool_size,
                    diversity_weight=self._cfg.diversity_weight,
                )
                selection_mode = "graph_disagreement"
            else:
                selected = _sample_points(
                    min(self._cfg.batch_size, self._cfg.budget - len(self._store)),
                    self._rng,
                )
                selection_mode = "random_warmup"

            if fits:
                best = fits[0]
                print(
                    f"[GRN-Graph] selection={selection_mode} "
                    f"best={best.summary} edges={best.graph['edges']}"
                )
            else:
                print(f"[GRN-Graph] selection={selection_mode} (no fitted graph yet)")

            for p in selected:
                if len(self._store) >= self._cfg.budget:
                    break
                self._store.add(self._oracle.run(p))

            hist_row = {
                "iteration": iteration,
                "n_experiments": len(self._store),
                "selection_mode": selection_mode,
            }
            if fits:
                hist_row["top_graphs"] = _fits_to_payload(self._oracle, fits)
                hist_row["best_graph_eval"] = self._oracle.evaluate_graph(fits[0].graph)
            history.append(hist_row)
            history_path.write_text(json.dumps(history, indent=2))

        if not last_fits:
            last_fits = self._fit_graphs()

        best_graph = last_fits[0].graph if last_fits else None
        final_graph_eval = self._oracle.evaluate_graph(best_graph) if best_graph else None

        self._store.save(out_dir / "experiments.json")
        history_path.write_text(json.dumps(history, indent=2))
        status = "completed"
        if final_graph_eval and final_graph_eval.get("exact_graph_accuracy") == 1.0:
            status = "exact_graph_recovered"
        summary = {
            "domain": self._cfg.domain_id,
            "difficulty": self._cfg.difficulty,
            "law_version": self._cfg.law_version,
            "budget": self._cfg.budget,
            "n_oracle_calls": len(self._store),
            "duration_s": time.time() - t0,
            "best_graph": best_graph,
            "final_graph_eval": final_graph_eval,
            "top_graphs": _fits_to_payload(self._oracle, last_fits),
            "status": status if final_graph_eval else "no_graph",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        return GraphDiscoveryResult(
            status=summary["status"],
            n_oracle_calls=len(self._store),
            duration_s=summary["duration_s"],
            best_graph=best_graph,
            final_graph_eval=final_graph_eval,
            history=history,
            result_dir=str(out_dir),
        )
