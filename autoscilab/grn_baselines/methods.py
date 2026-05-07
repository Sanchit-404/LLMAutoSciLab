from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from autoscilab.al.gp_model import GPSurrogate
from autoscilab.al.selector import ALSelector
from autoscilab.data.store import ExperimentStore
from autoscilab.grn.graph_search import search_graph_hypotheses
from autoscilab.grn_mei_v5.loop import GRNMEIGraphConfig, GRNMEIGraphLoop
from autoscilab.oracle.grnbench import GRNBenchOracle, GRN_GRAPH_NODES, GRN_INPUT_BOUNDS, GRN_INPUT_VARS

STATE_NODES = ["A", "B", "C", "R"]
ALL_GRAPH_NODES = list(GRN_GRAPH_NODES)
NODE_INDEX = {node: i for i, node in enumerate(ALL_GRAPH_NODES)}
ONLINE_METHODS = ["random_graphfit", "uncertainty_graphfit", "llm_mei_graphfit"]
OFFLINE_METHODS = ["genie3", "gies", "notears"]
ALL_METHODS = ONLINE_METHODS + OFFLINE_METHODS


@dataclass
class BaselineResult:
    method: str
    example_id: str
    status: str
    duration_s: float
    n_experiments: int
    best_graph: dict[str, Any] | None
    graph_eval: dict[str, Any] | None
    extras: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "example_id": self.example_id,
            "status": self.status,
            "duration_s": self.duration_s,
            "n_experiments": self.n_experiments,
            "best_graph": self.best_graph,
            "graph_eval": self.graph_eval,
            **self.extras,
        }


def _lhs_points(n: int, rng: np.random.Generator) -> list[dict[str, float]]:
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


def _store_from_random_design(oracle: GRNBenchOracle, budget: int, seed: int) -> ExperimentStore:
    store = ExperimentStore(oracle.domain, GRN_INPUT_VARS)
    rng = np.random.default_rng(seed)
    for p in _lhs_points(budget, rng):
        store.add(oracle.run(p))
    return store


def _store_from_saved_experiments(path: Path) -> ExperimentStore:
    payload = json.loads(Path(path).read_text())
    store = ExperimentStore(payload["domain"], payload["parameter_names"])
    from autoscilab.oracle.base import OracleResult

    for row in payload["results"]:
        store.add(
            OracleResult(
                params=row["params"],
                measurement=row["measurement"],
                domain=payload["domain"],
                noise_level=row["noise_level"],
                metadata=row.get("metadata", {}),
            )
        )
    return store


def _graph_from_adjacency(adj: np.ndarray) -> dict[str, Any]:
    edges: list[tuple[str, str, int]] = []
    for i, src in enumerate(ALL_GRAPH_NODES):
        for j, dst in enumerate(ALL_GRAPH_NODES):
            sign = int(np.sign(adj[i, j]))
            if sign != 0:
                edges.append((src, dst, sign))
    return {
        "adjacency": adj.astype(int).tolist(),
        "edges": edges,
    }


def _evaluate_graph(oracle: GRNBenchOracle, graph: dict[str, Any] | None) -> dict[str, Any] | None:
    if graph is None:
        return None
    return oracle.evaluate_graph(graph)


def _fit_graph_from_store(oracle: GRNBenchOracle, store: ExperimentStore) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fits = search_graph_hypotheses(
        store,
        beam_size=8,
        max_edges=6,
        top_k=5,
        restarts=3,
        max_indegree=3,
        per_depth_candidate_cap=48,
        state_weight=0.5,
    )
    best = fits[0]
    best_graph = best.graph
    top_graphs = []
    for fit in fits:
        top_graphs.append(
            {
                "graph": fit.graph,
                "summary": fit.summary,
                "train_rmsle": fit.train_rmsle,
                "graph_eval": oracle.evaluate_graph(fit.graph),
            }
        )
    return best_graph, top_graphs


def run_random_graphfit(
    example_id: str,
    oracle: GRNBenchOracle,
    budget: int,
    seed: int,
) -> BaselineResult:
    t0 = time.time()
    store = _store_from_random_design(oracle, budget, seed)
    best_graph, top_graphs = _fit_graph_from_store(oracle, store)
    return BaselineResult(
        method="random_graphfit",
        example_id=example_id,
        status="completed",
        duration_s=time.time() - t0,
        n_experiments=len(store),
        best_graph=best_graph,
        graph_eval=_evaluate_graph(oracle, best_graph),
        extras={"top_graphs": top_graphs, "design": "random_lhs", "observation_mode": "marker_panel"},
    )


def run_uncertainty_graphfit(
    example_id: str,
    oracle: GRNBenchOracle,
    budget: int,
    seed: int,
) -> BaselineResult:
    t0 = time.time()
    store = ExperimentStore(oracle.domain, GRN_INPUT_VARS)
    gp = GPSurrogate()
    selector = ALSelector(oracle=oracle, gp=gp)
    rng = np.random.default_rng(seed)
    init_n = min(5, budget)
    for p in _lhs_points(init_n, rng):
        store.add(oracle.run(p))
    full_region = {k: [float(v[0]), float(v[1])] for k, v in GRN_INPUT_BOUNDS.items()}
    while len(store) < budget:
        batch = min(4, budget - len(store))
        selected = selector.select(
            store=store,
            region_bounds=full_region,
            n_points=batch,
            acquisition="variance",
        )
        for p in selected:
            if len(store) >= budget:
                break
            store.add(oracle.run(p))
    best_graph, top_graphs = _fit_graph_from_store(oracle, store)
    return BaselineResult(
        method="uncertainty_graphfit",
        example_id=example_id,
        status="completed",
        duration_s=time.time() - t0,
        n_experiments=len(store),
        best_graph=best_graph,
        graph_eval=_evaluate_graph(oracle, best_graph),
        extras={"top_graphs": top_graphs, "design": "uncertainty_al", "observation_mode": "marker_panel"},
    )


def run_llm_mei_graphfit(
    example_id: str,
    oracle: GRNBenchOracle,
    budget: int,
    seed: int,
    llm_model: str,
    base_url: str | None,
    out_dir: Path,
) -> BaselineResult:
    t0 = time.time()
    cfg = GRNMEIGraphConfig(
        domain_id=oracle.domain,
        difficulty=oracle._difficulty,  # existing oracle instance already encapsulates benchmark choice
        law_version=oracle._law_version,
        noise_level=oracle._noise_level,
        budget=budget,
        llm_model=llm_model,
        results_dir=out_dir,
        seed=seed,
        use_internal_state=True,
        state_weight=0.5,
    )
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TOGETHER_API_KEY")
    result = GRNMEIGraphLoop(cfg, api_key=api_key, oracle=oracle, base_url=base_url).run()
    return BaselineResult(
        method="llm_mei_graphfit",
        example_id=example_id,
        status=result.status or "completed",
        duration_s=result.duration_seconds,
        n_experiments=result.n_oracle_calls,
        best_graph=result.best_graph,
        graph_eval=result.final_evaluation,
        extras={
            "n_llm_calls": result.n_llm_calls,
            "history_len": len(result.history),
            "observation_mode": "marker_panel",
        },
    )


def _state_panel(store: ExperimentStore) -> tuple[np.ndarray, list[str]]:
    rows = []
    for result in store.get_all():
        states = result.metadata.get("states")
        if not isinstance(states, dict):
            raise ValueError("Offline graph baselines require oracle state metadata.")
        rows.append([
            float(result.params["signal"]),
            float(states["A"]),
            float(states["B"]),
            float(states["C"]),
            float(states["R"]),
        ])
    return np.array(rows, dtype=float), ALL_GRAPH_NODES


def _coef_sign(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if X.shape[1] == 0:
        return np.empty(0, dtype=float)
    model = Ridge(alpha=1e-3)
    model.fit(X, y)
    coef = np.array(model.coef_, dtype=float)
    if np.allclose(coef, 0.0):
        lr = LinearRegression()
        lr.fit(X, y)
        coef = np.array(lr.coef_, dtype=float)
    return coef


def _threshold_topk(weights: np.ndarray, max_parents: int = 3, min_weight: float = 1e-6) -> np.ndarray:
    out = np.zeros_like(weights)
    for j in range(weights.shape[1]):
        col = np.abs(weights[:, j])
        order = np.argsort(col)[::-1]
        kept = 0
        for i in order:
            if i == j or col[i] <= min_weight:
                continue
            out[i, j] = weights[i, j]
            kept += 1
            if kept >= max_parents:
                break
    return out


def _collect_offline_dataset(
    oracle: GRNBenchOracle,
    budget: int,
    seed: int,
    dataset_source: str = "random",
    dataset_path: Path | None = None,
) -> ExperimentStore:
    if dataset_source == "random":
        return _store_from_random_design(oracle, budget, seed)
    if dataset_source == "mei":
        if dataset_path is None:
            raise ValueError("dataset_path is required for dataset_source='mei'")
        store = _store_from_saved_experiments(dataset_path)
        if len(store) != budget:
            raise ValueError(f"Expected {budget} experiments in {dataset_path}, got {len(store)}")
        return store
    raise ValueError(f"Unknown dataset_source: {dataset_source}")


def run_genie3(
    example_id: str,
    oracle: GRNBenchOracle,
    budget: int,
    seed: int,
    dataset_source: str = "random",
    dataset_path: Path | None = None,
) -> BaselineResult:
    t0 = time.time()
    store = _collect_offline_dataset(oracle, budget, seed, dataset_source=dataset_source, dataset_path=dataset_path)
    Xraw, columns = _state_panel(store)
    scaler = StandardScaler()
    X = scaler.fit_transform(Xraw)
    d = X.shape[1]
    weights = np.zeros((d, d), dtype=float)
    for j, dst in enumerate(columns):
        if dst == "signal":
            continue
        feat_idx = [i for i in range(d) if i != j]
        model = ExtraTreesRegressor(n_estimators=400, random_state=seed)
        model.fit(X[:, feat_idx], X[:, j])
        importances = np.array(model.feature_importances_, dtype=float)
        coef = _coef_sign(X[:, feat_idx], X[:, j])
        for local_idx, i in enumerate(feat_idx):
            sign = 1.0 if coef[local_idx] >= 0 else -1.0
            weights[i, j] = sign * importances[local_idx]
    weights[:, NODE_INDEX["signal"]] = 0.0
    np.fill_diagonal(weights, 0.0)
    sparse = _threshold_topk(weights, max_parents=3, min_weight=1e-3)
    graph = _graph_from_adjacency(np.sign(sparse).astype(int))
    return BaselineResult(
        method="genie3",
        example_id=example_id,
        status="completed",
        duration_s=time.time() - t0,
        n_experiments=len(store),
        best_graph=graph,
        graph_eval=_evaluate_graph(oracle, graph),
        extras={"design": dataset_source, "observation_mode": "marker_panel", "implementation": "internal_genie3_like"},
    )


def _bic_local_score(X: np.ndarray, target_idx: int, parent_indices: list[int]) -> float:
    y = X[:, target_idx]
    n = len(y)
    if n == 0:
        return float("inf")
    if not parent_indices:
        resid = y - np.mean(y)
        mse = float(np.mean(resid * resid)) + 1e-9
        return float(n * np.log(mse))
    reg = LinearRegression()
    reg.fit(X[:, parent_indices], y)
    pred = reg.predict(X[:, parent_indices])
    resid = y - pred
    mse = float(np.mean(resid * resid)) + 1e-9
    k = len(parent_indices) + 1
    return float(n * np.log(mse) + k * np.log(n + 1.0))


def _dag_score(X: np.ndarray, adj: np.ndarray) -> float:
    score = 0.0
    for j, node in enumerate(ALL_GRAPH_NODES):
        if node == "signal":
            continue
        parents = [i for i in range(adj.shape[0]) if adj[i, j] != 0]
        score += _bic_local_score(X, j, parents)
    return float(score)


def _is_acyclic(adj: np.ndarray) -> bool:
    g = nx.DiGraph()
    g.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] != 0:
                g.add_edge(i, j)
    return nx.is_directed_acyclic_graph(g)


def _linear_parent_signs(X: np.ndarray, adj: np.ndarray) -> np.ndarray:
    signed = np.zeros_like(adj, dtype=int)
    for j, node in enumerate(ALL_GRAPH_NODES):
        if node == "signal":
            continue
        parents = [i for i in range(adj.shape[0]) if adj[i, j] != 0]
        if not parents:
            continue
        reg = LinearRegression()
        reg.fit(X[:, parents], X[:, j])
        for coef, i in zip(reg.coef_, parents, strict=False):
            signed[i, j] = 1 if coef >= 0 else -1
    return signed


def run_gies(
    example_id: str,
    oracle: GRNBenchOracle,
    budget: int,
    seed: int,
    dataset_source: str = "random",
    dataset_path: Path | None = None,
) -> BaselineResult:
    t0 = time.time()
    store = _collect_offline_dataset(oracle, budget, seed, dataset_source=dataset_source, dataset_path=dataset_path)
    Xraw, _ = _state_panel(store)
    X = StandardScaler().fit_transform(Xraw)
    d = X.shape[1]
    adj = np.zeros((d, d), dtype=int)
    best_score = _dag_score(X, adj)
    improved = True
    while improved:
        improved = False
        best_move: tuple[str, int, int] | None = None
        best_adj = adj.copy()
        for i in range(d):
            for j in range(d):
                if i == j or j == NODE_INDEX["signal"]:
                    continue
                if adj[i, j] == 0:
                    cand = adj.copy()
                    cand[i, j] = 1
                    if not _is_acyclic(cand):
                        continue
                    score = _dag_score(X, cand)
                    if score < best_score:
                        best_score = score
                        best_adj = cand
                        best_move = ("add", i, j)
                        improved = True
                else:
                    cand = adj.copy()
                    cand[i, j] = 0
                    score = _dag_score(X, cand)
                    if score < best_score:
                        best_score = score
                        best_adj = cand
                        best_move = ("drop", i, j)
                        improved = True
                    if i != NODE_INDEX["signal"] and adj[j, i] == 0:
                        rev = adj.copy()
                        rev[i, j] = 0
                        rev[j, i] = 1
                        if j == NODE_INDEX["signal"] or not _is_acyclic(rev):
                            continue
                        score = _dag_score(X, rev)
                        if score < best_score:
                            best_score = score
                            best_adj = rev
                            best_move = ("flip", i, j)
                            improved = True
        adj = best_adj
        if best_move is None:
            break
    signed = _linear_parent_signs(X, adj)
    graph = _graph_from_adjacency(signed)
    return BaselineResult(
        method="gies",
        example_id=example_id,
        status="completed",
        duration_s=time.time() - t0,
        n_experiments=len(store),
        best_graph=graph,
        graph_eval=_evaluate_graph(oracle, graph),
        extras={
            "design": dataset_source,
            "observation_mode": "marker_panel",
            "implementation": "internal_gies_like",
            "score": best_score,
        },
    )


def _notears_objective_factory(X: np.ndarray, l1: float, penalty: float):
    n, d = X.shape
    mask = np.ones((d, d), dtype=float)
    np.fill_diagonal(mask, 0.0)
    mask[:, NODE_INDEX["signal"]] = 0.0

    def unpack(vec: np.ndarray) -> np.ndarray:
        W = vec.reshape(d, d) * mask
        np.fill_diagonal(W, 0.0)
        W[:, NODE_INDEX["signal"]] = 0.0
        return W

    def obj(vec: np.ndarray) -> float:
        W = unpack(vec)
        resid = X - X @ W
        loss = 0.5 / n * float(np.sum(resid * resid))
        h = float(np.trace(expm(W * W)) - d)
        return loss + l1 * float(np.abs(W).sum()) + penalty * h * h

    return obj, unpack


def run_notears(
    example_id: str,
    oracle: GRNBenchOracle,
    budget: int,
    seed: int,
    dataset_source: str = "random",
    dataset_path: Path | None = None,
) -> BaselineResult:
    t0 = time.time()
    store = _collect_offline_dataset(oracle, budget, seed, dataset_source=dataset_source, dataset_path=dataset_path)
    Xraw, _ = _state_panel(store)
    X = StandardScaler().fit_transform(Xraw)
    d = X.shape[1]
    obj, unpack = _notears_objective_factory(X, l1=0.02, penalty=10.0)
    x0 = np.zeros(d * d, dtype=float)
    res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": 200})
    W = unpack(np.array(res.x if res.success else x0, dtype=float))
    thresh = np.where(np.abs(W) >= 0.08, W, 0.0)
    graph = _graph_from_adjacency(np.sign(thresh).astype(int))
    return BaselineResult(
        method="notears",
        example_id=example_id,
        status="completed",
        duration_s=time.time() - t0,
        n_experiments=len(store),
        best_graph=graph,
        graph_eval=_evaluate_graph(oracle, graph),
        extras={
            "design": dataset_source,
            "observation_mode": "marker_panel",
            "implementation": "internal_linear_notears_like",
            "objective": float(obj(res.x if res.success else x0)),
        },
    )


def run_method(
    method: str,
    example_id: str,
    oracle: GRNBenchOracle,
    budget: int,
    seed: int,
    llm_model: str = "gpt-4o-mini",
    base_url: str | None = None,
    out_dir: Path | None = None,
    dataset_source: str = "random",
    dataset_path: Path | None = None,
) -> BaselineResult:
    if method == "random_graphfit":
        return run_random_graphfit(example_id, oracle, budget, seed)
    if method == "uncertainty_graphfit":
        return run_uncertainty_graphfit(example_id, oracle, budget, seed)
    if method == "llm_mei_graphfit":
        if out_dir is None:
            raise ValueError("llm_mei_graphfit requires an output directory")
        return run_llm_mei_graphfit(example_id, oracle, budget, seed, llm_model, base_url, out_dir)
    if method == "genie3":
        return run_genie3(example_id, oracle, budget, seed, dataset_source=dataset_source, dataset_path=dataset_path)
    if method == "gies":
        return run_gies(example_id, oracle, budget, seed, dataset_source=dataset_source, dataset_path=dataset_path)
    if method == "notears":
        return run_notears(example_id, oracle, budget, seed, dataset_source=dataset_source, dataset_path=dataset_path)
    raise ValueError(f"Unknown method: {method}")
