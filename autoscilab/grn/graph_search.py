from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from autoscilab.data.store import ExperimentStore
from autoscilab.oracle.grnbench import GRN_GRAPH_NODES

_TARGET_NODES = ["A", "B", "C", "R"]
_SOURCE_NODES = ["signal", "A", "B", "C", "R"]
_EDGE_SLOTS: list[tuple[str, str]] = [
    (src, dst)
    for dst in _TARGET_NODES
    for src in _SOURCE_NODES
    if src != dst
]
_EDGE_INDEX = {edge: i for i, edge in enumerate(_EDGE_SLOTS)}
_NODE_INDEX = {n: i for i, n in enumerate(GRN_GRAPH_NODES)}


@dataclass
class GraphHypothesisFit:
    adjacency: np.ndarray
    params: dict[str, float]
    train_rmsle: float
    reporter_rmsle: float
    state_rmsle: float
    complexity: int
    score: float

    @property
    def graph(self) -> dict[str, Any]:
        return {
            "adjacency": self.adjacency.astype(int).tolist(),
            "edges": adjacency_to_edges(self.adjacency),
        }

    @property
    def summary(self) -> str:
        return (
            f"edges={self.complexity} "
            f"train_rmsle={self.train_rmsle:.4f} "
            f"reporter_rmsle={self.reporter_rmsle:.4f} "
            f"state_rmsle={self.state_rmsle:.4f} "
            f"score={self.score:.4f}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _predict_from_graph(self.adjacency, self.params, X)[0]

    def predict_states(self, X: np.ndarray) -> dict[str, np.ndarray]:
        _, states = _predict_from_graph(self.adjacency, self.params, X)
        return states


def adjacency_to_edges(adj: np.ndarray) -> list[tuple[str, str, int]]:
    edges: list[tuple[str, str, int]] = []
    for i, src in enumerate(GRN_GRAPH_NODES):
        for j, dst in enumerate(GRN_GRAPH_NODES):
            sign = int(np.sign(adj[i, j]))
            if sign != 0:
                edges.append((src, dst, sign))
    return edges


def _empty_adjacency() -> np.ndarray:
    return np.zeros((len(GRN_GRAPH_NODES), len(GRN_GRAPH_NODES)), dtype=int)


def _canonical_key(adj: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in adj.reshape(-1))


def _has_path_to_reporter(adj: np.ndarray) -> bool:
    start = _NODE_INDEX["signal"]
    goal = _NODE_INDEX["C"]
    frontier = [start]
    seen = {start}
    while frontier:
        cur = frontier.pop()
        if cur == goal:
            return True
        for nxt in range(adj.shape[1]):
            if adj[cur, nxt] == 0 or nxt in seen:
                continue
            seen.add(nxt)
            frontier.append(nxt)
    return False


def _node_indegree(adj: np.ndarray, node: str) -> int:
    j = _NODE_INDEX[node]
    return int(np.count_nonzero(adj[:, j]))


def _graph_complexity(adj: np.ndarray) -> int:
    return int(np.count_nonzero(adj))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _params_from_vector(adj: np.ndarray, vec: np.ndarray) -> dict[str, float]:
    params: dict[str, float] = {}
    idx = 0
    for src, dst in _EDGE_SLOTS:
        sign = int(np.sign(adj[_NODE_INDEX[src], _NODE_INDEX[dst]]))
        if sign != 0:
            params[f"w_{src}_{dst}"] = float(max(vec[idx], 0.0))
            idx += 1
    for node in _TARGET_NODES:
        params[f"bias_{node}"] = float(vec[idx])
        idx += 1
    params["signal_gain"] = float(max(vec[idx], 0.01))
    idx += 1
    params["scale"] = float(max(vec[idx], 1e-3))
    return params


def _initial_vector(adj: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    x0: list[float] = []
    bounds: list[tuple[float, float]] = []
    for src, dst in _EDGE_SLOTS:
        sign = int(np.sign(adj[_NODE_INDEX[src], _NODE_INDEX[dst]]))
        if sign != 0:
            x0.append(1.0)
            bounds.append((0.01, 4.0))
    for _node in _TARGET_NODES:
        x0.append(-0.25)
        bounds.append((-4.0, 4.0))
    x0.append(1.5)   # signal gain
    bounds.append((0.05, 6.0))
    x0.append(60.0)  # scale
    bounds.append((1.0, 200.0))
    return np.array(x0, dtype=float), bounds


def _predict_from_graph(adj: np.ndarray, params: dict[str, float], X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    signal = X[:, 0]
    pert = {
        "A": X[:, 1],
        "B": X[:, 2],
        "C": X[:, 3],
        "R": X[:, 4],
    }
    n = len(X)
    state = {
        "signal": _sigmoid(params["signal_gain"] * np.log1p(np.maximum(signal, 1e-9))),
        "A": np.full(n, 0.1, dtype=float),
        "B": np.full(n, 0.1, dtype=float),
        "C": np.full(n, 0.1, dtype=float),
        "R": np.full(n, 0.1, dtype=float),
    }

    for _ in range(12):
        prev = {k: v.copy() for k, v in state.items()}
        for dst in _TARGET_NODES:
            acc = np.full(n, params[f"bias_{dst}"], dtype=float)
            for src in _SOURCE_NODES:
                sign = int(np.sign(adj[_NODE_INDEX[src], _NODE_INDEX[dst]]))
                if sign == 0:
                    continue
                weight = params[f"w_{src}_{dst}"]
                acc += sign * weight * prev[src]
            state[dst] = pert[dst] * _sigmoid(acc)
    reporter = np.maximum(params["scale"] * state["C"], 1e-6)
    return reporter, state


def _extract_targets(
    store: ExperimentStore,
    *,
    use_oracle_state_targets: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    X, y_obs = store.to_arrays()
    if len(X) == 0:
        return X, y_obs, None
    if not use_oracle_state_targets:
        return X, y_obs, None
    y_target = []
    state_targets = {node: [] for node in _TARGET_NODES}
    has_states = True
    for result in store.get_all():
        y_target.append(float(result.metadata.get("reporter_true", result.measurement)))
        states = result.metadata.get("states")
        if not isinstance(states, dict):
            has_states = False
            continue
        for node in _TARGET_NODES:
            if node not in states:
                has_states = False
                break
            state_targets[node].append(float(states[node]))
    if not has_states:
        return X, np.array(y_target, dtype=float), None
    return X, np.array(y_target, dtype=float), {
        node: np.array(vals, dtype=float) for node, vals in state_targets.items()
    }


def _rmsle(y_hat: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(y_hat) & np.isfinite(y) & (y_hat > 0) & (y > 0)
    if not np.any(mask):
        return 1e6
    return float(np.sqrt(np.mean((np.log1p(y_hat[mask]) - np.log1p(y[mask])) ** 2)))


def _loss(
    vec: np.ndarray,
    adj: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    state_targets: dict[str, np.ndarray] | None,
    state_weight: float,
) -> float:
    params = _params_from_vector(adj, vec)
    y_hat, states_hat = _predict_from_graph(adj, params, X)
    reporter_loss = _rmsle(y_hat, y)
    if state_targets is None:
        return reporter_loss
    state_losses = [_rmsle(np.maximum(states_hat[node], 1e-6), np.maximum(state_targets[node], 1e-6)) for node in _TARGET_NODES]
    return float((1.0 - state_weight) * reporter_loss + state_weight * float(np.mean(state_losses)))


def fit_graph_hypothesis(
    adj: np.ndarray,
    store: ExperimentStore,
    restarts: int = 3,
    complexity_penalty: float = 0.0,
    state_weight: float = 0.5,
    use_oracle_state_targets: bool = True,
) -> GraphHypothesisFit:
    X, y, state_targets = _extract_targets(
        store,
        use_oracle_state_targets=use_oracle_state_targets,
    )
    if len(X) == 0:
        raise ValueError("Cannot fit graph hypothesis with no data.")

    x0, bounds = _initial_vector(adj)
    best_x = x0.copy()
    best_loss = _loss(best_x, adj, X, y, state_targets, state_weight)
    rng = np.random.default_rng(42)

    for _ in range(restarts):
        start = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)
        res = minimize(
            _loss,
            start,
            args=(adj, X, y, state_targets, state_weight),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 60},
        )
        cand_x = np.array(res.x if res.success else start, dtype=float)
        cand_loss = _loss(cand_x, adj, X, y, state_targets, state_weight)
        if cand_loss < best_loss:
            best_x = cand_x
            best_loss = cand_loss

    complexity = int(np.count_nonzero(adj))
    params = _params_from_vector(adj, best_x)
    y_hat, states_hat = _predict_from_graph(adj, params, X)
    reporter_rmsle = _rmsle(y_hat, y)
    if state_targets is None:
        state_rmsle = float("nan")
    else:
        state_rmsle = float(np.mean([
            _rmsle(np.maximum(states_hat[node], 1e-6), np.maximum(state_targets[node], 1e-6))
            for node in _TARGET_NODES
        ]))
    n_obs = max(len(X), 1)
    bic_penalty = 0.5 * np.log(n_obs + 1.0) * complexity / max(n_obs, 1)
    score = float(best_loss + bic_penalty + complexity_penalty * complexity)
    return GraphHypothesisFit(
        adjacency=np.array(adj, dtype=int),
        params=params,
        train_rmsle=float(best_loss),
        reporter_rmsle=float(reporter_rmsle),
        state_rmsle=float(state_rmsle),
        complexity=complexity,
        score=score,
    )


def _neighbor_graphs(adj: np.ndarray, max_edges: int, max_indegree: int = 3) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    current_edges = int(np.count_nonzero(adj))
    for src, dst in _EDGE_SLOTS:
        i = _NODE_INDEX[src]
        j = _NODE_INDEX[dst]
        cur = int(adj[i, j])
        if cur == 0:
            if current_edges < max_edges and _node_indegree(adj, dst) < max_indegree:
                for sign in (-1, 1):
                    nxt = adj.copy()
                    nxt[i, j] = sign
                    if _has_path_to_reporter(nxt):
                        out.append(nxt)
        else:
            rem = adj.copy()
            rem[i, j] = 0
            if _has_path_to_reporter(rem):
                out.append(rem)
            flip = adj.copy()
            flip[i, j] = -cur
            if _has_path_to_reporter(flip):
                out.append(flip)
    return out


def search_graph_hypotheses(
    store: ExperimentStore,
    beam_size: int = 8,
    max_edges: int = 6,
    top_k: int = 4,
    restarts: int = 3,
    max_indegree: int = 3,
    per_depth_candidate_cap: int | None = None,
    state_weight: float = 0.5,
    use_oracle_state_targets: bool = True,
) -> list[GraphHypothesisFit]:
    beam: list[np.ndarray] = []
    for src, dst in _EDGE_SLOTS:
        if src == "signal":
            for sign in (1, -1):
                adj = _empty_adjacency()
                adj[_NODE_INDEX[src], _NODE_INDEX[dst]] = sign
                beam.append(adj)

    seen: set[tuple[int, ...]] = set()
    best_by_key: dict[tuple[int, ...], GraphHypothesisFit] = {}

    # Evaluate initial beam
    scored0: list[GraphHypothesisFit] = []
    for adj in beam:
        key = _canonical_key(adj)
        if key in seen:
            continue
        seen.add(key)
        fit = fit_graph_hypothesis(
            adj,
            store,
            restarts=restarts,
            state_weight=state_weight,
            use_oracle_state_targets=use_oracle_state_targets,
        )
        best_by_key[key] = fit
        scored0.append(fit)
    scored0.sort(key=lambda f: (f.score, f.train_rmsle))
    beam = [f.adjacency for f in scored0[:beam_size]]

    for _depth in range(max_edges):
        candidates: list[np.ndarray] = []
        for adj in beam:
            for nxt in _neighbor_graphs(adj, max_edges=max_edges, max_indegree=max_indegree):
                key = _canonical_key(nxt)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(nxt)
                if per_depth_candidate_cap is not None and len(candidates) >= per_depth_candidate_cap:
                    break
            if per_depth_candidate_cap is not None and len(candidates) >= per_depth_candidate_cap:
                break

        if not candidates:
            break

        scored: list[GraphHypothesisFit] = []
        for adj in candidates:
            fit = fit_graph_hypothesis(
                adj,
                store,
                restarts=restarts,
                state_weight=state_weight,
                use_oracle_state_targets=use_oracle_state_targets,
            )
            scored.append(fit)
            best_by_key[_canonical_key(adj)] = fit
        scored.sort(key=lambda f: (f.score, f.train_rmsle))
        beam = [f.adjacency for f in scored[:beam_size]]

    all_fits = list(best_by_key.values())
    all_fits.sort(key=lambda f: (f.score, f.train_rmsle))
    dedup: list[GraphHypothesisFit] = []
    used: set[tuple[int, ...]] = set()
    for fit in all_fits:
        key = _canonical_key(fit.adjacency)
        if key in used:
            continue
        used.add(key)
        dedup.append(fit)
        if len(dedup) >= top_k:
            break
    return dedup
