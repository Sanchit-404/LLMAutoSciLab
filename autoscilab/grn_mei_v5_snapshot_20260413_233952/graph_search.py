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


@dataclass
class GraphHypothesisLineage:
    hypothesis_id: str
    hypothesis_text: str
    exact_graph: dict[str, Any]
    exact_fit: GraphHypothesisFit
    best_fit: GraphHypothesisFit
    drift_steps: int
    drift_summary: str
    translation_rationale: str
    translation_assumptions: list[str]

    @property
    def selected_fit(self) -> GraphHypothesisFit:
        return self.best_fit


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


def canonicalize_edges(
    edges: list[tuple[str, str, int]] | list[dict[str, Any]],
) -> list[tuple[str, str, int]]:
    canonical: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if isinstance(edge, dict):
            src = str(edge["src"])
            dst = str(edge["dst"])
            sign = int(edge["sign"])
        else:
            src, dst, sign = edge
        if src not in _NODE_INDEX or dst not in _NODE_INDEX:
            raise ValueError(f"invalid edge nodes: {src}->{dst}")
        if src == dst:
            raise ValueError("self-loops are not allowed")
        if dst == "signal":
            raise ValueError("signal cannot be a destination node")
        key = (src, dst)
        if key in seen:
            raise ValueError(f"duplicate edge: {src}->{dst}")
        seen.add(key)
        canonical.append((src, dst, 1 if sign >= 0 else -1))
    canonical.sort(key=lambda item: (item[0], item[1], item[2]))
    return canonical


def edges_to_adjacency(edges: list[tuple[str, str, int]] | list[dict[str, Any]]) -> np.ndarray:
    adj = _empty_adjacency()
    for src, dst, sign in canonicalize_edges(edges):
        adj[_NODE_INDEX[src], _NODE_INDEX[dst]] = sign
    return adj


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


def validate_adjacency(adj: np.ndarray, max_edges: int | None = None) -> None:
    if adj.shape != (len(GRN_GRAPH_NODES), len(GRN_GRAPH_NODES)):
        raise ValueError("adjacency has wrong shape")
    if np.any(np.diag(adj) != 0):
        raise ValueError("self-loops are not allowed")
    if np.any(adj[:, _NODE_INDEX["signal"]] != 0):
        raise ValueError("signal cannot be a destination node")
    if max_edges is not None and int(np.count_nonzero(adj)) > max_edges:
        raise ValueError("graph exceeds max edge count")
    if not _has_path_to_reporter(adj):
        raise ValueError("graph must contain a path from signal to C")


def graph_edit_distance(adj_a: np.ndarray, adj_b: np.ndarray) -> int:
    return int(np.count_nonzero(adj_a != adj_b))


def summarize_graph_drift(adj_a: np.ndarray, adj_b: np.ndarray) -> str:
    changes: list[str] = []
    for src, dst in _EDGE_SLOTS:
        i = _NODE_INDEX[src]
        j = _NODE_INDEX[dst]
        old = int(np.sign(adj_a[i, j]))
        new = int(np.sign(adj_b[i, j]))
        if old == new:
            continue
        if old == 0 and new != 0:
            changes.append(f"add {src}->{dst}({new:+d})")
        elif old != 0 and new == 0:
            changes.append(f"drop {src}->{dst}({old:+d})")
        else:
            changes.append(f"flip {src}->{dst}({old:+d}->{new:+d})")
    return "; ".join(changes) if changes else "no structural drift"


def _node_indegree(adj: np.ndarray, node: str) -> int:
    j = _NODE_INDEX[node]
    return int(np.count_nonzero(adj[:, j]))


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
    x0.append(1.5)
    bounds.append((0.05, 6.0))
    x0.append(60.0)
    bounds.append((1.0, 200.0))
    return np.array(x0, dtype=float), bounds


def _predict_from_graph(adj: np.ndarray, params: dict[str, float], X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    signal = X[:, 0]
    pert = {"A": X[:, 1], "B": X[:, 2], "C": X[:, 3], "R": X[:, 4]}
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
                acc += sign * params[f"w_{src}_{dst}"] * prev[src]
            state[dst] = pert[dst] * _sigmoid(acc)
    reporter = np.maximum(params["scale"] * state["C"], 1e-6)
    return reporter, state


def _extract_targets(
    store: ExperimentStore,
    use_internal_state: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    X, _ = store.to_arrays()
    if len(X) == 0:
        return X, np.empty(0), None
    y_target = []
    for result in store.get_all():
        panel = result.metadata.get("marker_panel")
        if isinstance(panel, dict):
            y_target.append(float(panel.get("reporter_true", panel.get("reporter", result.measurement))))
        else:
            y_target.append(float(result.metadata.get("reporter_true", result.measurement)))
    if not use_internal_state:
        return X, np.array(y_target, dtype=float), None

    state_targets = {node: [] for node in _TARGET_NODES}
    has_states = True
    for result in store.get_all():
        panel = result.metadata.get("marker_panel")
        states = panel if isinstance(panel, dict) else result.metadata.get("states")
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
    state_losses = [
        _rmsle(np.maximum(states_hat[node], 1e-6), np.maximum(state_targets[node], 1e-6))
        for node in _TARGET_NODES
    ]
    return float((1.0 - state_weight) * reporter_loss + state_weight * float(np.mean(state_losses)))


def fit_graph_hypothesis(
    adj: np.ndarray,
    store: ExperimentStore,
    restarts: int = 3,
    complexity_penalty: float = 0.0,
    bic_penalty_scale: float = 0.0,
    state_weight: float = 0.5,
    use_internal_state: bool = False,
) -> GraphHypothesisFit:
    validate_adjacency(adj)
    X, y, state_targets = _extract_targets(store, use_internal_state=use_internal_state)
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
    bic_penalty = bic_penalty_scale * 0.5 * np.log(n_obs + 1.0) * complexity / max(n_obs, 1)
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


def search_local_graph_neighborhood(
    base_adj: np.ndarray,
    store: ExperimentStore,
    max_edges: int,
    restarts: int = 3,
    max_indegree: int = 3,
    max_edit_distance: int = 1,
    per_depth_candidate_cap: int | None = None,
    complexity_penalty: float = 0.0,
    bic_penalty_scale: float = 0.0,
    state_weight: float = 0.5,
    use_internal_state: bool = False,
) -> list[GraphHypothesisFit]:
    validate_adjacency(base_adj, max_edges=max_edges)
    seen: set[tuple[int, ...]] = {_canonical_key(base_adj)}
    fits_by_key: dict[tuple[int, ...], GraphHypothesisFit] = {}
    frontier = [np.array(base_adj, dtype=int)]
    fits_by_key[_canonical_key(base_adj)] = fit_graph_hypothesis(
        base_adj,
        store,
        restarts=restarts,
        complexity_penalty=complexity_penalty,
        bic_penalty_scale=bic_penalty_scale,
        state_weight=state_weight,
        use_internal_state=use_internal_state,
    )
    for _depth in range(1, max_edit_distance + 1):
        next_frontier: list[np.ndarray] = []
        for adj in frontier:
            for nxt in _neighbor_graphs(adj, max_edges=max_edges, max_indegree=max_indegree):
                key = _canonical_key(nxt)
                if key in seen:
                    continue
                seen.add(key)
                next_frontier.append(nxt)
                if per_depth_candidate_cap is not None and len(next_frontier) >= per_depth_candidate_cap:
                    break
            if per_depth_candidate_cap is not None and len(next_frontier) >= per_depth_candidate_cap:
                break
        if not next_frontier:
            break
        for adj in next_frontier:
            key = _canonical_key(adj)
            if key in fits_by_key:
                continue
            fits_by_key[key] = fit_graph_hypothesis(
                adj,
                store,
                restarts=restarts,
                complexity_penalty=complexity_penalty,
                bic_penalty_scale=bic_penalty_scale,
                state_weight=state_weight,
                use_internal_state=use_internal_state,
            )
        frontier = next_frontier
    fits = list(fits_by_key.values())
    fits.sort(key=lambda fit: (fit.score, fit.train_rmsle))
    return fits


def build_hypothesis_lineages(
    store: ExperimentStore,
    hypotheses: list[dict[str, Any]],
    restarts: int = 3,
    max_edges: int = 6,
    max_indegree: int = 3,
    max_edit_distance: int = 1,
    per_depth_candidate_cap: int | None = None,
    complexity_penalty: float = 0.0,
    bic_penalty_scale: float = 0.0,
    state_weight: float = 0.5,
    use_internal_state: bool = False,
) -> list[GraphHypothesisLineage]:
    lineages: list[GraphHypothesisLineage] = []
    for hypothesis in hypotheses:
        adj = edges_to_adjacency(hypothesis["edges"])
        validate_adjacency(adj, max_edges=max_edges)
        exact_fit = fit_graph_hypothesis(
            adj,
            store,
            restarts=restarts,
            complexity_penalty=complexity_penalty,
            bic_penalty_scale=bic_penalty_scale,
            state_weight=state_weight,
            use_internal_state=use_internal_state,
        )
        local_fits = search_local_graph_neighborhood(
            adj,
            store,
            max_edges=max_edges,
            restarts=restarts,
            max_indegree=max_indegree,
            max_edit_distance=max_edit_distance,
            per_depth_candidate_cap=per_depth_candidate_cap,
            complexity_penalty=complexity_penalty,
            bic_penalty_scale=bic_penalty_scale,
            state_weight=state_weight,
            use_internal_state=use_internal_state,
        )
        best_fit = local_fits[0] if local_fits else exact_fit
        drift_steps = graph_edit_distance(adj, best_fit.adjacency)
        lineages.append(
            GraphHypothesisLineage(
                hypothesis_id=str(hypothesis["hypothesis_id"]),
                hypothesis_text=str(hypothesis["hypothesis_text"]),
                exact_graph=exact_fit.graph,
                exact_fit=exact_fit,
                best_fit=best_fit,
                drift_steps=drift_steps,
                drift_summary=summarize_graph_drift(adj, best_fit.adjacency),
                translation_rationale=str(hypothesis.get("translation_rationale", "")),
                translation_assumptions=[str(item) for item in hypothesis.get("translation_assumptions", [])],
            )
        )
    lineages.sort(key=lambda item: (item.best_fit.score, item.exact_fit.score))
    return lineages


def select_committee_from_lineages(
    lineages: list[GraphHypothesisLineage],
    top_k: int,
) -> list[GraphHypothesisFit]:
    committee: list[GraphHypothesisFit] = []
    seen: set[tuple[int, ...]] = set()
    for lineage in lineages:
        for fit in (lineage.exact_fit, lineage.best_fit):
            key = _canonical_key(fit.adjacency)
            if key in seen:
                continue
            seen.add(key)
            committee.append(fit)
            if len(committee) >= top_k:
                return committee
    return committee
