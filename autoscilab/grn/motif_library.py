from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from autoscilab.data.store import ExperimentStore

MOTIF_LIBRARY = [
    "activation_chain",
    "coherent_feedforward",
    "incoherent_feedforward",
    "negative_feedback",
    "toggle_switch",
]

_PARAM_SPECS: dict[str, list[str]] = {
    "activation_chain": ["ks", "kab", "kbc", "alpha_a", "alpha_b", "alpha_c", "scale"],
    "coherent_feedforward": ["ks", "kab", "kac", "kbc", "alpha_a", "alpha_b", "alpha_c", "scale"],
    "incoherent_feedforward": ["ks", "kab", "kac", "kbc", "alpha_a", "alpha_b", "alpha_c", "scale"],
    "negative_feedback": ["ks", "kab", "kbc", "kcr", "krc", "alpha_a", "alpha_b", "alpha_c", "alpha_r", "scale"],
    "toggle_switch": ["ks", "kab", "kac", "kbc", "kar", "kra", "alpha_a", "alpha_b", "alpha_c", "alpha_r", "scale"],
}

_DEFAULTS = {
    "ks": 1.0,
    "kab": 0.7,
    "kac": 0.7,
    "kbc": 0.7,
    "kcr": 0.5,
    "krc": 0.5,
    "kar": 0.4,
    "kra": 0.4,
    "alpha_a": 1.0,
    "alpha_b": 1.0,
    "alpha_c": 1.0,
    "alpha_r": 1.0,
    "scale": 75.0,
}

_BND = {
    "ks": (0.05, 5.0),
    "kab": (0.05, 5.0),
    "kac": (0.05, 5.0),
    "kbc": (0.05, 5.0),
    "kcr": (0.05, 5.0),
    "krc": (0.05, 5.0),
    "kar": (0.05, 5.0),
    "kra": (0.05, 5.0),
    "alpha_a": (0.05, 4.0),
    "alpha_b": (0.05, 4.0),
    "alpha_c": (0.05, 4.0),
    "alpha_r": (0.05, 4.0),
    "scale": (1.0, 200.0),
}


@dataclass
class MotifTemplateFit:
    motif_id: str
    params: dict[str, float]
    train_rmsle: float
    graph: dict[str, Any]
    summary: str

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _predict_template(self.motif_id, self.params, X)


def _hill_act(x: np.ndarray, k: float, n: float = 2.0) -> np.ndarray:
    x = np.maximum(x, 1e-12)
    xn = np.power(x, n)
    kn = max(k, 1e-12) ** n
    return xn / (kn + xn)


def _hill_rep(x: np.ndarray, k: float, n: float = 2.0) -> np.ndarray:
    x = np.maximum(x, 0.0)
    xn = np.power(x, n)
    kn = max(k, 1e-12) ** n
    return kn / (kn + xn)


def _params_from_vector(motif_id: str, vec: np.ndarray) -> dict[str, float]:
    names = _PARAM_SPECS[motif_id]
    return {name: float(v) for name, v in zip(names, vec)}


def _predict_template(motif_id: str, params: dict[str, float], X: np.ndarray) -> np.ndarray:
    signal = X[:, 0]
    pert_a = X[:, 1]
    pert_b = X[:, 2]
    pert_c = X[:, 3]
    pert_r = X[:, 4]
    basal = 0.03

    a = pert_a * (basal + params.get("alpha_a", 1.0) * _hill_act(signal, params["ks"]))
    b = pert_b * (basal + params.get("alpha_b", 1.0) * _hill_act(a, params.get("kab", 0.7)))
    c = np.full_like(signal, basal)
    r = np.full_like(signal, basal)

    if motif_id == "activation_chain":
        c = pert_c * (basal + params["alpha_c"] * _hill_act(b, params["kbc"]))
    elif motif_id == "coherent_feedforward":
        c = pert_c * (
            basal
            + params["alpha_c"] * _hill_act(a, params["kac"]) * _hill_act(b, params["kbc"])
        )
    elif motif_id == "incoherent_feedforward":
        c = pert_c * (
            basal
            + params["alpha_c"] * _hill_act(a, params["kac"]) * _hill_rep(b, params["kbc"])
        )
    elif motif_id == "negative_feedback":
        c = pert_c * (basal + params["alpha_c"] * _hill_act(b, params["kbc"]))
        r = pert_r * basal
        for _ in range(12):
            c = pert_c * (
                basal
                + params["alpha_c"] * _hill_act(b, params["kbc"]) * _hill_rep(r, params["kcr"])
            )
            r = pert_r * (basal + params["alpha_r"] * _hill_act(c, params["krc"]))
    elif motif_id == "toggle_switch":
        r = pert_r * (basal + 0.5 * params["alpha_r"])
        for _ in range(16):
            a = pert_a * (
                basal + params["alpha_a"] * _hill_act(signal, params["ks"]) * _hill_rep(r, params["kar"])
            )
            r = pert_r * (basal + params["alpha_r"] * _hill_rep(a, params["kra"]))
            b = pert_b * (basal + params["alpha_b"] * _hill_act(a, params["kab"]))
            c = pert_c * (
                basal
                + params["alpha_c"] * _hill_act(a, params["kac"]) * _hill_act(b, params["kbc"])
            )
    else:
        raise ValueError(f"Unknown motif template: {motif_id}")

    return np.maximum(params["scale"] * c, 1e-6)


def _template_graph(motif_id: str) -> dict[str, Any]:
    edges = {
        "activation_chain": [("signal", "A", 1), ("A", "B", 1), ("B", "C", 1)],
        "coherent_feedforward": [("signal", "A", 1), ("A", "B", 1), ("A", "C", 1), ("B", "C", 1)],
        "incoherent_feedforward": [("signal", "A", 1), ("A", "B", 1), ("A", "C", 1), ("B", "C", -1)],
        "negative_feedback": [("signal", "A", 1), ("A", "B", 1), ("B", "C", 1), ("C", "R", 1), ("R", "C", -1)],
        "toggle_switch": [("signal", "A", 1), ("A", "R", -1), ("R", "A", -1), ("A", "B", 1), ("A", "C", 1), ("B", "C", 1)],
    }[motif_id]
    return {"motif_id": motif_id, "edges": edges}


def _loss(vec: np.ndarray, motif_id: str, X: np.ndarray, y: np.ndarray) -> float:
    params = _params_from_vector(motif_id, vec)
    y_hat = _predict_template(motif_id, params, X)
    mask = np.isfinite(y_hat) & np.isfinite(y) & (y_hat > 0) & (y > 0)
    if not np.any(mask):
        return 1e6
    return float(np.sqrt(np.mean((np.log1p(y_hat[mask]) - np.log1p(y[mask])) ** 2)))


def fit_motif_template(motif_id: str, store: ExperimentStore, restarts: int = 1) -> MotifTemplateFit:
    X, y = store.to_arrays()
    if len(X) == 0:
        raise ValueError("Cannot fit motif template with no data.")

    names = _PARAM_SPECS[motif_id]
    bounds = [_BND[n] for n in names]
    x0 = np.array([_DEFAULTS[n] for n in names], dtype=float)

    best_x = x0.copy()
    best_loss = _loss(best_x, motif_id, X, y)
    rng = np.random.default_rng(42)

    for _ in range(restarts):
        jitter = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)
        res = minimize(
            _loss,
            jitter,
            args=(motif_id, X, y),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 60},
        )
        candidate_x = np.array(res.x if res.success else jitter, dtype=float)
        candidate_loss = _loss(candidate_x, motif_id, X, y)
        if candidate_loss < best_loss:
            best_x = candidate_x
            best_loss = candidate_loss

    params = _params_from_vector(motif_id, best_x)
    return MotifTemplateFit(
        motif_id=motif_id,
        params=params,
        train_rmsle=float(best_loss),
        graph=_template_graph(motif_id),
        summary=f"{motif_id} train_rmsle={best_loss:.4f}",
    )


def fit_motif_library(store: ExperimentStore, top_k: int = 3) -> list[MotifTemplateFit]:
    fits = [fit_motif_template(motif_id, store) for motif_id in MOTIF_LIBRARY]
    fits.sort(key=lambda f: f.train_rmsle)
    return fits[:top_k]
