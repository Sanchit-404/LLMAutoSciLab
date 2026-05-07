"""
ALSelector: given an LLM-specified region (parameter bounds) and a fitted GP,
select the most informative parameter combinations to run next.

Key design: GP is always fitted on ALL historical data, but candidate points
are sampled only within the LLM-specified region bounds.

Sampling:
- When GP is not fitted (< min_data), use Latin Hypercube Sampling (LHS) for
  better space-filling coverage than pure random.
- When GP is fitted, sample 2000 random candidates within region and score
  by acquisition function (UCB early-phase, EI late-phase).
"""
import numpy as np
from scipy.stats import norm

from autoscilab.al.gp_model import GPSurrogate
from autoscilab.data.store import ExperimentStore
from autoscilab.oracle.base import BaseOracle


class ALSelector:
    def __init__(self, oracle: BaseOracle, gp: GPSurrogate):
        self._oracle = oracle
        self._gp = gp
        self._last_selection_info: dict[str, float | int | bool | str | None] = {}

    def select(
        self,
        store: ExperimentStore,
        region_bounds: dict[str, list[float]],  # {param: [min, max]} from LLM
        n_points: int = 5,
        acquisition: str = "ei",  # "ei" or "ucb"
        hypotheses: list[str] | None = None,
        n_candidates: int = 2000,
        kappa: float = 2.0,
        discrimination_binding: bool = False,
    ) -> list[dict[str, float]]:
        """
        Select n_points parameter combinations within region_bounds.

        If GP is not fitted (too few data), falls back to log-uniform random sampling.

        Args:
            store: full experiment history (GP is re-fitted on all of it)
            region_bounds: {param_name: [min, max]} — LLM-specified region
            n_points: how many parameter sets to return
            acquisition: "ei" or "ucb"
            n_candidates: number of random candidates to score
            kappa: UCB exploration weight

        Returns:
            List of parameter dicts to run through the oracle
        """
        # Clip region bounds to oracle's physical bounds
        oracle_bounds = self._oracle.parameter_bounds
        clipped_region = self._clip_region(region_bounds, oracle_bounds)

        # Fit GP on all historical data if enough points
        X_all, y_all = store.to_arrays()
        if len(X_all) >= 3 and not self._gp.is_fitted:
            self._gp.fit(X_all, y_all, oracle_bounds)
        elif len(X_all) >= 3:
            # Always refit on full history (solves GP transfer problem)
            self._gp.fit(X_all, y_all, oracle_bounds)

        if not self._gp.is_fitted or len(X_all) < 3:
            # Not enough data — fall back to random sampling in region
            return self._random_sample(clipped_region, n_points)

        # Sample candidates within region
        candidates = self._sample_region(clipped_region, n_candidates)

        # Score with acquisition function
        mean_log, std_log = self._gp.predict_log(candidates, oracle_bounds)
        is_optimization = getattr(self._oracle, "objective_type", "equation") == "optimization"
        objective_direction = getattr(self._oracle, "objective_direction", "maximize")

        disagree = None
        has_signal = False
        if acquisition == "disagree":
            # Hypothesis-disagreement sampling: prefer candidates where plausible
            # equations diverge most. If disagreement signal is weak/unusable,
            # fall back to objective-aware UCB (optimization) or uncertainty.
            disagree = self._disagreement_score(candidates, hypotheses, store=store)
            has_signal = bool(np.isfinite(disagree).any() and float(np.max(disagree)) > 1e-10)
            if has_signal:
                # Binding mode: optimize separation first; uncertainty is only tie-breaker.
                if discrimination_binding:
                    scores = disagree + 0.05 * std_log
                else:
                    scores = disagree + 0.25 * std_log
            elif is_optimization and not discrimination_binding:
                # Legacy fallback for optimization tasks when no hypothesis signal is available.
                scores = self._objective_ucb(
                    mean=mean_log,
                    std=std_log,
                    kappa=kappa,
                    direction=objective_direction,
                )
            else:
                scores = std_log
        elif acquisition == "ei":
            if is_optimization:
                scores = self._optimization_ei(
                    mean=mean_log,
                    std=std_log,
                    y_all=y_all,
                    direction=objective_direction,
                )
            else:
                # For equation discovery we want information about the functional SHAPE,
                # not the maximum output. Use f_best=mean(log y) so EI rewards uncertainty
                # over already-seen regions rather than biasing toward max(y) regions.
                y_best_log = float(np.mean(np.log(np.abs(y_all) + 1e-300)))
                scores = self._ei(mean_log, std_log, y_best_log)
        elif acquisition == "ucb":
            if is_optimization:
                scores = self._objective_ucb(
                    mean=mean_log,
                    std=std_log,
                    kappa=kappa,
                    direction=objective_direction,
                )
            else:
                scores = self._ucb(mean_log, std_log, kappa)
        elif acquisition == "variance":
            # Pure uncertainty sampling: pick wherever the GP is most uncertain
            scores = std_log
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")

        # Pick top n_points (no repeats via diverse selection)
        selected_indices = self._diverse_top_k(candidates, scores, n_points)
        self._record_selection_info(
            acquisition=acquisition,
            scores=scores,
            selected_indices=selected_indices,
            disagree=disagree,
            has_signal=has_signal,
            discrimination_binding=discrimination_binding,
        )
        param_names = self._oracle.parameter_names

        return [
            {p: float(candidates[i, j]) for j, p in enumerate(param_names)}
            for i in selected_indices
        ]

    def get_last_selection_info(self) -> dict[str, float | int | bool | str | None]:
        return dict(self._last_selection_info)

    def _record_selection_info(
        self,
        acquisition: str,
        scores: np.ndarray,
        selected_indices: list[int],
        disagree: np.ndarray | None = None,
        has_signal: bool = False,
        discrimination_binding: bool = False,
    ) -> None:
        info: dict[str, float | int | bool | str | None] = {
            "acquisition": acquisition,
            "n_candidates": int(len(scores)),
            "n_selected": int(len(selected_indices)),
            "score_pool_mean": float(np.nanmean(scores)) if len(scores) else 0.0,
            "score_selected_mean": (
                float(np.nanmean(scores[selected_indices]))
                if len(selected_indices)
                else 0.0
            ),
            "discrimination_binding": bool(discrimination_binding),
            "disagreement_has_signal": False,
            "disagreement_pool_mean": None,
            "disagreement_selected_mean": None,
            "disagreement_gain_ratio": None,
        }
        if acquisition == "disagree":
            info["disagreement_has_signal"] = bool(has_signal)
            if disagree is not None and len(disagree):
                pool_mean = float(np.nanmean(disagree))
                sel_mean = (
                    float(np.nanmean(disagree[selected_indices]))
                    if len(selected_indices)
                    else 0.0
                )
                gain = (sel_mean - pool_mean) / (abs(pool_mean) + 1e-12)
                info["disagreement_pool_mean"] = pool_mean
                info["disagreement_selected_mean"] = sel_mean
                info["disagreement_gain_ratio"] = float(gain)
        self._last_selection_info = info

    def _disagreement_score(
        self,
        X: np.ndarray,
        hypotheses: list[str] | None,
        store: ExperimentStore,
    ) -> np.ndarray:
        """
        Compute disagreement across multiple hypotheses.

        Two-stage strategy:
        1) fast power-law proxy (existing exponent parser)
        2) functional disagreement by compiling hypothesis expressions, fitting
           free constants on observed data, and evaluating predictions on X.
        """
        if not hypotheses or len(hypotheses) < 2 or X.size == 0:
            return np.zeros(X.shape[0], dtype=float)

        score_fn = self._functional_disagreement_score(X, hypotheses, store)
        if np.isfinite(score_fn).any() and float(np.max(score_fn)) > 1e-12:
            return score_fn

        param_names = self._oracle.parameter_names
        parsed = []
        for h in hypotheses:
            exps = self._parse_power_exponents(h or "", param_names)
            if exps is not None:
                parsed.append(exps)

        if len(parsed) < 2:
            return np.zeros(X.shape[0], dtype=float)

        logX = np.log(np.clip(X, 1e-300, None))
        preds = []
        for exp_vec in parsed:
            preds.append(logX @ exp_vec)
        M = np.vstack(preds)  # (n_hyp, n_points)
        return np.var(M, axis=0)

    def _normalise_hypothesis_expr(self, raw: str) -> str:
        import re

        s = (raw or "").strip()
        if not s:
            return ""

        # ── Strip pipe-delimited "| where ..." clauses ───────────────────────
        if "|" in s:
            s = s.split("|", 1)[0].strip()

        # ── Strip LHS assignment including function-call LHS: N(t) = ... ─────
        lhs_eq = re.match(r"^\s*[A-Za-z_]\w*(?:\([^)]*\))?\s*=\s*", s)
        if lhs_eq:
            s = s[lhs_eq.end():]

        # ── Strip trailing prose appended after the expression ────────────────
        # Handles: ", where n is...", " where k is a constant", " for some k"
        # ", with C a free constant", " (where ...)"
        s = re.sub(r",\s*(where|for\s+some|with\s+[A-Za-z])\b.*$", "", s,
                   flags=re.IGNORECASE)
        s = re.sub(r"\s+(where|for\s+some)\b.*$", "", s, flags=re.IGNORECASE)
        # Strip trailing parenthetical prose: " (units)" or " (where ...)"
        # Only strip if the content is purely alphabetic prose (no math operators/digits)
        s = re.sub(r"\s+\([A-Za-z][A-Za-z\s,_]{3,}\)\s*$", "", s)

        # ── Normalise common math notation ────────────────────────────────────
        # ln( → log(  (natural log alias)
        s = re.sub(r"\bln\s*\(", "log(", s)
        # arcsin/arccos/arctan → asin/acos/atan
        s = re.sub(r"\barc(sin|cos|tan)\s*\(", r"a\1(", s, flags=re.IGNORECASE)
        # ^ → ** (only when not already **) — use sentinel to avoid lookbehind (py3.12 bug)
        _dstar = "\x00DSTAR\x00"
        s = s.replace("**", _dstar)
        s = s.replace("^", "**")
        s = s.replace(_dstar, "**")

        # ── Strip trailing unit annotations ───────────────────────────────────
        s = re.sub(r"\s+\[[A-Za-z0-9_/%^*.+]+\]\s*$", "", s)

        return s.strip()

    def _extract_unbound_names(self, expr: str, param_names: list[str]) -> list[str]:
        import ast
        import math

        if not expr:
            return []
        try:
            tree = ast.parse(expr, mode="eval")
        except Exception:
            return []
        names = sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
        known = {
            *param_names,
            "np",
            "math",
            "abs",
            "min",
            "max",
            "pow",
            "pi",
            "e",
        }
        known.update(
            {
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "tanh",
                "exp",
                "log",
                "log10",
                "sqrt",
            }
        )
        known.update(
            name for name in dir(math)
            if not name.startswith("_")
        )
        return [n for n in names if n not in known and n not in param_names]

    def _functional_disagreement_score(
        self,
        X_candidates: np.ndarray,
        hypotheses: list[str],
        store: ExperimentStore,
    ) -> np.ndarray:
        from scipy.optimize import minimize
        import math

        X_obs, y_obs = store.to_arrays()
        if len(X_obs) < 4 or len(y_obs) < 4:
            return np.zeros(X_candidates.shape[0], dtype=float)

        param_names = self._oracle.parameter_names
        y_safe = np.where(y_obs > 0, y_obs, np.nan)
        gns = {
            "np": np,
            "math": math,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "sqrt": math.sqrt,
            "abs": abs,
            "pow": pow,
            "pi": math.pi,
            "e": math.e,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "min": min,
            "max": max,
        }

        pred_logs: list[np.ndarray] = []
        for raw in hypotheses:
            expr = self._normalise_hypothesis_expr(raw)
            if not expr:
                continue
            const_names = self._extract_unbound_names(expr, param_names)
            sig = ", ".join(param_names + const_names)
            fn_code = f"def _hyp_fn({sig}):\n    return {expr}\n"
            try:
                local_ns: dict[str, object] = {}
                exec(fn_code, gns, local_ns)
                fn = local_ns["_hyp_fn"]
            except Exception:
                continue

            def _log_mse(const_vals: np.ndarray) -> float:
                try:
                    y_pred = np.array(
                        [
                            fn(
                                *[float(X_obs[i, j]) for j in range(X_obs.shape[1])],
                                *[float(c) for c in const_vals],
                            )
                            for i in range(len(X_obs))
                        ],
                        dtype=float,
                    )
                except Exception:
                    return 1e9
                mask = np.isfinite(y_pred) & np.isfinite(y_safe) & (y_pred > 0)
                if int(np.sum(mask)) < 3:
                    return 1e9
                diff = np.log(y_pred[mask]) - np.log(y_safe[mask])
                return float(np.mean(diff**2))

            fitted_consts = np.array([], dtype=float)
            if const_names:
                best_loss = float("inf")
                best_x = None
                for seed in (0.3, 1.0, 3.0):
                    x0 = np.full(len(const_names), seed, dtype=float)
                    try:
                        res = minimize(
                            _log_mse,
                            x0,
                            method="Nelder-Mead",
                            options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4},
                        )
                        if float(res.fun) < best_loss:
                            best_loss = float(res.fun)
                            best_x = np.array(res.x, dtype=float)
                    except Exception:
                        continue
                if best_x is None or not np.isfinite(best_loss) or best_loss > 1e8:
                    continue
                fitted_consts = best_x
            else:
                if _log_mse(np.array([])) > 1e8:
                    continue

            try:
                y_hat = np.array(
                    [
                        fn(
                            *[float(X_candidates[i, j]) for j in range(X_candidates.shape[1])],
                            *[float(c) for c in fitted_consts],
                        )
                        for i in range(len(X_candidates))
                    ],
                    dtype=float,
                )
            except Exception:
                continue
            mask = np.isfinite(y_hat) & (y_hat > 0)
            if int(np.sum(mask)) < max(3, int(0.25 * len(y_hat))):
                continue
            y_hat = np.where(mask, y_hat, np.nan)
            pred_logs.append(np.log(y_hat))

        if len(pred_logs) < 2:
            return np.zeros(X_candidates.shape[0], dtype=float)
        M = np.vstack(pred_logs)  # (n_hyp, n_points)
        return np.nanvar(M, axis=0)

    def has_usable_disagreement_hypotheses(self, hypotheses: list[str] | None) -> bool:
        """Return True only when at least two hypotheses can be numerically parsed."""
        if not hypotheses or len(hypotheses) < 2:
            return False
        param_names = self._oracle.parameter_names
        parsed = 0
        for h in hypotheses:
            if self._parse_power_exponents(h or "", param_names) is not None:
                parsed += 1
            if parsed >= 2:
                return True
        return False

    def _parse_power_exponents(self, hypothesis: str, param_names: list[str]) -> np.ndarray | None:
        """
        Extract a vector of exponents for a multiplicative power-law hypothesis.
        Supports lightweight patterns:
          - p**a, p^a
          - "/ p**a" treated as negative exponent
        If nothing is found, returns None.
        """
        import re

        if not hypothesis:
            return None

        # Normalize a bit for matching
        h = " " + hypothesis.replace("^", "**") + " "
        exp = np.zeros(len(param_names), dtype=float)
        found = False

        for i, p in enumerate(param_names):
            # Capture an optional preceding "/" within a few chars
            # Examples matched:
            #   "* distance**2"
            #   "/ distance ** 1.5"
            pat = rf"(?P<div>/\s*)?\b{re.escape(p)}\s*\*\*\s*(?P<e>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            for m in re.finditer(pat, h):
                try:
                    e = float(m.group("e"))
                except Exception:
                    continue
                if m.group("div"):
                    e = -e
                exp[i] += e
                found = True

        return exp if found else None

    def _ei(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        f_best: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        """Expected Improvement acquisition function (in log-space)."""
        z = (mean - f_best - xi) / (std + 1e-9)
        return (mean - f_best - xi) * norm.cdf(z) + std * norm.pdf(z)

    def _ucb(self, mean: np.ndarray, std: np.ndarray, kappa: float) -> np.ndarray:
        """Upper Confidence Bound acquisition function (in log-space)."""
        return mean + kappa * std

    def _objective_ucb(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        kappa: float,
        direction: str,
    ) -> np.ndarray:
        """Objective-aware UCB on the GP's log-target scale."""
        if direction == "minimize":
            return -mean + kappa * std
        return mean + kappa * std

    def _optimization_ei(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_all: np.ndarray,
        direction: str,
    ) -> np.ndarray:
        """Objective-aware EI on the GP's log-target scale."""
        y_log = np.log(np.abs(y_all) + 1e-300)
        if direction == "minimize":
            mean_t = -mean
            f_best_t = float(np.max(-y_log))
        else:
            mean_t = mean
            f_best_t = float(np.max(y_log))
        return self._ei(mean_t, std, f_best_t)

    def _sample_region(
        self,
        region_bounds: dict[str, tuple[float, float]],
        n: int,
    ) -> np.ndarray:
        """Sample n points with mixed linear/log strategy within region_bounds."""
        param_names = self._oracle.parameter_names
        cols = []
        for p in param_names:
            lo, hi = region_bounds[p]
            ratio = hi / max(lo, 1e-12) if lo > 0 else np.inf
            use_linear = lo <= 0 or ratio < 20.0
            if use_linear:
                cols.append(np.random.uniform(lo, hi, size=n))
            else:
                lo = max(lo, 1e-300)
                hi = max(hi, lo * 1.01)
                log_samples = np.random.uniform(np.log(lo), np.log(hi), size=n)
                cols.append(np.exp(log_samples))
        return np.column_stack(cols)

    def _lhs_sample(
        self,
        region_bounds: dict[str, tuple[float, float]],
        n: int,
    ) -> np.ndarray:
        """
        Latin Hypercube Sample in log-space within region_bounds.

        LHS partitions each dimension into n equal intervals and picks one
        sample per interval, ensuring better space-filling than pure random.
        Falls back to log-uniform random if scipy.stats.qmc is unavailable.
        """
        param_names = self._oracle.parameter_names
        d = len(param_names)

        try:
            from scipy.stats.qmc import LatinHypercube
            sampler = LatinHypercube(d=d, seed=42)
            unit = sampler.random(n=n)  # (n, d) uniform in [0, 1]
            samples = np.zeros((n, d), dtype=float)
            for j, p in enumerate(param_names):
                lo, hi = region_bounds[p]
                ratio = hi / max(lo, 1e-12) if lo > 0 else np.inf
                use_linear = lo <= 0 or ratio < 20.0
                if use_linear:
                    samples[:, j] = lo + unit[:, j] * (hi - lo)
                else:
                    llo = np.log(max(lo, 1e-300))
                    lhi = np.log(max(hi, lo * 1.01))
                    samples[:, j] = np.exp(llo + unit[:, j] * (lhi - llo))
        except (ImportError, Exception):
            # Manual LHS fallback: stratified random in each dimension
            rng = np.random.default_rng(42)
            cuts = np.linspace(0, 1, n + 1)
            unit = np.column_stack([
                rng.uniform(cuts[:-1], cuts[1:])
                for _ in range(d)
            ])
            for col in range(d):
                rng.shuffle(unit[:, col])
            samples = np.zeros((n, d), dtype=float)
            for j, p in enumerate(param_names):
                lo, hi = region_bounds[p]
                ratio = hi / max(lo, 1e-12) if lo > 0 else np.inf
                use_linear = lo <= 0 or ratio < 20.0
                if use_linear:
                    samples[:, j] = lo + unit[:, j] * (hi - lo)
                else:
                    llo = np.log(max(lo, 1e-300))
                    lhi = np.log(max(hi, lo * 1.01))
                    samples[:, j] = np.exp(llo + unit[:, j] * (lhi - llo))

        return samples

    def _random_sample(
        self,
        region_bounds: dict[str, tuple[float, float]],
        n: int,
    ) -> list[dict[str, float]]:
        """Fallback: LHS log-space sample within region (better coverage than random)."""
        param_names = self._oracle.parameter_names
        candidates = self._lhs_sample(region_bounds, n)
        return [
            {p: float(candidates[i, j]) for j, p in enumerate(param_names)}
            for i in range(n)
        ]

    def _diverse_top_k(
        self,
        candidates: np.ndarray,
        scores: np.ndarray,
        k: int,
    ) -> list[int]:
        """
        Select k diverse high-scoring candidates.
        Uses a greedy approach: pick best, then pick next-best that's
        sufficiently different (L2 distance in normalized space > threshold).
        """
        sorted_idx = np.argsort(scores)[::-1]
        selected = [int(sorted_idx[0])]
        if k == 1:
            return selected

        # Normalize candidates for distance comparison
        c_min = candidates.min(axis=0)
        c_max = candidates.max(axis=0) + 1e-12
        c_norm = (candidates - c_min) / (c_max - c_min)

        for idx in sorted_idx[1:]:
            if len(selected) >= k:
                break
            c = c_norm[idx]
            # Check minimum distance to already-selected points
            dists = [np.linalg.norm(c - c_norm[s]) for s in selected]
            if min(dists) > 0.05:  # 5% of normalized range
                selected.append(int(idx))

        # If not enough diverse points, just fill with top remaining
        if len(selected) < k:
            for idx in sorted_idx:
                if int(idx) not in selected:
                    selected.append(int(idx))
                if len(selected) >= k:
                    break

        return selected[:k]

    def _clip_region(
        self,
        region: dict[str, list[float]],
        oracle_bounds: dict[str, tuple[float, float]],
    ) -> dict[str, tuple[float, float]]:
        """Clip LLM-specified region to oracle's physical bounds.

        Handles malformed LLM output gracefully: if a parameter's bounds
        are missing, not a 2-element list, or contain non-numeric values,
        falls back to the full oracle bounds for that parameter.
        """
        clipped = {}
        for p in self._oracle.parameter_names:
            oracle_lo, oracle_hi = oracle_bounds[p]
            lo, hi = oracle_lo, oracle_hi  # default fallback
            bounds = region.get(p)
            if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                try:
                    lo = max(float(bounds[0]), oracle_lo)
                    hi = min(float(bounds[1]), oracle_hi)
                    if lo >= hi:
                        lo, hi = oracle_lo, oracle_hi
                except (TypeError, ValueError):
                    pass  # non-numeric — keep oracle defaults
            clipped[p] = (lo, hi)
        return clipped
