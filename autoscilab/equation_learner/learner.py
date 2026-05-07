"""
Equation Learner: PySR symbolic regression.

PySR (evolutionary symbolic regression) searches for the best closed-form expression
by evolving a population of equations using genetic programming in Julia.

The LLM's current hypothesis biases the operator set:
- If the hypothesis mentions sin/cos/trig/oscillat → trig operators are included
- Always includes: sqrt, log, exp, abs, and arithmetic (+, -, *, /, ^)
- The ^ exponent constraint is (-4, 4): allows fractional and Euler-sized exponents (e≈2.718)

Confidence is computed via bootstrap sampling: PySR is run on B random subsets of
the training data. High prediction agreement across bootstrap runs → high confidence.
"""
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autoscilab.data.store import ExperimentStore
from autoscilab.oracle.base import BaseOracle


@dataclass
class FittedEquation:
    law_str: str          # Python function string for evaluate_law
    skeleton_str: str     # Human-readable symbolic form (sympy str)
    r2: float
    rmsle: float
    constants: dict[str, float]
    n_data_points: int
    expected_form: str = ""
    confidence: float = 0.0   # Bootstrap confidence in [0, 1]

    def is_valid(self) -> bool:
        return not (np.isnan(self.r2) or np.isnan(self.rmsle)) and self.r2 > -1e6

    def __str__(self) -> str:
        return (f"{self.expected_form} | R²={self.r2:.4f} | "
                f"RMSLE={self.rmsle:.4f} | conf={self.confidence:.2f}")


def _unary_operators_from_hypothesis(hypothesis: str, domain: str | None = None, tags: list[str] | None = None) -> list[str]:
    """
    Parse the LLM hypothesis string to select PySR unary operators.
    Domain profiles guarantee the right building blocks even if the hypothesis
    fails to mention them (e.g., exp/log for decay, sin/cos for oscillations).
    """
    h = hypothesis.lower()
    ops = ["sqrt", "log", "exp", "abs"]

    tag_profiles: dict[str, list[str]] = {
        "decay": ["exp", "log"],
        "oscillatory": ["sin", "cos", "exp", "log"],
        "trig": ["sin", "cos"],
        # ChemBench tags
        "saturation": ["exp", "log"],
        "arrhenius": ["exp", "log"],
        "hill": ["exp", "log"],
        "substrate_inhibition": [],
        "bisubstrate": [],
        "pingpong": [],
        "ph_ionization": ["exp", "log"],
        "competitive_inhibition": [],
        "uncompetitive_inhibition": [],
        "noncompetitive_inhibition": [],
        "product_inhibition": [],
    }
    for tag in tags or []:
        for op in tag_profiles.get(tag, []):
            if op not in ops:
                ops.append(op)

    trig_domains = {"m4_snell_law", "m6_underdamped_harmonic", "m7_malus_law"}
    if any(k in h for k in ["sin", "cos", "trig", "angle", "theta",
                             "harmonic", "oscillat", "wave", "snell", "malus"]):
        if domain in trig_domains:
            for op in ("sin", "cos"):
                if op not in ops:
                    ops.append(op)
    # Only allow tan when explicitly mentioned AND domain is trig-heavy
    if "tan" in h and domain in trig_domains and "tan" not in ops:
        ops.append("tan")
    return ops


def _exponent_range_from_hypothesis(hypothesis: str) -> tuple[float, float]:
    """
    Heuristically extract a reasonable exponent search range for PySR from the
    current LLM hypothesis.

    Looks for patterns like x**1.5 or x ** 2, collects all numeric exponents,
    and returns a (min-1, max+1) window, clipped to [-4, 4]. If nothing is
    found, falls back to the default (-4, 4).
    """
    import re

    if not hypothesis:
        return (-4.0, 4.0)

    # Match **<number> and ** <number>, including floats
    pattern = r"\*\*\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    exps: list[float] = []
    for match in re.findall(pattern, hypothesis):
        try:
            exps.append(float(match))
        except ValueError:
            continue

    if not exps:
        return (-4.0, 4.0)

    lo = min(exps) - 1.0
    hi = max(exps) + 1.0
    return (lo, hi)


def _exponent_range_from_slopes(store: ExperimentStore, oracle: BaseOracle) -> tuple[float, float] | None:
    """
    Estimate exponent range from empirical log-log slopes (mean across params).
    Returns (lo, hi) clipped to [-4, 4] or None if insufficient data/invalid.
    """
    if len(store) < 5:
        return None
    X, y = store.to_arrays()
    if np.any(y <= 0):
        return None
    log_y = np.log10(y)
    slopes = []
    for idx, p in enumerate(oracle.parameter_names):
        xi = X[:, idx]
        if np.any(xi <= 0):
            continue
        log_x = np.log10(xi)
        if np.std(log_x) < 1e-10:
            continue
        slope = float(np.polyfit(log_x, log_y, 1)[0])
        slopes.append(slope)
    if not slopes:
        return None
    mean_slope = float(np.mean(slopes))
    lo = max(-4.0, mean_slope - 1.5)
    hi = min(4.0, mean_slope + 1.5)
    return (lo, hi)


def _make_pysr_model(
    niterations: int,
    unary_ops: list[str],
    populations: int = 20,
    random_state: int = 0,
    exponent_range: tuple[float, float] = (-4.0, 4.0),
    maxsize: int = 25,
    binary_operators: list[str] | None = None,
    warm_start: bool = False,
):
    import warnings
    from pysr import PySRRegressor
    warnings.filterwarnings(
        "ignore",
        message=".*Setting `random_state`.*",
        category=UserWarning,
    )
    lo, hi = exponent_range
    lo = max(-4.0, float(lo))
    hi = min(4.0, float(hi))
    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/", "^"]

    # procs=1 / parallelism="serial" is the safe default when multiple outer
    # workers run concurrently (they'd fight over CPU otherwise).
    # Override via env vars for single-worker server runs:
    #   PYSR_PROCS=8 PYSR_PARALLELISM=multithreading
    procs = int(os.environ.get("PYSR_PROCS", "1"))
    parallelism = os.environ.get("PYSR_PARALLELISM", "serial")
    kwargs = dict(
        niterations=niterations,
        random_state=random_state,
        binary_operators=binary_operators,
        unary_operators=unary_ops,
        constraints={"^": (lo, hi)},
        populations=populations,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
        parsimony=0.001,
        maxsize=maxsize,
        procs=procs,
        parallelism=parallelism,
    )
    try:
        return PySRRegressor(warm_start=warm_start, **kwargs)
    except TypeError:
        # Backward compatibility with older PySR versions.
        return PySRRegressor(**kwargs)


class EquationLearner:
    def __init__(
        self,
        oracle: BaseOracle,
        n_iterations: int = 40,
        powerlaw_prefit: bool = True,
        warm_start_chaining: bool = False,
        warm_start_gate_min_r2: float = 0.85,
        warm_start_gate_max_rmsle: float = 2.5,
        separable_factor_locking: bool = False,
        separable_lock_max_factors: int = 1,
        separable_lock_min_r2: float = 0.92,
        separable_lock_min_span_decades: float = 0.8,
        separable_lock_max_slope_std: float = 0.35,
    ):
        """
        oracle: used to get parameter names.
        n_iterations: default PySR niterations (used when fit() is called without override).
        """
        self._oracle = oracle
        self._n_iterations = n_iterations
        self._powerlaw_prefit_enabled = powerlaw_prefit
        self._warm_start_chaining = warm_start_chaining
        self._warm_start_gate_min_r2 = float(warm_start_gate_min_r2)
        self._warm_start_gate_max_rmsle = float(warm_start_gate_max_rmsle)
        self._separable_factor_locking = separable_factor_locking
        self._separable_lock_max_factors = max(0, int(separable_lock_max_factors))
        self._separable_lock_min_r2 = float(separable_lock_min_r2)
        self._separable_lock_min_span_decades = float(separable_lock_min_span_decades)
        self._separable_lock_max_slope_std = float(separable_lock_max_slope_std)
        self._warm_models: dict[tuple[Any, ...], Any] = {}
        self._warm_model_quality: dict[tuple[Any, ...], dict[str, float]] = {}
        self._last_fit_key: tuple[Any, ...] | None = None
        self._best: FittedEquation | None = None
        # All skeleton (scipy) candidates from the most recent fit() call.
        # Populated alongside PySR so the caller can surface both to the LLM.
        self.skel_candidates: list[FittedEquation] = []

    @property
    def best(self) -> FittedEquation | None:
        return self._best

    def fit(
        self,
        store: ExperimentStore,
        goal: str,
        current_hypothesis: str,
        n_iterations: int | None = None,
        n_bootstrap: int = 3,
        n_diverse_runs: int = 1,
        relevant_vars: list[str] | None = None,
        skeleton_families: list[str] | None = None,
        skeleton_priority_threshold: float | None = None,
    ) -> FittedEquation | None:
        """
        Run PySR on all data in the store, then estimate confidence via bootstrap.

        Bootstrap confidence: run PySR on B resampled subsets of training data,
        measure how consistently they predict on the validation set.
        High agreement (low CV of predictions) → confidence close to 1.

        Bootstrap is gated on validation R²: low-R² equations are clearly wrong,
        so we set confidence=0 regardless of bootstrap consistency (prevents the
        "constant dominates variance" false-positive).

        Args:
            n_diverse_runs: run PySR n_diverse_runs times with different random
                seeds and keep the best result by validation R². Useful for hard
                domains where the search is sensitive to initialization.
            relevant_vars: if provided, PySR only sees these columns (derived
                from the LLM's current hypothesis). The generated law_str still
                uses the full parameter signature so evaluate_law() works correctly.
                Dramatically reduces PySR search space (e.g. 7D→2D for MM kinetics).
            skeleton_families: list of Python expression strings (using oracle param
                names + free constants C0/C1/...) to try fitting directly via
                scipy curve_fit before running PySR. If a skeleton fit achieves
                R²>0.95, PySR is skipped entirely (faster + symbolically exact).

        Returns the best FittedEquation found (or previous best if this run fails).
        """
        if len(store) < 5:
            return None

        try:
            import pandas as pd
            from pysr import PySRRegressor
        except ImportError as e:
            raise ImportError(
                "PySR not installed. Run:\n"
                "  pip install pysr\n"
                "  python -c 'import pysr; pysr.install()'"
            ) from e

        X, y = store.to_arrays()
        param_names = self._oracle.parameter_names
        full_param_names = list(param_names)   # always used for law_str signature
        n = len(X)

        # ── Direct skeleton fitting (before variable filtering) ───────────────
        # Try fitting each family expression directly with scipy curve_fit using
        # the FULL X (all variables). This is symbolically exact when the LLM's
        # hypothesis matches the true form, and gives a baseline to compare PySR against.
        # Uses full param_names so Arrhenius / pH families can reference T and pH
        # even if relevant_vars filtering would exclude them from PySR.
        self.skel_candidates = []   # reset every call so stale results never bleed through
        if skeleton_families and len(X) >= 10:
            all_skels = list(skeleton_families)
            if current_hypothesis and current_hypothesis not in (
                "unknown", "no data yet", "no data yet — broad exploration needed"
            ):
                all_skels = [current_hypothesis] + all_skels

            # Train/val split on full X for skeleton fitting
            rng_skel = np.random.default_rng(42)
            n_val_skel = max(1, int(0.2 * n))
            val_idx_skel = rng_skel.choice(n, n_val_skel, replace=False)
            val_mask_skel = np.zeros(n, dtype=bool)
            val_mask_skel[val_idx_skel] = True
            X_skel_train = X[~val_mask_skel]
            y_skel_train = y[~val_mask_skel]
            X_skel_val   = X[val_mask_skel]
            y_skel_val   = y[val_mask_skel]

            best_skel = self._skeleton_fit_families(
                all_skels,
                X_skel_train, y_skel_train,
                X_skel_val, y_skel_val,
                list(full_param_names),
                list(full_param_names),
            )
            if best_skel is not None:
                print(f"[EquationLearner] Best skeleton fit: {best_skel.expected_form} "
                      f"| R²={best_skel.r2:.4f} | RMSLE={best_skel.rmsle:.4f}")
                self.skel_candidates.append(best_skel)
                if self._best is None or best_skel.r2 > self._best.r2:
                    self._best = best_skel
                # Skeleton priority: if RMSLE is good enough, skip PySR entirely.
                # This ensures the structurally interpretable skeleton wins over a
                # numerically-tighter but structurally opaque PySR expression.
                if (skeleton_priority_threshold is not None
                        and best_skel.rmsle <= skeleton_priority_threshold):
                    print(
                        f"[EquationLearner] Skeleton priority: RMSLE={best_skel.rmsle:.4f} "
                        f"≤ {skeleton_priority_threshold} → skipping PySR"
                    )
                    return self._best
                # Otherwise continue to PySR — both will be offered to LLM for final selection.

        # ── LLM-guided variable filtering ────────────────────────────────────
        # If the LLM's current hypothesis only references a subset of variables,
        # pass only those columns to PySR. This collapses the search space
        # dramatically (e.g. 7D→2D for Michaelis-Menten) and prevents PySR from
        # picking up spurious secondary-effect correlations (T, pH background noise).
        # The law_str still uses the full signature so evaluate_law() works fine.
        if relevant_vars:
            col_mask = [i for i, p in enumerate(param_names) if p in relevant_vars]
            if col_mask:
                X = X[:, col_mask]
                param_names = [full_param_names[i] for i in col_mask]
                print(f"[EquationLearner] Variable filter active: {param_names} "
                      f"(dropped: {[v for v in full_param_names if v not in param_names]})")

        # Reproducible train/val split (80/20)
        rng = np.random.default_rng(42)
        n_val = max(1, int(0.2 * n))
        val_idx = rng.choice(n, n_val, replace=False)
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_idx] = True
        train_mask = ~val_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        # Named DataFrames so PySR uses the actual parameter names in equations.
        # Also inject euler_e = e ≈ 2.71828 as a constant feature column so PySR
        # can discover expressions like lambda^euler_e or t^euler_e directly.
        # After fitting, euler_e is substituted with sympy.E (exact Euler's number).
        _euler_train = np.full(len(X_train), float(np.e))
        _euler_val   = np.full(len(X_val),   float(np.e))
        param_names_aug = list(param_names) + ["euler_e"]
        X_train_df = pd.DataFrame(np.column_stack([X_train, _euler_train]), columns=param_names_aug)
        X_val_df   = pd.DataFrame(np.column_stack([X_val,   _euler_val]),   columns=param_names_aug)

        # ── Log-space PySR for extreme y-range ───────────────────────────────
        # When y spans >3 orders of magnitude and is all-positive, fitting
        # log10(y) avoids the extreme values that cause PySR to produce
        # interpolation artifacts. The resulting expression gives log10(y),
        # so the reconstructed law wraps it as 10**(...).
        y_train_pos = y_train[y_train > 0]
        use_log_space = (
            len(y_train_pos) == len(y_train)  # all positive
            and len(y_train_pos) >= 5
            and (np.log10(y_train_pos.max()) - np.log10(y_train_pos.min())) >= 3.0
        )
        if use_log_space:
            y_oom = float(np.log10(y_train.max()) - np.log10(y_train.min()))
            print(f"[EquationLearner] log-space mode: y spans {y_oom:.1f} OOM → fitting log10(y)")
            y_train_fit = np.log10(y_train)
            y_val_fit = np.log10(np.clip(y_val, 1e-300, None))
        else:
            y_train_fit = y_train
            y_val_fit = y_val

        # --- Fast power-law pre-fit (log-log linear regression) -----------------
        quick_fit = self._power_law_prefit(X_train, y_train, param_names, full_param_names)

        unary_ops = _unary_operators_from_hypothesis(
            current_hypothesis,
            getattr(self._oracle, "domain", None),
            getattr(self._oracle, "domain_tags", []),
        )
        slope_range = _exponent_range_from_slopes(store, self._oracle)
        exp_range = slope_range or _exponent_range_from_hypothesis(current_hypothesis)
        iters = n_iterations if n_iterations is not None else self._n_iterations
        n_runs = max(1, n_diverse_runs)

        print(f"[EquationLearner] PySR | n={len(X_train)} | niter={iters} | "
              f"runs={n_runs} | unary={unary_ops} | exp_range={exp_range}"
              f"{' | LOG-SPACE' if use_log_space else ''}")

        # ── Fast power-law prefit (log-log regression) ───────────────────────
        if self._powerlaw_prefit_enabled:
            quick_fit = self._power_law_prefit(X_train, y_train, param_names, full_param_names)
            if quick_fit and (self._best is None or quick_fit.r2 > self._best.r2):
                self._best = quick_fit

        # ── Main PySR fit (with optional diverse restarts) ───────────────────
        model = None
        best_run_r2 = -np.inf

        for run_idx in range(n_runs):
            m = _make_pysr_model(
                iters,
                unary_ops,
                populations=20,
                random_state=run_idx * 42,
                exponent_range=exp_range,
            )
            try:
                m.fit(X_train_df, y_train_fit)
                preds_run_fit = m.predict(X_val_df)
                if use_log_space:
                    preds_run = np.power(10.0, np.clip(preds_run_fit, -300, 300))
                else:
                    preds_run = preds_run_fit
                r2_run = self._r2(y_val, preds_run)
                if r2_run > best_run_r2:
                    best_run_r2 = r2_run
                    model = m
                    if n_runs > 1:
                        print(f"  [run {run_idx+1}/{n_runs}] R²={r2_run:.4f} ← new best")
            except Exception as e:
                print(f"[EquationLearner] PySR run {run_idx+1} failed: {e}")
                continue

        if model is None:
            print("[EquationLearner] All PySR runs failed")
            return self._best

        try:
            best_sympy = model.sympy()
            best_sympy = self._snap_constants(best_sympy)
            # Substitute the injected euler_e feature → exact sympy E (Euler's number),
            # so the recovered expression reads e.g. lambda_constant**E not lambda_constant**euler_e.
            import sympy as _sp
            _euler_sym = _sp.Symbol("euler_e")
            if _euler_sym in best_sympy.free_symbols:
                best_sympy = best_sympy.subs(_euler_sym, _sp.E)
                print("[EquationLearner] euler_e → E substituted in expression")
            if use_log_space:
                # Wrap: true_y = 10 ** (PySR_expr)
                from sympy import Pow, Integer
                best_sympy_wrapped = Pow(Integer(10), best_sympy)
                # Use full_param_names so the law_str signature has ALL variables
                # (evaluate_law() always calls with the full argument list)
                law_str = self._to_law_str(best_sympy_wrapped, full_param_names)
                # For display, show the log-space form clearly
                eq_str = f"10**({best_sympy})"
            else:
                law_str = self._to_law_str(best_sympy, full_param_names)
                eq_str = str(best_sympy)
            y_pred_val_fit = model.predict(X_val_df)
            if use_log_space:
                y_pred_val = np.power(10.0, np.clip(y_pred_val_fit, -300, 300))
            else:
                y_pred_val = y_pred_val_fit
            r2 = self._r2(y_val, y_pred_val)
            rmsle = self._rmsle(y_val, y_pred_val)
        except Exception as e:
            print(f"[EquationLearner] Failed to extract PySR result: {e}")
            return self._best

        # ── Bootstrap confidence (gated on R²) ──────────────────────────────
        # Gate: only run bootstrap if val R² > 0.5.
        # This prevents false positives where a nearly-constant equation has
        # low prediction variance (high "consistency") but is clearly wrong.
        # Even with a good bootstrap score, we down-weight by R² quality.
        confidence = 0.0
        if n_bootstrap > 0 and len(X_train) >= 10 and r2 > 0.5:
            raw_conf = self._bootstrap_confidence(
                X_train_df, y_train, X_val_df,
                unary_ops=unary_ops,
                boot_iters=max(5, iters // 3),
                n_bootstrap=n_bootstrap,
            )
            # Scale by R² quality: linearly ramp from r2=0.5 (conf→0) to r2=0.9 (full conf)
            r2_weight = float(np.clip((r2 - 0.5) / 0.4, 0.0, 1.0))
            confidence = raw_conf * r2_weight

        print(f"[EquationLearner] Best: {eq_str} | R²={r2:.4f} | RMSLE={rmsle:.4f} | conf={confidence:.2f}")

        fitted = FittedEquation(
            law_str=law_str,
            skeleton_str=eq_str,
            r2=r2,
            rmsle=rmsle,
            constants={},
            n_data_points=len(X_train),
            expected_form=eq_str,
            confidence=confidence,
        )
        if fitted.is_valid() and (self._best is None or fitted.r2 > self._best.r2):
            self._best = fitted

        return self._best

    # ── Shared helpers for policy-specific fit methods ─────────────────────

    def _prepare_data(self, store: ExperimentStore):
        """Prepare train/val split and DataFrames. Returns None if insufficient data."""
        import pandas as pd
        if len(store) < 5:
            return None
        X, y = store.to_arrays()
        param_names = self._oracle.parameter_names
        n = len(X)
        rng = np.random.default_rng(42)
        n_val = max(1, int(0.2 * n))
        val_idx = rng.choice(n, n_val, replace=False)
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_idx] = True
        train_mask = ~val_mask
        X_train_df = pd.DataFrame(X[train_mask], columns=param_names)
        X_val_df = pd.DataFrame(X[val_mask], columns=param_names)
        return X[train_mask], y[train_mask], X[val_mask], y[val_mask], X_train_df, X_val_df, param_names

    def _extract_equation(
        self,
        model,
        X_val_df,
        y_val,
        param_names,
        y_scale_val: np.ndarray | None = None,
        full_param_names: list[str] | None = None,
        locked_factor_expr: str | None = None,
    ) -> "FittedEquation | None":
        """Extract a FittedEquation from a fitted PySR model."""
        try:
            from sympy import pycode
            best_sympy = model.sympy()
            try:
                expr_code = pycode(best_sympy, strict=False)
            except TypeError:
                expr_code = pycode(best_sympy)
            except Exception:
                expr_code = (
                    str(best_sympy)
                    .replace("Abs(", "abs(")
                    .replace("re(", "(")
                    .replace("im(", "(")
                )
            y_pred_val = model.predict(X_val_df)
            if y_scale_val is not None:
                y_pred_val = y_pred_val * y_scale_val
            r2 = self._r2(y_val, y_pred_val)
            rmsle = self._rmsle(y_val, y_pred_val)
            eq_str = str(best_sympy)
            full_names = list(full_param_names or param_names)
            if locked_factor_expr:
                eq_str = f"({locked_factor_expr}) * ({eq_str})"
                expr_code = f"({locked_factor_expr}) * ({expr_code})"
            law_str = (
                f"def discovered_law({', '.join(full_names)}):\n"
                f"    return {expr_code}\n"
            )
            return FittedEquation(
                law_str=law_str,
                skeleton_str=eq_str,
                r2=r2,
                rmsle=rmsle,
                constants={},
                n_data_points=len(X_val_df) * 5,
                expected_form=eq_str,
                confidence=0.0,
            )
        except Exception as e:
            print(f"[EquationLearner] Failed to extract equation: {e}")
            return None

    def _fit_pysr_model(
        self,
        X_train_df,
        y_train: np.ndarray,
        *,
        niterations: int,
        unary_ops: list[str],
        exponent_range: tuple[float, float],
        maxsize: int,
        random_state: int,
        binary_operators: list[str] | None = None,
        cache_tag: str = "default",
    ):
        """Fit PySR with optional warm-start chaining across discovery iterations."""
        if not self._warm_start_chaining:
            self._last_fit_key = None
            model = _make_pysr_model(
                niterations,
                unary_ops,
                exponent_range=exponent_range,
                maxsize=maxsize,
                random_state=random_state,
                binary_operators=binary_operators,
                warm_start=False,
            )
            model.fit(X_train_df, y_train)
            return model

        key = (
            cache_tag,
            tuple(X_train_df.columns),
            tuple(unary_ops),
            tuple(binary_operators or ["+", "-", "*", "/", "^"]),
            round(float(exponent_range[0]), 3),
            round(float(exponent_range[1]), 3),
            int(maxsize),
        )
        self._last_fit_key = key
        model = self._warm_models.get(key)
        quality = self._warm_model_quality.get(key)
        warm_ok = bool(
            quality
            and np.isfinite(float(quality.get("r2", float("nan"))))
            and np.isfinite(float(quality.get("rmsle", float("nan"))))
            and float(quality.get("r2", -np.inf)) >= self._warm_start_gate_min_r2
            and float(quality.get("rmsle", np.inf)) <= self._warm_start_gate_max_rmsle
        )
        if model is not None and not warm_ok:
            # Previous population was low quality; reset instead of entrenching.
            self._warm_models.pop(key, None)
            model = None
        if model is None:
            model = _make_pysr_model(
                niterations,
                unary_ops,
                exponent_range=exponent_range,
                maxsize=maxsize,
                random_state=random_state,
                binary_operators=binary_operators,
                warm_start=True,
            )
            self._warm_models[key] = model
        else:
            # Continue evolution from previous population with updated iteration budget.
            try:
                setattr(model, "niterations", niterations)
            except Exception:
                pass

        try:
            model.fit(X_train_df, y_train)
            return model
        except Exception:
            # Corrupted warm state fallback: reset this track once.
            self._warm_models.pop(key, None)
            fresh = _make_pysr_model(
                niterations,
                unary_ops,
                exponent_range=exponent_range,
                maxsize=maxsize,
                random_state=random_state,
                binary_operators=binary_operators,
                warm_start=True,
            )
            self._warm_models[key] = fresh
            fresh.fit(X_train_df, y_train)
            return fresh

    def _update_warm_model_quality(
        self,
        key: tuple[Any, ...] | None,
        *,
        r2: float,
        rmsle: float,
    ) -> None:
        """Store last fit quality for warm-start gate decisions."""
        if not self._warm_start_chaining or key is None:
            return
        try:
            r2_f = float(r2)
        except Exception:
            r2_f = float("nan")
        try:
            rmsle_f = float(rmsle)
        except Exception:
            rmsle_f = float("nan")
        if not np.isfinite(r2_f):
            r2_f = -np.inf
        if not np.isfinite(rmsle_f):
            rmsle_f = np.inf
        self._warm_model_quality[key] = {"r2": r2_f, "rmsle": rmsle_f}

    def _infer_separable_factor_locks(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_names: list[str],
    ) -> list[dict[str, float]]:
        """
        Infer near-separable multiplicative factors y ≈ x_i^a * g(x_rest) from
        warm-sweep-like slices where other parameters are approximately fixed.
        """
        if not self._separable_factor_locking:
            return []
        if self._separable_lock_max_factors <= 0:
            return []
        if len(y_train) < 8 or np.any(y_train <= 0) or np.any(X_train <= 0):
            return []

        scored: list[dict[str, float]] = []
        logX = np.log(np.clip(X_train, 1e-300, None))
        logy = np.log(np.clip(y_train, 1e-300, None))
        d = X_train.shape[1]

        for j in range(d):
            others = [k for k in range(d) if k != j]
            if not others:
                continue
            slice_slopes: list[float] = []
            slice_r2: list[float] = []
            slice_span: list[float] = []
            for q in (0.2, 0.5, 0.8):
                mask = np.ones(len(y_train), dtype=bool)
                for k in others:
                    center = float(np.quantile(logX[:, k], q))
                    mask &= np.abs(logX[:, k] - center) <= 0.25
                if int(np.sum(mask)) < 4:
                    continue
                lx = logX[mask, j]
                ly = logy[mask]
                if lx.size < 4:
                    continue
                span_dec = float((np.max(lx) - np.min(lx)) / np.log(10))
                if span_dec < self._separable_lock_min_span_decades:
                    continue
                slope, intercept = np.polyfit(lx, ly, 1)
                pred = slope * lx + intercept
                ss_res = float(np.sum((ly - pred) ** 2))
                ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
                r2 = 1.0 - ss_res / (ss_tot + 1e-12)
                if r2 < self._separable_lock_min_r2:
                    continue
                slice_slopes.append(float(slope))
                slice_r2.append(float(r2))
                slice_span.append(float(span_dec))

            if len(slice_slopes) < 2:
                continue
            slope_std = float(np.std(slice_slopes))
            if slope_std > self._separable_lock_max_slope_std:
                continue
            slope = float(np.median(slice_slopes))
            r2 = float(np.median(slice_r2))
            span_dec = float(np.median(slice_span))

            scored.append(
                {
                    "param_index": int(j),
                    "param": str(param_names[j]),
                    "exponent": float(slope),
                    "r2": float(r2),
                    "slope_std": float(slope_std),
                    "span_decades": float(span_dec),
                    "quality": float((r2 * span_dec) / (1.0 + slope_std)),
                }
            )

        if not scored:
            return []

        scored.sort(key=lambda rec: float(rec["quality"]), reverse=True)
        max_keep = min(self._separable_lock_max_factors, max(0, len(param_names) - 1))
        chosen = scored[:max_keep]
        return chosen

    def fit_multi_complexity_cascade(
        self,
        store: ExperimentStore,
        goal: str,
        current_hypothesis: str,
        n_iterations: int | None = None,
        operator_profile: dict[str, Any] | None = None,
        run_unrestricted_parallel: bool = False,
    ) -> list["FittedEquation"]:
        """Policy BE: Two-tier PySR (standard + high complexity) followed by
        residual cascade on the best primary fit. Drops the low-complexity tier
        (power-law prefit handles simple forms) and gives more iterations to
        the high-complexity run for better convergence."""
        data = self._prepare_data(store)
        if data is None:
            return []
        X_train, y_train, X_val, y_val, X_train_df, X_val_df, param_names = data

        # Optional separable-factor locking:
        # If y ≈ x_i^a * g(x_rest) is strongly supported by warm-sweep slices,
        # fit PySR on g(x_rest) only and lift predictions back to full y.
        fit_param_names = list(param_names)
        X_train_fit = X_train
        X_val_fit = X_val
        y_train_fit = y_train
        y_scale_val: np.ndarray | None = None
        locked_factor_expr: str | None = None
        lock_records = self._infer_separable_factor_locks(X_train, y_train, param_names)
        if lock_records:
            lock_indices = [int(rec["param_index"]) for rec in lock_records]
            remain_indices = [i for i in range(len(param_names)) if i not in lock_indices]
            if remain_indices:
                train_scale = np.ones(len(y_train), dtype=float)
                val_scale = np.ones(len(y_val), dtype=float)
                parts: list[str] = []
                for rec in lock_records:
                    j = int(rec["param_index"])
                    name = str(rec["param"])
                    exp = float(rec["exponent"])
                    train_scale *= np.clip(X_train[:, j], 1e-300, None) ** exp
                    val_scale *= np.clip(X_val[:, j], 1e-300, None) ** exp
                    parts.append(f"{name}**({exp:.6g})")

                y_adj = y_train / np.clip(train_scale, 1e-300, None)
                ok = np.all(np.isfinite(y_adj)) and np.all(y_adj > 0)
                if ok:
                    import pandas as pd

                    X_train_fit = X_train[:, remain_indices]
                    X_val_fit = X_val[:, remain_indices]
                    fit_param_names = [param_names[i] for i in remain_indices]
                    y_train_fit = y_adj
                    y_scale_val = val_scale
                    locked_factor_expr = " * ".join(parts)
                    X_train_df = pd.DataFrame(X_train_fit, columns=fit_param_names)
                    X_val_df = pd.DataFrame(X_val_fit, columns=fit_param_names)
                    lock_desc = "; ".join(
                        f"{rec['param']}^{rec['exponent']:.3f} (r2={rec['r2']:.3f}, span={rec['span_decades']:.2f}dec)"
                        for rec in lock_records
                    )
                    print(
                        "[EquationLearner] Separable factor lock active: "
                        f"{lock_desc} | residual_params={fit_param_names}"
                    )

        unary_ops_default = _unary_operators_from_hypothesis(
            current_hypothesis,
            getattr(self._oracle, "domain", None),
            getattr(self._oracle, "domain_tags", []),
        )
        exp_range = _exponent_range_from_slopes(store, self._oracle) or _exponent_range_from_hypothesis(current_hypothesis)
        iters = n_iterations or self._n_iterations

        if self._powerlaw_prefit_enabled:
            quick = self._power_law_prefit(X_train, y_train, param_names)
            if quick and (self._best is None or quick.r2 > self._best.r2):
                self._best = quick

        candidates: list[FittedEquation] = []
        best_primary_model = None
        best_primary_eq = None
        selected_unary_ops = list(unary_ops_default)
        selected_binary_ops: list[str] | None = None

        # Phase 1: Two-tier PySR — standard (ms=25) + high complexity (ms=40)
        # High-complexity tier gets 50% more iterations to compensate for larger search space
        tier_configs = [
            (25, iters, 25),
            (40, int(iters * 1.5), 40),
        ]
        def _run_primary_track(
            unary_ops: list[str],
            binary_ops: list[str] | None,
            label: str,
        ) -> tuple[list[FittedEquation], Any | None, FittedEquation | None]:
            track_candidates: list[FittedEquation] = []
            track_best_model = None
            track_best_eq = None
            track_best_r2 = -np.inf
            for ms, tier_iters, seed in tier_configs:
                try:
                    m = self._fit_pysr_model(
                        X_train_df,
                        y_train_fit,
                        niterations=tier_iters,
                        unary_ops=unary_ops,
                        exponent_range=exp_range,
                        maxsize=ms,
                        random_state=seed,
                        binary_operators=binary_ops,
                        cache_tag=f"{label}-ms{ms}-lock{bool(locked_factor_expr)}",
                    )
                    warm_key = self._last_fit_key
                    eq = self._extract_equation(
                        m,
                        X_val_df,
                        y_val,
                        fit_param_names,
                        y_scale_val=y_scale_val,
                        full_param_names=param_names,
                        locked_factor_expr=locked_factor_expr,
                    )
                    if eq and eq.is_valid():
                        self._update_warm_model_quality(
                            warm_key,
                            r2=eq.r2,
                            rmsle=eq.rmsle,
                        )
                        track_candidates.append(eq)
                        if self._best is None or eq.r2 > self._best.r2:
                            self._best = eq
                        if eq.r2 > track_best_r2:
                            track_best_r2 = eq.r2
                            track_best_model = m
                            track_best_eq = eq
                        print(f"  [{label} ms={ms} iters={tier_iters}] {eq.skeleton_str} R²={eq.r2:.4f}")
                    else:
                        # Keep warm-start gate informed even when extraction fails.
                        try:
                            y_pred_val = np.asarray(m.predict(X_val_df), dtype=float)
                            if y_scale_val is not None:
                                y_pred_val = y_pred_val * y_scale_val
                            r2_fail = self._r2(y_val, y_pred_val)
                            rmsle_fail = self._rmsle(y_val, y_pred_val)
                        except Exception:
                            r2_fail = -np.inf
                            rmsle_fail = np.inf
                        self._update_warm_model_quality(
                            warm_key,
                            r2=r2_fail,
                            rmsle=rmsle_fail,
                        )
                except Exception as e:
                    print(f"[EquationLearner] {label} ms={ms} failed: {e}")
            return track_candidates, track_best_model, track_best_eq

        if operator_profile:
            restricted_unary = list(operator_profile.get("unary_ops") or unary_ops_default)
            restricted_binary = list(operator_profile.get("binary_operators") or ["+", "-", "*", "/", "^"])
            print(
                "[EquationLearner] Operator restriction active "
                f"(from validated hypothesis, unary={restricted_unary}, binary={restricted_binary}, "
                f"parallel_unrestricted={run_unrestricted_parallel})"
            )
            c_res, m_res, eq_res = _run_primary_track(
                restricted_unary,
                restricted_binary,
                label="BE restricted",
            )
            candidates.extend(c_res)
            best_primary_model = m_res
            best_primary_eq = eq_res
            selected_unary_ops = restricted_unary
            selected_binary_ops = restricted_binary

            if run_unrestricted_parallel:
                c_unres, m_unres, eq_unres = _run_primary_track(
                    unary_ops_default,
                    None,
                    label="BE unrestricted",
                )
                candidates.extend(c_unres)
                if eq_unres and (
                    best_primary_eq is None or float(eq_unres.r2) > float(best_primary_eq.r2)
                ):
                    best_primary_model = m_unres
                    best_primary_eq = eq_unres
                    selected_unary_ops = list(unary_ops_default)
                    selected_binary_ops = None
        else:
            c_unres, m_unres, eq_unres = _run_primary_track(
                unary_ops_default,
                None,
                label="BE tier",
            )
            candidates.extend(c_unres)
            best_primary_model = m_unres
            best_primary_eq = eq_unres

        # Phase 2: Residual cascade on the best primary model
        if best_primary_model is not None and best_primary_eq is not None and best_primary_eq.r2 > 0.3:
            try:
                y_pred_train = best_primary_model.predict(X_train_df)
                residuals = y_train_fit - y_pred_train
                resid_std = float(np.std(residuals))
                y_std = float(np.std(y_train_fit))

                if resid_std > 0.01 * y_std:
                    resid_model = self._fit_pysr_model(
                        X_train_df,
                        residuals,
                        niterations=iters,
                        unary_ops=selected_unary_ops,
                        exponent_range=exp_range,
                        maxsize=20,
                        random_state=77,
                        binary_operators=selected_binary_ops,
                        cache_tag=f"residual-lock{bool(locked_factor_expr)}",
                    )

                    y_main_val = best_primary_model.predict(X_val_df)
                    y_resid_val = resid_model.predict(X_val_df)
                    y_combined_val = y_main_val + y_resid_val
                    if y_scale_val is not None:
                        y_combined_val = y_combined_val * y_scale_val
                    r2_combined = self._r2(y_val, y_combined_val)

                    if r2_combined > best_primary_eq.r2:
                        main_sympy = best_primary_model.sympy()
                        resid_sympy = resid_model.sympy()
                        from sympy import Add, pycode
                        combined_sympy = Add(main_sympy, resid_sympy)
                        try:
                            local_code = pycode(combined_sympy, strict=False)
                        except TypeError:
                            local_code = pycode(combined_sympy)
                        except Exception:
                            local_code = (
                                str(combined_sympy)
                                .replace("Abs(", "abs(")
                                .replace("re(", "(")
                                .replace("im(", "(")
                            )
                        combined_expr = str(combined_sympy)
                        if locked_factor_expr:
                            local_code = f"({locked_factor_expr}) * ({local_code})"
                            combined_expr = f"({locked_factor_expr}) * ({combined_expr})"
                        combined_law = (
                            f"def discovered_law({', '.join(param_names)}):\n"
                            f"    return {local_code}\n"
                        )
                        rmsle_combined = self._rmsle(y_val, y_combined_val)

                        combined_eq = FittedEquation(
                            law_str=combined_law,
                            skeleton_str=combined_expr,
                            r2=r2_combined,
                            rmsle=rmsle_combined,
                            constants={},
                            n_data_points=len(X_train_df),
                            expected_form=f"({best_primary_eq.skeleton_str}) + ({str(resid_sympy)})",
                            confidence=0.0,
                        )
                        candidates.append(combined_eq)
                        if combined_eq.is_valid() and (self._best is None or combined_eq.r2 > self._best.r2):
                            self._best = combined_eq
                        print(f"  [BE residual_cascade] combined R²={r2_combined:.4f} vs primary R²={best_primary_eq.r2:.4f}")
                    else:
                        print(f"  [BE residual_cascade] no improvement (combined R²={r2_combined:.4f})")
            except Exception as e:
                print(f"[EquationLearner] BE residual cascade failed: {e}")

        return candidates

    def _bootstrap_confidence(
        self,
        X_train_df,
        y_train: np.ndarray,
        X_val_df,
        unary_ops: list[str],
        boot_iters: int,
        n_bootstrap: int,
    ) -> float:
        """
        Run PySR on B bootstrap samples (with replacement) of the training data.
        Measure the coefficient of variation of predictions on the validation set.

        confidence = 1 - mean(std(preds) / |mean(preds)|)

        If all bootstrap models agree on predictions → CV ≈ 0 → confidence ≈ 1.
        If they disagree wildly → CV ≈ 1 → confidence ≈ 0.
        """
        import pandas as pd

        n = len(X_train_df)
        rng = np.random.default_rng(99)
        all_preds = []

        for b in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            X_boot = X_train_df.iloc[idx].reset_index(drop=True)
            y_boot = y_train[idx]

            try:
                m = _make_pysr_model(boot_iters, unary_ops, populations=8)
                m.fit(X_boot, y_boot)
                preds = m.predict(X_val_df)
                if np.all(np.isfinite(preds)):
                    all_preds.append(preds)
            except Exception:
                continue

        if len(all_preds) < 2:
            return 0.0

        all_preds = np.array(all_preds)          # (n_bootstrap, n_val)
        std_preds = np.std(all_preds, axis=0)
        mean_preds = np.abs(np.mean(all_preds, axis=0)) + 1e-10
        mean_cv = float(np.mean(std_preds / mean_preds))

        return float(np.clip(1.0 - mean_cv, 0.0, 1.0))

    def _power_law_prefit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_names: list[str],
        full_param_names: list[str] | None = None,
    ) -> FittedEquation | None:
        """
        Quick log-log linear regression to seed the search with a pure power law:
            y ≈ C * prod_i param_i ** exp_i
        Returns a FittedEquation if valid; else None.

        full_param_names: if provided (and different from param_names), the
            generated law_str will use the full signature so evaluate_law()
            can call discovered_law with all variables.
        """
        if len(y) < 5:
            return None
        if np.any(y <= 0):
            return None
        if np.any(X <= 0):
            return None

        log_y = np.log(y)
        log_X = np.log(X)
        A = np.column_stack([np.ones(len(log_y)), log_X])
        try:
            coeffs, *_ = np.linalg.lstsq(A, log_y, rcond=None)
        except Exception:
            return None
        c0 = float(np.exp(coeffs[0]))
        exps = coeffs[1:]
        y_pred = np.exp(A @ coeffs)
        r2 = self._r2(y, y_pred)
        rmsle = self._rmsle(y, y_pred)

        skeleton_terms = [f"{p}^{exp:.3f}" for p, exp in zip(param_names, exps)]
        skeleton = f"{c0:.6g} * " + " * ".join(skeleton_terms)
        # Use full_param_names for the function signature so evaluate_law()
        # can always call discovered_law with the complete argument list.
        sig_names = full_param_names if full_param_names is not None else param_names
        sig_str = ", ".join(sig_names)
        law_str = (
            f"def discovered_law({sig_str}):\n"
            f"    return {c0:.10g}"
            + "".join([f" * {p}**({exp:.6g})" for p, exp in zip(param_names, exps)])
            + "\n"
        )

        return FittedEquation(
            law_str=law_str,
            skeleton_str=skeleton,
            r2=r2,
            rmsle=rmsle,
            constants={"C0": c0},
            n_data_points=len(y),
            expected_form=skeleton,
            confidence=0.5 if r2 > 0 else 0.1,
        )

    @staticmethod
    def _snap_constants(expr):
        """Replace Float nodes close to known irrationals with their exact sympy values.

        Checks each numeric leaf in the sympy expression tree. If it is within
        0.5% of e, π, √2, ln(2), or φ, it is replaced with the exact symbol.
        This is domain-agnostic: it fires whenever PySR approximates an irrational.
        """
        import sympy as sp
        import numpy as np

        KNOWN = [
            (np.e,               sp.E,           "e"),
            (np.pi,              sp.pi,          "π"),
            (np.sqrt(2),         sp.sqrt(2),     "√2"),
            (np.log(2),          sp.log(2),      "ln2"),
            (np.e + 1.5,         sp.E + sp.Rational(3, 2), "e+1.5"),
            ((1 + np.sqrt(5))/2, (1 + sp.sqrt(5))/2,       "φ"),
        ]
        TOL = 0.005  # 0.5% relative tolerance

        def _snap(node):
            if isinstance(node, sp.Float):
                val = float(node)
                for numeric, symbolic, name in KNOWN:
                    if numeric != 0 and abs(val - numeric) / abs(numeric) < TOL:
                        return symbolic
            return node

        try:
            return expr.xreplace({
                node: _snap(node)
                for node in expr.atoms(sp.Float)
            })
        except Exception:
            return expr

    def _to_law_str(self, sympy_expr, param_names: list[str]) -> str:
        """Convert a sympy expression to a discovered_law Python function string.

        The resulting string is exec'd by evaluate_law with np and math in scope.
        evaluate_law calls discovered_law with scalar values, so math.xxx works fine.
        """
        try:
            from sympy import pycode
            expr_code = pycode(sympy_expr)
            # pycode uses math.xxx which is available in eval scope; no replacement needed
        except Exception:
            # Fallback: sympy str() uses **, Abs, etc. — should parse OK for scalars
            expr_code = str(sympy_expr).replace("Abs(", "abs(")

        params_str = ", ".join(param_names)
        return (
            f"def discovered_law({params_str}):\n"
            f"    return {expr_code}\n"
        )

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if not np.all(np.isfinite(y_pred)) or len(y_true) == 0:
            return float("nan")
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else float("nan")
        return float(1 - ss_res / ss_tot)

    @staticmethod
    def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if not np.all(np.isfinite(y_pred)) or len(y_true) == 0:
            return float("nan")
        y_pred_safe = np.abs(y_pred) + 1e-300
        y_true_safe = np.abs(y_true) + 1e-300
        return float(np.sqrt(np.mean((np.log(y_pred_safe) - np.log(y_true_safe)) ** 2)))

    # ── Direct skeleton fitting (scipy curve_fit on LLM hypothesis forms) ────

    def _skeleton_fit_families(
        self,
        exprs: list[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_names: list[str],
        full_param_names: list[str],
    ) -> "FittedEquation | None":
        """Try fitting each expression in exprs; return the best by validation R²."""
        best: FittedEquation | None = None
        for expr in exprs:
            if not expr or len(expr.strip()) < 3:
                continue
            try:
                result = self._skeleton_fit_one(
                    expr, X_train, y_train, X_val, y_val, param_names, full_param_names
                )
            except Exception as e:
                print(f"[SkeletonFit] exception on '{expr[:40]}': {e}")
                continue
            if result is None:
                continue
            if best is None or result.r2 > best.r2:
                best = result
        return best

    def _skeleton_fit_one(
        self,
        expr: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_names: list[str],
        full_param_names: list[str],
    ) -> "FittedEquation | None":
        """
        Fit a single symbolic expression template via scipy curve_fit.

        The expression uses oracle parameter names as variables and free constants
        (any identifier not in param_names or Python/numpy builtins) as unknowns.

        Strategy:
        - All free constants are assumed positive (enzyme kinetics: kcat, Km, Ki, n, Ea/R).
        - Multiple log-uniform initial conditions are tried; Arrhenius-specific p0
          values (Ea/R ~ 1000–20000) are added when exp() + T are present.
        - Returns None on parse failure, convergence failure, or if all predictions
          are non-finite.
        """
        import re
        import keyword
        from scipy.optimize import curve_fit

        if not expr or len(expr.strip()) < 3:
            return None

        # Pre-substitute known fixed physical constants so they don't appear as free params
        KNOWN_CONSTANTS = {"R": "8.314", "T_ref": "310.0"}
        h = str(expr)
        for name, val in KNOWN_CONSTANTS.items():
            h = re.sub(r"\b" + re.escape(name) + r"\b", val, h)
        # math.e → numeric so it broadcasts in numpy ops
        h = h.replace("math.e", "(2.718281828459045)")

        # Identify free constants = identifiers not in oracle params / Python builtins
        BUILTINS = set(keyword.kwlist) | {
            "np", "math", "exp", "log", "log10", "log2", "sqrt", "abs",
            "sin", "cos", "tan", "pi", "e", "E", "inf", "nan",
            "True", "False", "None", "max", "min", "pow",
            "round", "int", "float", "str", "alpha", "beta", "gamma", "delta",
        }
        oracle_params = set(param_names)
        identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", h))
        free_constants = sorted(identifiers - oracle_params - BUILTINS)

        if not free_constants:
            return None
        if len(free_constants) > 6:
            # Too many unknowns relative to typical experiment budgets
            return None

        n_free = len(free_constants)

        # Build vectorized function: _fn(_X, C0, C1, ...) where _X is (n_features, n_samples)
        const_sig = ", ".join(free_constants)
        param_lines = "\n    ".join(
            f"{p} = _X[{i}]" for i, p in enumerate(param_names)
        )
        fn_code = (
            f"import numpy as _np\n"
            f"def _fn(_X, {const_sig}):\n"
            f"    {param_lines}\n"
            f"    exp = _np.exp\n"
            f"    log = _np.log\n"
            f"    sqrt = _np.sqrt\n"
            f"    pi = _np.pi\n"
            f"    try:\n"
            f"        _r = _np.asarray({h}, dtype=float)\n"
            f"        return _np.where(_np.isfinite(_r) & (_r >= 0), _r, _np.nan)\n"
            f"    except Exception:\n"
            f"        return _np.full(len(_X[0]), _np.nan)\n"
        )
        try:
            ns: dict = {}
            exec(fn_code, ns)
            fn = ns["_fn"]
            # Quick smoke test
            test_X = np.ones((len(param_names), 3))
            test_out = fn(test_X, *([1.0] * n_free))
            if test_out is None or np.all(np.isnan(test_out)):
                return None
        except Exception as e:
            return None

        X_T = X_train.T      # (n_features, n_train)
        X_val_T = X_val.T    # (n_features, n_val)

        # Use log-space fitting when all y > 0 (enzyme kinetics always positive).
        # This makes the optimization objective align with RMSLE evaluation and
        # avoids linear-space local minima that give high R² but terrible RMSLE
        # (e.g., substrate inhibition bell curve with wrong Km/Ki magnitude).
        use_log_fit = np.all(y_train > 0) and np.all(y_val > 0)
        if use_log_fit:
            # Build a log-wrapper: _fn_log returns log(f(X, params))
            fn_log_code = (
                f"import numpy as _np\n"
                f"def _fn_log(_X, {const_sig}):\n"
                f"    _r = _fn(_X, {const_sig})\n"
                f"    _r = _np.asarray(_r, dtype=float)\n"
                f"    return _np.where((_r > 0) & _np.isfinite(_r), _np.log(_r), -1e6)\n"
            )
            ns_log: dict = {"_fn": fn, "_np": np}
            exec(fn_log_code, ns_log)
            fn_fit = ns_log["_fn_log"]
            y_fit_train = np.log(y_train)
            y_fit_val = np.log(y_val)
        else:
            fn_fit = fn
            y_fit_train = y_train
            y_fit_val = y_val

        # Build diverse initial conditions
        rng_s = np.random.default_rng(99)
        p0_list: list[list[float]] = [
            [1.0] * n_free,
            [0.1] * n_free,
            [5.0] * n_free,
            [10.0] * n_free,
            [0.5] * n_free,
            [100.0] * n_free,
        ]
        # Log-uniform random restarts over [10^-2, 10^4]
        for _ in range(20):
            p0_list.append((10 ** rng_s.uniform(-2, 4, n_free)).tolist())
        # Arrhenius-specific: if exp() and T both present, try large Ea/R values
        has_exp_T = ("exp" in h) and ("T" in oracle_params)
        if has_exp_T:
            for ea_r in (500.0, 1000.0, 3000.0, 6000.0, 10000.0, 15000.0, 20000.0):
                for idx in range(n_free):
                    p = [1.0] * n_free
                    p[idx] = ea_r
                    p0_list.append(p)

        best_popt: np.ndarray | None = None
        best_rmsle_train = np.inf

        for p0 in p0_list:
            try:
                popt, _ = curve_fit(
                    fn_fit, X_T, y_fit_train,
                    p0=p0,
                    bounds=(1e-8, 1e7),
                    maxfev=5000,
                    ftol=1e-8,
                    xtol=1e-8,
                )
                y_pred_tr = fn(X_T, *popt)
                if not np.all(np.isfinite(y_pred_tr)):
                    continue
                rmsle_tr = self._rmsle(y_train, y_pred_tr)
                if rmsle_tr < best_rmsle_train:
                    best_rmsle_train = rmsle_tr
                    best_popt = popt.copy()
            except Exception:
                continue

        if best_popt is None or not np.isfinite(best_rmsle_train):
            return None

        # Evaluate on validation set
        try:
            y_pred_val = fn(X_val_T, *best_popt)
        except Exception:
            return None
        if not np.all(np.isfinite(y_pred_val)) or len(y_pred_val) == 0:
            return None

        r2_val = self._r2(y_val, y_pred_val)
        rmsle_val = self._rmsle(y_val, y_pred_val)
        if not np.isfinite(r2_val):
            return None

        # Substitute fitted constants back into expression → human-readable
        substituted = h
        for cname, val in zip(free_constants, best_popt):
            substituted = re.sub(
                r"\b" + re.escape(cname) + r"\b",
                f"({val:.6g})",
                substituted,
            )

        # Build law_str callable for evaluate_law() (uses full param signature, scalars)
        full_names_str = ", ".join(full_param_names)
        law_str = (
            f"def discovered_law({full_names_str}):\n"
            f"    from math import exp, log, sqrt, pi, e\n"
            f"    return {substituted}\n"
        )

        print(
            f"[SkeletonFit] '{expr[:55]}' → "
            f"R²={r2_val:.4f} RMSLE={rmsle_val:.4f} "
            f"[{', '.join(f'{c}={v:.4g}' for c,v in zip(free_constants, best_popt))}]"
        )

        return FittedEquation(
            law_str=law_str,
            skeleton_str=substituted[:120],
            r2=r2_val,
            rmsle=rmsle_val,
            constants=dict(zip(free_constants, best_popt)),
            n_data_points=len(X_train),
            expected_form=f"skel:{expr[:60]}",
            confidence=0.9 if r2_val > 0.9 else (0.6 if r2_val > 0.7 else 0.3),
        )
