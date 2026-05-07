"""
LLMAgent: orchestrates LLM calls for experiment proposal and equation suggestion.
"""
import numpy as np

from autoscilab.data.store import ExperimentStore
from autoscilab.llm.client import LLMClient
from autoscilab.llm.memory import LLMMemory
from autoscilab.llm.prompts import (
    SYSTEM_PROMPT,
    OPTIMIZATION_SYSTEM_PROMPT,
    CONFIDENCE_SYSTEM_PROMPT,
    LLMProposal,
    HypothesisBundle,
    EquationSkeleton,
    EquationConfidenceRating,
    DiscriminationPlan,
    build_proposal_prompt,
    build_hypothesis_generation_prompt,
    build_equation_prompt,
    build_confidence_prompt,
    build_validation_prompt,
    build_discrimination_prompt,
)
from autoscilab.oracle.base import BaseOracle


class LLMAgent:
    def __init__(self, client: LLMClient, oracle: BaseOracle, memory: LLMMemory | None = None):
        self._client = client
        self._oracle = oracle
        self._memory = memory or LLMMemory()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def memory(self) -> LLMMemory:
        return self._memory

    def _compute_hypothesis_residuals(
        self,
        store: ExperimentStore,
        current_hypothesis: str,
        n_worst: int = 5,
    ) -> str:
        """
        Evaluate current hypothesis on stored data and return:
        - A DATA RANGE ALERT if y spans >4 orders of magnitude (likely singularity).
        - The top-N worst-fit data points so the LLM sees exactly where it's wrong.

        This is the most direct signal the LLM can get: "your formula predicts 8
        but the true value is 7.8e8 at theta=1.569" is unambiguous evidence that
        the functional form is wrong near a boundary.
        """
        if not current_hypothesis or len(store) < 4:
            return ""
        if current_hypothesis in ("unknown", "no data yet — broad exploration needed"):
            return ""

        X, y = store.to_arrays()
        param_names = self._oracle.parameter_names
        lines = []

        # ── Structural similarity + per-variable slope analysis ──────────────
        # Evaluates the hypothesis on stored data and provides two diagnostics:
        #
        # 1. STRUCTURAL SIMILARITY: Pearson corr(log y_obs, log y_pred).
        #    High correlation (>0.85) means the form captures the right trends
        #    but exponents/constants are off → LLM should refine, not replace.
        #    Low correlation (<0.5) means the functional form is fundamentally wrong.
        #
        # 2. PER-VARIABLE MONOTONICITY CHECK: for each input variable, compare
        #    the sign of the marginal effect in the data vs the hypothesis.
        #    A sign flip means the variable's role is fundamentally misrepresented.
        #    Report as fractional slope discrepancy to avoid false alarms on
        #    exponential laws where absolute slopes are not constant.
        #
        # Also: near-singularity detection for trig laws.
        try:
            import math as _math_sl
            _ns_sl = {
                'sin': _math_sl.sin, 'cos': _math_sl.cos, 'tan': _math_sl.tan,
                'log': _math_sl.log, 'log10': _math_sl.log10, 'exp': _math_sl.exp,
                'sqrt': _math_sl.sqrt, 'asin': _math_sl.asin, 'acos': _math_sl.acos,
                'atan': _math_sl.atan, 'pi': _math_sl.pi, 'e': _math_sl.e,
                'abs': abs, 'sinh': _math_sl.sinh, 'cosh': _math_sl.cosh,
                'tanh': _math_sl.tanh, '__builtins__': {},
            }
            code_sl = compile(current_hypothesis, '<hyp_sl>', 'eval')
            y_pred_sl, y_obs_sl, X_sl = [], [], []
            for xi, yi in zip(X, y):
                if yi <= 0:
                    continue
                try:
                    ns = {**_ns_sl, **{p: float(xi[j]) for j, p in enumerate(param_names)}}
                    yp = float(eval(code_sl, ns))
                    if np.isfinite(yp) and yp > 0:
                        y_pred_sl.append(yp)
                        y_obs_sl.append(float(yi))
                        X_sl.append(xi.copy())
                except Exception:
                    pass

            if len(y_pred_sl) >= 8:
                yp_arr = np.array(y_pred_sl)
                yo_arr = np.array(y_obs_sl)
                Xv = np.array(X_sl)
                log_yo = np.log10(yo_arr)
                log_yp = np.log10(yp_arr)
                log_ratios = log_yo - log_yp

                # ── 1. Structural similarity ──────────────────────────────────
                if np.std(log_yp) > 1e-6 and np.std(log_yo) > 1e-6:
                    corr = float(np.corrcoef(log_yo, log_yp)[0, 1])
                    rms_lr = float(np.sqrt(np.mean(log_ratios**2)))

                    if corr > 0.85 and rms_lr > 0.3:
                        lines.append(
                            f"\n📐 STRUCTURAL HINT: corr(log y_obs, log y_pred) = {corr:.2f} "
                            f"(high — your form captures the right trends). "
                            f"RMS log-error = {rms_lr:.2f} orders. "
                            f"The FUNCTIONAL FORM is likely correct but exponents or "
                            f"coefficients need adjustment. DO NOT change the overall "
                            f"structure (e.g. keep exp if you have exp) — refine the "
                            f"exponents on individual variables instead."
                        )
                    elif corr < 0.5 and rms_lr > 0.5:
                        lines.append(
                            f"\n❌ STRUCTURAL HINT: corr(log y_obs, log y_pred) = {corr:.2f} "
                            f"(low — your form does NOT capture the right trends). "
                            f"The functional form itself is wrong. Consider a fundamentally "
                            f"different structure."
                        )

                # ── 2. Per-variable monotonicity + fractional slope check ─────
                slope_lines = []
                for i, p in enumerate(param_names):
                    col = Xv[:, i]
                    if np.std(col) < 1e-6:
                        continue

                    use_log = np.all(col > 0) and np.std(np.log10(col)) > 0.1
                    x_feat = np.log10(col) if use_log else col

                    emp_slope = float(np.polyfit(x_feat, log_yo, 1)[0])
                    hyp_slope = float(np.polyfit(x_feat, log_yp, 1)[0])

                    # Sign flip: fundamentally wrong direction
                    if np.sign(emp_slope) != np.sign(hyp_slope) and abs(emp_slope) > 0.3:
                        slope_lines.append(
                            f"    {p}: data slope={emp_slope:+.2f} but hypothesis "
                            f"slope={hyp_slope:+.2f} — WRONG SIGN. "
                            f"Your hypothesis has {p} going the wrong direction."
                        )
                    # Large fractional discrepancy (>40% of empirical slope)
                    elif abs(hyp_slope) > 0.3 and abs(emp_slope) > 0.3:
                        frac = (emp_slope - hyp_slope) / abs(emp_slope)
                        if abs(frac) > 0.4:
                            direction = "too strong" if frac < 0 else "too weak"
                            slope_lines.append(
                                f"    {p}: hypothesis effect is {direction} "
                                f"({frac:+.0%} fractional discrepancy vs data). "
                                f"{'Reduce' if frac < 0 else 'Increase'} the exponent "
                                f"or coefficient of {p}."
                            )

                # Near-singularity: extreme log-ratios concentrated near π/2
                lr_range = float(np.max(log_ratios) - np.min(log_ratios))
                if lr_range > 2.0:
                    worst_idx = int(np.argmax(np.abs(log_ratios)))
                    worst_xi = Xv[worst_idx]
                    for i, p in enumerate(param_names):
                        val = float(worst_xi[i])
                        if abs(val - np.pi / 2) < 0.15:
                            slope_lines.append(
                                f"    ⚠ Largest residual at {p}≈{val:.4g} (near π/2). "
                                f"Likely a missing denominator vanishing at {p}→π/2. "
                                f"Try sin({p})**a / cos({p})**b."
                            )

                if slope_lines:
                    lines.append("\n⚠ PER-VARIABLE DISCREPANCY:")
                    lines.extend(slope_lines)
        except Exception:
            pass

        # ── Hypothesis residuals ───────────────────────────────────────────────
        try:
            import math as _math
            _ns_base = {
                'sin': _math.sin, 'cos': _math.cos, 'tan': _math.tan,
                'log': _math.log, 'log10': _math.log10, 'exp': _math.exp,
                'sqrt': _math.sqrt, 'asin': _math.asin, 'acos': _math.acos,
                'atan': _math.atan, 'pi': _math.pi, 'e': _math.e,
                'abs': abs, 'sinh': _math.sinh, 'cosh': _math.cosh,
                'tanh': _math.tanh, '__builtins__': {},
            }
            code = compile(current_hypothesis, '<hypothesis>', 'eval')
            errors = []
            for xi, yi in zip(X, y):
                if yi <= 0:
                    continue
                try:
                    ns = {**_ns_base, **{p: float(xi[j]) for j, p in enumerate(param_names)}}
                    y_pred = eval(code, ns)
                    if y_pred is not None and np.isfinite(float(y_pred)) and float(y_pred) > 0:
                        log_err = abs(float(np.log10(float(y_pred)) - np.log10(yi)))
                        errors.append((log_err, yi, float(y_pred), xi.copy()))
                except Exception:
                    pass

            if errors:
                errors.sort(reverse=True)
                worst = errors[:n_worst]
                n_bad_1oom = sum(1 for e in errors if e[0] > 1.0)
                n_bad_2oom = sum(1 for e in errors if e[0] > 2.0)
                residual_lines = [
                    f"\nHYPOTHESIS RESIDUAL CHECK — current: {current_hypothesis}",
                    f"  {n_bad_1oom}/{len(errors)} points off by >1 order of magnitude  |  "
                    f"{n_bad_2oom}/{len(errors)} off by >2 orders",
                    "  Worst-fit data points (true vs predicted):",
                ]
                for log_err, yi_true, yi_pred, xi in worst:
                    param_str = ", ".join(f"{p}={xi[j]:.4g}" for j, p in enumerate(param_names))
                    residual_lines.append(
                        f"    [{param_str}]  true={yi_true:.4g}  predicted={yi_pred:.4g}"
                        f"  error={log_err:.1f} log-orders {'⚠⚠' if log_err > 3 else '⚠' if log_err > 1 else ''}"
                    )
                if n_bad_1oom > len(errors) // 2:
                    residual_lines.append(
                        "  → MAJORITY of points are poorly fit. The functional FORM is likely wrong, "
                        "not just the constants. Consider a structurally different hypothesis."
                    )
                lines.extend(residual_lines)
        except Exception:
            pass

        return "\n".join(lines) if lines else ""

    @staticmethod
    def _compute_irrational_hint(best_equation_fit: str | None) -> str:
        """
        Option C: scan the best PySR equation string for Float literals that are
        within 0.5% of a known mathematical constant (e, π, √2, ln2, φ).
        Returns a warning string for the LLM prompt so it knows to use the exact
        irrational rather than a fitted approximation.

        Example output:
            ⚠ IRRATIONAL CONSTANT DETECTED: a fitted constant ≈ 2.719 in your
            best equation is very close to Euler's number e ≈ 2.71828.
            Consider writing e (math.e / np.e) explicitly in your hypothesis
            instead of a free constant, as the true law likely uses e exactly.
        """
        if not best_equation_fit:
            return ""
        import re
        import numpy as np

        KNOWN = [
            (np.e,               "Euler's number e ≈ 2.71828",  "e (math.e / np.e)"),
            (np.pi,              "π ≈ 3.14159",                  "pi (math.pi / np.pi)"),
            (np.sqrt(2),         "√2 ≈ 1.41421",                "sqrt(2)"),
            (np.log(2),          "ln(2) ≈ 0.69315",             "log(2)"),
            (np.e + 1.5,         "e+1.5 ≈ 4.21828",             "e + 1.5"),
            ((1 + np.sqrt(5))/2, "φ (golden ratio) ≈ 1.61803",  "(1 + sqrt(5)) / 2"),
        ]
        TOL = 0.005  # 0.5% relative tolerance

        # Find all numeric literals (including negatives, decimals, scientific notation)
        token_re = re.compile(r'(?<![a-zA-Z_])(-?\d+\.?\d*(?:[eE][+-]?\d+)?)')
        hits = []
        for m in token_re.finditer(best_equation_fit):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            if val <= 0:
                continue
            for numeric, name, symbol in KNOWN:
                if abs(val - numeric) / numeric < TOL:
                    hits.append((val, name, symbol))
                    break  # only report the closest match per token

        if not hits:
            return ""

        lines = []
        seen = set()
        for val, name, symbol in hits:
            if name in seen:
                continue
            seen.add(name)
            lines.append(
                f"⚠ IRRATIONAL CONSTANT DETECTED: a fitted constant ≈ {val:.5g} in your "
                f"current best equation is very close to {name}. "
                f"Consider using {symbol} EXPLICITLY in your hypothesis rather than a "
                f"free constant — the true law likely uses this irrational exactly."
            )
        return "\n".join(lines)

    def _compute_slope_hints(self, store: ExperimentStore) -> str:
        """
        Auto-compute per-parameter log-log slopes from stored data and return
        a human-readable hint string.

        For each parameter, fits log10(y) ~ slope * log10(x) + const via linear
        regression. Negative slopes flag inverse/decreasing relationships so the
        LLM knows to consider negative exponents or 1/param terms.

        Returns empty string if fewer than 5 data points or if y has non-positive
        values (can't take log).
        """
        if len(store) < 5:
            return ""

        X, y = store.to_arrays()
        param_names = self._oracle.parameter_names

        # Need positive y for log
        if np.any(y <= 0):
            return ""

        log_y = np.log10(y)
        lines = [
            "EMPIRICAL LOG-LOG SLOPES (auto-computed — use to read off exponents):",
        ]
        any_negative = False

        for i, p in enumerate(param_names):
            xi = X[:, i]
            if np.any(xi <= 0):
                continue
            log_xi = np.log10(xi)
            if np.std(log_xi) < 1e-10:
                continue  # no variation in this param in current data

            # Least-squares slope: d(log10 y)/d(log10 xi)
            slope = float(np.polyfit(log_xi, log_y, 1)[0])

            if slope < -0.3:
                flag = "⚠ NEGATIVE — law likely inverts this param (e.g. y ∝ 1/{p}^{abs(slope):.1f})"
                any_negative = True
            elif slope > 0.3:
                flag = "positive"
            else:
                flag = "near-zero (weak dependence)"
            lines.append(f"  {p}: Δlog10(y)/Δlog10({p}) ≈ {slope:+.2f}  [{flag}]")

        if any_negative:
            lines.append(
                "⚠ INVERSE RELATIONSHIP DETECTED. Include regions with the inverted "
                "parameter in the denominator, e.g. try y = C / (param^n * ...) form."
            )

        return "\n".join(lines)

    def _compute_forward_discrimination_hints(
        self,
        fitted_laws: list[str],
        n_samples: int = 2000,
    ) -> str:
        """
        Sample the FULL parameter space and find where competing fitted hypotheses disagree most.

        Domain-agnostic: evaluates Python function strings on a random grid spanning the
        oracle's full parameter bounds (not just observed data). The top-disagreement regions
        point to unexplored areas that would best discriminate between structural forms.

        Parameters
        ----------
        fitted_laws : list[str]
            Python function definition strings (def discovered_law(...): ...) with fitted
            constants already embedded. At least 2 required.
        n_samples : int
            Number of random candidate points to evaluate.

        Returns
        -------
        str
            Human-readable hint for injection into the proposal prompt, or "" if
            fewer than 2 laws are evaluable.
        """
        if len(fitted_laws) < 2:
            return ""

        import math as _math

        bounds = self._oracle.parameter_bounds
        param_names = self._oracle.parameter_names
        lo = np.array([bounds[p][0] for p in param_names], dtype=float)
        hi = np.array([bounds[p][1] for p in param_names], dtype=float)

        rng = np.random.default_rng(42)
        X = rng.uniform(lo, hi, size=(n_samples, len(param_names)))

        _ns_base: dict = {
            "sin": _math.sin, "cos": _math.cos, "tan": _math.tan,
            "log": _math.log, "log10": _math.log10, "exp": _math.exp,
            "sqrt": _math.sqrt, "asin": _math.asin, "acos": _math.acos,
            "atan": _math.atan, "pi": _math.pi, "e": _math.e,
            "abs": abs, "sinh": _math.sinh, "cosh": _math.cosh,
            "tanh": _math.tanh, "np": np, "__builtins__": {},
        }

        predictions: list[np.ndarray] = []
        for law_str in fitted_laws[:5]:
            try:
                ns = dict(_ns_base)
                exec(law_str, ns)  # noqa: S102
                fn = ns.get("discovered_law")
                if fn is None:
                    continue
                preds = np.array([
                    fn(**{p: float(X[i, j]) for j, p in enumerate(param_names)})
                    for i in range(n_samples)
                ], dtype=float)
                valid = np.isfinite(preds) & (preds > 0)
                if valid.sum() < n_samples * 0.4:
                    continue
                preds[~valid] = np.nan
                predictions.append(preds)
            except Exception:
                continue

        if len(predictions) < 2:
            return ""

        preds_matrix = np.stack(predictions, axis=0)  # (n_laws, n_samples)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_preds = np.log10(preds_matrix)

        # Per-sample disagreement: std of log10 predictions across laws
        disagreement = np.nanstd(log_preds, axis=0)  # (n_samples,)
        valid_mask = np.isfinite(disagreement)
        if valid_mask.sum() < 20:
            return ""

        mean_dis = float(np.nanmean(disagreement[valid_mask]))
        max_dis = float(np.nanmax(disagreement[valid_mask]))

        # Top-disagreement points
        top_k = min(30, int(valid_mask.sum() * 0.05) + 5)
        top_idx = np.where(valid_mask)[0][
            np.argsort(disagreement[valid_mask])[-top_k:][::-1]
        ]
        X_top = X[top_idx]

        # Per-variable: correlation of disagreement with that variable's value
        # (identifies which variables drive the form divergence)
        driving_vars: list[tuple[float, str, str]] = []
        for j, p in enumerate(param_names):
            col = X[valid_mask, j]
            dis_col = disagreement[valid_mask]
            if np.std(col) < 1e-12:
                continue
            corr = float(np.corrcoef(col, dis_col)[0, 1])
            abs_corr = abs(corr)
            if abs_corr > 0.15:
                direction = "high" if corr > 0 else "low"
                driving_vars.append((abs_corr, p, direction))
        driving_vars.sort(reverse=True)

        lines = [
            "\nFORWARD DISAGREEMENT PROFILE (hypotheses evaluated on full parameter space):",
            f"  Disagreement (std log10 predictions): mean={mean_dis:.3f} | peak={max_dis:.3f} log-orders",
        ]

        if max_dis < 0.05:
            lines.append(
                "  Note: hypotheses are nearly identical across parameter space — "
                "structural ambiguity may not be resolvable experimentally."
            )
            return "\n".join(lines)

        lines.append(
            "  Top-disagreement region (propose experiments here to discriminate):"
        )
        for j, p in enumerate(param_names):
            col = X_top[:, j]
            lo_v, hi_v = float(np.min(col)), float(np.max(col))
            center = (lo_v + hi_v) / 2.0
            full_span = float(hi[j] - lo[j])
            pct = (center - float(lo[j])) / (full_span + 1e-300)
            region_label = "low" if pct < 0.33 else ("high" if pct > 0.67 else "mid")
            lines.append(
                f"    {p}: [{lo_v:.4g}, {hi_v:.4g}]  "
                f"(centre={center:.4g} — {region_label} end of full range)"
            )

        if driving_vars:
            top3 = driving_vars[:3]
            var_msgs = [
                f"{v} (forms diverge at {d} values, corr={c:.2f})"
                for c, v, d in top3
            ]
            lines.append(
                "  Variables whose extreme values drive structural divergence: "
                + ", ".join(var_msgs)
            )
            lines.append(
                "  → Sweep these variables to their extremes while fixing others "
                "to reveal which structural form is correct."
            )

        return "\n".join(lines)

    def _compute_prediction_table(
        self,
        fitted_laws: list[str],
        law_labels: list[str] | None = None,
        n_samples: int = 2000,
        n_show: int = 3,
    ) -> str:
        """
        Find the single point of maximum disagreement between competing hypotheses
        and return a concrete prediction table showing what each hypothesis predicts
        there. This lets the LLM reason mechanistically about WHY they diverge.

        Returns a formatted string like:
            Top discriminating point: C_A=9.5, C_I=0.10, Enz=1.00, ...
              H1 (Vmax*Enz*C_A/(Km*(1+C_I/Ki)+C_A)):  4.82  ← highest
              H2 (Vmax*Enz*C_A/(Km+C_A)/(1+C_I/Ki)):  2.31  (2.09× below H1)
              H3 (... arrhenius ...):                   1.94  (2.49× below H1)
            → Measuring here distinguishes H1 from H2/H3.
              If outcome ≈ 4.8 → H2/H3 are ruled out.
              If outcome ≈ 2.3 → H1 is ruled out.
        """
        if len(fitted_laws) < 2:
            return ""

        import math as _math

        bounds = self._oracle.parameter_bounds
        param_names = self._oracle.parameter_names
        lo = np.array([bounds[p][0] for p in param_names], dtype=float)
        hi = np.array([bounds[p][1] for p in param_names], dtype=float)
        labels = (law_labels or [f"H{i+1}" for i in range(len(fitted_laws))])[:len(fitted_laws)]

        rng = np.random.default_rng(7)
        X = rng.uniform(lo, hi, size=(n_samples, len(param_names)))

        _ns_base: dict = {
            "sin": _math.sin, "cos": _math.cos, "tan": _math.tan,
            "log": _math.log, "log10": _math.log10, "exp": _math.exp,
            "sqrt": _math.sqrt, "asin": _math.asin, "acos": _math.acos,
            "atan": _math.atan, "pi": _math.pi, "e": _math.e,
            "abs": abs, "sinh": _math.sinh, "cosh": _math.cosh,
            "tanh": _math.tanh, "np": np, "__builtins__": {},
        }

        predictions: list[np.ndarray] = []
        valid_labels: list[str] = []
        valid_laws: list[str] = []
        for law_str, label in zip(fitted_laws[:5], labels):
            try:
                ns = dict(_ns_base)
                exec(law_str, ns)  # noqa: S102
                fn = ns.get("discovered_law")
                if fn is None:
                    continue
                preds = np.array([
                    fn(**{p: float(X[i, j]) for j, p in enumerate(param_names)})
                    for i in range(n_samples)
                ], dtype=float)
                valid = np.isfinite(preds) & (preds > 0)
                if valid.sum() < n_samples * 0.4:
                    continue
                preds[~valid] = np.nan
                predictions.append(preds)
                valid_labels.append(label)
                valid_laws.append(law_str)
            except Exception:
                continue

        if len(predictions) < 2:
            return ""

        preds_matrix = np.stack(predictions, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_preds = np.log10(preds_matrix)
        disagreement = np.nanstd(log_preds, axis=0)
        valid_mask = np.isfinite(disagreement)
        if valid_mask.sum() < 5:
            return ""

        # Find single top-disagreement point
        best_idx = int(np.nanargmax(disagreement))
        best_point = X[best_idx]
        best_preds = [float(p[best_idx]) if np.isfinite(p[best_idx]) else None
                      for p in predictions]

        point_str = ", ".join(
            f"{p}={best_point[j]:.3g}" for j, p in enumerate(param_names)
        )

        # Determine min/max predictions for the reference comparison
        valid_preds = [(v, lbl) for v, lbl in zip(best_preds, valid_labels) if v is not None]
        if len(valid_preds) < 2:
            return ""
        max_pred = max(v for v, _ in valid_preds)
        min_pred = min(v for v, _ in valid_preds)

        lines = [
            "\nFORWARD PREDICTION TABLE — maximum-disagreement point across competing hypotheses:",
            f"  Point: {point_str}",
        ]
        for pred_val, lbl in zip(best_preds, valid_labels):
            if pred_val is None:
                lines.append(f"  {lbl}: [invalid]")
                continue
            ratio_note = ""
            if pred_val != max_pred:
                ratio_note = f"  ({max_pred/pred_val:.2f}× below top)"
            lines.append(f"  {lbl}: {pred_val:.4g}{ratio_note}")

        lines.append(
            f"  Log-disagreement at this point: {float(disagreement[best_idx]):.3f} orders of magnitude"
        )
        lines.append(
            "DISCRIMINATION REASONING TASK:"
        )
        lines.append(
            f"  A measurement at the above point ({point_str}) would distinguish the competing forms."
        )
        lines.append(
            f"  If measured ≈ {max_pred:.3g} → the lower-predicting hypothesis/hypotheses are ruled out."
        )
        lines.append(
            f"  If measured ≈ {min_pred:.3g} → the higher-predicting hypothesis/hypotheses are ruled out."
        )
        lines.append(
            "  Reason about WHY the hypotheses diverge here (e.g. which structural term causes the gap)"
            " and propose additional experiments in this region to confirm."
        )

        return "\n".join(lines)

    def propose_experiments(
        self,
        goal: str,
        store: ExperimentStore,
        current_hypothesis: str,
        budget_remaining: int,
        total_budget: int,
        current_phase: str = "discovery",
        best_equation_fit: str | None = None,
        discrimination_hints: str = "",
        objective_aware: bool = True,
    ) -> LLMProposal:
        """
        Given current data + hypothesis, propose search regions for the next batch.
        The memory (equation history) is always injected so the LLM can update its hypothesis.
        Auto-computed log-log slope hints are injected to surface inverse relationships.
        """
        slope_hints = self._compute_slope_hints(store)

        objective_type = getattr(self._oracle, "objective_type", "equation")
        objective_direction = getattr(self._oracle, "objective_direction", "maximize")
        objective_profile = getattr(self._oracle, "objective_profile", {})
        if not objective_aware:
            objective_type = "equation"
            objective_direction = "maximize"
            objective_profile = {}

        # Append residual diagnostics to slope_hints for equation tasks
        if objective_type == "equation" and current_hypothesis:
            residual_str = self._compute_hypothesis_residuals(store, current_hypothesis)
            if residual_str:
                slope_hints = (slope_hints + "\n" + residual_str).strip()

        # Option C: inject irrational-constant hint when PySR finds a near-irrational
        irrational_hint = self._compute_irrational_hint(best_equation_fit)
        if irrational_hint:
            slope_hints = (slope_hints + "\n" + irrational_hint).strip()

        prompt = build_proposal_prompt(
            goal=goal,
            domain_id=self._oracle.domain,
            param_names=self._oracle.parameter_names,
            param_description=self._oracle.param_description,
            function_signature=self._oracle.function_signature,
            data_table=store.to_text_table(max_rows=30),
            current_hypothesis=current_hypothesis,
            budget_remaining=budget_remaining,
            total_budget=total_budget,
            current_phase=current_phase,
            best_equation_fit=best_equation_fit,
            memory_str=self._memory.to_prompt_str(),
            slope_hints=slope_hints,
            discrimination_hints=discrimination_hints,
            objective_type=objective_type,
            objective_direction=objective_direction,
            objective_profile=objective_profile,
        )
        system_prompt = SYSTEM_PROMPT
        if objective_aware and objective_type == "optimization":
            system_prompt = OPTIMIZATION_SYSTEM_PROMPT

        self._call_count += 1
        proposal = self._client.complete_json(
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt,
            schema=LLMProposal,
            tool_name="propose_experiments",
        )
        return proposal

    def generate_hypotheses(
        self,
        goal: str,
        store: ExperimentStore,
        current_hypothesis: str,
        budget_remaining: int,
        total_budget: int,
        current_phase: str = "discovery",
        best_equation_fit: str | None = None,
        objective_aware: bool = True,
    ) -> HypothesisBundle:
        slope_hints = self._compute_slope_hints(store)
        objective_type = getattr(self._oracle, "objective_type", "equation")

        # Append residual diagnostics for equation tasks
        if objective_type == "equation" and current_hypothesis:
            residual_str = self._compute_hypothesis_residuals(store, current_hypothesis)
            if residual_str:
                slope_hints = (slope_hints + "\n" + residual_str).strip()

        # Option C: inject irrational-constant hint when PySR finds a near-irrational
        irrational_hint = self._compute_irrational_hint(best_equation_fit)
        if irrational_hint:
            slope_hints = (slope_hints + "\n" + irrational_hint).strip()

        objective_direction = getattr(self._oracle, "objective_direction", "maximize")
        objective_profile = getattr(self._oracle, "objective_profile", {})
        if not objective_aware:
            objective_type = "equation"
            objective_direction = "maximize"
            objective_profile = {}

        prompt = build_hypothesis_generation_prompt(
            goal=goal,
            domain_id=self._oracle.domain,
            param_names=self._oracle.parameter_names,
            param_description=self._oracle.param_description,
            function_signature=self._oracle.function_signature,
            data_table=store.to_text_table(max_rows=30),
            current_hypothesis=current_hypothesis,
            budget_remaining=budget_remaining,
            total_budget=total_budget,
            current_phase=current_phase,
            best_equation_fit=best_equation_fit,
            memory_str=self._memory.to_prompt_str(),
            slope_hints=slope_hints,
            objective_type=objective_type,
            objective_direction=objective_direction,
            objective_profile=objective_profile,
        )
        system_prompt = SYSTEM_PROMPT
        if objective_aware and objective_type == "optimization":
            system_prompt = OPTIMIZATION_SYSTEM_PROMPT

        self._call_count += 1
        return self._client.complete_json(
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt,
            schema=HypothesisBundle,
            tool_name="generate_hypotheses",
        )

    def propose_equation(
        self,
        goal: str,
        store: ExperimentStore,
        current_hypothesis: str,
        previous_skeletons: list[str] | None = None,
    ) -> EquationSkeleton:
        """
        Given accumulated data, propose a Python equation skeleton for fitting.
        (Used when PySR is unavailable; not called in the normal PySR pipeline.)
        """
        prompt = build_equation_prompt(
            goal=goal,
            domain_id=self._oracle.domain,
            param_names=self._oracle.parameter_names,
            function_signature=self._oracle.function_signature,
            data_table=store.to_text_table(max_rows=40),
            previous_attempts=previous_skeletons or [],
            current_hypothesis=current_hypothesis,
        )

        self._call_count += 1
        return self._client.complete_json(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
            schema=EquationSkeleton,
            tool_name="propose_equation",
        )

    def assess_equation_confidence(
        self,
        equation_str: str,
        store: ExperimentStore,
        n_samples: int = 3,
    ) -> float:
        """
        LLM-sampled confidence: call the LLM n_samples times asking it to rate
        how confident it is that equation_str is the correct governing law.
        Returns the mean confidence score across all samples.

        This is independent of bootstrap — it tests whether the LLM's physics
        reasoning agrees with the symbolic regression result.
        """
        prompt = build_confidence_prompt(
            equation_str=equation_str,
            data_table=store.to_text_table(max_rows=25),
            param_names=self._oracle.parameter_names,
        )

        scores = []
        for _ in range(n_samples):
            try:
                self._call_count += 1
                rating = self._client.complete_json(
                    messages=[{"role": "user", "content": prompt}],
                    system=CONFIDENCE_SYSTEM_PROMPT,
                    schema=EquationConfidenceRating,
                    tool_name="rate_equation_confidence",
                )
                scores.append(rating.confidence)
            except Exception as e:
                print(f"[LLMAgent] confidence sample failed: {e}")
                continue

        if not scores:
            return 0.0

        mean_conf = float(np.mean(scores))
        print(f"[LLMAgent] LLM confidence samples={scores} → mean={mean_conf:.3f}")
        return mean_conf

    def propose_validation_experiments(
        self,
        goal: str,
        store: ExperimentStore,
        best_hypothesis: str,
        budget_remaining: int,
        gt_metrics_str: str = "",
    ) -> LLMProposal:
        """
        Adversarial validation: propose experiments that could break the current hypothesis.
        gt_metrics_str: optional ground-truth evaluation string to show the LLM.
        """
        prompt = build_validation_prompt(
            goal=goal,
            domain_id=self._oracle.domain,
            param_names=self._oracle.parameter_names,
            param_description=self._oracle.param_description,
            function_signature=self._oracle.function_signature,
            data_table=store.to_text_table(max_rows=30),
            best_hypothesis=best_hypothesis,
            budget_remaining=budget_remaining,
            gt_metrics_str=gt_metrics_str,
        )

        self._call_count += 1
        proposal = self._client.complete_json(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
            schema=LLMProposal,
            tool_name="propose_experiments",
        )
        # Override phase to validation regardless of what LLM said
        proposal.phase = "validation"
        return proposal

    def analyze_disagreements(
        self,
        goal: str,
        store: ExperimentStore,
        candidates: list[dict],
        budget_remaining: int,
        residual_stats: list[dict] | None = None,
        disagreement_stats: list[dict] | None = None,
    ) -> DiscriminationPlan:
        prompt = build_discrimination_prompt(
            goal=goal,
            domain_id=self._oracle.domain,
            param_names=self._oracle.parameter_names,
            function_signature=self._oracle.function_signature,
            data_table=store.to_text_table(max_rows=25),
            candidates=candidates,
            budget_remaining=budget_remaining,
            residual_stats=residual_stats,
            disagreement_stats=disagreement_stats,
        )
        self._call_count += 1
        return self._client.complete_json(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
            schema=DiscriminationPlan,
            tool_name="analyze_disagreements",
        )
