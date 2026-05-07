"""
StrongModelConsultant — a high-capacity LLM (e.g. GPT-4o) that the main
discovery agent can query at most `max_calls` times per run.

Acts as an external scientific advisor: it receives a structured description
of observed data patterns and returns a free-text hypothesis about the
underlying mechanism. It does NOT receive the domain name or any vocabulary
list — it must reason from the data alone.

Three consultation points (triggered by DiscoveryLoop):
  1. PATTERNS  — after initial exploration (~20 % of budget):
                  "What mechanism could produce these variable effects?"
  2. RESIDUALS — after first equation fit (~50 % of budget):
                  "What's wrong with this equation? What should we change?"
  3. FINAL     — near end of budget (~80 % of budget):
                  "What is your best guess at the rate law given all the data?"
"""
from __future__ import annotations

import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert biochemist and enzyme kineticist with deep knowledge of
    reaction mechanisms, rate laws, and experimental enzyme kinetics.

    You will be shown a summary of experimental data from an unknown enzyme
    reaction. Your job is to reason from the data patterns alone — without
    being told the mechanism family — and propose a specific rate law.

    Guidelines:
    - Write the rate law as a Python expression (not pseudocode).
    - Name all free parameters clearly (e.g. kcat, Km, Ki, n, Ea, alpha).
    - If the data contradicts standard Michaelis-Menten kinetics, say so
      explicitly and explain what the anomaly implies mechanistically.
    - Be concise. Prioritise precision over completeness.
""")

_CALL1_TEMPLATE = textwrap.dedent("""\
    === ENZYME KINETICS DATA SUMMARY (Consultation 1 of 3) ===

    Inputs measured: C_A [substrate, mM], C_I [putative inhibitor, mM],
    C_B [second substrate / cofactor, mM], C_P [product, mM],
    Enz [enzyme concentration], T [temperature, K], pH.
    Output: r0 [initial rate, mM/min].

    Observed variable effects (from {n_obs} experiments):
    {effects_block}

    Notable anomalies vs. standard Michaelis-Menten:
    {anomalies_block}

    Data range summary:
    {stats_block}

    Based solely on these observations, propose:
    1. The most likely functional form for r0 (Python expression).
    2. Which variables are genuinely relevant (non-trivial effect on r0).
    3. Any unusual features you would design follow-up experiments to confirm.
""")

_CALL2_TEMPLATE = textwrap.dedent("""\
    === ENZYME KINETICS DATA SUMMARY (Consultation 2 of 3) ===

    Observations so far ({n_obs} experiments):
    {effects_block}

    Current best equation found by symbolic regression:
        {current_eq}
    Training RMSLE: {train_rmsle:.4f}

    Systematic residuals (actual − predicted, grouped by variable):
    {residuals_block}

    What specific modification to the equation above would reduce the
    systematic residuals? Write the revised rate law as a Python expression
    with named parameters. If the current equation is fundamentally wrong,
    propose a replacement from scratch.
""")

_CALL3_TEMPLATE = textwrap.dedent("""\
    === ENZYME KINETICS DATA SUMMARY (Consultation 3 of 3) ===

    All {n_obs} experiments collected:
    {effects_block}

    Best equation found so far:
        {current_eq}
    Training RMSLE: {train_rmsle:.4f}

    Remaining systematic patterns not captured by the current equation:
    {residuals_block}

    This is the final consultation. Write your best guess at the exact
    rate law as a Python function:

        def rate(C_A, C_I, C_B, C_P, Enz, T, pH):
            ...

    Include all relevant variables and explain each term briefly.
""")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _summarise_effects(X: np.ndarray, y: np.ndarray,
                        param_names: list[str]) -> tuple[str, str, str]:
    """
    Return (effects_block, anomalies_block, stats_block) strings.
    X: (n, p) array of inputs,  y: (n,) array of log-rates.
    """
    r = np.exp(y)   # back to linear space
    effects = []
    anomalies = []

    for i, name in enumerate(param_names):
        xi = X[:, i]
        if xi.std() < 1e-6:
            continue
        # Pearson correlation with rate
        corr = float(np.corrcoef(xi, r)[0, 1])
        # Spearman (monotonicity)
        rank_x = np.argsort(np.argsort(xi)).astype(float)
        rank_r = np.argsort(np.argsort(r)).astype(float)
        spear = float(np.corrcoef(rank_x, rank_r)[0, 1])

        if abs(corr) < 0.05 and abs(spear) < 0.05:
            direction = "no apparent effect"
        elif spear > 0.5:
            direction = "positive (rate increases)"
        elif spear < -0.5:
            direction = "negative (rate decreases)"
        else:
            direction = f"non-monotonic (corr={corr:.2f}, spear={spear:.2f})"

        effects.append(f"  {name:<8}: {direction}")

        # Flag anomalies
        if name == "C_I" and spear > 0.3:
            anomalies.append(
                f"  ⚠ C_I INCREASES rate (spear={spear:.2f}) — "
                "suggests activation, not inhibition"
            )
        if name == "C_P" and spear > 0.3:
            anomalies.append(
                f"  ⚠ C_P INCREASES rate (spear={spear:.2f}) — "
                "product activation / autocatalysis?"
            )
        if name == "C_A" and spear > 0.1:
            # Check for saturation vs power law
            hi_mask = xi > np.percentile(xi, 75)
            lo_mask = xi < np.percentile(xi, 25)
            if hi_mask.sum() > 2 and lo_mask.sum() > 2:
                hi_r = r[hi_mask].mean()
                lo_r = r[lo_mask].mean()
                hi_x = xi[hi_mask].mean()
                lo_x = xi[lo_mask].mean()
                if hi_x > 0 and lo_x > 0:
                    log_slope = (np.log(hi_r + 1e-9) - np.log(lo_r + 1e-9)) / \
                                (np.log(hi_x) - np.log(lo_x))
                    if log_slope < 0.25:
                        effects.append(
                            f"  {'':8}  (strong saturation — log-log slope≈{log_slope:.2f})"
                        )
                    elif log_slope > 0.75:
                        anomalies.append(
                            f"  ⚠ C_A log-log slope≈{log_slope:.2f} — "
                            "near power-law, minimal saturation"
                        )

    stats = []
    for i, name in enumerate(param_names):
        xi = X[:, i]
        stats.append(
            f"  {name:<8}: [{xi.min():.3g}, {xi.max():.3g}]  "
            f"mean={xi.mean():.3g}"
        )

    effects_str   = "\n".join(effects) if effects else "  (insufficient variation)"
    anomalies_str = "\n".join(anomalies) if anomalies else "  None detected"
    stats_str     = "\n".join(stats)
    return effects_str, anomalies_str, stats_str


def _summarise_residuals(X: np.ndarray, resid: np.ndarray,
                          param_names: list[str]) -> str:
    """Describe where residuals are largest, grouped by variable quartile."""
    lines = []
    for i, name in enumerate(param_names):
        xi = X[:, i]
        if xi.std() < 1e-6:
            continue
        q1, q3 = np.percentile(xi, 25), np.percentile(xi, 75)
        lo_resid = resid[xi <= q1].mean() if (xi <= q1).sum() > 0 else 0.0
        hi_resid = resid[xi >= q3].mean() if (xi >= q3).sum() > 0 else 0.0
        if abs(lo_resid) > 0.05 or abs(hi_resid) > 0.05:
            lines.append(
                f"  {name:<8}: low-quartile bias={lo_resid:+.3f}  "
                f"high-quartile bias={hi_resid:+.3f}"
            )
    return "\n".join(lines) if lines else "  Residuals appear unstructured"


# ---------------------------------------------------------------------------
# Consultant
# ---------------------------------------------------------------------------

@dataclass
class ConsultationRecord:
    call_number: int          # 1, 2, or 3
    trigger:     str          # "patterns" | "residuals" | "final"
    n_obs:       int
    prompt:      str
    response:    str
    elapsed_s:   float


class StrongModelConsultant:
    """
    Wraps a high-capacity LLM (GPT-4o or similar) as a limited-budget
    scientific advisor for the discovery loop.

    Parameters
    ----------
    model      : OpenAI model name, e.g. "gpt-4o" or "o3"
    max_calls  : Hard cap on consultations per run (default 3)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_calls: int = 3,
        api_key: Optional[str] = None,
    ):
        self._model    = model
        self._max      = max_calls
        self._n_calls  = 0
        self._log: list[ConsultationRecord] = []

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "StrongModelConsultant requires OPENAI_API_KEY in environment."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "openai package required: pip install openai"
            ) from e
        self._client = OpenAI(api_key=key)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def calls_remaining(self) -> int:
        return self._max - self._n_calls

    @property
    def calls_used(self) -> int:
        return self._n_calls

    @property
    def log(self) -> list[ConsultationRecord]:
        return list(self._log)

    def consult_on_patterns(
        self,
        X: np.ndarray,
        log_y: np.ndarray,
        param_names: list[str],
    ) -> str:
        """
        Consultation 1 — after initial exploration.
        Returns the strong model's mechanism hypothesis as free text.
        """
        if self._n_calls >= self._max:
            return ""
        effects, anomalies, stats = _summarise_effects(X, log_y, param_names)
        prompt = _CALL1_TEMPLATE.format(
            n_obs=len(log_y),
            effects_block=effects,
            anomalies_block=anomalies,
            stats_block=stats,
        )
        return self._call(prompt, trigger="patterns")

    def consult_on_residuals(
        self,
        X: np.ndarray,
        log_y: np.ndarray,
        param_names: list[str],
        current_eq: str,
        train_rmsle: float,
        pred_log_y: np.ndarray,
    ) -> str:
        """
        Consultation 2 — after first equation fit.
        Returns suggestions for improving the current equation.
        """
        if self._n_calls >= self._max:
            return ""
        effects, _, _ = _summarise_effects(X, log_y, param_names)
        resid = log_y - pred_log_y        # log-space residuals (actual − pred)
        resid_block = _summarise_residuals(X, resid, param_names)
        prompt = _CALL2_TEMPLATE.format(
            n_obs=len(log_y),
            effects_block=effects,
            current_eq=current_eq,
            train_rmsle=train_rmsle,
            residuals_block=resid_block,
        )
        return self._call(prompt, trigger="residuals")

    def consult_on_final(
        self,
        X: np.ndarray,
        log_y: np.ndarray,
        param_names: list[str],
        current_eq: str,
        train_rmsle: float,
        pred_log_y: np.ndarray,
    ) -> str:
        """
        Consultation 3 — near end of budget.
        Returns the strong model's final rate law proposal.
        """
        if self._n_calls >= self._max:
            return ""
        effects, anomalies, _ = _summarise_effects(X, log_y, param_names)
        resid = log_y - pred_log_y
        resid_block = _summarise_residuals(X, resid, param_names)
        prompt = _CALL3_TEMPLATE.format(
            n_obs=len(log_y),
            effects_block=effects,
            current_eq=current_eq,
            train_rmsle=train_rmsle,
            residuals_block=resid_block,
        )
        return self._call(prompt, trigger="final")

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _call(self, user_prompt: str, trigger: str) -> str:
        self._n_calls += 1
        call_num = self._n_calls
        print(f"\n  [Consultant] call {call_num}/{self._max} ({trigger}) → {self._model}")
        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.2,
            )
            response_text = resp.choices[0].message.content or ""
        except Exception as e:
            response_text = f"[Consultant error: {e}]"

        elapsed = time.time() - t0
        self._log.append(ConsultationRecord(
            call_number=call_num,
            trigger=trigger,
            n_obs=int(user_prompt.count("\n")),
            prompt=user_prompt,
            response=response_text,
            elapsed_s=elapsed,
        ))
        print(f"  [Consultant] response ({elapsed:.1f}s):\n"
              + "\n".join(f"    {l}" for l in response_text.splitlines()[:8]))
        return response_text
