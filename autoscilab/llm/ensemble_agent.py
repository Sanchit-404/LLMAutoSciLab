"""
EnsembleAgent: orchestrates the Mechanistic Ensemble Inference (MEI) pipeline.

Flow per round:
  1. Call small LLM K times in parallel (EnsembleClient) to sample K hypotheses.
  2. Build HypothesisDistribution — cluster by functional form, score agreement.
  3. Pass distribution summary to the large LLM for synthesis:
       - Updated best hypothesis (draws on ensemble majority + data)
       - Discriminating experiment proposal (targets high-disagreement regions)
  4. Return the synthesis result to DiscoveryLoop for oracle calls.

The large LLM synthesis step is what makes this different from pure ensemble
voting: the strong model sees the full distribution and can reason about WHY
the small models disagree, proposing experiments that will best resolve the
ambiguity (implicit BOED via language).
"""
from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import numpy as np

from autoscilab.llm.ensemble_client import EnsembleClient
from autoscilab.llm.hypothesis_dist import HypothesisDistribution
from autoscilab.llm.prompts import (
    SYSTEM_PROMPT,
    HypothesisBundle,
    LLMProposal,
    build_hypothesis_generation_prompt,
    build_proposal_prompt,
)

if TYPE_CHECKING:
    from autoscilab.data.store import ExperimentStore
    from autoscilab.llm.client import LLMClient
    from autoscilab.llm.memory import LLMMemory
    from autoscilab.oracle.base import BaseOracle


# ──────────────────────────────────────────────────────────────────────────── #
#  Helper: filter nonsensical ensemble hypotheses                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _filter_valid_expressions(
    expressions: list[str],
    store: "ExperimentStore",
    param_names: list[str],
    max_r2_threshold: float = -1.0,
) -> list[str]:
    """
    Filter ensemble hypotheses that are clearly invalid or nonsensical.

    Drops expressions that:
    - Raise an exception when evaluated on any data point
    - Produce NaN/inf values on more than half the data points
    - Have R² < max_r2_threshold (default: -1.0 = worse than predicting the mean)

    Parameters
    ----------
    expressions : list[str]
        Raw hypothesis strings from the small ensemble.
    store : ExperimentStore
        Current experiment data.
    param_names : list[str]
        Variable names expected by each expression.
    max_r2_threshold : float
        Expressions with R² below this are filtered out.
    """
    if len(store) < 3:
        return expressions  # not enough data to evaluate meaningfully

    try:
        import numpy as np
        X, y = store.to_xy()
        if len(y) < 3:
            return expressions
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot < 1e-15:
            return expressions  # degenerate data, can't compute R²

        valid = []
        for expr in expressions:
            try:
                # Build a simple callable from the expression string
                fn_code = f"def _f({', '.join(param_names)}): return {expr}"
                _ns: dict = {}
                exec(fn_code, {"__builtins__": {"abs": abs, "__import__": __import__}}, _ns)  # noqa: S102
                import math
                _ns["_f"].__globals__.update({"np": np, "exp": np.exp, "log": np.log,
                                               "sqrt": np.sqrt, "abs": np.abs, "math": math})
                exec(fn_code, _ns["_f"].__globals__, _ns)
                _f = _ns["_f"]

                # Evaluate on all data points
                preds = []
                for row in X:
                    kwargs = {name: float(val) for name, val in zip(param_names, row)}
                    try:
                        v = float(_f(**kwargs))
                        preds.append(v)
                    except Exception:
                        preds.append(float("nan"))

                preds_arr = np.array(preds, dtype=float)
                n_bad = np.sum(~np.isfinite(preds_arr))
                if n_bad > len(preds_arr) // 2:
                    continue  # too many NaN/inf

                # Replace remaining NaN with 0 for R² computation
                preds_clean = np.where(np.isfinite(preds_arr), preds_arr, 0.0)
                ss_res = np.sum((y - preds_clean) ** 2)
                r2 = 1.0 - ss_res / ss_tot
                if r2 >= max_r2_threshold:
                    valid.append(expr)
            except Exception:
                pass  # silently drop unparseable expressions

        # Always return at least 1 expression (the least-bad one) if all filtered
        return valid if valid else expressions[:1]
    except Exception:
        return expressions  # fallback: return all


# ──────────────────────────────────────────────────────────────────────────── #
#  Synthesis prompts                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

_SYNTHESIS_SYSTEM = textwrap.dedent("""\
    You are an expert scientist performing symbolic law discovery.
    You will be shown:
      (a) A summary of experimental data collected so far.
      (b) An ensemble of K hypotheses proposed by smaller, less capable models.

    Your job is to synthesise this information and:
      1. Identify the most likely correct functional form, drawing on both the
         data evidence and the ensemble's structural hypotheses.
      2. Propose the single most discriminating experiment — the parameter
         combination where the ensemble hypotheses disagree most strongly —
         so that the next oracle call maximally narrows down the true law.

    CRITICAL GUIDANCE for interpreting the ensemble:
    - Small models (7B parameters) may propose structurally incorrect or
      physically implausible hypotheses. Treat them as weak priors, not answers.
    - A hypothesis that produces NaN/inf on the observed data, or that has
      clearly negative predictive R², should be rejected — it adds no information.
    - ALWAYS use the data as the primary evidence. The ensemble is a hint only.
    - If ensemble members all agree on a structure, verify it fits the data
      before adopting it — agreement among weak models is not strong evidence.

    Be concise and precise. Write all expressions as valid Python.
""")

_SYNTHESIS_USER_TEMPLATE = textwrap.dedent("""\
    {ensemble_summary}

    ---
    DATA COLLECTED SO FAR ({n_obs} experiments):
    {data_table}

    CURRENT BEST HYPOTHESIS: {current_hypothesis}

    ORACLE BUDGET REMAINING: {budget_remaining}

    ---
    Based on the ensemble above and the data, provide:
    1. Your best hypothesis for the governing law (Python expression).
    2. The parameter region most likely to discriminate between the competing
       structures — describe it as a JSON search region.
""")


# ──────────────────────────────────────────────────────────────────────────── #
#  EnsembleAgent                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class EnsembleAgent:
    """
    Orchestrates one MEI round: small LLM ensemble → distribution → synthesis.

    Parameters
    ----------
    ensemble_client : EnsembleClient
        Client for the small local vLLM model (K samples in parallel).
    large_llm_client : LLMClient
        Client for the large strong model (Together/OpenAI).
    oracle : BaseOracle
        The experiment oracle (provides param names, description, etc.).
    memory : LLMMemory
        Shared memory injected into all prompts.
    """

    def __init__(
        self,
        ensemble_client: EnsembleClient,
        large_llm_client: "LLMClient",
        oracle: "BaseOracle",
        memory: "LLMMemory",
        adaptive: bool = False,
        k_max: int = 20,
        stability_threshold: float = 0.1,
    ):
        self._ensemble  = ensemble_client
        self._large_llm = large_llm_client
        self._oracle    = oracle
        self._memory    = memory
        self._call_count = 0
        self._last_distribution: HypothesisDistribution | None = None
        self._adaptive = adaptive
        self._k_max = k_max
        self._stability_threshold = stability_threshold

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_distribution(self) -> HypothesisDistribution | None:
        return self._last_distribution

    # ------------------------------------------------------------------ #
    #  Step 1: Sample K hypotheses from small LLM                         #
    # ------------------------------------------------------------------ #

    def sample_hypothesis_distribution(
        self,
        goal: str,
        store: "ExperimentStore",
        current_hypothesis: str,
        budget_remaining: int,
        total_budget: int,
    ) -> HypothesisDistribution:
        """
        Call the small LLM K times in parallel and build a HypothesisDistribution.

        Each call asks the small model to generate ONE hypothesis expression.
        Temperature sampling provides structural diversity across the K calls.

        Returns
        -------
        HypothesisDistribution
            Distribution over K functional forms with agreement/cluster analysis.
        """
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
            current_phase="discovery",
            best_equation_fit=None,
            memory_str=self._memory.to_prompt_str(),
            slope_hints="",
            objective_type="equation",
            objective_direction="maximize",
            objective_profile={},
        )

        # Sample K hypotheses — fixed or adaptive
        _msg = [{"role": "user", "content": prompt}]
        self._call_count += 1

        if self._adaptive:
            expressions_raw = self._sample_adaptive(_msg)
        else:
            bundles: list[HypothesisBundle] = self._ensemble.complete_json_k(
                messages=_msg, system=SYSTEM_PROMPT, schema=HypothesisBundle,
            )
            expressions_raw = [
                b.hypothesis for b in bundles
                if b.hypothesis and b.hypothesis not in ("unknown", "no data yet — broad exploration needed")
            ]

        # Filter: drop expressions that are not evaluable or have clearly negative R²
        expressions = _filter_valid_expressions(expressions_raw, store, self._oracle.parameter_names)
        n_dropped = len(expressions_raw) - len(expressions)
        if n_dropped > 0:
            print(f"  [EnsembleAgent] dropped {n_dropped}/{len(expressions_raw)} invalid/nonsensical ensemble hypotheses")

        dist = HypothesisDistribution.from_expressions(expressions)
        self._last_distribution = dist

        print(
            f"  [EnsembleAgent] sampled {len(expressions)} hypotheses "
            f"({n_dropped} filtered) — "
            f"{dist.n_unique_structures()} unique structures, "
            f"agreement={dist.agreement_score:.0%}, entropy={dist.entropy():.2f} bits"
        )
        return dist

    def _sample_adaptive(self, messages: list[dict]) -> list[str]:
        """
        Adaptive sampling (v3): sample in batches of ensemble_k until the
        Shannon entropy of the cluster distribution stabilises.
        Stops when |H_new - H_old| < stability_threshold or k_max is reached.
        """
        k_batch = self._ensemble._k
        all_expressions: list[str] = []
        prev_entropy: float | None = None
        total_sampled = 0

        while total_sampled < self._k_max:
            remaining = self._k_max - total_sampled
            batch = min(k_batch, remaining)
            bundles: list[HypothesisBundle] = self._ensemble.complete_json_k(
                messages=messages, system=SYSTEM_PROMPT, schema=HypothesisBundle,
                k_override=batch,
            )
            new_exprs = [
                b.hypothesis for b in bundles
                if b.hypothesis and b.hypothesis not in ("unknown", "no data yet — broad exploration needed")
            ]
            all_expressions.extend(new_exprs)
            total_sampled += batch

            if len(all_expressions) < k_batch:
                continue  # need at least one batch worth before checking stability

            dist = HypothesisDistribution.from_expressions(all_expressions)
            h = dist.entropy()
            if prev_entropy is not None:
                delta = abs(h - prev_entropy)
                print(
                    f"  [MEI-adaptive] k={total_sampled}/{self._k_max}  "
                    f"entropy={h:.3f} bits  Δ={delta:.3f} "
                    f"({'stable ✓' if delta < self._stability_threshold else 'sampling more...'})"
                )
                if delta < self._stability_threshold:
                    break
            prev_entropy = h

        return all_expressions

    # ------------------------------------------------------------------ #
    #  Step 2: Large LLM synthesises from distribution                    #
    # ------------------------------------------------------------------ #

    def synthesize_proposal(
        self,
        goal: str,
        store: "ExperimentStore",
        distribution: HypothesisDistribution,
        current_hypothesis: str,
        budget_remaining: int,
        total_budget: int,
        best_equation_fit: str | None = None,
    ) -> tuple[LLMProposal, str]:
        """
        Pass the ensemble distribution to the large LLM for synthesis.

        The large model sees:
          - The full ensemble summary (structures, agreement, disagreement regions)
          - All experimental data collected so far
          - The current best hypothesis

        It returns:
          - An updated hypothesis (string) — the large model's best guess
          - An experiment proposal (LLMProposal) targeting discriminating regions

        Returns
        -------
        (proposal, updated_hypothesis_str)
        """
        # Build a synthesis-aware proposal prompt
        synthesis_context = (
            f"\n\n=== SMALL-LLM ENSEMBLE ===\n"
            f"{distribution.synthesis_prompt}\n"
            f"=== END ENSEMBLE ===\n\n"
            f"The ensemble above shows {distribution.n_unique_structures()} competing "
            f"functional forms. Use this as a prior when proposing experiments — "
            f"target regions where these structures predict DIFFERENT values.\n"
        )

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
            current_phase="discovery",
            best_equation_fit=best_equation_fit,
            memory_str=self._memory.to_prompt_str(),
            slope_hints=synthesis_context,
            discrimination_hints="",
            objective_type="equation",
            objective_direction="maximize",
            objective_profile={},
        )

        self._call_count += 1
        proposal = self._large_llm.complete_json(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
            schema=LLMProposal,
        )

        # Extract the hypothesis from the proposal (large LLM's synthesis)
        updated_hypothesis = proposal.hypothesis or current_hypothesis

        print(
            f"  [EnsembleAgent] synthesis complete — "
            f"hypothesis: {updated_hypothesis[:80]}..."
        )
        return proposal, updated_hypothesis

    # ------------------------------------------------------------------ #
    #  Combined round                                                      #
    # ------------------------------------------------------------------ #

    def run_round(
        self,
        goal: str,
        store: "ExperimentStore",
        current_hypothesis: str,
        budget_remaining: int,
        total_budget: int,
        best_equation_fit: str | None = None,
    ) -> tuple[LLMProposal, str, HypothesisDistribution]:
        """
        Full MEI round: sample K → build distribution → synthesize.

        Returns
        -------
        (proposal, updated_hypothesis, distribution)
        """
        dist = self.sample_hypothesis_distribution(
            goal=goal,
            store=store,
            current_hypothesis=current_hypothesis,
            budget_remaining=budget_remaining,
            total_budget=total_budget,
        )

        proposal, updated_hyp = self.synthesize_proposal(
            goal=goal,
            store=store,
            distribution=dist,
            current_hypothesis=current_hypothesis,
            budget_remaining=budget_remaining,
            total_budget=total_budget,
            best_equation_fit=best_equation_fit,
        )

        return proposal, updated_hyp, dist
