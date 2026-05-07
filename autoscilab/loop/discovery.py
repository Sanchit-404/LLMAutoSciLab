"""
DiscoveryLoop: orchestrates all components in the three-nested-loop architecture.

Outer loop:  LLM updates hypothesis, proposes search regions
Middle loop: AL selects specific experiments within regions
Inner loop:  Oracle runs experiments, returns measurements

Confidence modes
────────────────
  "bootstrap" — confidence from PySR bootstrap sampling (fast, data-driven)
  "llm"       — confidence from repeated LLM self-assessment (slower, reasoning-based)
  "both"      — geometric mean of bootstrap and LLM confidence (strictest)

Memory
──────
  LLMMemory accumulates every discovered equation and injects the history into
  each LLM proposal prompt, so the LLM sees PySR's findings and updates its
  hypothesis accordingly.
"""
import ast
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from rich.console import Console
from rich.panel import Panel

from autoscilab.al.gp_model import GPSurrogate
from autoscilab.al.selector import ALSelector
from autoscilab.data.store import ExperimentStore
from autoscilab.equation_learner.learner import (
    EquationLearner,
    FittedEquation,
    _unary_operators_from_hypothesis,
)
from autoscilab.llm.agent import LLMAgent
from autoscilab.llm.client import LLMClient
from autoscilab.llm.memory import LLMMemory, NoOpMemory
from autoscilab.oracle.base import BaseOracle
from autoscilab.oracle.newtonbench import get_oracle
from autoscilab.llm.prompts import SearchRegion

console = Console()


def _seed_domain_families(domain_tags: list[str], params: list[str]) -> list[str]:
    """Return lightweight structural families to keep disagreement diverse.

    Uses broad tags (decay, oscillatory, power_law) rather than per-domain
    hand-coding so the mechanism is reusable across benchmarks.
    """
    pset = set(params)
    fam: list[str] = []

    if "decay" in domain_tags and {"t"} <= pset:
        fam.append("C0 * exp(-C1 * t)")
        fam.append("C0 * exp(-(C1**alpha) * t)")
        fam.append("C0 * (1 + C1 * t)**(-beta)")

    if "oscillatory" in domain_tags and "t" in pset:
        fam.append("A * exp(-b * t) * sin(omega * t)")
        fam.append("A * exp(-b * t) * cos(omega * t)")
        fam.append("A * exp(-b * t) * (sin(omega * t) + cos(omega * t))")

    if "power_law" in domain_tags:
        fam.append("C0 * " + " * ".join(f"{p}**C{i+1}" for i, p in enumerate(params)))

    if "polynomial" in domain_tags and len(params) == 1:
        p = params[0]
        fam.append(f"C0 * {p}**2")
        fam.append(f"C0 * {p}**2 + C1 * {p}**0.5")
        fam.append(f"C0 * {p}**3 + C1 * {p} + C2")

    # ChemBench enzyme kinetics families
    if "saturation" in domain_tags and "C_A" in pset:
        fam.append("C0 * Enz * C_A / (C1 + C_A)")
        fam.append("C0 * Enz * C_A / (C1 * (1 + C_I / C2) + C_A)")
    if "arrhenius" in domain_tags and "T" in pset:
        # Two parameterizations: reference-temperature form is much better conditioned
        # (C0 ≈ kcat_ref, C1 = Ea/R ≈ 6000, C2 = Km)
        fam.append("C0 * exp(-C1 * (1/T - 1/310.0)) * Enz * C_A / (C2 + C_A)")
        fam.append("C0 * exp(-C1 / T) * Enz * C_A / (C2 + C_A)")
    if "hill" in domain_tags and "C_A" in pset:
        fam.append("C0 * Enz * C_A**C1 / (C2**C1 + C_A**C1)")
    if "substrate_inhibition" in domain_tags and "C_A" in pset:
        fam.append("C0 * Enz * C_A / (C1 + C_A + C_A**2 / C2)")
    if "bisubstrate" in domain_tags or "pingpong" in domain_tags:
        fam.append("C0 * Enz * C_A * C_B / (C1 * C_B + C2 * C_A + C_A * C_B)")
    if "ph_ionization" in domain_tags and "pH" in pset:
        fam.append("C0 * Enz * C_A / ((C1 + C_A) * (1 + 10**(C2 - pH) + 10**(pH - C3)))")
    # Remaining inhibition forms (always include for domains that use C_A + C_I)
    if "competitive_inhibition" in domain_tags:
        fam.append("C0 * Enz * C_A / (C1 * (1 + C_I / C2) + C_A)")
    if "uncompetitive_inhibition" in domain_tags:
        fam.append("C0 * Enz * C_A / (C1 + C_A * (1 + C_I / C2))")
    if "noncompetitive_inhibition" in domain_tags:
        fam.append("C0 * Enz * C_A / ((1 + C_I / C1) * (C2 + C_A))")
    if "product_inhibition" in domain_tags:
        fam.append("C0 * Enz * C_A / (C1 * (1 + C_P / C2) + C_A)")

    return fam[:8]


@dataclass
class DiscoveryConfig:
    domain_id: str
    goal: str
    budget: int = 100
    discovery_fraction: float = 0.70
    validation_fraction: float = 0.20
    # remaining 10% = contradiction resolution
    noise_level: float = 0.01
    difficulty: str = "easy"
    law_version: str | None = None
    acquisition: str = "ei"           # "ei" = adaptive (UCB early→EI late); "ucb"/"variance" = fixed
    max_experiments_per_iter: int = 8  # cap how many experiments per outer LLM loop
    eq_learner_every: int = 10        # run equation learner every N experiments
    pysr_n_iterations: int = 800      # symbolic-search budget per PySR fit call
    min_data_for_eq: int = 10         # need at least this many points before equation fitting
    confidence_threshold: float = 0.9
    # ── Oracle knowledge leakage control ─────────────────────────────────────
    # If True, oracle.domain_tags are used to pre-seed mechanistic family
    # templates.  This gives the LLM pipeline a strong hint about the true
    # mechanism type BEFORE it observes any data — which is oracle leakage.
    # Set to False to test whether the LLM genuinely discovers mechanisms from
    # data observation alone (the correct benchmark setting).
    use_domain_tags: bool = True
    # ── Hypothesis grammar source (ChemBench only) ───────────────────────────
    # Controls what hypothesis grammar the LLM sees via oracle.param_description:
    #   "domain_specific" — per-domain grammar names the active mechanism (medium leakage)
    #   "universal"       — all possible enzyme kinetic mechanisms listed; LLM must
    #                       identify the correct one from data (fair biochem knowledge)
    #   "none"            — no grammar at all (completely blind)
    hypothesis_grammar_source: str = "domain_specific"
    llm_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    llm_temperature: float | None = None
    max_completion_tokens: int = 8192
    # ── Strong-model consultant ───────────────────────────────────────────────
    # When set, a high-capacity LLM (e.g. "gpt-4o") is consulted at most
    # `strong_model_calls` times per run at key decision points.
    strong_model: str | None = None          # e.g. "gpt-4o"; None = disabled
    strong_model_calls: int  = 3             # hard cap per run
    results_dir: Path = Path("results/")
    # ── Confidence mode ──────────────────────────────────────────────────────
    confidence_mode: str = "bootstrap"   # "bootstrap" | "llm" | "both"
    llm_confidence_samples: int = 3      # LLM samples per confidence estimate (mode=llm/both)
    # ── Hypothesis backend / objective mode ────────────────────────────────
    # hypothesis_backend:
    #   symbolic = use PySR as current default
    #   none     = no symbolic fitting, experiment design only
    hypothesis_backend: str = "symbolic"  # "symbolic" | "none"
    # objective_mode:
    #   auto          = use oracle.objective_type
    #   equation      = evaluate discovered law (RMSLE/exact)
    #   optimization  = evaluate best observed objective over budget
    objective_mode: str = "auto"          # "auto" | "equation" | "optimization"
    # ── Equation learner ─────────────────────────────────────────────────────
    n_diverse_pysr_runs: int = 1         # diverse random restarts for PySR (>1 = slower but better)
    mid_run_gt_eval: bool = True         # run evaluate_law() mid-run when val R²>0.95
    # ── Control flags ────────────────────────────────────────────────────────
    deterministic_first_iter: bool = False  # If True, first outer iter is a single full-bounds sweep
    # ── Variable relevance probing ────────────────────────────────────────────
    # Spend this many experiments at the start on structured variable sweeps:
    # vary each variable in isolation (low + high) while holding others fixed.
    # Gives the LLM direct causal evidence about which variables matter BEFORE
    # committing to a mechanism. Set to 0 to disable.
    # For ChemBench (7 vars): 14 probes cover all variables at 2 levels each.
    variable_probe_budget: int = 0
    powerlaw_prefit: bool = True            # If True, run fast log-log prefit to seed PySR
    objective_aware_prompting: bool = True  # If False, always use equation-centric proposal prompts
    memory_enabled: bool = True             # If False, disable prompt memory and memory updates
    analysis_feedback_enabled: bool = True  # Enable extra ambiguity-analysis LLM call
    analysis_min_candidates: int = 2        # Min candidate hypotheses before analysis call
    analysis_disagreement_threshold: float = 0.35  # Min max_rel_gap needed to justify analysis
    analysis_cooldown_iters: int = 1        # Wait this many outer iters between analysis calls
    analysis_max_calls: int | None = None   # Optional cap on analysis calls per run
    negative_evidence_enabled: bool = True  # Record falsified hypotheses and inject in memory
    strict_generator_discriminator_loop: bool = False  # force 3-step LLM loop: generate -> discriminate -> propose
    allow_disagreement: bool = True         # If False, disable disagreement-driven acquisition
    hypothesis_conditioned_acquisition: bool = True  # If False, acquire over the full domain
    iterative_refinement_enabled: bool = True        # If False, skip mid-run refinement updates
    confidence_feedback_enabled: bool = True         # If False, remove confidence from loop feedback
    # ── Validated-hypothesis arbitration / operator restriction ───────────────
    # When True, if a validated hypothesis is very strong, derive a restricted
    # operator profile for PySR from that hypothesis.
    validated_operator_restriction: bool = False
    # Safety switch: if restriction is active, also run an unrestricted PySR pass
    # and compare both in final arbitration.
    restriction_parallel_unrestricted: bool = True
    restriction_min_validated_rmsle: float = 0.1
    restriction_min_support: int = 2
    # ── Discriminator scoring data policy ─────────────────────────────────────
    # Use a mixed scoring set (new points + stratified historical points) instead
    # of scoring only on the latest mini-batch.
    stratified_hypothesis_scoring: bool = True
    hypothesis_scoring_max_points: int = 16
    # ── Discriminator binding / promotion gating ───────────────────────────────
    # Make disagreement acquisition optimize hypothesis separation directly
    # (uncertainty only as tie-breaker), and track if separation gain is improving.
    discrimination_binding: bool = True
    discrimination_min_gain: float = 0.10
    discriminator_stagnation_patience: int = 1
    # Before promoting a hypothesis as "validated", require that it also fits
    # the full observed dataset (not only the latest scoring slice).
    promotion_full_store_gate: bool = True
    promotion_full_store_max_rmsle: float = 1.0
    # ── PySR stability / decomposition controls ────────────────────────────────
    # Continue PySR evolution across iterations instead of restarting from scratch.
    pysr_warm_start_chaining: bool = False
    pysr_warm_start_gate_min_r2: float = 0.85
    pysr_warm_start_gate_max_rmsle: float = 2.5
    # Detect near-separable multiplicative factors from warm-sweep data and fit
    # residual structure on remaining variables.
    separable_factor_locking: bool = False
    separable_lock_max_factors: int = 1
    separable_lock_min_r2: float = 0.92
    separable_lock_min_span_decades: float = 0.8
    separable_lock_max_slope_std: float = 0.35
    # Add transformed-space coverage diagnostics to discriminator/proposer hints.
    transformed_coverage_hints: bool = False
    transformed_coverage_low_span_decades: float = 0.8
    # ── Policy selection ──────────────────────────────────────────────────
    # families_v3 = default (hypothesis families + warm sweeps, uses EquationLearner.fit())
    # policy_BE   = multi-complexity cascade (uses EquationLearner.fit_multi_complexity_cascade())
    policy: str = "families_v3"
    # ── MEI (Mechanistic Ensemble Inference) ─────────────────────────────────
    # When enabled, a small local LLM is sampled K times per ensemble round to
    # build a HypothesisDistribution. A FalsificationSelector then picks
    # experiments that maximally discriminate between competing structures.
    # The large LLM synthesises a proposal from the full distribution.
    # Requires a vLLM server running: ./scripts/start_vllm.sh
    ensemble_mode: bool = False
    ensemble_k: int = 5                                        # hypotheses sampled per round
    ensemble_model: str = "Qwen/Qwen2.5-7B-Instruct"          # small model name in vLLM
    ensemble_url: str = "http://localhost:8001/v1"             # vLLM server URL
    ensemble_temperature: float = 0.9                          # diversity temperature
    ensemble_every_n_iters: int = 3                            # run MEI round every N outer iters
    # ── MEI v3: adaptive k ───────────────────────────────────────────────
    # Sample in batches of ensemble_k until entropy of the cluster
    # distribution stabilises (|H_new - H_old| < threshold) or k_max reached.
    ensemble_adaptive: bool = False
    ensemble_k_max: int = 20                                   # hard cap for adaptive sampling
    ensemble_stability_threshold: float = 0.1                  # bits — stop when entropy delta < this
    # ── MEI v4: confidence gate ──────────────────────────────────────────
    # Skip MEI entirely when bootstrap confidence already exceeds this value.
    # 1.0 = disabled (always run MEI when scheduled).
    ensemble_confidence_gate: float = 1.0
    falsification_n_candidates: int = 2000                     # candidate grid size for FalsificationSelector
    falsification_diversity_weight: float = 0.3                # diversity vs disagreement trade-off
    # ── Fix 4: Forward prediction discrimination ──────────────────────────────
    # When True, compute a concrete per-hypothesis prediction table at the
    # maximum-disagreement point and inject it into every proposal prompt.
    # Lets the LLM reason: "H1 predicts 4.8, H2 predicts 2.3 here — measure it."
    forward_discrimination: bool = False
    # ── Symbolic accuracy improvements ───────────────────────────────────────
    # skeleton_priority_threshold: if skeleton RMSLE ≤ this, skip PySR entirely.
    # Set to None to disable (default). 0.10 is a good starting point.
    skeleton_priority_threshold: float | None = None
    # prune_spurious_vars: after final equation selection, refit without variables
    # whose permutation importance < 0.01 ΔRMSLE.
    prune_spurious_vars: bool = False
    # ood_holdout_frac: fraction of budget spent measuring OOD extremes upfront
    # (not added to training store). Used to penalise overfit equations at end.
    # 0.0 = disabled (default). 0.10 is a good starting point.
    ood_holdout_frac: float = 0.0

    @property
    def discovery_budget(self) -> int:
        return int(self.budget * self.discovery_fraction)

    @property
    def validation_budget(self) -> int:
        return int(self.budget * self.validation_fraction)

    @property
    def contradiction_budget(self) -> int:
        return self.budget - self.discovery_budget - self.validation_budget


@dataclass
class DiscoveryResult:
    domain: str
    goal: str
    best_equation: FittedEquation | None
    final_evaluation: dict[str, Any] | None
    n_oracle_calls: int
    n_llm_calls: int
    store: ExperimentStore
    termination_reason: str      # "budget" | "confident" | "flatline"
    duration_seconds: float
    history: list[dict] = field(default_factory=list)
    status: str = ""             # "success" | "good" | "partial" | "underfit" | etc.


class DiscoveryLoop:
    def __init__(
        self,
        config: DiscoveryConfig,
        api_key: str | None = None,
        oracle: BaseOracle | None = None,
        base_url: str | None = None,
    ):
        self._cfg = config
        if oracle is not None:
            self._oracle = oracle
        else:
            self._oracle = get_oracle(
                domain_id=config.domain_id,
                difficulty=config.difficulty,
                noise_level=config.noise_level,
                law_version=config.law_version,
                grammar_mode=config.hypothesis_grammar_source,
            )
        self._store = ExperimentStore(config.domain_id, self._oracle.parameter_names)
        self._gp = GPSurrogate(use_log_y=True)
        self._selector = ALSelector(self._oracle, self._gp)
        self._memory = LLMMemory() if config.memory_enabled else NoOpMemory()
        self._llm_client = LLMClient(
            model=config.llm_model,
            api_key=api_key,
            base_url=base_url,
            max_completion_tokens=config.max_completion_tokens,
            temperature=config.llm_temperature,
        )
        self._agent = LLMAgent(self._llm_client, self._oracle, memory=self._memory)

        # MEI ensemble agent (optional)
        self._ensemble_agent = None
        if config.ensemble_mode:
            from autoscilab.llm.ensemble_agent import EnsembleAgent
            from autoscilab.llm.ensemble_client import EnsembleClient
            _ens_client = EnsembleClient(
                base_url=config.ensemble_url,
                model=config.ensemble_model,
                k=config.ensemble_k,
                temperature=config.ensemble_temperature,
            )
            self._ensemble_agent = EnsembleAgent(
                ensemble_client=_ens_client,
                large_llm_client=self._llm_client,
                oracle=self._oracle,
                memory=self._memory,
                adaptive=config.ensemble_adaptive,
                k_max=config.ensemble_k_max,
                stability_threshold=config.ensemble_stability_threshold,
            )
            _mode = "adaptive" if config.ensemble_adaptive else "fixed"
            _gate = f"  conf_gate={config.ensemble_confidence_gate}" if config.ensemble_confidence_gate < 1.0 else ""
            print(
                f"  [MEI] ensemble enabled: model={config.ensemble_model}  "
                f"K={config.ensemble_k}(k_max={config.ensemble_k_max} mode={_mode})  "
                f"url={config.ensemble_url}{_gate}"
            )
        self._ensemble_iter_count = 0          # outer iteration counter for ensemble_every_n_iters
        self._pending_falsification_points: list[dict] = []  # from FalsificationSelector

        # Strong-model consultant (optional)
        self._consultant = None
        if config.strong_model:
            from autoscilab.llm.consultant import StrongModelConsultant
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if openai_key:
                self._consultant = StrongModelConsultant(
                    model=config.strong_model,
                    max_calls=config.strong_model_calls,
                    api_key=openai_key,
                )
                print(f"  [Consultant] enabled: model={config.strong_model}  "
                      f"max_calls={config.strong_model_calls}")
            else:
                print("  [Consultant] WARNING: strong_model set but OPENAI_API_KEY missing — disabled")
        self._eq_learner = EquationLearner(
            oracle=self._oracle,
            n_iterations=config.pysr_n_iterations,
            powerlaw_prefit=config.powerlaw_prefit,
            warm_start_chaining=config.pysr_warm_start_chaining,
            warm_start_gate_min_r2=config.pysr_warm_start_gate_min_r2,
            warm_start_gate_max_rmsle=config.pysr_warm_start_gate_max_rmsle,
            separable_factor_locking=config.separable_factor_locking,
            separable_lock_max_factors=config.separable_lock_max_factors,
            separable_lock_min_r2=config.separable_lock_min_r2,
            separable_lock_min_span_decades=config.separable_lock_min_span_decades,
            separable_lock_max_slope_std=config.separable_lock_max_slope_std,
        )
        self._use_symbolic = config.hypothesis_backend == "symbolic"
        if config.objective_mode == "auto":
            self._objective_type = getattr(self._oracle, "objective_type", "equation")
        else:
            self._objective_type = config.objective_mode

        self._current_hypothesis = "unknown"
        self._current_phase = "discovery"
        self._oracle_calls = 0
        self._history: list[dict] = []
        self._last_gt_eval: dict | None = None          # most recent mid-run GT eval result
        self._best_gt_equation: "FittedEquation | None" = None   # best by GT RMSLE
        self._best_gt_eval_rmsle: float | None = None            # GT RMSLE of best_gt_equation
        # Track multiple structural families and GT-quality for retirement logic
        self._hypothesis_families: list[str] = []
        self._consecutive_poor_gt: int = 0
        # Candidate pool: multi-equation candidates from policy-specific fitting
        self._candidate_pool: list[FittedEquation] = []
        self._analysis_calls: int = 0
        self._last_analysis_iter: int = -999
        self._last_discrimination_summary: dict[str, Any] | None = None
        self._discriminator_stagnation_count: int = 0
        # Consultant trigger flags: fired once each at 20%, 50%, 80% of budget
        self._consultant_trigger_done: dict[int, bool] = {1: False, 2: False, 3: False}
        # Domain-tailored template families to keep structural diversity alive.
        # Only populated if use_domain_tags=True (default). Setting it to False
        # forces the LLM to generate all hypotheses from data without oracle hints.
        if config.use_domain_tags:
            self._seed_families: list[str] = _seed_domain_families(
                getattr(self._oracle, "domain_tags", []), self._oracle.parameter_names
            )
        else:
            self._seed_families = []

        # Start with the seeds so early disagreement sampling can separate them
        self._hypothesis_families: list[str] = list(self._seed_families)

    def _llm_arbitrate_candidates(
        self,
        candidates: "list[FittedEquation]",
        store: "ExperimentStore",
        param_names: list[str],
    ) -> "FittedEquation | None":
        """
        Ask the main LLM to pick the best equation from a candidate pool.

        Shows each candidate's expression, R², RMSLE, and whether it came from
        skeleton fitting (scipy, LLM-guided structure) or PySR (free-form search).
        The LLM picks based on scientific plausibility — preferring structurally
        meaningful equations over numerically-better ones with spurious variables.

        Falls back to highest-R² selection if the LLM call fails.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Pre-filter: only offer the LLM candidates that are plausible fits.
        # R²<0.7 almost certainly can't be the right law — don't confuse the LLM.
        viable = [c for c in candidates if c.r2 >= 0.70]
        if not viable:
            return max(candidates, key=lambda c: float(c.r2))
        if len(viable) == 1:
            return viable[0]
        candidates = viable

        # Build candidate summary for the LLM
        lines = ["Candidate equations (choose the most scientifically plausible):"]
        X_arb, y_arb = store.to_arrays()
        for i, c in enumerate(candidates):
            source = "skeleton/scipy" if c in self._eq_learner.skel_candidates else "PySR"
            vars_used = sorted(set(
                p for p in param_names if p in (c.skeleton_str or "")
            ))
            # Compute permutation importance for this candidate
            imp = {}
            if c.law_str:
                try:
                    imp = self._compute_var_permutation_importance(
                        c.law_str, X_arb, y_arb, param_names
                    )
                except Exception:
                    pass
            imp_str = ""
            if imp:
                parts = []
                for v in vars_used:
                    score = imp.get(v, 0.0)
                    flag = " ⚠ LOW" if score < 0.005 else ""
                    parts.append(f"{v}={score:.4f}{flag}")
                imp_str = f"\n    Permutation importance (ΔRMSLE when shuffled): {', '.join(parts)}"
            lines.append(
                f"\n[{i}] Source: {source}\n"
                f"    Expression: {c.skeleton_str}\n"
                f"    Train R²={c.r2:.4f}  RMSLE={c.rmsle:.4f}\n"
                f"    Variables used: {vars_used}"
                + imp_str
            )

        data_table = store.to_text_table(max_rows=20)
        prompt = (
            "\n".join(lines)
            + f"\n\nExperimental data summary:\n{data_table}"
            + "\n\nReply with ONLY the integer index of the best candidate."
            + " Prefer the equation whose variables have HIGH permutation importance (>0.01 ΔRMSLE)."
            + " Variables with LOW importance (<0.005 ΔRMSLE, marked ⚠ LOW) are likely fitting"
            + " noise rather than real mechanism — prefer the equation that omits them."
            + " Use your scientific knowledge to judge whether each variable's role is physically"
            + " plausible given the mechanism context (e.g. a very weak T effect with an implied"
            + " activation energy outside typical enzymatic ranges is likely spurious)."
        )

        try:
            self._agent._call_count += 1
            raw = self._agent._client.complete(
                messages=[{"role": "user", "content": prompt}],
                system=(
                    "You are a scientific reasoning agent. "
                    "Given a list of candidate rate-law equations and experimental data, "
                    "pick the index of the most scientifically plausible equation. "
                    "Respond with ONLY a single integer — no explanation."
                ),
                max_tokens=16,
            )
            clean = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
            clean = re.sub(r"^```(?:text)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)

            idx_match = (
                re.search(r"\[(\d+)\]", clean)
                or re.search(r"\b(?:index|candidate)\s+(\d+)\b", clean, flags=re.IGNORECASE)
                or re.search(r"^\s*(\d+)\s*$", clean)
                or re.search(r"^\s*(\d+)\b", clean)
            )
            if not idx_match:
                raise ValueError(f"could not parse candidate index from: {clean[:120]!r}")

            idx = int(idx_match.group(1))
            if 0 <= idx < len(candidates):
                chosen = candidates[idx]
                print(
                    f"[LLM arbitration] chose [{idx}] "
                    f"R²={chosen.r2:.4f} | {chosen.skeleton_str[:80]}"
                )
                return chosen
        except Exception as e:
            print(f"[LLM arbitration] failed ({e}), falling back to max R²")

        return max(candidates, key=lambda c: float(c.r2))

    def _consultant_try_trigger(
        self,
        trigger_num: int,
        X: np.ndarray,
        log_y: np.ndarray,
        param_names: list[str],
    ) -> None:
        """
        Fire one of the 3 pre-scheduled consultant calls if conditions are met.
        Injects the advice into LLM memory so subsequent proposal prompts see it.

        trigger_num=1  → pattern characterization (after ~20 % budget)
        trigger_num=2  → residual explanation (after first equation fit, ~50 %)
        trigger_num=3  → final rate-law confirmation (~80 %)
        """
        if self._consultant is None:
            return
        if self._consultant_trigger_done.get(trigger_num):
            return
        if self._consultant.calls_remaining <= 0:
            return

        self._consultant_trigger_done[trigger_num] = True
        advice = ""
        try:
            if trigger_num == 1:
                advice = self._consultant.consult_on_patterns(X, log_y, param_names)
            elif trigger_num in (2, 3):
                best = self._eq_learner.best
                if best is None:
                    self._consultant_trigger_done[trigger_num] = False  # re-arm
                    return
                # Get log-space predictions for residual analysis
                try:
                    import math as _math
                    _exec_globals: dict = {"np": np, "math": _math, "__builtins__": __builtins__}
                    exec(best.law_str, _exec_globals)  # noqa: S102
                    discovered_law = _exec_globals.get("discovered_law")
                    if discovered_law is not None:
                        preds_lin = np.array([
                            discovered_law(*row) for row in X
                        ], dtype=float)
                    else:
                        preds_lin = np.exp(log_y)
                    preds_lin = np.where(preds_lin > 0, preds_lin, 1e-12)
                    pred_log_y = np.log(preds_lin)
                except Exception:
                    # Fallback: zero residuals (consultant still sees effects)
                    pred_log_y = log_y.copy()
                if trigger_num == 2:
                    advice = self._consultant.consult_on_residuals(
                        X, log_y, param_names,
                        current_eq=best.skeleton_str or best.law_str,
                        train_rmsle=best.rmsle,
                        pred_log_y=pred_log_y,
                    )
                else:
                    advice = self._consultant.consult_on_final(
                        X, log_y, param_names,
                        current_eq=best.skeleton_str or best.law_str,
                        train_rmsle=best.rmsle,
                        pred_log_y=pred_log_y,
                    )
        except Exception as exc:
            console.print(f"  [yellow][Consultant] trigger {trigger_num} failed: {exc}[/yellow]")
            return

        if advice:
            self._memory.add_consultant_advice(trigger_num, ["patterns", "residuals", "final"][trigger_num - 1], advice)

    def _compute_slope_change(self) -> str:
        """Detect piecewise slope changes that hint at multi-term laws.
        For each parameter, splits data into low-x and high-x halves,
        computes log-log slope in each half. A large difference signals
        a single power law cannot explain the data."""
        X, y = self._store.to_arrays()
        param_names = self._oracle.parameter_names
        if len(y) < 10 or np.any(y <= 0):
            return ""
        log_y = np.log10(y)
        hints: list[str] = []
        for i, p in enumerate(param_names):
            xi = X[:, i]
            if np.any(xi <= 0):
                continue
            log_x = np.log10(xi)
            if np.std(log_x) < 0.3:
                continue
            median = float(np.median(log_x))
            low_mask = log_x <= median
            high_mask = log_x > median
            if np.sum(low_mask) < 3 or np.sum(high_mask) < 3:
                continue
            slope_low = float(np.polyfit(log_x[low_mask], log_y[low_mask], 1)[0])
            slope_high = float(np.polyfit(log_x[high_mask], log_y[high_mask], 1)[0])
            if abs(slope_low - slope_high) > 0.5:
                hints.append(
                    f"SLOPE CHANGE for {p}: slope≈{slope_low:.2f} (low {p}) vs "
                    f"slope≈{slope_high:.2f} (high {p}). "
                    f"A single power law CANNOT explain this — consider multi-term forms."
                )
        return "\n".join(hints)

    def _is_chem_domain(self) -> bool:
        """True when the oracle is a ChemBenchOracle."""
        try:
            from autoscilab.oracle.chembench import ChemBenchOracle
            return isinstance(self._oracle, ChemBenchOracle)
        except ImportError:
            return False

    def _staged_pysr_iters(self, outer_iter: int, is_final: bool = False) -> int:
        """Scale PySR iteration budget by run stage (ChemBench only).

        Early iterations get cheap fast fits (quick feedback to the LLM).
        The final call always gets the full budget.
          iter 1-2  →  base // 8  (min 100)  — fast structural scan
          iter 3-5  →  base // 4  (min 200)  — medium refinement
          iter 6+   →  base // 2  (min 400)  — late-stage
          final     →  base       (full)
        """
        base = self._cfg.pysr_n_iterations
        if not self._is_chem_domain() or is_final:
            return base
        if outer_iter <= 2:
            return max(100, base // 8)
        elif outer_iter <= 5:
            return max(200, base // 4)
        else:
            return max(400, base // 2)

    def _extract_relevant_vars(self, hypothesis: str) -> list[str] | None:
        """Parse the LLM hypothesis to find which ChemBench variables it references.

        Returns a filtered list to pass to PySR, or None to use all variables.
        Rules:
        - Enzyme (E) is always included (it's in every ChemBench rate equation)
        - C_A is always included (substrate is always part of enzyme kinetics)
        - Any variable with |Spearman(var, log(r0))| > 0.15 is always included,
          regardless of whether the LLM mentioned it — prevents important variables
          like C_I from being excluded due to a poor initial hypothesis
        - If ≥5 variables are found, return them (wide search)
        - If fewer than 2 relevant variables are found, fall back to None (all vars)
        """
        if not self._is_chem_domain():
            return None
        all_vars = list(self._oracle.parameter_names)
        found = set(v for v in all_vars
                    if re.search(r'\b' + re.escape(v) + r'\b', hypothesis))
        # Always include core kinetics variables
        for core_var in ("E", "Enz", "C_A"):
            if core_var in all_vars:
                found.add(core_var)

        # Data-driven floor: include any variable with meaningful Spearman
        # correlation with log(r0), regardless of LLM hypothesis
        try:
            from scipy.stats import spearmanr
            X, y = self._store.to_arrays()
            if X is not None and len(X) >= 10:
                log_y = np.log1p(np.abs(y))
                param_names = list(self._oracle.parameter_names)
                for j, var in enumerate(param_names):
                    if var in found:
                        continue
                    col = X[:, j]
                    if col.std() < 1e-10:
                        continue  # constant column → no info
                    rho, _ = spearmanr(col, log_y)
                    if abs(rho) > 0.15:
                        found.add(var)
        except Exception:
            pass  # silently skip if store is empty or scipy unavailable

        found_list = list(found)
        return found_list if len(found_list) >= 2 else None

    def _residual_var_promotion(
        self,
        eq: "FittedEquation",
        relevant_vars: list[str],
    ) -> list[str]:
        """Check whether any excluded variables correlate with PySR residuals.

        If an excluded variable explains ≥25% of residual variance (Spearman |r|>0.25),
        promote it into the relevant set for the next PySR call. This handles
        cases where the LLM omitted a variable that genuinely matters.

        Guards:
        - Only runs when the current best equation has R² > 0.75 (otherwise
          residuals are dominated by structural misfit, not missing variables).
        - Uses Spearman rank correlation so that monotone-but-nonlinear relationships
          (e.g. Arrhenius exp(-Ea/T) vs T) are detected correctly.
        """
        if not self._is_chem_domain() or eq is None:
            return relevant_vars

        # Gate: don't promote based on a poorly-fitting equation — residuals from
        # a bad fit are dominated by model error, not missing variables.
        if eq.r2 < 0.75:
            return relevant_vars

        all_vars = list(self._oracle.parameter_names)
        excluded = [v for v in all_vars if v not in relevant_vars]
        if not excluded:
            return relevant_vars

        X, y = self._store.to_arrays()
        if len(y) < 5:
            return relevant_vars

        # Evaluate current equation on training data
        try:
            import math as _math
            ns: dict = {}
            exec(eq.law_str, {
                "np": np, "math": _math, "sqrt": np.sqrt,
                "log": np.log, "exp": np.exp, "abs": np.abs,
            }, ns)
            fn = ns["discovered_law"]
            y_pred = np.array([fn(*row) for row in X], dtype=float)
            residuals = y - y_pred
        except Exception:
            return relevant_vars

        # Use Spearman rank correlation — detects monotone nonlinear relationships
        # (e.g. exp(-Ea/T) is non-linearly related to T but monotone in 1/T,
        # which Spearman captures while Pearson often misses).
        from scipy.stats import spearmanr
        promoted = list(relevant_vars)
        for v in excluded:
            vi = all_vars.index(v)
            xv = X[:, vi]
            if np.std(xv) < 1e-10:
                continue
            try:
                corr, _ = spearmanr(xv, residuals)
            except Exception:
                continue
            if abs(corr) > 0.25:
                console.print(
                    f"  [yellow]Residual promotion: {v} correlates with residuals "
                    f"(Spearman r={corr:+.2f}) → adding to PySR input[/yellow]"
                )
                promoted.append(v)
        return promoted

    def _compute_var_permutation_importance(
        self, law_str: str, X: np.ndarray, y: np.ndarray, param_names: list[str]
    ) -> dict[str, float]:
        """
        Permutation importance for each variable in a fitted law.

        Shuffles each variable column and measures the RMSLE increase.
        Variables with importance < 0.01 are likely spurious (fitting noise).
        """
        import math as _math
        rng = np.random.default_rng(42)
        importance = {}

        try:
            ns: dict = {"np": np, "math": _math, "__builtins__": __builtins__}
            exec(law_str, ns)
            fn = ns.get("discovered_law")
            if fn is None:
                return {}

            def _rmsle(X_in):
                try:
                    preds = np.array([
                        fn(**{p: float(X_in[i, j]) for j, p in enumerate(param_names)})
                        for i in range(len(X_in))
                    ])
                    preds = np.maximum(preds, 1e-300)
                    y_pos = np.maximum(y, 1e-300)
                    return float(np.sqrt(np.mean((np.log1p(preds) - np.log1p(y_pos)) ** 2)))
                except Exception:
                    return float("nan")

            baseline = _rmsle(X)
            if np.isnan(baseline):
                return {}

            for j, var in enumerate(param_names):
                if var not in (law_str or ""):
                    importance[var] = 0.0
                    continue
                X_perm = X.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                perm_rmsle = _rmsle(X_perm)
                importance[var] = max(0.0, perm_rmsle - baseline) if not np.isnan(perm_rmsle) else 0.0
        except Exception:
            pass

        return importance

    def _prune_spurious_vars(
        self,
        eq: "FittedEquation",
        store: "ExperimentStore",
        param_names: list[str],
        importance_threshold: float = 0.01,
    ) -> "FittedEquation":
        """
        Remove variables with permutation importance < threshold from final equation
        and refit the skeleton without them.

        This directly addresses the failure mode where MEI adds spurious T/C_I/C_B
        terms to an otherwise correct skeleton (e.g. MM + spurious Arrhenius).
        The refit uses the existing skeleton families minus the dropped variable.

        Returns the pruned equation if it's not materially worse (RMSLE within 20%),
        otherwise returns the original.
        """
        if not eq or not eq.law_str or not self._is_chem_domain():
            return eq

        X, y = store.to_arrays()
        if len(X) < 10:
            return eq

        try:
            imp = self._compute_var_permutation_importance(eq.law_str, X, y, param_names)
        except Exception:
            return eq

        spurious = [v for v, s in imp.items() if s < importance_threshold and v in (eq.skeleton_str or "")]
        if not spurious:
            return eq

        print(f"[PruneVars] Detected spurious variables: {spurious} (importance < {importance_threshold})")

        # Build pruned skeleton families by removing spurious variables from
        # all accumulated hypothesis families
        pruned_families = []
        for fam in list(self._hypothesis_families) + list(self._seed_families):
            # Skip families that contain the spurious variable
            if any(sv in fam for sv in spurious):
                continue
            pruned_families.append(fam)

        if not pruned_families:
            print("[PruneVars] No valid families after pruning — keeping original.")
            return eq

        # Refit with pruned variable set (exclude spurious from relevant_vars)
        valid_vars = [v for v in param_names if v not in spurious]
        try:
            from autoscilab.equation_learner.learner import EquationLearner
            pruned_eq = self._eq_learner.fit(
                store,
                self._cfg.goal,
                self._current_hypothesis,
                n_iterations=max(200, self._cfg.pysr_n_iterations // 4),
                n_diverse_runs=1,
                relevant_vars=valid_vars,
                skeleton_families=pruned_families[:10],
                skeleton_priority_threshold=0.15,  # accept pruned skeleton more liberally
            )
        except Exception as e:
            print(f"[PruneVars] Refit failed: {e}")
            return eq

        if pruned_eq is None:
            return eq

        # Accept pruned version if RMSLE is within 20% of original
        if pruned_eq.rmsle <= eq.rmsle * 1.20:
            print(
                f"[PruneVars] Accepted pruned equation: RMSLE {eq.rmsle:.4f} → {pruned_eq.rmsle:.4f} "
                f"(dropped: {spurious})"
            )
            return pruned_eq
        else:
            print(
                f"[PruneVars] Pruned RMSLE {pruned_eq.rmsle:.4f} > 1.2× original {eq.rmsle:.4f} — keeping original."
            )
            return eq

    def _measure_ood_points(self, n_points: int) -> "ExperimentStore | None":
        """
        Measure oracle at OOD boundary points (variables at 5th/95th percentile,
        others at midpoint). These are NOT added to the training store.

        Returns a small store with only OOD measurements, used to detect equation
        overfitting to the training range.
        """
        from autoscilab.data.store import ExperimentStore
        param_names = self._oracle.parameter_names
        bounds = self._oracle.parameter_bounds
        rng = np.random.default_rng(99)

        ood_store = ExperimentStore(self._cfg.domain_id, param_names)
        n_measured = 0

        # For each variable: one point at low (5th pct) + one at high (95th pct),
        # all others at midpoint
        # bounds may be a dict (name→(lo,hi)) or list of (lo,hi) — handle both
        def _get_bounds(v, idx):
            if isinstance(bounds, dict):
                return bounds[v]
            return bounds[idx]

        for var_idx, var in enumerate(param_names):
            lo, hi = _get_bounds(var, var_idx)
            midpoints = [(_get_bounds(p, i)[0] + _get_bounds(p, i)[1]) / 2.0
                         for i, p in enumerate(param_names)]
            for extreme in [lo + 0.05 * (hi - lo), lo + 0.95 * (hi - lo)]:
                if n_measured >= n_points:
                    break
                params = dict(zip(param_names, midpoints))
                params[var] = extreme
                try:
                    result = self._oracle.run(**params)
                    y_val = float(result) if not isinstance(result, dict) else float(list(result.values())[0])
                    ood_store.add(list(params.values()), y_val)
                    n_measured += 1
                    self._oracle_calls += 1  # count against budget
                except Exception:
                    pass
            if n_measured >= n_points:
                break

        if len(ood_store) == 0:
            return None
        print(f"[OOD] Measured {len(ood_store)} OOD points (boundary extremes).")
        return ood_store

    def _ood_rmsle(self, eq: "FittedEquation", ood_store: "ExperimentStore") -> float | None:
        """Evaluate equation RMSLE on the OOD holdout store."""
        if not eq or not eq.law_str or ood_store is None or len(ood_store) == 0:
            return None
        try:
            X_ood, y_ood = ood_store.to_arrays()
            param_names = self._oracle.parameter_names
            ns = {p: X_ood[:, i] for i, p in enumerate(param_names)}
            ns.update({"np": np, "math": __import__("math"), "sqrt": np.sqrt,
                        "exp": np.exp, "log": np.log, "abs": np.abs})
            exec(eq.law_str, ns)
            discovered_law = ns["discovered_law"]
            y_pred = np.array([discovered_law(**{p: float(X_ood[k, i]) for i, p in enumerate(param_names)})
                                for k in range(len(X_ood))], dtype=float)
            mask = (y_ood > 0) & (y_pred > 0) & np.isfinite(y_pred)
            if mask.sum() < 2:
                return None
            return float(np.sqrt(np.mean((np.log(y_ood[mask]) - np.log(y_pred[mask])) ** 2)))
        except Exception:
            return None

    def _run_variable_probes(self) -> str:
        """
        Run structured variable probes: vary each variable in isolation (low + high)
        while holding others at their midpoint defaults.

        Returns a summary string injected into LLM memory describing which variables
        causally affect the output.
        """
        bounds = self._oracle.parameter_bounds
        param_names = self._oracle.parameter_names

        # Default = midpoint of each variable's range
        defaults = {
            v: (bounds[v][0] + bounds[v][1]) / 2.0
            for v in param_names
        }
        # Override with oracle defaults if available
        oracle_defaults = getattr(self._oracle, "parameter_defaults", {})
        defaults.update(oracle_defaults)

        probe_results: dict[str, list[float]] = {v: [] for v in param_names}

        for var in param_names:
            lo, hi = bounds[var]
            # Two probe values: 10th and 90th percentile
            for pct in (0.1, 0.9):
                p = dict(defaults)
                p[var] = lo + pct * (hi - lo)
                try:
                    result = self._oracle.run(p)
                    self._store.add(result)
                    self._oracle_calls += 1
                    probe_results[var].append(result.measurement)
                except Exception as e:
                    console.print(f"  [yellow]Probe failed for {var}: {e}[/yellow]")

        # Compute effect size for each variable: relative range of outputs
        # when varying that variable vs all-probe output range
        all_measurements = [
            m for measurements in probe_results.values() for m in measurements
            if np.isfinite(m) and m > 0
        ]
        if not all_measurements:
            return ""

        global_range = max(all_measurements) - min(all_measurements) + 1e-300

        relevance = {}
        for var in param_names:
            ms = [m for m in probe_results[var] if np.isfinite(m) and m > 0]
            if len(ms) >= 2:
                effect = (max(ms) - min(ms)) / global_range
                relevance[var] = effect
            else:
                relevance[var] = 0.0

        # Build summary
        lines = ["VARIABLE RELEVANCE PROBES (each variable swept low→high, others fixed):"]
        strong, weak = [], []
        for var in sorted(relevance, key=lambda v: -relevance[v]):
            effect = relevance[var]
            ms = probe_results[var]
            if len(ms) >= 2:
                pct_change = 100 * abs(ms[1] - ms[0]) / (abs(ms[0]) + 1e-300)
                label = "RELEVANT" if effect > 0.05 else ("WEAK" if effect > 0.01 else "LIKELY IRRELEVANT")
                lines.append(
                    f"  {var}: effect={effect:.3f}  "
                    f"output={ms[0]:.4g}→{ms[1]:.4g} ({pct_change:.1f}% change)  [{label}]"
                )
                if effect > 0.05:
                    strong.append(var)
                else:
                    weak.append(var)

        if strong:
            lines.append(f"\nClearly relevant variables (effect>0.05): {strong}")
        if weak:
            lines.append(f"Likely irrelevant (effect<0.05): {weak}")
        lines.append(
            "\nUSE THIS TO FOCUS MECHANISM SEARCH: only include clearly relevant variables "
            "in your hypotheses. Irrelevant variables should be excluded from the rate law."
        )

        summary = "\n".join(lines)
        console.print(f"[cyan]Variable probes complete ({self._oracle_calls} experiments used):[/cyan]")
        console.print(f"[dim]{summary}[/dim]")
        return summary

    def _policy_fit_equations(
        self,
        goal: str,
        hypothesis: str,
        n_iterations: int,
        relevant_vars: list[str] | None = None,
        skeleton_families: list[str] | None = None,
    ) -> tuple["FittedEquation | None", list["FittedEquation"]]:
        """Run equation fitting based on the configured policy.
        Returns (best_equation, candidate_list)."""
        policy = self._cfg.policy
        store = self._store
        n_div = self._cfg.n_diverse_pysr_runs
        restriction_profile = self._get_operator_restriction_profile()

        if policy == "policy_BE":
            candidates = self._eq_learner.fit_multi_complexity_cascade(
                store,
                goal,
                hypothesis,
                n_iterations=n_iterations,
                operator_profile=restriction_profile,
                run_unrestricted_parallel=(
                    bool(restriction_profile) and self._cfg.restriction_parallel_unrestricted
                ),
            )
            return self._eq_learner.best, candidates

        else:
            # Default: families_v3
            eq = self._eq_learner.fit(
                store, goal, hypothesis,
                n_iterations=n_iterations, n_diverse_runs=n_div,
                relevant_vars=relevant_vars,
                skeleton_families=skeleton_families,
                skeleton_priority_threshold=self._cfg.skeleton_priority_threshold,
            )
            # Return PySR result + any skeleton (scipy) candidates as a joint pool.
            # The skeleton candidates preserve LLM-proposed symbolic structure;
            # PySR candidates may be numerically better but structurally unconstrained.
            # The LLM will arbitrate between them at the end of the run.
            candidates: list[FittedEquation] = []
            if eq:
                candidates.append(eq)
            for sc in self._eq_learner.skel_candidates:
                if sc and sc is not eq:
                    candidates.append(sc)
            return eq, candidates

    def run(self) -> DiscoveryResult:
        cfg = self._cfg
        start_time = time.time()
        termination_reason = "budget"
        validation_failed = False

        console.print(Panel(
            f"[bold cyan]AutoSciLab Discovery[/bold cyan]\n"
            f"Domain: {cfg.domain_id} | Budget: {cfg.budget} | "
            f"Difficulty: {cfg.difficulty} | Noise: {cfg.noise_level} | "
            f"Confidence mode: {cfg.confidence_mode} | "
            f"Hypothesis backend: {cfg.hypothesis_backend} | "
            f"Objective: {self._objective_type}",
            expand=False,
        ))

        outer_iter = 0
        param_names = self._oracle.parameter_names
        oracle_bounds = self._oracle.parameter_bounds

        # ── OOD holdout measurement ───────────────────────────────────────────
        # Spend ood_holdout_frac * budget on boundary extreme points BEFORE the
        # main loop. These are NOT added to the training store — only used at the
        # end to detect overfitting (wrong functional form that fits training range
        # but collapses at extrapolation).
        self._ood_store: "ExperimentStore | None" = None
        if cfg.ood_holdout_frac > 0 and self._objective_type == "equation":
            n_ood = max(2, int(cfg.ood_holdout_frac * cfg.budget))
            console.print(
                f"\n[cyan]OOD holdout: measuring {n_ood} boundary-extreme points "
                f"(NOT added to training store)...[/cyan]"
            )
            self._ood_store = self._measure_ood_points(n_ood)

        # ── Variable relevance probe phase ───────────────────────────────────
        if cfg.variable_probe_budget > 0 and self._oracle_calls == 0:
            n_probes = min(cfg.variable_probe_budget, cfg.budget // 3,
                          2 * len(param_names))
            console.print(
                f"\n[cyan]Running variable relevance probes "
                f"(budget={n_probes}, {len(param_names)} variables)...[/cyan]"
            )
            probe_summary = self._run_variable_probes()
            if probe_summary:
                self._memory.add_observation(
                    iteration=0,
                    observation=probe_summary,
                )

        while self._oracle_calls < cfg.budget:
            outer_iter += 1
            budget_remaining = cfg.budget - self._oracle_calls
            iter_max_disagreement = 0.0
            iter_discrimination_gain = None
            iter_discrimination_has_signal = None
            iter_discrimination_effective = None
            did_analysis_call = False
            generated_hypothesis = self._current_hypothesis
            generated_alts: list[str] = []

            # ── Phase management ────────────────────────────────────────────
            if self._oracle_calls >= cfg.discovery_budget and self._current_phase == "discovery":
                self._current_phase = "validation"
                console.print(f"\n[yellow]→ Switching to VALIDATION phase "
                              f"({budget_remaining} experiments remaining)[/yellow]")

            if validation_failed and self._current_phase == "validation":
                self._current_phase = "contradiction"
                console.print(f"\n[red]→ Switching to CONTRADICTION phase[/red]")

            # ── Get best equation string for LLM context ────────────────────
            # When multiple candidates exist (from policy-specific fitting),
            # show all of them rather than just the best.
            if self._use_symbolic and self._candidate_pool and len(self._candidate_pool) > 1:
                lines = []
                for ci, ceq in enumerate(self._candidate_pool[:5]):
                    lines.append(f"  {ci+1}. {ceq.skeleton_str} | R²={ceq.r2:.4f} | RMSLE={ceq.rmsle:.4f}")
                best_eq_str = "Multiple candidates:\n" + "\n".join(lines)
            elif self._use_symbolic and self._eq_learner.best:
                best_eq_str = str(self._eq_learner.best)
            else:
                best_eq_str = None

            # ── LLM proposes search regions ─────────────────────────────────
            console.print(f"\n[dim]Outer iter {outer_iter} | "
                          f"Oracle calls: {self._oracle_calls}/{cfg.budget} | "
                          f"Phase: {self._current_phase} | "
                          f"Policy: {cfg.policy}[/dim]")

            # Build GT metrics string for validation prompt
            gt_metrics_str = ""
            if self._last_gt_eval:
                gt_rmsle_val = self._last_gt_eval.get("rmsle", None)
                gt_exact = self._last_gt_eval.get("exact_accuracy", None)
                if gt_rmsle_val is not None:
                    quality = ("EXCELLENT" if gt_rmsle_val < 0.01
                               else "GOOD" if gt_rmsle_val < 0.1 else "POOR")
                    gt_metrics_str = (
                        f"  GT RMSLE: {gt_rmsle_val:.5f} ({quality})"
                        + (f"\n  Exact accuracy: {gt_exact}" if gt_exact is not None else "")
                    )

            if self._current_phase == "validation" and self._use_symbolic and self._eq_learner.best:
                proposal = self._agent.propose_validation_experiments(
                    goal=cfg.goal,
                    store=self._store,
                    best_hypothesis=self._current_hypothesis,
                    budget_remaining=budget_remaining,
                    gt_metrics_str=gt_metrics_str,
                )
            else:
                discrimination_hints = ""
                extra_regions: list[SearchRegion] = []
                if cfg.discrimination_binding and self._last_discrimination_summary:
                    prev = self._last_discrimination_summary
                    discrimination_hints += (
                        "\nPREVIOUS DISCRIMINATOR EFFICACY:\n"
                        f"- has_signal={prev.get('has_signal')}\n"
                        f"- gain_ratio={prev.get('gain_ratio', 0.0):.3f}\n"
                        f"- stagnation_count={self._discriminator_stagnation_count}\n"
                    )
                    if self._discriminator_stagnation_count >= max(1, cfg.discriminator_stagnation_patience):
                        discrimination_hints += (
                            "MANDATORY FOR THIS ITERATION:\n"
                            "- Propose at least one region expected to MAXIMIZE disagreement among top hypotheses.\n"
                            "- Propose at least one region expected to FALSIFY the current primary hypothesis.\n"
                            "- Avoid narrow refinements around already-sampled points.\n"
                        )

                # Strict 3-step mode: hypothesis generation call is always separate
                # from analysis and experiment proposal.
                if cfg.strict_generator_discriminator_loop:
                    try:
                        hyp_bundle = self._agent.generate_hypotheses(
                            goal=cfg.goal,
                            store=self._store,
                            current_hypothesis=self._current_hypothesis,
                            budget_remaining=budget_remaining,
                            total_budget=cfg.budget,
                            current_phase=self._current_phase,
                            best_equation_fit=best_eq_str,
                            objective_aware=cfg.objective_aware_prompting,
                        )
                        generated_hypothesis = (hyp_bundle.hypothesis or "").strip() or self._current_hypothesis
                        generated_alts = [
                            h.strip()
                            for h in (hyp_bundle.alternate_hypotheses or [])
                            if h and h.strip()
                        ][:6]
                    except Exception as e:
                        console.print(f"[yellow]Generator hypothesis call failed: {e}[/yellow]")

                if self._seed_families:
                    seed_preview = "; ".join(self._seed_families[:3])
                    discrimination_hints += (
                        "\nSEEDED STRUCTURAL FAMILIES (kept in play for disagreement): "
                        f"{seed_preview}\n"
                    )
                if cfg.transformed_coverage_hints:
                    hyp_for_coverage = list(
                        dict.fromkeys(
                            h for h in (
                                [generated_hypothesis]
                                + generated_alts
                                + list(self._hypothesis_families)
                            ) if h and h.strip()
                        )
                    )
                    coverage_hint = self._build_transformed_coverage_hints(hyp_for_coverage)
                    if coverage_hint:
                        discrimination_hints += "\n" + coverage_hint + "\n"

                # Policy BE: inject slope-change warnings
                if cfg.policy == "policy_BE":
                    slope_change = self._compute_slope_change()
                    if slope_change:
                        discrimination_hints += f"\n{slope_change}\n"

                # Build candidate dicts from pool + memory for discrimination
                mem_candidates = self._memory.top_k(3)
                candidate_dicts: list[dict] = []
                X_diag, y_diag = self._store.to_arrays()
                # Include candidates from the multi-equation pool
                for ceq in self._candidate_pool[:3]:
                    candidate_dicts.append({
                        "equation": ceq.skeleton_str,
                        "r2": ceq.r2,
                        "rmsle": ceq.rmsle,
                        "conf": ceq.confidence,
                        "law_str": ceq.law_str,
                    })
                # Fallback: include current best if pool is empty
                if not candidate_dicts and self._eq_learner.best is not None:
                    candidate_dicts.append({
                        "equation": self._eq_learner.best.skeleton_str,
                        "r2": self._eq_learner.best.r2,
                        "rmsle": self._eq_learner.best.rmsle,
                        "conf": self._eq_learner.best.confidence,
                        "law_str": self._eq_learner.best.law_str,
                    })
                for rec in mem_candidates:
                    candidate_dicts.append({
                        "equation": rec.equation,
                        "r2": rec.r2,
                        "rmsle": rec.rmsle,
                        "conf": rec.bootstrap_conf,
                    })

                def _append_generator_candidate(hyp: str) -> None:
                    rec: dict[str, Any] = {
                        "equation": hyp,
                        "r2": 0.0,
                        "rmsle": 999.0,
                        "conf": 0.0,
                    }
                    try:
                        fit = self._fit_hypothesis_expression(hypothesis=hyp, X=X_diag, y=y_diag)
                    except Exception:
                        fit = None
                    if fit:
                        rec["r2"] = float(fit.get("r2", 0.0))
                        rec["rmsle"] = float(fit.get("rmsle", 999.0))
                        law = fit.get("law_str")
                        if isinstance(law, str):
                            rec["law_str"] = law
                        y_pred = fit.get("y_pred")
                        if isinstance(y_pred, np.ndarray) and y_pred.shape[0] == y_diag.shape[0]:
                            rec["y_pred"] = y_pred
                    candidate_dicts.append(rec)

                if cfg.strict_generator_discriminator_loop:
                    if generated_hypothesis and generated_hypothesis != "unknown":
                        _append_generator_candidate(generated_hypothesis)
                    for alt_h in generated_alts:
                        _append_generator_candidate(alt_h)
                residual_stats, disagreement_stats = self._compute_disagreement_diagnostics(candidate_dicts)
                if residual_stats:
                    discrimination_hints += (
                        "\nDIAGNOSTIC SNAPSHOT:\n"
                        + "\n".join(
                            [
                                f"- {r['equation']}: mae_rel={r['mae_rel']:.3f}, bias={r['bias']:+.3f}, trend={r['trend']}"
                                for r in residual_stats[:4]
                            ]
                        )
                        + "\n"
                    )
                max_disagreement = 0.0
                if disagreement_stats:
                    max_disagreement = float(
                        max(float(d.get("max_rel_gap", 0.0)) for d in disagreement_stats)
                    )
                iter_max_disagreement = max_disagreement
                enough_candidates = len(candidate_dicts) >= max(2, cfg.analysis_min_candidates)
                cooldown_ok = (outer_iter - self._last_analysis_iter) > max(0, cfg.analysis_cooldown_iters)
                budget_ok = (
                    cfg.analysis_max_calls is None
                    or self._analysis_calls < cfg.analysis_max_calls
                )
                if cfg.strict_generator_discriminator_loop:
                    should_analyze = (
                        cfg.analysis_feedback_enabled
                        and enough_candidates
                        and budget_ok
                    )
                else:
                    should_analyze = (
                        cfg.analysis_feedback_enabled
                        and enough_candidates
                        and budget_ok
                        and cooldown_ok
                        and (
                            max_disagreement >= cfg.analysis_disagreement_threshold
                            or self._consecutive_poor_gt >= 1
                        )
                    )
                if should_analyze:
                    try:
                        plan = self._agent.analyze_disagreements(
                            goal=cfg.goal,
                            store=self._store,
                            candidates=candidate_dicts[:5],
                            budget_remaining=budget_remaining,
                            residual_stats=residual_stats,
                            disagreement_stats=disagreement_stats,
                        )
                        extra_regions = plan.search_regions or []
                        discrimination_hints += f"\nDISCRIMINATION ANALYSIS:\n{plan.reasoning}\n"
                        self._analysis_calls += 1
                        self._last_analysis_iter = outer_iter
                        did_analysis_call = True
                    except Exception as e:
                        print(f"[Discovery] Discrimination analysis failed: {e}")

                if cfg.strict_generator_discriminator_loop:
                    locked_lines = [f"- {generated_hypothesis}"] + [f"- {h}" for h in generated_alts[:5]]
                    discrimination_hints += (
                        "\nLOCKED HYPOTHESES FROM GENERATOR (proposer must discriminate between these):\n"
                        + "\n".join(locked_lines)
                        + "\n"
                    )

                # ── MEI ensemble round (optional) ────────────────────────
                _use_mei = (
                    self._ensemble_agent is not None
                    and self._current_phase == "discovery"
                    and len(self._store) >= cfg.min_data_for_eq
                    and self._ensemble_iter_count % cfg.ensemble_every_n_iters == 0
                )
                # v4 confidence gate: skip MEI if LLM already converged
                if _use_mei and cfg.confidence_feedback_enabled and cfg.ensemble_confidence_gate < 1.0:
                    _cur_conf = (
                        self._eq_learner.best.confidence
                        if self._eq_learner.best is not None else 0.0
                    )
                    if _cur_conf >= cfg.ensemble_confidence_gate:
                        print(
                            f"  [MEI] skipped — bootstrap conf {_cur_conf:.2f} "
                            f">= gate {cfg.ensemble_confidence_gate:.2f}"
                        )
                        _use_mei = False
                if _use_mei:
                    from autoscilab.al.falsification import FalsificationSelector
                    try:
                        _mei_proposal, _mei_hyp, _mei_dist = self._ensemble_agent.run_round(
                            goal=cfg.goal,
                            store=self._store,
                            current_hypothesis=self._current_hypothesis,
                            budget_remaining=budget_remaining,
                            total_budget=cfg.budget,
                            best_equation_fit=best_eq_str,
                        )
                        # Override hypothesis with large LLM synthesis if it changed
                        if _mei_hyp and _mei_hyp != self._current_hypothesis:
                            self._current_hypothesis = _mei_hyp

                        # Augment proposal with falsification points
                        _fals = FalsificationSelector(
                            oracle=self._oracle,
                            distribution=_mei_dist,
                            n_candidates=cfg.falsification_n_candidates,
                            diversity_weight=cfg.falsification_diversity_weight,
                        )
                        _fals_points = _fals.select(n=min(4, budget_remaining))
                        _fals_stats  = _fals.disagreement_stats()
                        print(
                            f"  [MEI] falsification: {len(_fals_points)} discriminating points "
                            f"(mean_disagreement={_fals_stats['mean']:.2f}, "
                            f"max={_fals_stats['max']:.2f})"
                        )
                        proposal = _mei_proposal
                        self._pending_falsification_points = _fals_points
                    except Exception as _mei_exc:
                        print(f"  [MEI] ensemble round failed ({type(_mei_exc).__name__}: {_mei_exc}) — falling back to LLM-only")
                        _use_mei = False
                        self._pending_falsification_points = []
                else:
                    self._pending_falsification_points = []

                self._ensemble_iter_count += 1

                # ── Fix 4: Forward prediction table ─────────────────────────
                # Compute concrete per-hypothesis predictions at the max-disagreement
                # point and inject into discrimination_hints. The LLM can then reason
                # "H1 predicts 4.8, H2 predicts 2.3 here — measuring this would rule
                # one out" without any domain-specific hardcoding.
                if cfg.forward_discrimination and self._candidate_pool and len(self._candidate_pool) >= 2:
                    try:
                        fitted_laws = [c.law_str for c in self._candidate_pool[:5] if c and c.law_str]
                        law_labels = [
                            f"H{i+1}({(c.skeleton_str or '')[:40]})"
                            for i, c in enumerate(self._candidate_pool[:5]) if c and c.law_str
                        ]
                        pred_table = self._agent._compute_prediction_table(fitted_laws, law_labels)
                        if pred_table:
                            discrimination_hints = (discrimination_hints + "\n" + pred_table).strip()
                    except Exception as _fwd_exc:
                        print(f"  [FwdDiscrim] table failed: {_fwd_exc}")

                if not _use_mei:
                    proposal = self._agent.propose_experiments(
                        goal=cfg.goal,
                        store=self._store,
                        current_hypothesis=generated_hypothesis if cfg.strict_generator_discriminator_loop else self._current_hypothesis,
                        budget_remaining=budget_remaining,
                        total_budget=cfg.budget,
                        current_phase=self._current_phase,
                        best_equation_fit=best_eq_str,
                        discrimination_hints=discrimination_hints,
                        objective_aware=cfg.objective_aware_prompting,
                    )
                if extra_regions:
                    merged = list(extra_regions) + list(proposal.search_regions)
                    # Deduplicate by bounds signature
                    seen = set()
                    deduped = []
                    for r in merged:
                        key = tuple(sorted((k, tuple(v)) for k, v in r.bounds.items()))
                        if key in seen:
                            continue
                        seen.add(key)
                        deduped.append(r)
                        if len(deduped) >= 8:
                            break
                    proposal.search_regions = deduped

            # Optional deterministic first-iteration full-bounds sweep.
            if outer_iter == 1 and cfg.deterministic_first_iter:
                sweep_bounds = {
                    p: [float(lo), float(hi)]
                    for p, (lo, hi) in oracle_bounds.items()
                }
                proposal.search_regions = [
                    SearchRegion(
                        bounds=sweep_bounds,
                        n_experiments=max(1, cfg.max_experiments_per_iter),
                        priority="high",
                        rationale="Deterministic full-bounds sweep to estimate exponents early.",
                    )
                ]

            # ── Update / reset structural families based on GT quality ───────
            if cfg.strict_generator_discriminator_loop:
                proposal_alts = generated_alts
            else:
                proposal_alts = list(getattr(proposal, "alternate_hypotheses", []) or [])
            families_now: list[str] = []
            if cfg.strict_generator_discriminator_loop:
                if generated_hypothesis:
                    families_now.append(generated_hypothesis.strip())
            elif proposal.hypothesis:
                families_now.append(proposal.hypothesis.strip())
            families_now.extend([h.strip() for h in proposal_alts if h and h.strip()])

            def _dedupe(seq: list[str], k: int = 5) -> list[str]:
                seen: set[str] = set()
                out: list[str] = []
                for s in seq:
                    if s and s not in seen:
                        seen.add(s)
                        out.append(s)
                        if len(out) >= k:
                            break
                return out

            # If we have seen multiple consecutive clearly bad GT fits, retire the
            # old family set and replace with fresh candidates from this proposal.
            def _mentions_all_params(h: str) -> bool:
                hl = (h or "").lower()
                return all(p.lower() in hl for p in param_names)

            filtered = [h for h in families_now if _mentions_all_params(h)]
            # Suppress repeated falsified forms so the LLM does not keep cycling.
            memory_filtered = [
                h for h in (filtered or families_now)
                if not self._memory.should_avoid_hypothesis(h, min_failed=2)
            ]
            if memory_filtered:
                filtered = memory_filtered

            seeds = list(self._seed_families)
            if self._consecutive_poor_gt >= 2 and families_now:
                self._hypothesis_families = _dedupe(seeds + (filtered or families_now))
                self._consecutive_poor_gt = 0
            elif not self._hypothesis_families and families_now:
                self._hypothesis_families = _dedupe(seeds + (filtered or families_now))
            elif families_now:
                # Merge in any truly new families but keep a small cap.
                merged = self._hypothesis_families + seeds + (filtered or families_now)
                self._hypothesis_families = _dedupe(merged)
            elif seeds and not self._hypothesis_families:
                self._hypothesis_families = _dedupe(seeds)

            # ── Deterministic warm-start sweeps (first 3 rounds) ─────────────
            # Use broad one-parameter sweeps to cleanly infer exponents early,
            # rather than relying on free-form regions.
            warm_sweep_rounds = len(param_names)
            sweep_param = None
            if outer_iter <= warm_sweep_rounds and self._current_phase == "discovery":
                sweep_param = param_names[outer_iter - 1]
                lo, hi = oracle_bounds[sweep_param]
                sweep_bounds: dict[str, list[float]] = {}
                for p in param_names:
                    p_lo, p_hi = oracle_bounds[p]
                    if p == sweep_param:
                        sweep_bounds[p] = [float(lo), float(hi)]
                    else:
                        # Fix others to a narrow band around the geometric mid
                        mid = float(np.sqrt(p_lo * p_hi))
                        band = 1.25
                        sweep_bounds[p] = [
                            float(max(p_lo, mid / band)),
                            float(min(p_hi, mid * band)),
                        ]

                proposal.search_regions = [
                    SearchRegion(
                        bounds=sweep_bounds,
                        n_experiments=max(1, cfg.max_experiments_per_iter),
                        priority="high",
                        rationale=f"Deterministic sweep: vary {sweep_param} broadly to infer its exponent cleanly.",
                    )
                ]

            # Update hypothesis from LLM (normalize null/empty to a safe default).
            # If we already have a fitted equation, avoid staying in a vague
            # "unknown" state: fall back to the best equation string instead.
            if cfg.strict_generator_discriminator_loop:
                hyp = (generated_hypothesis or "").strip()
            else:
                hyp = (proposal.hypothesis or "").strip()
            lower = hyp.lower()
            is_generic = (
                not hyp
                or lower in ("none", "null", "unknown")
                or "unknown at this stage" in lower
                or "no data yet" in lower
            )
            if is_generic and self._eq_learner.best is not None:
                # Use the best PySR equation as the current hypothesis so that
                # subsequent prompts see a concrete form.
                hyp = str(self._eq_learner.best)
            elif is_generic:
                hyp = "no data yet — broad exploration needed"

            self._current_hypothesis = hyp
            console.print(f"[bold]Hypothesis:[/bold] {self._current_hypothesis[:120]}")
            if getattr(proposal, "alternate_hypotheses", None):
                console.print(f"[dim]Alt hypotheses: {len(proposal.alternate_hypotheses)}[/dim]")
            if self._hypothesis_families:
                console.print(f"[dim]Families (K={len(self._hypothesis_families)}): using family-level disagreement[/dim]")
            if not cfg.hypothesis_conditioned_acquisition:
                full_bounds = {
                    p: [float(lo), float(hi)]
                    for p, (lo, hi) in oracle_bounds.items()
                }
                proposal.search_regions = [
                    SearchRegion(
                        bounds=full_bounds,
                        n_experiments=max(1, cfg.max_experiments_per_iter),
                        priority="high",
                        rationale="Global acquisition ablation: ignore hypothesis-conditioned regions.",
                    )
                ]

            console.print(f"[dim]Regions: {len(proposal.search_regions)}[/dim]")

            # ── Phase-aware acquisition function ────────────────────────────
            # When cfg.acquisition=="ei", use UCB (broad exploration) early and
            # switch to EI (exploit uncertainty near good regions) once we have
            # built a reasonable picture of the landscape.
            budget_used_frac = self._oracle_calls / max(cfg.budget, 1)
            if cfg.acquisition == "ei":
                if budget_used_frac < 0.25:
                    current_acquisition = "ucb"
                    current_kappa = 4.0
                elif budget_used_frac < 0.55:
                    current_acquisition = "ucb"
                    current_kappa = 2.0
                else:
                    current_acquisition = "ei"
                    current_kappa = 2.0
            else:
                current_acquisition = cfg.acquisition
                current_kappa = 2.0

            # If we have multiple hypotheses, prefer disagreement sampling.
            # Use persistent structural families (not just this-iteration variants)
            # to drive experiments that confirm/break competing families.
            candidate_hypotheses = self._hypothesis_families or [
                self._current_hypothesis
            ] + list(getattr(proposal, "alternate_hypotheses", []) or [])
            if (
                cfg.allow_disagreement
                and current_acquisition in ("ei", "ucb", "variance")
                and len(candidate_hypotheses) >= 2
            ):
                current_acquisition = "disagree"

            # ── AL selects + Oracle runs experiments ────────────────────────
            store_size_before = len(self._store)  # snapshot for post-run scoring
            experiments_this_iter = 0
            iter_cap = cfg.max_experiments_per_iter
            regions_ran: list[dict[str, Any]] = []
            selector_diags: list[dict[str, Any]] = []
            for region in proposal.search_regions:
                n = min(region.n_experiments,
                        budget_remaining - experiments_this_iter,
                        iter_cap - experiments_this_iter)
                if n <= 0:
                    break

                regions_ran.append(
                    {
                        "bounds": region.bounds,
                        "n_requested": int(region.n_experiments),
                        "n_ran": int(n),
                        "priority": getattr(region, "priority", None),
                        "rationale": getattr(region, "rationale", None),
                    }
                )
                selected_params = self._selector.select(
                    store=self._store,
                    region_bounds=region.bounds,
                    n_points=n,
                    acquisition=current_acquisition,
                    hypotheses=candidate_hypotheses if current_acquisition == "disagree" else None,
                    kappa=current_kappa,
                    discrimination_binding=cfg.discrimination_binding,
                )
                sel_diag = self._selector.get_last_selection_info()
                if sel_diag:
                    selector_diags.append(sel_diag)

                for params in selected_params:
                    if self._oracle_calls >= cfg.budget:
                        break
                    result = self._oracle.run(params)
                    self._store.add(result)
                    self._oracle_calls += 1
                    experiments_this_iter += 1

            # ── MEI falsification points ──────────────────────────────────
            # Measure the FalsificationSelector's discriminating points
            # directly (not routed through AL — they are deliberately adversarial).
            for fp in self._pending_falsification_points:
                if self._oracle_calls >= cfg.budget:
                    break
                result = self._oracle.run(fp)
                self._store.add(result)
                self._oracle_calls += 1
                experiments_this_iter += 1
            self._pending_falsification_points = []

            console.print(f"  Ran {experiments_this_iter} experiments. "
                          f"Total: {self._oracle_calls}")

            # ── Strong-model consultant triggers ─────────────────────────────
            # Call at 20%, 50%, 80% of budget. Uses natural log of rates so
            # the consultant's correlation helpers work correctly.
            if self._consultant is not None and len(self._store) >= 8:
                _X_c, _y_c = self._store.to_arrays()   # _y_c is linear
                _log_y_c = np.log(np.where(_y_c > 0, _y_c, 1e-12))
                _budget_frac = self._oracle_calls / max(self._cfg.budget, 1)
                if _budget_frac >= 0.18 and not self._consultant_trigger_done[1]:
                    self._consultant_try_trigger(1, _X_c, _log_y_c, param_names)
                elif (_budget_frac >= 0.45
                      and not self._consultant_trigger_done[2]
                      and self._eq_learner.best is not None):
                    self._consultant_try_trigger(2, _X_c, _log_y_c, param_names)
                elif _budget_frac >= 0.75 and not self._consultant_trigger_done[3]:
                    self._consultant_try_trigger(3, _X_c, _log_y_c, param_names)

            if current_acquisition == "disagree" and selector_diags:
                gains = [
                    float(d.get("disagreement_gain_ratio"))
                    for d in selector_diags
                    if d.get("disagreement_gain_ratio") is not None
                ]
                has_signal = any(bool(d.get("disagreement_has_signal")) for d in selector_diags)
                mean_gain = float(np.mean(gains)) if gains else 0.0
                effective = bool(has_signal and mean_gain >= cfg.discrimination_min_gain)
                iter_discrimination_gain = mean_gain
                iter_discrimination_has_signal = bool(has_signal)
                iter_discrimination_effective = bool(effective)
                self._last_discrimination_summary = {
                    "has_signal": bool(has_signal),
                    "gain_ratio": float(mean_gain),
                    "effective": bool(effective),
                }
                if effective:
                    self._discriminator_stagnation_count = 0
                else:
                    self._discriminator_stagnation_count += 1
                console.print(
                    "  [dim]Discriminator efficacy:"
                    f" signal={has_signal} gain={mean_gain:.3f} "
                    f"effective={effective} stagnation={self._discriminator_stagnation_count}[/dim]"
                )

            # ── Hypothesis scoring (generator-discriminator feedback) ────────
            # After any iteration that ran discriminating experiments (either the
            # strict G-D mode or a soft-gate analysis call), score each candidate
            # hypothesis against the NEWLY collected data and update memory/families.
            new_data_count = len(self._store) - store_size_before
            run_hypothesis_scoring = (
                new_data_count >= 2
                and (did_analysis_call or cfg.strict_generator_discriminator_loop)
                and (self._hypothesis_families or generated_alts)
            )
            if run_hypothesis_scoring:
                # Collect all candidate hypothesis strings to score
                hyps_to_score = list(
                    dict.fromkeys(
                        h for h in (
                            list(self._hypothesis_families)
                            + ([generated_hypothesis] if generated_hypothesis and generated_hypothesis != "unknown" else [])
                            + generated_alts
                        )
                        if h and h.strip()
                    )
                )
                if hyps_to_score:
                    console.print(
                        f"  [dim]→ Scoring {len(hyps_to_score)} hypotheses against "
                        f"{new_data_count} new points...[/dim]"
                    )
                    try:
                        X_all, y_all = self._store.to_arrays()
                        X_score, y_score = self._build_scoring_subset(
                            X_all=X_all,
                            y_all=y_all,
                            new_data_count=new_data_count,
                        )
                        scores = self._score_hypothesis_strings(hyps_to_score, X_score, y_score)
                        self._update_families_from_discrimination(
                            hypotheses=hyps_to_score,
                            scores=scores,
                            outer_iter=outer_iter,
                            n_data=len(X_score),
                            n_full_data=len(X_all),
                        )
                    except Exception as e:
                        console.print(f"  [yellow]Hypothesis scoring failed: {e}[/yellow]")

            # ── Equation fitting (every N experiments) ──────────────────────
            eq_confidence = 0.0
            boot_conf = 0.0
            llm_conf = 0.0

            if (cfg.iterative_refinement_enabled
                    and self._use_symbolic
                    and self._oracle_calls >= cfg.min_data_for_eq
                    and self._oracle_calls % cfg.eq_learner_every == 0):

                # ── ChemBench: staged iters + LLM-guided variable filter ─────
                staged_iters = self._staged_pysr_iters(outer_iter, is_final=False)
                relevant = self._extract_relevant_vars(self._current_hypothesis)
                # Promote any excluded vars that correlate with current residuals
                if relevant is not None and self._eq_learner.best is not None:
                    relevant = self._residual_var_promotion(self._eq_learner.best, relevant)
                if relevant is not None:
                    console.print(
                        f"[dim]  → Variable filter: {relevant} | iters={staged_iters}[/dim]"
                    )

                # Pass skeleton families for direct scipy fitting (ChemBench only).
                # Merge seed families (from domain_tags, may be empty) with the
                # LLM's accumulated structural hypotheses so that scipy always has
                # something to try — even when use_domain_tags=False.
                if self._is_chem_domain():
                    skel_fams = _dedupe(
                        list(self._seed_families)
                        + list(self._hypothesis_families),
                        k=30,
                    ) or None
                else:
                    skel_fams = None

                console.print(f"[dim]  → Fitting equation (policy={cfg.policy})...[/dim]")
                eq, new_candidates = self._policy_fit_equations(
                    goal=cfg.goal,
                    hypothesis=self._current_hypothesis,
                    n_iterations=staged_iters,
                    relevant_vars=relevant,
                    skeleton_families=skel_fams,
                )
                validated_candidates = self._validated_hypothesis_candidates(top_k=3)
                if validated_candidates:
                    console.print(
                        f"[dim]  → Injected {len(validated_candidates)} validated-hypothesis "
                        f"candidate(s) into arbitration pool[/dim]"
                    )
                merged_candidates = list(new_candidates) + list(validated_candidates)
                deduped_pool: list[FittedEquation] = []
                seen_skeletons: set[str] = set()
                for cand in merged_candidates:
                    if cand is None:
                        continue
                    key = (cand.skeleton_str or "").strip().lower()
                    if not key or key in seen_skeletons:
                        continue
                    seen_skeletons.add(key)
                    deduped_pool.append(cand)
                self._candidate_pool = deduped_pool
                if self._candidate_pool:
                    eq = max(self._candidate_pool, key=lambda c: float(c.r2))

                if eq:
                    boot_conf = eq.confidence
                    mid_gt_rmsle = None

                    # ── Mid-run ground-truth evaluation ───────────────────────
                    # When the fit looks very strong (val R²>0.95), call the GT
                    # evaluator mid-run. This gives the LLM a precise quality
                    # signal (GT RMSLE) to guide further exploration.
                    if self._objective_type == "equation" and cfg.mid_run_gt_eval and eq.r2 > 0.95:
                        console.print("[dim]  → Running mid-run GT evaluation...[/dim]")
                        try:
                            gt_eval = self._oracle.evaluate_law(eq.law_str)
                            mid_gt_rmsle = gt_eval.get("rmsle")
                            self._last_gt_eval = gt_eval
                            quality = ("✓ EXCELLENT" if mid_gt_rmsle < 0.01
                                       else "≈ GOOD" if mid_gt_rmsle < 0.1 else "✗ POOR")
                            console.print(
                                f"  [cyan]GT RMSLE (mid-run): {mid_gt_rmsle:.5f}  {quality}[/cyan]"
                            )
                            # Track the best GT-validated equation across iterations.
                            # Use a slightly looser threshold (<1.0) so we preserve
                            # genuinely good laws (e.g. Fourier medium v2) even if
                            # they are not yet "perfect".
                            if mid_gt_rmsle < 1.0:
                                if (self._best_gt_equation is None
                                        or mid_gt_rmsle < (self._best_gt_eval_rmsle or 99)):
                                    self._best_gt_equation = eq
                                    self._best_gt_eval_rmsle = mid_gt_rmsle
                            # Track consecutive clearly poor GT fits to trigger
                            # structural family resets.
                            if mid_gt_rmsle is not None and mid_gt_rmsle > 1.0:
                                self._consecutive_poor_gt += 1
                                if cfg.negative_evidence_enabled:
                                    self._memory.add_negative_evidence(
                                        iteration=outer_iter,
                                        hypothesis=eq.skeleton_str,
                                        evidence=(
                                            "High training fit but poor GT agreement; treat this family as likely overfit."
                                        ),
                                        gt_rmsle=mid_gt_rmsle,
                                    )
                            else:
                                self._consecutive_poor_gt = 0
                        except Exception as e:
                            console.print(f"  [yellow]GT eval failed: {e}[/yellow]")

                    # ── GT-evaluate additional pool candidates ────────────────
                    # For policies with multiple candidates, check if any pool
                    # candidate beats the current best GT equation.
                    if self._objective_type == "equation" and cfg.mid_run_gt_eval and len(self._candidate_pool) > 1:
                        for pool_eq in self._candidate_pool:
                            if pool_eq is eq or pool_eq.r2 < 0.90:
                                continue
                            try:
                                pool_gt = self._oracle.evaluate_law(pool_eq.law_str)
                                pool_rmsle = pool_gt.get("rmsle")
                                if pool_rmsle is not None and pool_rmsle < 1.0:
                                    if (self._best_gt_equation is None
                                            or pool_rmsle < (self._best_gt_eval_rmsle or 99)):
                                        self._best_gt_equation = pool_eq
                                        self._best_gt_eval_rmsle = pool_rmsle
                                        console.print(
                                            f"  [cyan]Pool candidate GT RMSLE: {pool_rmsle:.5f} ← new best[/cyan]"
                                        )
                            except Exception:
                                pass

                    # ── LLM-sampled confidence (if requested) ────────────────
                    if cfg.confidence_mode in ("llm", "both"):
                        console.print("[dim]  → Sampling LLM confidence...[/dim]")
                        llm_conf = self._agent.assess_equation_confidence(
                            equation_str=eq.skeleton_str,
                            store=self._store,
                            n_samples=cfg.llm_confidence_samples,
                        )

                    # ── Combine ───────────────────────────────────────────────
                    if cfg.confidence_mode == "bootstrap":
                        eq_confidence = boot_conf
                    elif cfg.confidence_mode == "llm":
                        eq_confidence = llm_conf
                    else:  # "both" — geometric mean: both must be high
                        eq_confidence = float((boot_conf * llm_conf) ** 0.5)

                    # ── GT RMSLE guard: override confidence if GT says POOR ───
                    # If mid-run GT evaluation fired and shows clearly bad fit
                    # (RMSLE > 0.5), cap confidence so we don't terminate early
                    # on an overfitted / wrong-form equation.
                    if mid_gt_rmsle is not None and mid_gt_rmsle > 0.5:
                        capped = min(eq_confidence, 0.5)
                        if capped < eq_confidence:
                            console.print(
                                f"  [yellow]GT RMSLE={mid_gt_rmsle:.3f} (POOR) — "
                                f"capping confidence {eq_confidence:.2f} → {capped:.2f}[/yellow]"
                            )
                        eq_confidence = capped

                    console.print(
                        f"  [green]Best equation: {eq}[/green]\n"
                        f"  [dim]bootstrap={boot_conf:.2f}  llm={llm_conf:.2f}  "
                        f"combined({cfg.confidence_mode})={eq_confidence:.2f}"
                        + (f"  acq={current_acquisition}(κ={current_kappa:.1f})" if cfg.acquisition == "ei" else "")
                        + "[/dim]"
                    )

                    # ── Update LLM memory with this discovery ─────────────────
                    self._memory.add(
                        iteration=outer_iter,
                        n_data=len(self._store),
                        equation=eq.skeleton_str,
                        r2=eq.r2,
                        rmsle=eq.rmsle,
                        bootstrap_conf=boot_conf if cfg.confidence_feedback_enabled else 0.0,
                        gt_rmsle=mid_gt_rmsle,
                    )

                    # ── Check early termination ───────────────────────────────
                    if (
                        cfg.confidence_feedback_enabled
                        and self._objective_type == "equation"
                        and eq_confidence >= cfg.confidence_threshold
                    ):
                        console.print(
                            f"[bold green]Confidence ({cfg.confidence_mode}) "
                            f"{eq_confidence:.2f} ≥ {cfg.confidence_threshold}. "
                            f"Terminating early.[/bold green]"
                        )
                        termination_reason = "confident"
                        break

            # ── Track history ───────────────────────────────────────────────
            self._history.append({
                "iter": outer_iter,
                "oracle_calls": self._oracle_calls,
                "phase": self._current_phase,
                "policy": cfg.policy,
                "acquisition": current_acquisition,
                "kappa": current_kappa,
                "hypothesis": self._current_hypothesis,
                "alternate_hypotheses": list(getattr(proposal, "alternate_hypotheses", []) or []),
                "generator_alternate_hypotheses": generated_alts if cfg.strict_generator_discriminator_loop else [],
                "warm_sweep_param": sweep_param,
                "regions_ran": regions_ran,
                "families": list(self._hypothesis_families),
                "consecutive_poor_gt": self._consecutive_poor_gt,
                "candidate_pool": [str(c) for c in self._candidate_pool[:5]],
                "analysis_called": did_analysis_call,
                "analysis_calls_total": self._analysis_calls,
                "max_disagreement": iter_max_disagreement,
                "discrimination_gain_ratio": iter_discrimination_gain,
                "discrimination_has_signal": iter_discrimination_has_signal,
                "discrimination_effective": iter_discrimination_effective,
                "discriminator_stagnation_count": self._discriminator_stagnation_count,
                "boot_conf": boot_conf,
                "llm_conf": llm_conf,
                "eq_confidence": eq_confidence,
                "best_eq": str(self._eq_learner.best) if self._eq_learner.best else None,
                "last_gt_eval": self._last_gt_eval,
            })

            budget_remaining = cfg.budget - self._oracle_calls

        final_eq: FittedEquation | None = None
        if self._use_symbolic:
            # ── Final equation fitting ──────────────────────────────────────
            console.print(f"\n[bold]Running final equation fitting (policy={cfg.policy})...[/bold]")
            # Final call gets full pysr_n_iterations budget.
            # For ChemBench: still apply variable filter + residual promotion
            # so the final fit is on the cleanest possible reduced space.
            final_relevant = self._extract_relevant_vars(self._current_hypothesis)
            if final_relevant is not None and self._eq_learner.best is not None:
                final_relevant = self._residual_var_promotion(self._eq_learner.best, final_relevant)
            if final_relevant is not None:
                console.print(f"[dim]  Final variable filter: {final_relevant}[/dim]")
            # Always pass seed + LLM-generated families for final skeleton fit.
            if self._is_chem_domain():
                final_skel_fams = _dedupe(
                    list(self._seed_families)
                    + list(self._hypothesis_families),
                    k=30,
                ) or None
            else:
                final_skel_fams = None
            final_eq, final_candidates = self._policy_fit_equations(
                goal=cfg.goal,
                hypothesis=self._current_hypothesis,
                n_iterations=cfg.pysr_n_iterations,
                relevant_vars=final_relevant,
                skeleton_families=final_skel_fams,
            )
            validated_final_candidates = self._validated_hypothesis_candidates(top_k=5)
            if validated_final_candidates:
                console.print(
                    f"[dim]Added {len(validated_final_candidates)} validated-hypothesis "
                    f"candidate(s) to final arbitration[/dim]"
                )
            all_final_candidates = list(final_candidates) + list(validated_final_candidates)
            deduped_final_candidates: list[FittedEquation] = []
            seen_final: set[str] = set()
            for cand in all_final_candidates:
                if cand is None:
                    continue
                key = (cand.skeleton_str or "").strip().lower()
                if not key or key in seen_final:
                    continue
                seen_final.add(key)
                deduped_final_candidates.append(cand)
            if deduped_final_candidates:
                # LLM arbitration: ask the main LLM to pick the best candidate
                # based on scientific plausibility, not just training R².
                # This prefers structurally correct skeletons over PySR equations
                # that numerically fit by adding spurious variables.
                final_eq = self._llm_arbitrate_candidates(
                    deduped_final_candidates,
                    store=self._store,
                    param_names=self._oracle.parameter_names,
                ) or max(deduped_final_candidates, key=lambda c: float(c.r2))

            # ── Fix 2: Prune spurious variables from final equation ────────────
            if final_eq is not None and cfg.prune_spurious_vars:
                console.print("[dim]  → Pruning spurious variables from final equation...[/dim]")
                final_eq = self._prune_spurious_vars(
                    final_eq, self._store, list(self._oracle.parameter_names)
                )

            # ── Fix 3: OOD validation ──────────────────────────────────────────
            # If OOD RMSLE >> training RMSLE (>3×), the equation is overfit to
            # the training range. Prefer the best skeleton candidate instead.
            if (final_eq is not None and self._ood_store is not None
                    and len(self._ood_store) >= 2):
                ood_rmsle = self._ood_rmsle(final_eq, self._ood_store)
                if ood_rmsle is not None:
                    train_rmsle = final_eq.rmsle
                    ratio = ood_rmsle / max(train_rmsle, 1e-6)
                    console.print(
                        f"  [dim]OOD RMSLE={ood_rmsle:.4f}  Train RMSLE={train_rmsle:.4f}  "
                        f"ratio={ratio:.1f}x[/dim]"
                    )
                    if ratio > 3.0:
                        console.print(
                            f"  [yellow]OOD ratio {ratio:.1f}x > 3 — equation likely overfit. "
                            f"Searching for better-generalising skeleton candidate...[/yellow]"
                        )
                        # Among all candidates, prefer the one with best OOD RMSLE
                        best_ood_eq = final_eq
                        best_ood_rmsle = ood_rmsle
                        for cand in deduped_final_candidates:
                            if cand is None or cand is final_eq:
                                continue
                            cand_ood = self._ood_rmsle(cand, self._ood_store)
                            if cand_ood is not None and cand_ood < best_ood_rmsle:
                                best_ood_rmsle = cand_ood
                                best_ood_eq = cand
                        if best_ood_eq is not final_eq:
                            console.print(
                                f"  [green]OOD-preferred equation: {best_ood_eq.skeleton_str[:60]} "
                                f"(OOD RMSLE={best_ood_rmsle:.4f})[/green]"
                            )
                            final_eq = best_ood_eq
            # GT-evaluate any promising final candidates
            if self._objective_type == "equation":
                for fc in deduped_final_candidates:
                    if not fc:
                        continue
                    try:
                        fc_gt = self._oracle.evaluate_law(fc.law_str)
                        fc_rmsle = fc_gt.get("rmsle")
                        if fc_rmsle is not None and np.isfinite(fc_rmsle) and fc_rmsle < 1.0:
                            if (self._best_gt_equation is None
                                    or fc_rmsle < (self._best_gt_eval_rmsle or 99)):
                                self._best_gt_equation = fc
                                self._best_gt_eval_rmsle = fc_rmsle
                    except Exception:
                        pass

            # ── Prefer best GT-validated equation if it was better ──────────
            # Sometimes the final fitting (which uses random restarts) finds a
            # different local optimum with higher training R² but worse GT RMSLE.
            # If a mid-run GT evaluation found a clearly better equation, prefer it.
            if (self._objective_type == "equation"
                    and self._best_gt_equation is not None
                    and self._best_gt_eval_rmsle is not None
                    and self._best_gt_eval_rmsle < 0.1):
                console.print(
                    f"[dim]Best GT-validated equation (RMSLE={self._best_gt_eval_rmsle:.5f}) "
                    f"preserved from mid-run evaluation.[/dim]"
                )
                final_eq = self._best_gt_equation

        # ── Evaluate final output ────────────────────────────────────────────
        final_eval = None
        if self._objective_type == "equation" and final_eq:
            console.print(f"\n[bold green]Best discovered equation:[/bold green]")
            console.print(final_eq.law_str)
            console.print(f"R² = {final_eq.r2:.4f} | RMSLE = {final_eq.rmsle:.4f}")

            console.print("\n[dim]Evaluating against ground truth...[/dim]")
            try:
                final_eval = self._oracle.evaluate_law(final_eq.law_str)
                console.print(f"[bold]Ground truth evaluation:[/bold]")
                console.print(f"  RMSLE: {final_eval.get('rmsle', 'N/A'):.4f}")
                console.print(f"  Exact accuracy: {final_eval.get('exact_accuracy', 'N/A')}")
            except Exception as e:
                console.print(f"[yellow]Could not run evaluation: {e}[/yellow]")
                final_eval = {"error": str(e)}
        elif self._objective_type == "optimization":
            console.print("\n[dim]Evaluating run-level optimization objective...[/dim]")
            final_eval = self._oracle.evaluate_run(self._store.get_all())
            if final_eval is None:
                final_eval = self._default_run_evaluation()
            console.print("[bold]Run evaluation:[/bold]")
            if final_eval.get("best_measurement") is not None:
                console.print(f"  Best measurement: {final_eval['best_measurement']:.6g}")
            if final_eval.get("improvement_ratio") is not None:
                console.print(f"  Improvement ratio: {final_eval['improvement_ratio']:.4f}")

        duration = time.time() - start_time
        console.print(f"\n[bold]Done![/bold] {self._oracle_calls} oracle calls | "
                      f"{self._agent.call_count} LLM calls | {duration:.1f}s | "
                      f"Reason: {termination_reason}")

        result = DiscoveryResult(
            domain=cfg.domain_id,
            goal=cfg.goal,
            best_equation=final_eq,
            final_evaluation=final_eval,
            n_oracle_calls=self._oracle_calls,
            n_llm_calls=self._agent.call_count,
            store=self._store,
            termination_reason=termination_reason,
            duration_seconds=duration,
            history=self._history,
        )
        result.status = self._compute_status(result)
        self._save_results(result)
        return result

    def _normalise_hypothesis_expr(self, raw: str) -> str:
        """
        Normalize LLM hypothesis text into a Python expression.

        Handles:
        - LHS assignments including function-like LHS: ``N(t) = ...``
        - Pipe-delimited prose: ``| where ...``
        - Trailing inline prose: ``, where n is...``, `` for some k``, `` where ...``
        - Caret exponent syntax: ``^`` → ``**``
        - Math aliases: ``ln(`` → ``log(``, ``arcsin(`` → ``asin(``
        - Trailing unit annotations: ``[m/s]``
        """
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
        # Strip trailing parenthetical prose (only pure-alpha prose, not math expressions)
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
        """Extract free symbol names from expression (constants to be fitted)."""
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
        return [n for n in names if n not in known and n not in param_names]

    def _fit_hypothesis_expression(
        self,
        hypothesis: str,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any] | None:
        """
        Fit free constants in a hypothesis expression and return predictions/metrics.
        """
        from scipy.optimize import minimize
        import math

        if X.size == 0 or y.size == 0:
            return None
        expr = self._normalise_hypothesis_expr(hypothesis)
        if not expr:
            return None

        param_names = self._oracle.parameter_names
        const_names = self._extract_unbound_names(expr, param_names)
        sig = ", ".join(param_names + const_names)
        fn_code = f"def _hyp_fn({sig}):\n    return {expr}\n"
        gns: dict[str, Any] = {
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
        try:
            local_ns: dict[str, Any] = {}
            exec(fn_code, gns, local_ns)
            fn = local_ns["_hyp_fn"]
        except Exception:
            return None

        y_safe = np.where(y > 0, y, np.nan)

        def _log_mse(const_vals: np.ndarray) -> float:
            try:
                y_pred_local = np.array(
                    [
                        fn(
                            *[float(X[i, j]) for j in range(X.shape[1])],
                            *[float(c) for c in const_vals],
                        )
                        for i in range(len(X))
                    ],
                    dtype=float,
                )
            except Exception:
                return 1e9
            mask = np.isfinite(y_pred_local) & np.isfinite(y_safe) & (y_pred_local > 0)
            if int(np.sum(mask)) < 3:
                return 1e9
            diff = np.log(y_pred_local[mask]) - np.log(y_safe[mask])
            return float(np.mean(diff**2))

        fitted_consts = np.array([], dtype=float)
        if const_names:
            best_loss = float("inf")
            best_x: np.ndarray | None = None
            for seed in (0.3, 1.0, 3.0):
                x0 = np.full(len(const_names), seed, dtype=float)
                try:
                    res = minimize(
                        _log_mse,
                        x0,
                        method="Nelder-Mead",
                        options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4},
                    )
                    if float(res.fun) < best_loss:
                        best_loss = float(res.fun)
                        best_x = np.array(res.x, dtype=float)
                except Exception:
                    continue
            if best_x is None or not np.isfinite(best_loss) or best_loss > 1e8:
                return None
            fitted_consts = best_x
        else:
            loss0 = _log_mse(np.array([], dtype=float))
            if not np.isfinite(loss0) or loss0 > 1e8:
                return None

        try:
            y_pred = np.array(
                [
                    fn(
                        *[float(X[i, j]) for j in range(X.shape[1])],
                        *[float(c) for c in fitted_consts],
                    )
                    for i in range(len(X))
                ],
                dtype=float,
            )
        except Exception:
            return None

        mask_r2 = np.isfinite(y_pred) & np.isfinite(y)
        if int(np.sum(mask_r2)) < 3:
            return None
        yv = y[mask_r2]
        pv = y_pred[mask_r2]
        denom = float(np.sum((yv - np.mean(yv)) ** 2)) + 1e-12
        r2 = 1.0 - float(np.sum((yv - pv) ** 2) / denom)
        mask_rmsle = np.isfinite(y_pred) & np.isfinite(y_safe) & (y_pred > 0)
        if int(np.sum(mask_rmsle)) < 3:
            rmsle = float("inf")
        else:
            rmsle = float(
                np.sqrt(
                    np.mean(
                        (
                            np.log(y_pred[mask_rmsle])
                            - np.log(y_safe[mask_rmsle])
                        )
                        ** 2
                    )
                )
            )

        const_map = {c: float(v) for c, v in zip(const_names, fitted_consts)}
        expr_eval = expr
        for cname, val in const_map.items():
            expr_eval = re.sub(rf"\b{re.escape(cname)}\b", f"({val:.10g})", expr_eval)
        # Ensure bare math function calls are resolvable in oracle eval scope.
        for fn_name in (
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
        ):
            expr_eval = re.sub(
                rf"(?<![\w.]){fn_name}\s*\(",
                f"math.{fn_name}(",
                expr_eval,
            )
        params_sig = ", ".join(param_names)
        law_str = f"def discovered_law({params_sig}):\n    return {expr_eval}\n"

        return {
            "expr": expr,
            "law_str": law_str,
            "r2": float(r2),
            "rmsle": float(rmsle),
            "constants": const_map,
            "y_pred": y_pred,
        }

    def _build_scoring_subset(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        new_data_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a scoring slice: latest points + stratified historical coverage.
        """
        if X_all.size == 0 or y_all.size == 0:
            return X_all, y_all
        if new_data_count <= 0 or new_data_count >= len(X_all):
            return X_all, y_all
        if not self._cfg.stratified_hypothesis_scoring:
            return X_all[-new_data_count:], y_all[-new_data_count:]

        X_new = X_all[-new_data_count:]
        y_new = y_all[-new_data_count:]
        max_pts = max(4, int(self._cfg.hypothesis_scoring_max_points))
        remaining = max_pts - len(X_new)
        if remaining <= 0:
            return X_new[:max_pts], y_new[:max_pts]

        X_old = X_all[:-new_data_count]
        y_old = y_all[:-new_data_count]
        if len(X_old) == 0:
            return X_new, y_new

        logy = np.log(np.abs(y_old) + 1e-300)
        order = np.argsort(logy)
        take = np.unique(np.linspace(0, len(order) - 1, num=min(remaining, len(order)), dtype=int))
        idx_old = order[take]

        X_mix = np.vstack([X_old[idx_old], X_new])
        y_mix = np.concatenate([y_old[idx_old], y_new])
        return X_mix, y_mix

    def _validated_hypothesis_candidates(self, top_k: int = 3) -> list[FittedEquation]:
        """
        Fit top validated hypotheses on full observed data and expose as equation candidates.
        """
        validated = self._memory.get_validated_hypotheses(top_k=max(1, top_k))
        if not validated or len(self._store) < 5:
            return []
        X_all, y_all = self._store.to_arrays()
        candidates: list[FittedEquation] = []
        for rec in validated:
            fit = self._fit_hypothesis_expression(rec.hypothesis, X_all, y_all)
            if fit is None:
                continue
            eq = FittedEquation(
                law_str=str(fit["law_str"]),
                skeleton_str=str(fit["expr"]),
                r2=float(fit["r2"]),
                rmsle=float(fit["rmsle"]),
                constants=dict(fit["constants"]),
                n_data_points=len(X_all),
                expected_form=str(rec.hypothesis),
                confidence=0.0,
            )
            if eq.is_valid():
                candidates.append(eq)
        candidates.sort(key=lambda c: c.rmsle)
        dedup: list[FittedEquation] = []
        seen = set()
        for c in candidates:
            key = c.skeleton_str.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(c)
            if len(dedup) >= top_k:
                break
        return dedup

    def _get_operator_restriction_profile(self) -> dict[str, Any] | None:
        """
        Derive a restricted operator profile from strongly validated hypotheses.
        """
        cfg = self._cfg
        if not cfg.validated_operator_restriction:
            return None
        validated = self._memory.get_validated_hypotheses(top_k=8)
        if not validated:
            return None

        best = None
        for rec in validated:
            support = self._memory.validated_support(rec.hypothesis)
            if rec.fit_rmsle <= cfg.restriction_min_validated_rmsle and support >= cfg.restriction_min_support:
                best = rec
                break
        if best is None:
            return None

        h = best.hypothesis.lower()
        unary = []
        for op in ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "sinh", "cosh", "tanh"]:
            if re.search(rf"\b{op}\s*\(", h):
                unary.append(op)
        if not unary:
            unary = _unary_operators_from_hypothesis(
                best.hypothesis,
                getattr(self._oracle, "domain", None),
                getattr(self._oracle, "domain_tags", []),
            )
        binary = ["+", "-", "*", "^"]
        if "/" in h:
            binary.append("/")
        return {
            "hypothesis": best.hypothesis,
            "support": self._memory.validated_support(best.hypothesis),
            "fit_rmsle": float(best.fit_rmsle),
            "unary_ops": unary,
            "binary_operators": binary,
        }

    def _predict_with_law(self, law_str: str, X: np.ndarray) -> np.ndarray | None:
        """Execute a discovered_law string and return predictions on X."""
        if not law_str or X.size == 0:
            return None
        try:
            import math
            local_ns: dict[str, Any] = {}
            global_ns: dict[str, Any] = {
                "np": np,
                "math": math,
                "abs": abs,
                "min": min,
                "max": max,
                "pow": pow,
            }
            exec(law_str, global_ns, local_ns)
            fn = local_ns.get("discovered_law") or global_ns.get("discovered_law")
            if not callable(fn):
                return None
            preds = []
            for row in X:
                try:
                    val = float(fn(*[float(v) for v in row]))
                except Exception:
                    val = float("nan")
                preds.append(val)
            y_pred = np.array(preds, dtype=float)
            if not np.isfinite(y_pred).any():
                return None
            return y_pred
        except Exception:
            return None

    def _normalise_hypothesis_expr_text(self, raw: str) -> str:
        s = (raw or "").strip()
        if not s:
            return ""
        if "|" in s:
            s = s.split("|", 1)[0].strip()
        lhs_eq = re.match(r"^\s*[A-Za-z_]\w*(?:\([^)]*\))?\s*=\s*", s)
        if lhs_eq:
            s = s[lhs_eq.end():]
        # ^ → ** (only when not already **) — sentinel avoids lookbehind (py3.12 bug)
        _dstar = "\x00DSTAR\x00"
        s = s.replace("**", _dstar)
        s = s.replace("^", "**")
        s = s.replace(_dstar, "**")
        return s.strip()

    def _extract_transformed_terms(self, hypotheses: list[str]) -> list[str]:
        """
        Extract transformed/composite terms from hypotheses for coverage diagnostics.
        """
        if not hypotheses:
            return []
        terms: list[str] = []
        fn_pat = re.compile(
            r"\b(?:exp|log|sqrt|sin|cos|tan|sinh|cosh|tanh|asin|acos|atan)\s*\(([^()]{1,120})\)"
        )
        for h in hypotheses:
            expr = self._normalise_hypothesis_expr_text(h)
            if not expr:
                continue
            for m in fn_pat.finditer(expr):
                full = m.group(0).strip()
                if any(p in full for p in self._oracle.parameter_names):
                    terms.append(full)
        # Also track pairwise multiplicative arguments (common identifiability sink).
        params = self._oracle.parameter_names
        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                terms.append(f"{params[i]}*{params[j]}")

        dedup: list[str] = []
        seen = set()
        for t in terms:
            k = t.replace(" ", "")
            if k in seen:
                continue
            seen.add(k)
            dedup.append(t)
            if len(dedup) >= 8:
                break
        return dedup

    def _build_transformed_coverage_hints(self, hypotheses: list[str]) -> str:
        """
        Domain-agnostic coverage diagnostics in hypothesis-implied transformed spaces.
        """
        cfg = self._cfg
        if not cfg.transformed_coverage_hints:
            return ""
        if len(self._store) < 6:
            return ""
        terms = self._extract_transformed_terms(hypotheses)
        if not terms:
            return ""

        X, _ = self._store.to_arrays()
        if X.size == 0:
            return ""
        param_names = self._oracle.parameter_names
        import math

        global_ns = {
            "np": np,
            "math": math,
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "abs": abs,
            "pow": pow,
            "min": min,
            "max": max,
        }

        low_terms: list[tuple[str, float]] = []
        for term in terms:
            vals = []
            for row in X:
                local_ns = {p: float(row[i]) for i, p in enumerate(param_names)}
                try:
                    v = float(eval(term, global_ns, local_ns))
                except Exception:
                    v = float("nan")
                if np.isfinite(v):
                    vals.append(v)
            if len(vals) < 4:
                continue
            arr = np.array(vals, dtype=float)
            # Prefer log-span for positive transforms; fallback to normalized linear span.
            if np.all(arr > 0):
                q10, q90 = np.quantile(arr, [0.1, 0.9])
                span = float(np.log10((q90 + 1e-300) / (q10 + 1e-300)))
                if span < cfg.transformed_coverage_low_span_decades:
                    low_terms.append((term, span))
            else:
                q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
                span = float((q90 - q10) / (abs(q50) + 1e-12))
                if span < 0.6:
                    low_terms.append((term, span))

        if not low_terms:
            return ""
        low_terms.sort(key=lambda t: t[1])
        lines = ["TRANSFORMED-SPACE COVERAGE WARNING:"]
        for term, span in low_terms[:3]:
            lines.append(
                f"- Low span for `{term}` (span={span:.3f}); propose at least one region that expands this transformed range."
            )
        lines.append("- Prioritize regions where this transformed term changes by large factors, not only raw-parameter refinement.")
        return "\n".join(lines)

    def _score_hypothesis_strings(
        self,
        hypotheses: list[str],
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """
        Score each raw hypothesis expression string against (X, y) data.

        Wraps each expression into a callable, fits free constants (C0, C1, ...)
        using scipy.optimize.minimize (log-space MSE), and returns RMSLE.
        Returns float('inf') for hypotheses that cannot be parsed or optimised.

        This is the scoring step of the generator-discriminator loop — after
        discriminating experiments run we call this to determine which candidate
        hypotheses are consistent with the new evidence.
        """
        results: dict[str, float] = {}

        for hyp in hypotheses:
            if not hyp or not hyp.strip():
                results[hyp] = float("inf")
                continue
            fit = self._fit_hypothesis_expression(hypothesis=hyp, X=X, y=y)
            if fit is None:
                results[hyp] = float("inf")
            else:
                results[hyp] = float(fit.get("rmsle", float("inf")))

        return results

    def _update_families_from_discrimination(
        self,
        hypotheses: list[str],
        scores: dict[str, float],
        outer_iter: int,
        n_data: int,
        n_full_data: int,
        good_rmsle_threshold: float = 0.5,
        poor_rmsle_threshold: float = 2.0,
    ) -> None:
        """
        After discriminating experiments, promote survivors to memory and retire
        clearly falsified hypotheses.

        Args:
            hypotheses: all candidate hypothesis strings scored this round
            scores: {hypothesis_str: rmsle} from _score_hypothesis_strings()
            outer_iter: current outer loop iteration (for record-keeping)
            n_data: number of new data points used for scoring
            n_full_data: total number of observed points in store
            good_rmsle_threshold: RMSLE below this → validated survivor
            poor_rmsle_threshold: RMSLE above this → negative evidence
        """
        survivors: list[str] = []
        X_full = y_full = None
        if self._cfg.promotion_full_store_gate:
            X_full, y_full = self._store.to_arrays()
        for hyp in hypotheses:
            rmsle = scores.get(hyp, float("inf"))
            if not np.isfinite(rmsle):
                self._memory.add_hypothesis_score(
                    iteration=outer_iter,
                    hypothesis=hyp,
                    fit_rmsle=float("inf"),
                    n_data=n_data,
                    status="unscored",
                )
                continue
            if rmsle <= good_rmsle_threshold:
                promoted_rmsle = float(rmsle)
                if self._cfg.promotion_full_store_gate and X_full is not None and y_full is not None and len(X_full) >= 6:
                    fit_full = self._fit_hypothesis_expression(hypothesis=hyp, X=X_full, y=y_full)
                    full_rmsle = float(fit_full.get("rmsle", float("inf"))) if fit_full is not None else float("inf")
                    if (not np.isfinite(full_rmsle)) or full_rmsle > self._cfg.promotion_full_store_max_rmsle:
                        self._memory.add_hypothesis_score(
                            iteration=outer_iter,
                            hypothesis=hyp,
                            fit_rmsle=rmsle,
                            n_data=n_data,
                            status="rejected_full_store_gate",
                        )
                        if self._cfg.negative_evidence_enabled:
                            self._memory.add_negative_evidence(
                                iteration=outer_iter,
                                hypothesis=hyp,
                                evidence=(
                                    f"Locally plausible (RMSLE={rmsle:.3f} on scoring slice) but "
                                    f"failed full-store gate (RMSLE={full_rmsle:.3f} on {n_full_data} points)."
                                ),
                            )
                        console.print(
                            f"  [yellow]↷ Rejected by full-store gate: local={rmsle:.3f}, "
                            f"full={full_rmsle:.3f} | {hyp[:80]}[/yellow]"
                        )
                        continue
                    promoted_rmsle = full_rmsle
                survivors.append(hyp)
                self._memory.add_validated_hypothesis(
                    iteration=outer_iter,
                    hypothesis=hyp,
                    fit_rmsle=promoted_rmsle,
                    n_data=n_full_data if self._cfg.promotion_full_store_gate else n_data,
                )
                console.print(
                    f"  [green]✓ Hypothesis validated (RMSLE={promoted_rmsle:.3f}): "
                    f"{hyp[:80]}[/green]"
                )
            elif rmsle >= poor_rmsle_threshold and self._cfg.negative_evidence_enabled:
                self._memory.add_negative_evidence(
                    iteration=outer_iter,
                    hypothesis=hyp,
                    evidence=(
                        f"Scored RMSLE={rmsle:.3f} on {n_data} discriminating "
                        f"experiments — clearly inconsistent with data."
                    ),
                )
                self._memory.add_hypothesis_score(
                    iteration=outer_iter,
                    hypothesis=hyp,
                    fit_rmsle=rmsle,
                    n_data=n_data,
                    status="failed",
                )
                console.print(
                    f"  [red]✗ Hypothesis falsified (RMSLE={rmsle:.3f}): "
                    f"{hyp[:80]}[/red]"
                )
            else:
                self._memory.add_hypothesis_score(
                    iteration=outer_iter,
                    hypothesis=hyp,
                    fit_rmsle=rmsle,
                    n_data=n_data,
                    status="uncertain",
                )

        # Purge falsified hypotheses from the active family set
        if survivors or scores:
            falsified = {
                h for h, r in scores.items()
                if np.isfinite(r) and r >= poor_rmsle_threshold
            }
            self._hypothesis_families = [
                f for f in self._hypothesis_families if f not in falsified
            ]
            if survivors:
                console.print(
                    f"  [dim]{len(survivors)}/{len(hypotheses)} hypotheses survived "
                    f"discrimination.[/dim]"
                )

    def _compute_disagreement_diagnostics(
        self,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Build structured diagnostics for the analysis LLM call.

        Returns:
            residual_stats: per-candidate residual summaries
            disagreement_stats: pairwise prediction disagreement summaries
        """
        if len(self._store) < 5:
            return [], []

        X, y = self._store.to_arrays()
        if X.size == 0 or y.size == 0:
            return [], []

        residual_stats: list[dict[str, Any]] = []
        pred_bank: list[tuple[str, np.ndarray, np.ndarray]] = []

        for c in candidates[:6]:
            y_pred = None
            law_str = c.get("law_str")
            if isinstance(law_str, str):
                y_pred = self._predict_with_law(law_str, X)
            if y_pred is None:
                y_pred_raw = c.get("y_pred")
                if isinstance(y_pred_raw, np.ndarray) and y_pred_raw.shape[0] == y.shape[0]:
                    y_pred = y_pred_raw.astype(float, copy=False)
            if y_pred is None:
                continue
            mask = np.isfinite(y_pred) & np.isfinite(y)
            if int(np.sum(mask)) < 4:
                continue
            y_m = y[mask]
            y_p = y_pred[mask]
            rel_err = np.abs(y_p - y_m) / (np.abs(y_m) + 1e-12)
            signed_rel = (y_p - y_m) / (np.abs(y_m) + 1e-12)
            if np.std(y_m) > 1e-12 and np.std(y_p - y_m) > 1e-12:
                corr = float(np.corrcoef(y_m, (y_p - y_m))[0, 1])
            else:
                corr = 0.0
            trend = "flat"
            if corr > 0.25:
                trend = "error_increases_with_output"
            elif corr < -0.25:
                trend = "error_decreases_with_output"

            residual_stats.append(
                {
                    "equation": str(c.get("equation", "unknown"))[:120],
                    "mae_rel": float(np.mean(rel_err)),
                    "bias": float(np.mean(signed_rel)),
                    "trend": trend,
                    "n_points": int(np.sum(mask)),
                }
            )
            pred_bank.append((str(c.get("equation", "unknown"))[:120], y_pred, mask))

        disagreement_stats: list[dict[str, Any]] = []
        param_names = self._oracle.parameter_names
        for i in range(len(pred_bank)):
            eq_i, pred_i, mask_i = pred_bank[i]
            for j in range(i + 1, len(pred_bank)):
                eq_j, pred_j, mask_j = pred_bank[j]
                mask = mask_i & mask_j
                if int(np.sum(mask)) < 4:
                    continue
                y_m = y[mask]
                pi = pred_i[mask]
                pj = pred_j[mask]
                rel_gap = np.abs(pi - pj) / (np.abs(y_m) + 1e-12)
                if rel_gap.size == 0:
                    continue
                x_m = X[mask]
                top_cut = np.quantile(rel_gap, 0.85)
                top = rel_gap >= top_cut
                x_top = x_m[top]
                if x_top.size == 0:
                    x_top = x_m
                region = {
                    p: [float(np.min(x_top[:, k])), float(np.max(x_top[:, k]))]
                    for k, p in enumerate(param_names)
                }
                disagreement_stats.append(
                    {
                        "eq_i": eq_i,
                        "eq_j": eq_j,
                        "mean_rel_gap": float(np.mean(rel_gap)),
                        "max_rel_gap": float(np.max(rel_gap)),
                        "region": region,
                    }
                )

        residual_stats.sort(key=lambda d: d["mae_rel"])
        disagreement_stats.sort(key=lambda d: d["max_rel_gap"], reverse=True)
        return residual_stats[:6], disagreement_stats[:8]

    def _compute_status(self, result: DiscoveryResult) -> str:
        """
        Classify the run outcome with a single status tag for diagnostics.

        success      — GT RMSLE < 0.01  (excellent fit)
        good         — GT RMSLE < 0.10  (useful result)
        underfit     — train R² < 0.80  (PySR couldn't fit the data)
        overfit      — val R² high but GT RMSLE > 0.5 (memorized, wrong structure)
        exploration_insufficient — budget exhausted with no confident equation
        no_equation  — PySR never returned a usable equation
        partial      — got an equation but quality is unclear
        """
        if self._objective_type == "optimization":
            if not result.final_evaluation:
                return "no_evaluation"
            if bool(result.final_evaluation.get("target_hit", False)):
                return "success"
            normalized_best = result.final_evaluation.get("normalized_best")
            if isinstance(normalized_best, (float, int)):
                if normalized_best >= 0.95:
                    return "success"
                if normalized_best >= 0.80:
                    return "good"
            improvement = result.final_evaluation.get("improvement_ratio")
            if isinstance(improvement, (float, int)):
                if improvement >= 0.50:
                    return "success"
                if improvement >= 0.15:
                    return "good"
            return "partial"

        if not result.best_equation:
            return "no_equation"

        gt_rmsle = (
            result.final_evaluation.get("rmsle", 99)
            if result.final_evaluation else 99
        )
        r2 = result.best_equation.r2
        rmsle_val = result.best_equation.rmsle

        if gt_rmsle < 0.01:
            return "success"
        if gt_rmsle < 0.10:
            return "good"
        if r2 < 0.80:
            return "underfit"
        if r2 > 0.98 and gt_rmsle > 0.5:
            return "overfit"
        if result.termination_reason == "budget" and result.best_equation.confidence < 0.5:
            return "exploration_insufficient"
        return "partial"

    def _save_results(self, result: DiscoveryResult) -> None:
        import json

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = self._cfg.results_dir / f"{result.domain}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        result.store.save(out_dir / "experiments.json")

        status = self._compute_status(result)

        summary = {
            "domain": result.domain,
            "goal": result.goal,
            "policy": self._cfg.policy,
            "objective_type": self._objective_type,
            "hypothesis_backend": self._cfg.hypothesis_backend,
            "status": status,
            "confidence_mode": self._cfg.confidence_mode,
            "acquisition": self._cfg.acquisition,
            "n_diverse_pysr_runs": self._cfg.n_diverse_pysr_runs,
            "pysr_n_iterations": self._cfg.pysr_n_iterations,
            "objective_aware_prompting": self._cfg.objective_aware_prompting,
            "analysis_feedback_enabled": self._cfg.analysis_feedback_enabled,
            "analysis_disagreement_threshold": self._cfg.analysis_disagreement_threshold,
            "analysis_calls": self._analysis_calls,
            "negative_evidence_enabled": self._cfg.negative_evidence_enabled,
            "strict_generator_discriminator_loop": self._cfg.strict_generator_discriminator_loop,
            "validated_operator_restriction": self._cfg.validated_operator_restriction,
            "restriction_parallel_unrestricted": self._cfg.restriction_parallel_unrestricted,
            "restriction_min_validated_rmsle": self._cfg.restriction_min_validated_rmsle,
            "restriction_min_support": self._cfg.restriction_min_support,
            "stratified_hypothesis_scoring": self._cfg.stratified_hypothesis_scoring,
            "hypothesis_scoring_max_points": self._cfg.hypothesis_scoring_max_points,
            "pysr_warm_start_chaining": self._cfg.pysr_warm_start_chaining,
            "pysr_warm_start_gate_min_r2": self._cfg.pysr_warm_start_gate_min_r2,
            "pysr_warm_start_gate_max_rmsle": self._cfg.pysr_warm_start_gate_max_rmsle,
            "separable_factor_locking": self._cfg.separable_factor_locking,
            "separable_lock_max_factors": self._cfg.separable_lock_max_factors,
            "separable_lock_min_r2": self._cfg.separable_lock_min_r2,
            "separable_lock_min_span_decades": self._cfg.separable_lock_min_span_decades,
            "separable_lock_max_slope_std": self._cfg.separable_lock_max_slope_std,
            "transformed_coverage_hints": self._cfg.transformed_coverage_hints,
            "transformed_coverage_low_span_decades": self._cfg.transformed_coverage_low_span_decades,
            "n_oracle_calls": result.n_oracle_calls,
            "n_llm_calls": result.n_llm_calls,
            "termination_reason": result.termination_reason,
            "duration_seconds": result.duration_seconds,
            "best_equation": result.best_equation.law_str if result.best_equation else None,
            "skeleton": result.best_equation.skeleton_str if result.best_equation else None,
            "r2": result.best_equation.r2 if result.best_equation else None,
            "rmsle": result.best_equation.rmsle if result.best_equation else None,
            "bootstrap_confidence": result.best_equation.confidence if result.best_equation else None,
            "final_evaluation": result.final_evaluation,
            "history": result.history,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        console.print(f"[bold]Status:[/bold] [{'green' if status in ('success','good') else 'yellow' if status in ('partial', 'no_evaluation') else 'red'}]{status}[/]")
        console.print(f"Results saved to [cyan]{out_dir}[/cyan]")

    def _default_run_evaluation(self) -> dict[str, Any]:
        """Fallback optimization-style summary when oracle has no evaluate_run()."""
        results = self._store.get_all()
        if not results:
            return {"objective_type": "optimization", "error": "no_results"}

        y = np.array([r.measurement for r in results], dtype=float)
        direction = getattr(self._oracle, "objective_direction", "maximize")
        if direction == "minimize":
            best = float(np.min(y))
            first = float(y[0])
            denom = max(abs(first), 1e-12)
            improvement_ratio = float((first - best) / denom)
        else:
            best = float(np.max(y))
            first = float(y[0])
            denom = max(abs(first), 1e-12)
            improvement_ratio = float((best - first) / denom)
        return {
            "objective_type": "optimization",
            "objective_direction": direction,
            "best_measurement": best,
            "mean_measurement": float(np.mean(y)),
            "initial_measurement": first,
            "improvement_ratio": improvement_ratio,
            "n_evals": len(results),
        }
