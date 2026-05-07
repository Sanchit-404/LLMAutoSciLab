#!/usr/bin/env python3
"""
ChemBench comparison: multiple baselines vs LLM pipeline.

Baselines
---------
bo          GP-EI + ARD variable selection + PySR          (current best algorithmic)
random      Uniform random sampling + PySR                  (floor)
uncertainty GP-variance (pure exploration) + ARD + PySR    (exploration-focused BO)
oracle_vars Random sampling + PySR on ground-truth vars     (variable selection oracle)
bed         Bayesian Experimental Design over mechanism lib  (model-directed BO)

LLM conditions
--------------
llm         Full LLM+AL+SR pipeline  (grammar/tag flags control leakage)

Usage:
    python scripts/run_chembench_comparison.py \\
        --domain c2_product_inhibition --budget 60 --difficulty easy --law-version v0

    # Run only specific baselines:
    python scripts/run_chembench_comparison.py --domain c2_product_inhibition \\
        --baselines random uncertainty oracle_vars --llm-only
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube

from autoscilab.oracle.chembench import (
    ChemBenchOracle, CHEM_INPUT_BOUNDS, CHEM_INPUT_VARS,
    CHEM_LOG_VARS, CHEM_RMSLE_THRESHOLD, CHEM_DOMAIN_REGISTRY,
)
from autoscilab.data.store import ExperimentStore
from autoscilab.al.gp_model import GPSurrogate
from autoscilab.al.selector import ALSelector


# ---------------------------------------------------------------------------
# BO-only baseline (no LLM)
# ---------------------------------------------------------------------------

def run_bo_baseline(oracle: ChemBenchOracle, budget: int, seed: int = 42) -> dict:
    """
    GP-EI active learning over the full 7D input space. No LLM.

    Experiment design:
    - 10 initial LHS points across all 7 variables
    - Then GP-EI selects 1 point per step for remaining budget
    - Single PySR call at end on all collected data (all 7 columns, no hints)

    This is deliberately not clever: it represents what a pure BO+SR system
    does without any mechanism hypothesis.
    """
    from autoscilab.equation_learner.learner import EquationLearner

    print("\n" + "="*60)
    print("BO-ONLY BASELINE (no LLM)")
    print("="*60)

    store = ExperimentStore(oracle.domain, CHEM_INPUT_VARS)
    gp = GPSurrogate()
    rng = np.random.default_rng(seed)
    bounds = CHEM_INPUT_BOUNDS
    t0 = time.time()

    # --- Initial LHS sampling (10 points) ---
    n_init = min(10, budget)
    sampler = LatinHypercube(d=len(CHEM_INPUT_VARS), seed=seed)
    lhs = sampler.random(n=n_init)
    for i in range(n_init):
        params = {}
        for j, var in enumerate(CHEM_INPUT_VARS):
            lo, hi = bounds[var]
            u = lhs[i, j]
            if var in CHEM_LOG_VARS and lo > 0:
                params[var] = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
            else:
                params[var] = lo + u * (hi - lo)
        result = oracle.run(params)
        store.add(result)
        print(f"  init {i+1:2d}/{n_init}: r0={result.measurement:.4f}  "
              f"C_A={params['C_A']:.2f} C_I={params['C_I']:.2f} "
              f"C_B={params['C_B']:.2f} C_P={params['C_P']:.2f} T={params['T']:.0f}")

    # --- GP-EI acquisition loop ---
    remaining = budget - n_init
    for step in range(remaining):
        X, y = store.to_arrays()
        gp.fit(X, y, bounds)

        # Sample 2000 candidates over full bounds (EI acquisition)
        n_cand = 2000
        cand_raw = rng.random((n_cand, len(CHEM_INPUT_VARS)))
        candidates = []
        cand_X = np.zeros((n_cand, len(CHEM_INPUT_VARS)))
        for j, var in enumerate(CHEM_INPUT_VARS):
            lo, hi = bounds[var]
            u = cand_raw[:, j]
            if var in CHEM_LOG_VARS and lo > 0:
                vals = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
            else:
                vals = lo + u * (hi - lo)
            cand_X[:, j] = vals

        # EI acquisition in log-space (consistent with GPSurrogate log-y)
        mu_log, sigma_log = gp.predict_log(cand_X)
        y_best_log = float(np.max(np.log(np.maximum(y, 1e-300))))
        sigma_log = np.maximum(sigma_log, 1e-8)
        z = (mu_log - y_best_log) / sigma_log
        ei = (mu_log - y_best_log) * norm.cdf(z) + sigma_log * norm.pdf(z)
        best_idx = int(np.argmax(ei))

        params = {var: float(cand_X[best_idx, j])
                  for j, var in enumerate(CHEM_INPUT_VARS)}
        result = oracle.run(params)
        store.add(result)

        if step % 10 == 0 or step == remaining - 1:
            print(f"  step {step+1:3d}/{remaining}: r0={result.measurement:.4f}  "
                  f"EI={ei[best_idx]:.4f}  "
                  f"C_A={params['C_A']:.2f} C_I={params['C_I']:.2f} "
                  f"C_B={params['C_B']:.2f} C_P={params['C_P']:.2f} T={params['T']:.0f}")

    # --- ARD variable selection + PySR (no mechanistic hints) ---
    # After collecting data, fit a final ARD GP to identify relevant variables.
    # ARD (Automatic Relevance Determination) assigns a per-dimension length-scale:
    # large length-scale → variable explains little variance → likely irrelevant.
    # This is a standard algorithmic technique requiring no chemical knowledge.
    # Then run PySR on the reduced variable set — much easier for symbolic regression.
    #
    # The LLM pipeline still has two structural advantages BO cannot replicate:
    #   (1) Mechanism-guided exploration: LLM knows to vary C_P for product inhibition,
    #       cold T for Arrhenius, etc. — EI never visits these low-rate regions.
    #   (2) Hypothesis generation: LLM proposes the correct skeleton form, enabling
    #       exact constant fitting rather than relying on PySR to rediscover the form.
    X_final, y_final = store.to_arrays()
    gp_ard = GPSurrogate()
    gp_ard.fit(X_final, y_final, bounds)
    bo_relevant = gp_ard.relevant_variables(list(CHEM_INPUT_VARS), threshold=2.0)
    print(f"\n  ARD variable selection: {bo_relevant} (of {list(CHEM_INPUT_VARS)})")
    print(f"  Running PySR on {len(store)} data points with ARD-selected variables...")
    learner = EquationLearner(oracle=oracle, n_iterations=800)
    eq = learner.fit(
        store,
        goal="unknown",
        current_hypothesis="unknown",
        skeleton_families=None,       # BO gets no mechanistic family knowledge
        relevant_vars=bo_relevant,    # ARD-derived variable pruning (domain-agnostic)
    )

    return _evaluate_and_return("bo_baseline", oracle, store, eq, t0,
                                extra={"ard_vars_selected": bo_relevant})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sample_random_points(n: int, bounds: dict, rng: np.random.Generator) -> list[dict]:
    """Sample n points uniformly (log-uniform for log-scale vars) from bounds."""
    points = []
    for _ in range(n):
        p = {}
        for var in CHEM_INPUT_VARS:
            lo, hi = bounds[var]
            u = rng.random()
            if var in CHEM_LOG_VARS and lo > 0:
                p[var] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
            else:
                p[var] = float(lo + u * (hi - lo))
        points.append(p)
    return points


def _run_pysr_on_store(store, oracle, relevant_vars=None):
    """Fit PySR on all data in store, optionally restricting to relevant_vars."""
    from autoscilab.equation_learner.learner import EquationLearner
    learner = EquationLearner(oracle=oracle, n_iterations=800)
    return learner.fit(
        store,
        goal="unknown",
        current_hypothesis="unknown",
        skeleton_families=None,
        relevant_vars=relevant_vars,
    )


def _evaluate_and_return(method_name: str, oracle, store, eq, t0: float, extra: dict | None = None) -> dict:
    """Evaluate equation against ground truth and return standard result dict."""
    eval_result = {"rmsle": float("nan"), "exact_accuracy": 0.0}
    if eq and eq.is_valid():
        eval_result = oracle.evaluate_law(eq.law_str)
        print(f"\n  Best equation: {eq.skeleton_str}")
        print(f"  R²={eq.r2:.4f}  train_RMSLE={eq.rmsle:.4f}")
        print(f"  GT RMSLE={eval_result['rmsle']:.4f}  "
              f"Exact={'YES' if eval_result['exact_accuracy'] > 0 else 'NO'}")
    else:
        print("  PySR failed to find valid equation.")

    all_results = store.get_all()
    result = {
        "method":        method_name,
        "n_experiments": len(store),
        "gt_rmsle":      eval_result.get("rmsle", float("nan")),
        "exact":         eval_result.get("exact_accuracy", 0.0) > 0,
        "equation":      eq.skeleton_str if eq else None,
        "duration_s":    time.time() - t0,
        "C_P_range":     [float(np.min([r.params["C_P"] for r in all_results])),
                          float(np.max([r.params["C_P"] for r in all_results]))],
        "C_I_range":     [float(np.min([r.params["C_I"] for r in all_results])),
                          float(np.max([r.params["C_I"] for r in all_results]))],
        "C_B_range":     [float(np.min([r.params["C_B"] for r in all_results])),
                          float(np.max([r.params["C_B"] for r in all_results]))],
        "T_range":       [float(np.min([r.params["T"] for r in all_results])),
                          float(np.max([r.params["T"] for r in all_results]))],
    }
    if extra:
        result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Baseline: random sampling + PySR  (floor)
# ---------------------------------------------------------------------------

def run_random_baseline(oracle: ChemBenchOracle, budget: int, seed: int = 42) -> dict:
    """
    Uniform random sampling + ARD variable selection + PySR.  No GP, no LLM.
    Establishes the floor: does ANY active learning help over random?
    """
    print("\n" + "="*60)
    print("BASELINE: RANDOM + PySR")
    print("="*60)

    store = ExperimentStore(oracle.domain, CHEM_INPUT_VARS)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    points = _sample_random_points(budget, CHEM_INPUT_BOUNDS, rng)
    for i, p in enumerate(points):
        result = oracle.run(p)
        store.add(result)
        if i % 15 == 0 or i == budget - 1:
            print(f"  {i+1:3d}/{budget}: r0={result.measurement:.4f}")

    # ARD variable selection then PySR
    gp_ard = GPSurrogate()
    gp_ard.fit(*store.to_arrays(), CHEM_INPUT_BOUNDS)
    relevant = gp_ard.relevant_variables(list(CHEM_INPUT_VARS), threshold=2.0)
    print(f"\n  ARD selected vars: {relevant}")
    eq = _run_pysr_on_store(store, oracle, relevant_vars=relevant)

    return _evaluate_and_return("random_baseline", oracle, store, eq, t0,
                                extra={"ard_vars_selected": relevant})


# ---------------------------------------------------------------------------
# Baseline: GP uncertainty sampling + PySR  (exploration-focused BO)
# ---------------------------------------------------------------------------

def run_uncertainty_baseline(oracle: ChemBenchOracle, budget: int, seed: int = 42) -> dict:
    """
    GP with pure-variance (max uncertainty) acquisition + ARD + PySR.
    Tests whether the EI failure is due to rate-seeking or lack of exploration.
    Unlike EI, this will explore CP>0, low T, and extreme pH — but without
    mechanistic reasoning it still fails on regime collapse (c5).
    """
    print("\n" + "="*60)
    print("BASELINE: GP UNCERTAINTY SAMPLING + PySR")
    print("="*60)

    store = ExperimentStore(oracle.domain, CHEM_INPUT_VARS)
    gp = GPSurrogate()
    rng = np.random.default_rng(seed)
    bounds = CHEM_INPUT_BOUNDS
    t0 = time.time()

    # 10 LHS init points
    n_init = min(10, budget)
    from scipy.stats.qmc import LatinHypercube
    lhs_pts = LatinHypercube(d=len(CHEM_INPUT_VARS), seed=seed).random(n=n_init)
    for i in range(n_init):
        p = {}
        for j, var in enumerate(CHEM_INPUT_VARS):
            lo, hi = bounds[var]
            u = lhs_pts[i, j]
            p[var] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
                           if var in CHEM_LOG_VARS and lo > 0
                           else lo + u * (hi - lo))
        result = oracle.run(p)
        store.add(result)
        print(f"  init {i+1:2d}/{n_init}: r0={result.measurement:.4f}")

    # GP variance acquisition loop
    for step in range(budget - n_init):
        X, y = store.to_arrays()
        gp.fit(X, y, bounds)

        # Sample candidates, pick max GP variance (pure exploration)
        n_cand = 2000
        cand_X = np.zeros((n_cand, len(CHEM_INPUT_VARS)))
        raw = rng.random((n_cand, len(CHEM_INPUT_VARS)))
        for j, var in enumerate(CHEM_INPUT_VARS):
            lo, hi = bounds[var]
            cand_X[:, j] = (np.exp(np.log(lo) + raw[:, j] * (np.log(hi) - np.log(lo)))
                            if var in CHEM_LOG_VARS and lo > 0
                            else lo + raw[:, j] * (hi - lo))

        _, sigma_log = gp.predict_log(cand_X)
        best_idx = int(np.argmax(sigma_log))

        params = {var: float(cand_X[best_idx, j]) for j, var in enumerate(CHEM_INPUT_VARS)}
        result = oracle.run(params)
        store.add(result)

        if step % 10 == 0 or step == budget - n_init - 1:
            print(f"  step {step+1:3d}/{budget-n_init}: r0={result.measurement:.4f}  "
                  f"σ={sigma_log[best_idx]:.4f}  "
                  f"C_P={params['C_P']:.2f}  T={params['T']:.0f}")

    # ARD + PySR
    gp_ard = GPSurrogate()
    gp_ard.fit(*store.to_arrays(), bounds)
    relevant = gp_ard.relevant_variables(list(CHEM_INPUT_VARS), threshold=2.0)
    print(f"\n  ARD selected vars: {relevant}")
    eq = _run_pysr_on_store(store, oracle, relevant_vars=relevant)

    return _evaluate_and_return("uncertainty_baseline", oracle, store, eq, t0,
                                extra={"ard_vars_selected": relevant})


# ---------------------------------------------------------------------------
# Baseline: oracle variable selection + PySR
# ---------------------------------------------------------------------------

def run_oracle_vars_baseline(oracle: ChemBenchOracle, budget: int, seed: int = 42) -> dict:
    """
    Random sampling + PySR restricted to ground-truth relevant variables.
    Tests whether BO failure is about variable selection or experiment design.
    Uses CHEM_DOMAIN_REGISTRY['relevant_vars'] — not available to BO or LLM.
    """
    print("\n" + "="*60)
    print("BASELINE: ORACLE VARIABLE SELECTION + PySR")
    print("="*60)

    store = ExperimentStore(oracle.domain, CHEM_INPUT_VARS)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Get ground-truth relevant variables
    relevant = CHEM_DOMAIN_REGISTRY.get(oracle.domain, {}).get("relevant_vars", list(CHEM_INPUT_VARS))
    print(f"  Oracle relevant vars: {relevant}")

    points = _sample_random_points(budget, CHEM_INPUT_BOUNDS, rng)
    for i, p in enumerate(points):
        result = oracle.run(p)
        store.add(result)
        if i % 15 == 0 or i == budget - 1:
            print(f"  {i+1:3d}/{budget}: r0={result.measurement:.4f}")

    print(f"\n  Running PySR on oracle-selected vars: {relevant}")
    eq = _run_pysr_on_store(store, oracle, relevant_vars=relevant)

    return _evaluate_and_return("oracle_vars_baseline", oracle, store, eq, t0,
                                extra={"oracle_vars": relevant})


# ---------------------------------------------------------------------------
# Baseline: BED (Bayesian Experimental Design) with hypothesis library
# ---------------------------------------------------------------------------

def run_bed_baseline(oracle: ChemBenchOracle, budget: int, seed: int = 42) -> dict:
    """
    Bayesian Experimental Design over a pre-specified library of all 10 mechanism families.

    At each step:
      1. Fit each candidate rate law to current data (scipy curve_fit, log-space objective).
      2. Compute each model's predictions at 2000 candidate experiment points.
      3. Pick the experiment where the candidate models MOST DISAGREE (max variance
         across model predictions) — this maximally discriminates between models.
      4. Run the experiment, update data, repeat.

    After budget exhausted:
      - Pick the model with lowest residual on all data.
      - Run PySR initialised from that model's skeleton to refine constants.

    The hypothesis library is pre-specified here as symbolic functions — the same
    mechanistic knowledge the LLM generates dynamically.  This directly answers
    the question: if you give an algorithm the same hypothesis space, does it
    match the LLM?
    """
    from scipy.optimize import curve_fit
    from autoscilab.equation_learner.learner import EquationLearner

    print("\n" + "="*60)
    print("BASELINE: BED + HYPOTHESIS LIBRARY")
    print("="*60)

    # ---- Hypothesis library: all 10 ChemBench mechanism families ----
    # Each entry: (name, fn(X_row, *params) -> r0, param_names, p0_fn)
    # X_row = [C_A, C_I, C_B, C_P, Enz, T, pH]  (indexed 0-6)
    R_GAS = 8.314
    T_REF = 310.0

    def _idx(v):
        return CHEM_INPUT_VARS.index(v)

    iCA = _idx("C_A"); iCI = _idx("C_I"); iCB = _idx("C_B")
    iCP = _idx("C_P"); iE  = _idx("Enz"); iT  = _idx("T"); ipH = _idx("pH")

    HYPOTHESIS_LIBRARY = [
        # (name, callable, param_names, initial_guess)
        ("MM",
         lambda X, kcat, Km:
             kcat * X[:, iE] * X[:, iCA] / (Km + X[:, iCA]),
         ["kcat", "Km"], [5.0, 1.0]),

        ("Competitive",
         lambda X, kcat, Km, Ki:
             kcat * X[:, iE] * X[:, iCA] / (Km * (1 + X[:, iCI]/Ki) + X[:, iCA]),
         ["kcat", "Km", "Ki"], [5.0, 1.0, 1.0]),

        ("Product",
         lambda X, kcat, Km, Kp:
             kcat * X[:, iE] * X[:, iCA] / (Km * (1 + X[:, iCP]/Kp) + X[:, iCA]),
         ["kcat", "Km", "Kp"], [5.0, 1.0, 5.0]),

        ("Arrhenius",
         lambda X, kcat_ref, Ea, Km:
             kcat_ref * np.exp(-Ea/R_GAS * (1/X[:, iT] - 1/T_REF))
             * X[:, iE] * X[:, iCA] / (Km + X[:, iCA]),
         ["kcat_ref", "Ea", "Km"], [5.0, 50000.0, 1.0]),

        ("pH_bell",
         lambda X, kcat, Km, pKa1, pKa2:
             kcat * X[:, iE] * X[:, iCA] /
             ((Km + X[:, iCA]) * (1 + 10**(pKa1 - X[:, ipH]) + 10**(X[:, ipH] - pKa2))),
         ["kcat", "Km", "pKa1", "pKa2"], [10.0, 1.0, 5.5, 8.5]),

        ("Pingpong",
         lambda X, kcat, KmA, KmB:
             kcat * X[:, iE] * X[:, iCA] * X[:, iCB] /
             (KmA*X[:, iCB] + KmB*X[:, iCA] + X[:, iCA]*X[:, iCB]),
         ["kcat", "KmA", "KmB"], [5.0, 1.0, 2.0]),

        ("Uncompetitive",
         lambda X, kcat, Km, Ki:
             kcat * X[:, iE] * X[:, iCA] / (Km + X[:, iCA] * (1 + X[:, iCI]/Ki)),
         ["kcat", "Km", "Ki"], [5.0, 1.0, 1.0]),

        ("SubstrateInhibition",
         lambda X, kcat, Km, Ki_s:
             kcat * X[:, iE] * X[:, iCA] / (Km + X[:, iCA] + X[:, iCA]**2/Ki_s),
         ["kcat", "Km", "Ki_s"], [8.0, 0.5, 50.0]),

        ("Hill",
         lambda X, kcat, K_half, n:
             kcat * X[:, iE] * X[:, iCA]**n / (K_half**n + X[:, iCA]**n),
         ["kcat", "K_half", "n"], [5.0, 2.0, 2.0]),

        ("Noncompetitive",
         lambda X, kcat, Km, Ki:
             kcat * X[:, iE] * X[:, iCA] / ((1 + X[:, iCI]/Ki) * (Km + X[:, iCA])),
         ["kcat", "Km", "Ki"], [5.0, 1.0, 1.0]),
    ]

    def _fit_model(fn, p0, X, log_y):
        """Fit model to log-space data. Returns (params, residual_rmsle)."""
        def objective(X_, *params):
            pred = fn(X_, *params)
            pred = np.maximum(pred, 1e-300)
            return np.log(pred)
        try:
            popt, _ = curve_fit(objective, X, log_y, p0=p0,
                                maxfev=5000, bounds=(0, np.inf))
            pred = fn(X, *popt)
            pred = np.maximum(pred, 1e-300)
            rmsle = float(np.sqrt(np.mean((np.log(pred) - log_y)**2)))
            return popt, rmsle
        except Exception:
            return p0, float("inf")

    store = ExperimentStore(oracle.domain, CHEM_INPUT_VARS)
    rng = np.random.default_rng(seed)
    bounds = CHEM_INPUT_BOUNDS
    t0 = time.time()

    # 10 LHS init points
    n_init = min(10, budget)
    from scipy.stats.qmc import LatinHypercube
    lhs_pts = LatinHypercube(d=len(CHEM_INPUT_VARS), seed=seed).random(n=n_init)
    for i in range(n_init):
        p = {}
        for j, var in enumerate(CHEM_INPUT_VARS):
            lo, hi = bounds[var]
            u = lhs_pts[i, j]
            p[var] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
                           if var in CHEM_LOG_VARS and lo > 0 else lo + u * (hi - lo))
        result = oracle.run(p)
        store.add(result)
        print(f"  init {i+1:2d}/{n_init}: r0={result.measurement:.4f}")

    model_params = {name: p0[:] for name, _, _, p0 in HYPOTHESIS_LIBRARY}

    for step in range(budget - n_init):
        X, y = store.to_arrays()
        log_y = np.log(np.maximum(y, 1e-300))

        # Fit all models to current data
        fitted = {}
        for name, fn, pnames, p0 in HYPOTHESIS_LIBRARY:
            popt, rmsle = _fit_model(fn, model_params[name], X, log_y)
            model_params[name] = list(popt)
            fitted[name] = (fn, popt, rmsle)

        # Generate 2000 candidate experiments
        n_cand = 2000
        cand_X = np.zeros((n_cand, len(CHEM_INPUT_VARS)))
        raw = rng.random((n_cand, len(CHEM_INPUT_VARS)))
        for j, var in enumerate(CHEM_INPUT_VARS):
            lo, hi = bounds[var]
            cand_X[:, j] = (np.exp(np.log(lo) + raw[:, j] * (np.log(hi) - np.log(lo)))
                            if var in CHEM_LOG_VARS and lo > 0
                            else lo + raw[:, j] * (hi - lo))

        # Score each candidate: variance of log-predictions across all models
        all_preds = []
        for name, (fn, popt, rmsle) in fitted.items():
            pred = fn(cand_X, *popt)
            pred = np.maximum(pred, 1e-300)
            all_preds.append(np.log(pred))
        pred_matrix = np.stack(all_preds, axis=0)   # (n_models, n_cand)
        disagreement = np.var(pred_matrix, axis=0)   # (n_cand,)
        best_idx = int(np.argmax(disagreement))

        params = {var: float(cand_X[best_idx, j]) for j, var in enumerate(CHEM_INPUT_VARS)}
        result = oracle.run(params)
        store.add(result)

        # Best model so far
        best_model = min(fitted.items(), key=lambda kv: kv[1][2])
        if step % 10 == 0 or step == budget - n_init - 1:
            print(f"  step {step+1:3d}/{budget-n_init}: r0={result.measurement:.4f}  "
                  f"disagree={disagreement[best_idx]:.4f}  "
                  f"best_model={best_model[0]} (rmsle={best_model[1][2]:.4f})  "
                  f"C_P={params['C_P']:.2f}  T={params['T']:.0f}")

    # Final: pick winning model, use its skeleton to seed PySR
    X, y = store.to_arrays()
    log_y = np.log(np.maximum(y, 1e-300))
    final_fits = {}
    for name, fn, pnames, p0 in HYPOTHESIS_LIBRARY:
        popt, rmsle = _fit_model(fn, model_params[name], X, log_y)
        final_fits[name] = (fn, popt, rmsle)
        print(f"  {name:20s}: train_rmsle={rmsle:.4f}")

    winner_name, (winner_fn, winner_popt, winner_rmsle) = min(
        final_fits.items(), key=lambda kv: kv[1][2]
    )
    print(f"\n  BED selected model: {winner_name} (rmsle={winner_rmsle:.4f})")

    # Map winning model → relevant variables for PySR (avoids ARD missing key vars)
    _BED_VAR_MAP = {
        "MM":                 ["C_A", "Enz"],
        "Competitive":        ["C_A", "C_I", "Enz"],
        "Product":            ["C_A", "C_P", "Enz"],
        "Arrhenius":          ["C_A", "T", "Enz"],
        "pH_bell":            ["C_A", "pH", "Enz"],
        "Pingpong":           ["C_A", "C_B", "Enz"],
        "Uncompetitive":      ["C_A", "C_I", "Enz"],
        "SubstrateInhibition":["C_A", "Enz"],
        "Hill":               ["C_A", "Enz"],
        "Noncompetitive":     ["C_A", "C_I", "Enz"],
    }
    relevant = _BED_VAR_MAP.get(winner_name, list(CHEM_INPUT_VARS))
    eq = _run_pysr_on_store(store, oracle, relevant_vars=relevant)

    return _evaluate_and_return("bed_baseline", oracle, store, eq, t0,
                                extra={"bed_winner": winner_name,
                                       "bed_winner_rmsle": winner_rmsle,
                                       "bed_vars": relevant})


# ---------------------------------------------------------------------------
# LLM pipeline
# ---------------------------------------------------------------------------

def run_llm_pipeline(
    domain: str, difficulty: str, law_version: str, budget: int,
    model: str = "gpt-4o-mini",
    noise: float = 0.01,
    use_domain_tags: bool = True,
    hypothesis_grammar_source: str = "domain_specific",
    strong_model: str | None = None,
    strong_model_calls: int = 3,
    ensemble_mode: bool = False,
    ensemble_k: int = 5,
    ensemble_model: str = "Qwen/Qwen2.5-7B-Instruct",
    ensemble_url: str = "http://localhost:8001/v1",
    ensemble_every: int = 3,
    main_url: str | None = None,
    max_completion_tokens: int = 8192,
    ensemble_adaptive: bool = False,
    ensemble_k_max: int = 20,
    ensemble_stability_threshold: float = 0.1,
    ensemble_confidence_gate: float = 1.0,
    confidence_threshold: float = 0.9,
    skeleton_priority_threshold: float | None = None,
    prune_spurious_vars: bool = False,
    ood_holdout_frac: float = 0.0,
    forward_discrimination: bool = False,
    results_dir: Path | None = None,
    max_experiments_per_iter: int = 10,
) -> dict:
    """Run the full LLM+AL+SR DiscoveryLoop."""
    from autoscilab.loop.discovery import DiscoveryConfig, DiscoveryLoop

    CHEM_GOALS = {
        "c0_michaelis_menten":         "Discover the enzyme kinetic rate law (Michaelis-Menten).",
        "c1_competitive_inhibition":   "Discover the rate law including competitive inhibition by I.",
        "c2_product_inhibition":       "Discover the rate law including product feedback inhibition.",
        "c3_arrhenius_temperature":    "Discover the rate law including Arrhenius temperature dependence.",
        "c4_ph_activity":              "Discover the rate law including pH-dependent activity.",
        "c5_pingpong_bisubstrate":     "Discover the rate law for a two-substrate pingpong mechanism.",
        "c6_uncompetitive_inhibition": "Discover the rate law including uncompetitive inhibition.",
        "c7_substrate_inhibition":     "Discover the rate law with substrate self-inhibition.",
        "c8_hill_cooperativity":       "Discover the cooperative Hill-type rate law.",
        "c9_noncompetitive_inhibition":"Discover the rate law with noncompetitive inhibition.",
    }

    print("\n" + "="*60)
    print(f"LLM PIPELINE (main={model}"
          + (f"  strong={strong_model}×{strong_model_calls}" if strong_model else "")
          + ")")
    print("="*60)

    # If a custom OpenAI-compatible URL is provided, route credentials based on
    # the endpoint instead of assuming a local unauthenticated server.
    if main_url:
        if "deepinfra.com" in main_url.lower():
            api_key = os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("OPENAI_API_KEY")
            key_name = "DEEPINFRA_API_KEY"
            if not api_key:
                print(f"  ERROR: No {key_name} found in environment.")
                return {"method": "llm_pipeline", "error": "no_api_key"}
        else:
            api_key = "local"
    else:
        # Route the API key based on model type.
        # LLMClient already handles routing internally (OPENAI_API_KEY for gpt-*,
        # TOGETHER_API_KEY for Together-hosted models), so pass api_key=None and let
        # LLMClient read the correct env var.  Passing OPENAI_API_KEY to a Together
        # model causes a 401.
        use_openai = "gpt-" in model or model.startswith("o1") or model.startswith("o3")
        if use_openai:
            api_key = os.environ.get("OPENAI_API_KEY")
            key_name = "OPENAI_API_KEY"
        else:
            api_key = os.environ.get("TOGETHER_API_KEY")
            key_name = "TOGETHER_API_KEY"

        if not api_key:
            print(f"  ERROR: No {key_name} found in environment.")
            return {"method": "llm_pipeline", "error": "no_api_key"}

    if not use_domain_tags:
        print("  [ABLATION] use_domain_tags=False — LLM must generate all hypotheses from data")
    if hypothesis_grammar_source != "domain_specific":
        print(f"  [ABLATION] grammar_mode={hypothesis_grammar_source!r}")

    config = DiscoveryConfig(
        domain_id=domain,
        goal=CHEM_GOALS.get(domain, f"Discover the governing rate law for {domain}."),
        budget=budget,
        noise_level=noise,
        difficulty=difficulty,
        law_version=law_version,
        acquisition="ei",
        eq_learner_every=20,
        pysr_n_iterations=800,
        max_experiments_per_iter=max_experiments_per_iter,
        llm_model=model,
        results_dir=results_dir or Path("results/chembench_comparison"),
        policy="families_v3",
        use_domain_tags=use_domain_tags,
        hypothesis_grammar_source=hypothesis_grammar_source,
        strong_model=strong_model,
        strong_model_calls=strong_model_calls,
        ensemble_mode=ensemble_mode,
        ensemble_k=ensemble_k,
        ensemble_model=ensemble_model,
        ensemble_url=ensemble_url,
        ensemble_every_n_iters=ensemble_every,
        max_completion_tokens=max_completion_tokens,
        ensemble_adaptive=ensemble_adaptive,
        ensemble_k_max=ensemble_k_max,
        ensemble_stability_threshold=ensemble_stability_threshold,
        ensemble_confidence_gate=ensemble_confidence_gate,
        confidence_threshold=confidence_threshold,
        skeleton_priority_threshold=skeleton_priority_threshold,
        prune_spurious_vars=prune_spurious_vars,
        ood_holdout_frac=ood_holdout_frac,
        forward_discrimination=forward_discrimination,
    )

    if strong_model:
        if not os.environ.get("OPENAI_API_KEY"):
            print("  WARNING: --strong-model set but OPENAI_API_KEY not found — consultant disabled")

    t0 = time.time()
    loop = DiscoveryLoop(config, api_key=api_key, base_url=main_url)
    result = loop.run()
    duration = time.time() - t0

    eval_result = result.final_evaluation or {}
    eq = result.best_equation

    if eq:
        print(f"\n  Best equation: {eq.skeleton_str}")
        print(f"  R²={eq.r2:.4f}  train_RMSLE={eq.rmsle:.4f}")
        print(f"  GT RMSLE={eval_result.get('rmsle','N/A')}  "
              f"Exact={'YES' if eval_result.get('exact_accuracy',0) > 0 else 'NO'}")
        print(f"  LLM calls: {result.n_llm_calls}  Oracle calls: {result.n_oracle_calls}")

    consultant_calls = 0
    consultant_model_used = None
    if loop._consultant is not None:
        consultant_calls = loop._consultant.calls_used
        consultant_model_used = strong_model

    return {
        "method":            "llm_pipeline",
        "n_experiments":     result.n_oracle_calls,
        "n_llm_calls":       result.n_llm_calls,
        "gt_rmsle":          eval_result.get("rmsle", float("nan")),
        "exact":             eval_result.get("exact_accuracy", 0.0) > 0,
        "equation":          eq.skeleton_str if eq else None,
        "duration_s":        duration,
        "termination":       result.termination_reason,
        "consultant_calls":  consultant_calls,
        "consultant_model":  consultant_model_used,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BASELINE_RUNNERS = {
    "bo":          run_bo_baseline,
    "random":      run_random_baseline,
    "uncertainty": run_uncertainty_baseline,
    "oracle_vars": run_oracle_vars_baseline,
    "bed":         run_bed_baseline,
}

ALL_BASELINES = list(BASELINE_RUNNERS.keys())


def main():
    parser = argparse.ArgumentParser(description="ChemBench: multiple baselines vs LLM")
    parser.add_argument("--domain",      default="c2_product_inhibition")
    parser.add_argument("--budget",      type=int, default=60)
    parser.add_argument("--difficulty",  default="easy")
    parser.add_argument("--law-version", default="v0")
    parser.add_argument("--noise",       type=float, default=0.01)
    parser.add_argument("--model",       default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        help="(deprecated alias for --main-model)")
    parser.add_argument("--main-model",  default=None,
                        help="Main (cheap) LLM for experiment proposals. "
                             "Default: gpt-4o-mini")
    parser.add_argument("--main-url", default=None,
                        help="Custom base_url for main model (e.g. http://localhost:8003/v1 "
                             "for local transformers server). Skips Together/OpenAI routing.")
    parser.add_argument("--result-key", default="llm",
                        help="Key to store LLM results under in the output JSON (default: llm)")
    parser.add_argument("--strong-model", default=None,
                        help="Strong (expensive) LLM consultant, called ≤3× per run. "
                             "Set to 'gpt-4o' to enable. Default: disabled (None).")
    parser.add_argument("--strong-model-calls", type=int, default=3,
                        help="Max times the strong consultant can be queried per run "
                             "(default: 3)")
    # Baseline selection
    parser.add_argument("--baselines",   nargs="+", choices=ALL_BASELINES + ["all", "none"],
                        default=["bo"],
                        help="Which algorithmic baselines to run. "
                             "Choices: bo random uncertainty oracle_vars bed all none")
    parser.add_argument("--bo-only",     action="store_true",
                        help="Shorthand: run only bo baseline, skip LLM")
    parser.add_argument("--llm-only",    action="store_true",
                        help="Shorthand: skip all baselines, run only LLM")
    # LLM flags
    parser.add_argument("--no-domain-tags", action="store_true",
                        help="Ablation: remove oracle domain_tags from LLM pipeline")
    parser.add_argument("--grammar",
                        choices=["domain_specific", "universal", "none"],
                        default="domain_specific",
                        help="Hypothesis grammar: domain_specific | universal | none")
    parser.add_argument("--no-llm",      action="store_true",
                        help="Skip LLM pipeline entirely (run baselines only)")
    parser.add_argument("--ensemble",    action="store_true",
                        help="Enable MEI ensemble mode (requires vLLM server)")
    parser.add_argument("--ensemble-k",  type=int, default=5,
                        help="Number of small-LLM samples per MEI round")
    parser.add_argument("--ensemble-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Small model name for ensemble sampling")
    parser.add_argument("--ensemble-url", default="http://localhost:8001/v1",
                        help="vLLM server URL")
    parser.add_argument("--ensemble-every", type=int, default=3,
                        help="Run MEI round every N outer iterations")
    parser.add_argument("--ensemble-adaptive", action="store_true",
                        help="v3: sample in batches until entropy stabilises")
    parser.add_argument("--ensemble-k-max", type=int, default=20,
                        help="v3: max samples for adaptive mode (default 20)")
    parser.add_argument("--ensemble-stability-threshold", type=float, default=0.1,
                        help="v3: entropy delta threshold to stop sampling (default 0.1 bits)")
    parser.add_argument("--ensemble-confidence-gate", type=float, default=1.0,
                        help="v4: skip MEI when bootstrap conf >= this value (default 1.0 = off)")
    parser.add_argument("--confidence-threshold", type=float, default=0.9,
                        help="Bootstrap confidence required for early termination (default 0.9; set >1 to disable)")
    parser.add_argument("--max-completion-tokens", type=int, default=8192,
                        help="Max completion tokens for local models (default 8192)")
    # ── Symbolic accuracy improvements ────────────────────────────────────────
    parser.add_argument("--skeleton-priority-threshold", type=float, default=None,
                        help="If skeleton RMSLE ≤ this, skip PySR (e.g. 0.10). Default: disabled.")
    parser.add_argument("--prune-spurious-vars", action="store_true",
                        help="After final selection, refit without vars with permutation importance <0.01.")
    parser.add_argument("--ood-holdout-frac", type=float, default=0.0,
                        help="Fraction of budget for OOD boundary holdout (e.g. 0.10). Default: 0 (off).")
    parser.add_argument("--forward-discrimination", action="store_true",
                        help="Fix 4: inject per-hypothesis prediction table at max-disagreement point.")
    args = parser.parse_args()

    domain      = args.domain
    difficulty  = args.difficulty
    law_version = args.law_version
    budget      = args.budget

    # --main-model takes priority; fall back to legacy --model
    main_model = args.main_model or args.model
    strong_model = args.strong_model          # None = disabled
    strong_model_calls = args.strong_model_calls

    # Resolve which baselines to run
    if args.llm_only:
        baselines_to_run = []
    elif args.bo_only:
        baselines_to_run = ["bo"]
    elif "all" in args.baselines:
        baselines_to_run = ALL_BASELINES
    elif "none" in args.baselines:
        baselines_to_run = []
    else:
        baselines_to_run = args.baselines

    run_llm = not args.bo_only and not args.no_llm

    print(f"\nChemBench Comparison")
    print(f"  Domain:      {domain}")
    print(f"  Difficulty:  {difficulty}")
    print(f"  Law version: {law_version}")
    print(f"  Budget:      {budget} oracle calls")
    print(f"  Baselines:   {baselines_to_run or 'none'}")
    print(f"  LLM:         {'yes (main=' + main_model + ')' if run_llm else 'no'}")
    if run_llm and strong_model:
        print(f"  Consultant:  {strong_model} (max {strong_model_calls} calls/run)")

    oracle = ChemBenchOracle(domain, difficulty, args.noise, law_version)
    gt_law = oracle.get_ground_truth_law_str()
    print(f"\nGround truth:\n{gt_law}\n")

    results = {}

    for bl in baselines_to_run:
        results[bl] = BASELINE_RUNNERS[bl](oracle, budget)

    if run_llm:
        result_key = getattr(args, "result_key", "llm")
        results[result_key] = run_llm_pipeline(
            domain, difficulty, law_version, budget, model=main_model,
            noise=args.noise,
            use_domain_tags=not args.no_domain_tags,
            hypothesis_grammar_source=args.grammar,
            strong_model=strong_model,
            strong_model_calls=strong_model_calls,
            ensemble_mode=args.ensemble,
            ensemble_k=args.ensemble_k,
            ensemble_model=args.ensemble_model,
            ensemble_url=args.ensemble_url,
            ensemble_every=args.ensemble_every,
            main_url=getattr(args, "main_url", None),
            max_completion_tokens=args.max_completion_tokens,
            ensemble_adaptive=args.ensemble_adaptive,
            ensemble_k_max=args.ensemble_k_max,
            ensemble_stability_threshold=args.ensemble_stability_threshold,
            ensemble_confidence_gate=args.ensemble_confidence_gate,
            confidence_threshold=args.confidence_threshold,
            skeleton_priority_threshold=getattr(args, "skeleton_priority_threshold", None),
            prune_spurious_vars=getattr(args, "prune_spurious_vars", False),
            ood_holdout_frac=getattr(args, "ood_holdout_frac", 0.0),
            forward_discrimination=getattr(args, "forward_discrimination", False),
        )

    # --- Summary ---
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for method, r in results.items():
        exact_str = "EXACT ✓" if r.get("exact") else "NOT EXACT ✗"
        rmsle_val = r.get("gt_rmsle", float("nan"))
        rmsle_str = f"{rmsle_val:.4f}" if rmsle_val == rmsle_val else "nan"
        print(f"\n  [{method.upper():16s}]")
        print(f"    GT RMSLE:   {rmsle_str}")
        print(f"    Result:     {exact_str}")
        print(f"    Equation:   {r.get('equation', 'N/A')}")
        print(f"    Duration:   {r.get('duration_s', 0):.0f}s")
        if "n_llm_calls" in r:
            print(f"    LLM calls:  {r['n_llm_calls']}")
        if "consultant_calls" in r:
            print(f"    Consultant: {r['consultant_calls']} calls ({r.get('consultant_model','?')})")
        if "ard_vars_selected" in r:
            print(f"    ARD vars:   {r['ard_vars_selected']}")
        if "bed_winner" in r:
            print(f"    BED winner: {r['bed_winner']} (rmsle={r.get('bed_winner_rmsle','?'):.4f})")
        if "oracle_vars" in r:
            print(f"    Oracle vars:{r['oracle_vars']}")
        if "C_P_range" in r:
            print(f"    C_P range:  {r['C_P_range']}")
            print(f"    C_I range:  {r['C_I_range']}")
            print(f"    C_B range:  {r['C_B_range']}")

    # Save — merge with existing results so multiple policies accumulate
    out_dir = Path("results/chembench_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{domain}_{difficulty}_{law_version}_comparison.json"
    existing = {}
    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text()).get("results", {})
        except Exception:
            pass
    existing.update(results)  # new results overwrite same-key entries
    with open(out_file, "w") as f:
        json.dump({
            "domain": domain, "difficulty": difficulty,
            "law_version": law_version, "budget": budget,
            "ground_truth": gt_law, "results": existing,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")


if __name__ == "__main__":
    main()
