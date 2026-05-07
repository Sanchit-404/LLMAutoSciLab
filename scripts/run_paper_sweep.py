#!/usr/bin/env python3
"""
Paper experiment sweep: single (model, budget) configuration.

Runs all 12 NewtonBench domains × 3 difficulties × 3 law versions × 2 seeds = 216 runs.
Fully checkpointed — safe to kill and resume.

Usage:
    python scripts/run_paper_sweep.py --model gpt-4o-mini --budget 20
    python scripts/run_paper_sweep.py --model meta-llama/Llama-3.3-8B-Instruct-Turbo --budget 60
    python scripts/run_paper_sweep.py --model Qwen/Qwen3-30B-A3B-Instruct --budget 100 --workers 32

    # Override via env vars:
    AUTOSCILAB_WORKERS=16 python scripts/run_paper_sweep.py --model gpt-4o-mini --budget 20
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Cap all threading libraries to 1 thread per worker so that N parallel
# workers each use ~1 CPU core instead of all available cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoscilab.loop.discovery import DiscoveryConfig, DiscoveryLoop
from autoscilab.oracle.newtonbench import NewtonBenchOracle

# ── Domains ───────────────────────────────────────────────────────────────────

ALL_DOMAINS = [
    "m0_gravity",
    "m1_coulomb_force",
    "m2_magnetic_force",
    "m3_fourier_law",
    "m4_snell_law",
    "m5_radioactive_decay",
    "m6_underdamped_harmonic",
    "m7_malus_law",
    "m8_sound_speed",
    "m9_hooke_law",
    "m10_be_distribution",
    "m11_heat_transfer",
]

DIFFICULTIES  = ["easy", "medium", "hard"]
LAW_VERSIONS  = ["v0", "v1", "v2"]
SEEDS         = [0, 1]

# ── Model routing ─────────────────────────────────────────────────────────────

def is_openai_model(model: str) -> bool:
    return model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3")

def get_api_key(model: str, base_url: str | None = None) -> str:
    if base_url and "deepinfra.com" in base_url.lower():
        key = os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("DEEPINFRA_API_KEY not set for DeepInfra model")
        return key
    if is_openai_model(model):
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI model")
        return key
    else:
        key = os.environ.get("TOGETHER_API_KEY", "")
        if not key:
            raise ValueError("TOGETHER_API_KEY not set for Together model")
        return key

def short_model_name(model: str) -> str:
    """Compact name for directory and display."""
    mapping = {
        "gpt-4o-mini": "gpt4o-mini",
        "gpt-4o": "gpt4o",
        "meta-llama/Llama-3.3-8B-Instruct-Turbo": "llama3.3-8b",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": "llama3.3-70b",
        "Qwen/Qwen3-30B-A3B-Instruct": "qwen3-30b",
        "Qwen/Qwen3-30B-A3B-Instruct-Turbo": "qwen3-30b",
    }
    return mapping.get(model, model.split("/")[-1].lower().replace(" ", "-"))

# ── Worker function (runs in subprocess) ─────────────────────────────────────

def run_one(
    domain: str,
    difficulty: str,
    version: str,
    seed: int,
    model: str,
    budget: int,
    max_per_iter: int,
    eq_every: int,
    noise: float,
    pysr_n_iterations: int,
    out_dir: Path,
    main_url: str | None = None,
) -> dict:
    # Cap threading BEFORE any imports — env vars set in the parent process do
    # NOT propagate through ProcessPoolExecutor on macOS (spawn start method).
    import os as _os
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_v] = "1"

    import random
    import numpy as np
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    rng_seed = seed * 1000 + hash(domain + difficulty + version) % 10000
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    try:
        oracle = NewtonBenchOracle(
            domain,
            difficulty=difficulty,
            law_version=version,
            noise_level=noise,
        )
        domain_label = domain.split("_", 1)[1].replace("_", " ") if "_" in domain else domain
        goal = f"Discover the governing law for {domain_label} ({version}, {difficulty})."

        cfg = DiscoveryConfig(
            domain_id=domain,
            goal=goal,
            budget=budget,
            noise_level=noise,
            difficulty=difficulty,
            law_version=version,
            max_experiments_per_iter=max_per_iter,
            eq_learner_every=eq_every,
            pysr_n_iterations=pysr_n_iterations,
            llm_model=model,
            policy="families_v3",
            analysis_feedback_enabled=True,
            negative_evidence_enabled=True,
            stratified_hypothesis_scoring=True,
            promotion_full_store_gate=True,
            strict_generator_discriminator_loop=True,
            mid_run_gt_eval=True,
            results_dir=out_dir,
        )

        api_key = get_api_key(model, main_url)
        loop = DiscoveryLoop(cfg, api_key=api_key, base_url=main_url, oracle=oracle)
        result = loop.run()

        return {
            "domain": domain,
            "difficulty": difficulty,
            "version": version,
            "seed": seed,
            "model": model,
            "budget": budget,
            "max_per_iter": max_per_iter,
            "status": result.status,
            "gt_rmsle": result.final_evaluation.get("rmsle") if result.final_evaluation else None,
            "exact_accuracy": result.final_evaluation.get("exact_accuracy") if result.final_evaluation else None,
            "oracle_calls": result.n_oracle_calls,
            "llm_calls": result.n_llm_calls,
            "duration_s": result.duration_seconds,
            # best_equation: human-readable skeleton string (for display)
            # law_str: executable Python function (for symbolic eval)
            "best_equation": result.best_equation,
            "law_str": result.best_equation.law_str if result.best_equation else None,
            "error": None,
        }
    except Exception:
        return {
            "domain": domain,
            "difficulty": difficulty,
            "version": version,
            "seed": seed,
            "model": model,
            "budget": budget,
            "max_per_iter": max_per_iter,
            "status": "error",
            "gt_rmsle": None,
            "exact_accuracy": None,
            "oracle_calls": 0,
            "llm_calls": 0,
            "duration_s": 0,
            "best_equation": None,
            "error": traceback.format_exc(),
        }

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_file: Path):
    if not checkpoint_file.exists():
        return set(), []
    try:
        data = json.loads(checkpoint_file.read_text())
        done = {
            (r["domain"], r["difficulty"], r["version"], r["seed"])
            for r in data
            if r.get("status") not in (None, "error")
        }
        return done, data
    except Exception:
        return set(), []

def save_checkpoint(results: list, checkpoint_file: Path):
    checkpoint_file.write_text(json.dumps(results, indent=2, default=str))

# ── Aggregate stats ───────────────────────────────────────────────────────────

def print_stats(results: list, domains: list):
    if not results:
        return
    n = len(results)
    n_success = sum(1 for r in results if r["status"] == "success")
    n_good    = sum(1 for r in results if r["status"] in ("success", "good"))
    n_error   = sum(1 for r in results if r["status"] == "error")
    rmsles    = [r["gt_rmsle"] for r in results if r.get("gt_rmsle") is not None]
    mean_rmsle = sum(rmsles) / len(rmsles) if rmsles else float("nan")

    print(f"\n{'─'*70}")
    print(f"  Runs: {n} | ✅ success={n_success} ({100*n_success/n:.0f}%) | "
          f"🟡 good+={n_good} | ❌ error={n_error} | mean_RMSLE={mean_rmsle:.4f}")

    # Per-domain
    for domain in domains:
        dr = [r for r in results if r["domain"] == domain]
        if not dr:
            continue
        ds  = sum(1 for r in dr if r["status"] == "success")
        dg  = sum(1 for r in dr if r["status"] in ("success", "good"))
        drl = [r["gt_rmsle"] for r in dr if r.get("gt_rmsle") is not None]
        rstr = f"rmsle={sum(drl)/len(drl):.3f}" if drl else ""
        print(f"    {domain:35s} success={ds}/{len(dr)} good+={dg}/{len(dr)} {rstr}")
    print(f"{'─'*70}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper sweep for one (model, budget) config.")
    parser.add_argument("--model",   required=True,
                        help="Model identifier, e.g. gpt-4o-mini or meta-llama/Llama-3.3-8B-Instruct-Turbo")
    parser.add_argument("--budget",  type=int, required=True,
                        help="Total oracle-call budget (20, 60, or 100)")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("AUTOSCILAB_WORKERS", "16")),
                        help="Parallel workers (default 16, override with AUTOSCILAB_WORKERS)")
    parser.add_argument("--noise",   type=float,
                        default=float(os.environ.get("AUTOSCILAB_NOISE", "0.0")))
    parser.add_argument("--pysr-iters", type=int,
                        default=int(os.environ.get("AUTOSCILAB_PYSR_ITERS", "800")),
                        help="PySR iterations per equation-learner call (default 800)")
    parser.add_argument("--seeds",   type=int, nargs="+", default=SEEDS)
    parser.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (auto-named if not set)")
    args = parser.parse_args()

    # Derive max_per_iter and eq_every from budget and 5 LLM calls
    n_llm_calls  = 5
    max_per_iter = args.budget // n_llm_calls
    eq_every     = max_per_iter  # PySR runs once per LLM call

    if max_per_iter < 1:
        parser.error(f"budget={args.budget} too small for {n_llm_calls} LLM calls")

    model_short = short_model_name(args.model)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir     = args.out_dir or Path(f"results/paper_sweep_{model_short}_b{args.budget}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_dir / "checkpoint.json"

    # Build task list
    all_tasks = [
        (domain, diff, ver, seed)
        for domain in args.domains
        for diff   in DIFFICULTIES
        for ver    in LAW_VERSIONS
        for seed   in args.seeds
    ]
    total = len(all_tasks)

    print("=" * 70)
    print(f"  Paper Sweep: {model_short}  budget={args.budget}")
    print("=" * 70)
    print(f"  Domains    : {len(args.domains)} × {len(DIFFICULTIES)} difficulties "
          f"× {len(LAW_VERSIONS)} versions × {len(args.seeds)} seeds = {total} runs")
    print(f"  LLM calls  : {n_llm_calls}  max_per_iter={max_per_iter}  eq_every={eq_every}")
    print(f"  PySR iters : {args.pysr_iters}")
    print(f"  Workers    : {args.workers}")
    print(f"  Output     : {out_dir}")
    print()

    completed, all_results = load_checkpoint(checkpoint_file)
    pending = [t for t in all_tasks if t not in completed]
    print(f"  Completed  : {len(completed)}  |  Pending: {len(pending)}")
    print()

    if not pending:
        print("All tasks already completed.")
        print_stats(all_results, args.domains)
        return

    done = len(completed)
    t0   = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                domain, diff, ver, seed,
                args.model, args.budget, max_per_iter, eq_every,
                args.noise, args.pysr_iters, out_dir,
            ): (domain, diff, ver, seed)
            for domain, diff, ver, seed in pending
        }

        for future in as_completed(futures):
            domain, diff, ver, seed = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "domain": domain, "difficulty": diff, "version": ver, "seed": seed,
                    "model": args.model, "budget": args.budget,
                    "status": "error", "gt_rmsle": None, "exact_accuracy": None,
                    "oracle_calls": 0, "llm_calls": 0, "duration_s": 0,
                    "best_equation": None, "error": str(e),
                }

            all_results.append(result)
            save_checkpoint(all_results, checkpoint_file)
            done += 1

            elapsed  = time.time() - t0
            finished = done - len(completed)
            rate     = finished / max(elapsed, 1)
            eta_s    = (total - done) / max(rate, 1e-9)
            icon     = {"success": "✅", "good": "🟡", "error": "❌"}.get(result["status"], "⚠️")
            rstr     = f"RMSLE={result['gt_rmsle']:.4f}" if result.get("gt_rmsle") is not None else "no-rmsle"
            print(
                f"[{done:4d}/{total}] {icon} {domain:35s} {diff:6s} {ver} s={seed} "
                f"| {result['status']:8s} | {rstr:16s} "
                f"| {elapsed/60:.1f}m  ETA {eta_s/60:.1f}m"
            )

            if done % 24 == 0:
                print_stats(all_results, args.domains)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  SWEEP COMPLETE: {model_short}  budget={args.budget}")
    print("=" * 70)
    print_stats(all_results, args.domains)

    # Per-difficulty
    print("Per-difficulty:")
    for diff in DIFFICULTIES:
        dr = [r for r in all_results if r["difficulty"] == diff]
        ds = sum(1 for r in dr if r["status"] == "success")
        dg = sum(1 for r in dr if r["status"] in ("success", "good"))
        print(f"  {diff:8s}  success={ds}/{len(dr)}  good+={dg}/{len(dr)}")

    agg = {
        "model": args.model,
        "model_short": model_short,
        "budget": args.budget,
        "max_per_iter": max_per_iter,
        "n_llm_calls": n_llm_calls,
        "pysr_n_iterations": args.pysr_iters,
        "n_runs": len(all_results),
        "n_success": sum(1 for r in all_results if r["status"] == "success"),
        "n_good_or_better": sum(1 for r in all_results if r["status"] in ("success", "good")),
        "n_error": sum(1 for r in all_results if r["status"] == "error"),
        "success_rate": sum(1 for r in all_results if r["status"] == "success") / max(len(all_results), 1),
    }
    rmsles = [r["gt_rmsle"] for r in all_results if r.get("gt_rmsle") is not None]
    agg["mean_gt_rmsle"] = sum(rmsles) / len(rmsles) if rmsles else None

    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"\nResults → {out_dir}")


if __name__ == "__main__":
    main()
