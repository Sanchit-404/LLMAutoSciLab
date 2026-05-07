#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from autoscilab.grn_autoscilab.loop import GRNMEIGraphConfig, GRNMEIGraphLoop
from autoscilab.oracle.grnbench import GRNBenchOracle

ROOT = Path(__file__).parent.parent
DEFAULT_EXAMPLES_FILE = ROOT / "configs" / "grnbench_llm_autoscilab_examples.json"


def _load_examples(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _run_one(example: dict, args: argparse.Namespace, out_dir: Path) -> dict:
    # Preserve explicitly exported credentials (e.g. per-run DeepInfra keys)
    # and only backfill missing variables from .env.
    load_dotenv(override=False)
    run_dir = out_dir / example["id"]
    try:
        oracle = GRNBenchOracle(
            domain_id=example["domain"],
            difficulty=example["difficulty"],
            law_version=example["law_version"],
            noise_level=args.noise,
            grammar_mode="none",
        )
        restarts = args.restarts
        max_graph_edit_distance = args.max_graph_edit_distance
        per_depth_candidate_cap = args.per_depth_candidate_cap
        candidate_pool_size = args.candidate_pool_size
        if args.experiment_mode == "search":
            restarts = max(restarts, 6)
            max_graph_edit_distance = max(max_graph_edit_distance, 2)
            per_depth_candidate_cap = max(per_depth_candidate_cap, 128)
        if args.experiment_mode == "acquisition":
            candidate_pool_size = max(candidate_pool_size, 768)

        cfg = GRNMEIGraphConfig(
            domain_id=example["domain"],
            difficulty=example["difficulty"],
            law_version=example["law_version"],
            noise_level=args.noise,
            budget=int(args.budget or example.get("budget", 20)),
            max_experiments_per_iter=args.max_per_iter,
            min_data_for_graph_fit=args.min_data_for_graph_fit,
            acquisition=args.acquisition,
            llm_model=args.main_model,
            max_completion_tokens=args.max_completion_tokens,
            candidate_pool_size=candidate_pool_size,
            max_edges=args.max_edges,
            top_k_graphs=args.top_k_graphs,
            restarts=restarts,
            max_indegree=args.max_indegree,
            max_graph_edit_distance=max_graph_edit_distance,
            per_depth_candidate_cap=per_depth_candidate_cap,
            diversity_weight=args.diversity_weight,
            state_weight=args.state_weight,
            use_internal_state=args.use_internal_state,
            complexity_penalty=args.complexity_penalty,
            bic_penalty_scale=args.bic_penalty_scale,
            experiment_mode=args.experiment_mode,
            results_dir=run_dir,
            seed=args.seed,
        )
        if args.main_url and "deepinfra.com" in args.main_url.lower():
            api_key = os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        elif args.main_url:
            api_key = os.environ.get("OPENAI_API_KEY") or "local"
        else:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TOGETHER_API_KEY")

        result = GRNMEIGraphLoop(cfg, api_key=api_key, oracle=oracle, base_url=args.main_url).run()
        payload = {
            "example_id": example["id"],
            "domain": example["domain"],
            "difficulty": example["difficulty"],
            "law_version": example["law_version"],
            "budget": cfg.budget,
            "status": result.status,
            "oracle_calls": result.n_oracle_calls,
            "llm_calls": result.n_llm_calls,
            "duration_s": result.duration_seconds,
            "best_graph": result.best_graph,
            "final_graph_eval": result.final_evaluation,
        }
        (run_dir / "llm_autoscilab_graph_summary.json").write_text(json.dumps(payload, indent=2))
        return payload
    except Exception as exc:
        tb = traceback.format_exc()
        err_payload = {
            "example_id": example["id"],
            "domain": example["domain"],
            "difficulty": example["difficulty"],
            "law_version": example["law_version"],
            "status": "error",
            "error": str(exc),
            "traceback": tb,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.json").write_text(json.dumps(err_payload, indent=2))
        return err_payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    # Preserve explicitly exported credentials (e.g. per-run DeepInfra keys)
    # and only backfill missing variables from .env.
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(description="Run LLM-AutoSciLab GRN graph discovery on examples.")
    parser.add_argument("--examples-file", default=str(DEFAULT_EXAMPLES_FILE))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--main-model", default="gpt-4o-mini")
    parser.add_argument("--main-url", default=None)
    parser.add_argument("--acquisition", choices=["ei", "variance", "ucb"], default="ei")
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--max-per-iter", type=int, default=4)
    parser.add_argument("--min-data-for-graph-fit", type=int, default=8)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--max-edges", type=int, default=6)
    parser.add_argument("--top-k-graphs", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--max-indegree", type=int, default=3)
    parser.add_argument("--max-graph-edit-distance", type=int, default=1)
    parser.add_argument("--per-depth-candidate-cap", type=int, default=48)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument("--use-internal-state", action="store_true", default=True)
    parser.add_argument("--complexity-penalty", type=float, default=0.0)
    parser.add_argument("--bic-penalty-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-mode", choices=["prompt", "search", "acquisition"], required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    examples = _load_examples(Path(args.examples_file))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / f"grn_llm_autoscilab_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GRN-LLM-AutoSciLab] Running {len(examples)} examples with workers={args.workers}")
    print(f"[GRN-LLM-AutoSciLab] Output: {out_dir}")

    rows: list[dict] = []
    if args.workers <= 1:
        for ex in examples:
            try:
                row = _run_one(ex, args, out_dir)
            except Exception as exc:
                tb = traceback.format_exc()
                err_payload = {
                    "example_id": ex["id"],
                    "domain": ex["domain"],
                    "difficulty": ex["difficulty"],
                    "law_version": ex["law_version"],
                    "error": str(exc),
                    "traceback": tb,
                }
                err_dir = out_dir / ex["id"]
                err_dir.mkdir(parents=True, exist_ok=True)
                (err_dir / "error.json").write_text(json.dumps(err_payload, indent=2))
                row = {
                    "example_id": ex["id"],
                    "domain": ex["domain"],
                    "difficulty": ex["difficulty"],
                    "law_version": ex["law_version"],
                    "status": "error",
                    "error": str(exc),
                    "traceback": tb,
                }
            rows.append(row)
            print(f"[GRN-LLM-AutoSciLab] {row['example_id']}: {row['status']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one, ex, args, out_dir): ex for ex in examples}
            for fut in as_completed(futures):
                ex = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    tb = traceback.format_exc()
                    err_payload = {
                        "example_id": ex["id"],
                        "domain": ex["domain"],
                        "difficulty": ex["difficulty"],
                        "law_version": ex["law_version"],
                        "error": str(exc),
                        "traceback": tb,
                    }
                    err_dir = out_dir / ex["id"]
                    err_dir.mkdir(parents=True, exist_ok=True)
                    (err_dir / "error.json").write_text(json.dumps(err_payload, indent=2))
                    row = {
                        "example_id": ex["id"],
                        "domain": ex["domain"],
                        "difficulty": ex["difficulty"],
                        "law_version": ex["law_version"],
                        "status": "error",
                        "error": str(exc),
                        "traceback": tb,
                    }
                rows.append(row)
                print(f"[GRN-LLM-AutoSciLab] {row['example_id']}: {row['status']}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"[GRN-LLM-AutoSciLab] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
