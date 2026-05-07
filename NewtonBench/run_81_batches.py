#!/usr/bin/env python3
"""
Run 81 experiments in 3 batches of 27.

Config: 3 domains × 3 difficulties × 3 systems × 3 law versions = 81.
Batch 1 = configs 0–26, Batch 2 = 27–53, Batch 3 = 54–80.

Usage:
  python run_81_batches.py --batch 0 [--model_name gpt41mini] [--agent_backend vanilla_agent] [--no_prompt]
  python run_81_batches.py --batch 1
  python run_81_batches.py --batch 2
"""
import argparse
import subprocess
import sys
from pathlib import Path

# 3 domains: Hooke, Malus, Fourier
DOMAINS = ["m9_hooke_law", "m7_malus_law", "m3_fourier_law"]
DIFFICULTIES = ["easy", "medium", "hard"]
SYSTEMS = ["vanilla_equation", "simple_system", "complex_system"]
LAW_VERSIONS = ["v0", "v1", "v2"]

def build_configs():
    configs = []
    for module in DOMAINS:
        for difficulty in DIFFICULTIES:
            for system in SYSTEMS:
                for law_version in LAW_VERSIONS:
                    configs.append({
                        "module": module,
                        "equation_difficulty": difficulty,
                        "model_system": system,
                        "law_version": law_version,
                    })
    return configs


def main():
    parser = argparse.ArgumentParser(description="Run 81 configs in batches of 27")
    parser.add_argument("--batch", type=int, required=True, choices=[0, 1, 2],
                        help="Batch index: 0 (configs 0-26), 1 (27-53), 2 (54-80)")
    parser.add_argument("--model_name", type=str, default="gpt41mini", help="LLM model name")
    parser.add_argument("--agent_backend", type=str, default="vanilla_agent",
                        choices=["vanilla_agent", "code_assisted_agent"])
    parser.add_argument("--trials_per_law", type=int, default=4, help="Trials per config")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--no_prompt", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    configs = build_configs()
    if len(configs) != 81:
        print(f"Expected 81 configs, got {len(configs)}", file=sys.stderr)
        sys.exit(1)

    start = args.batch * 27
    end = start + 27
    batch_configs = configs[start:end]

    repo_root = Path(__file__).resolve().parent
    run_experiments = repo_root / "run_experiments.py"

    print(f"Batch {args.batch}: configs {start}-{end-1} (27 configs)")
    for i, c in enumerate(batch_configs):
        print(f"  {start + i + 1}. {c['module']} {c['equation_difficulty']} {c['model_system']} {c['law_version']}")

    if not args.no_prompt:
        response = input("\nProceed? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Cancelled.")
            return

    failed = []
    for i, c in enumerate(batch_configs):
        idx = start + i + 1
        print(f"\n[{idx}/81] {c['module']} / {c['equation_difficulty']} / {c['model_system']} / {c['law_version']}")
        cmd = [
            sys.executable,
            str(run_experiments),
            "--module", c["module"],
            "--equation_difficulty", c["equation_difficulty"],
            "--model_system", c["model_system"],
            "--law_version", c["law_version"],
            "--model_name", args.model_name,
            "--agent_backend", args.agent_backend,
            "--trials", str(args.trials_per_law),
            "--noise", str(args.noise),
        ]
        try:
            subprocess.run(cmd, check=True, cwd=repo_root)
        except subprocess.CalledProcessError:
            failed.append((idx, c))

    if failed:
        print(f"\nFailed: {len(failed)} configs")
        for idx, c in failed:
            print(f"  {idx}: {c['module']} {c['equation_difficulty']} {c['model_system']} {c['law_version']}")
        sys.exit(1)
    print(f"\nBatch {args.batch} complete: 27/27 succeeded.")


if __name__ == "__main__":
    main()
