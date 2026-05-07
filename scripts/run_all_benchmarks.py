#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _runner(name: str) -> Path:
    mapping = {
        "newton": ROOT / "scripts" / "run_newton_llm_autoscilab_budget.py",
        "chem": ROOT / "scripts" / "run_chembench_llm_autoscilab_budget.py",
        "grn": ROOT / "scripts" / "run_grn_prompt_budget.py",
    }
    return mapping[name]


def _append_common_args(
    cmd: list[str],
    *,
    workers: int,
    limit: int | None,
    out_dir: Path | None,
) -> list[str]:
    cmd.extend(["--workers", str(workers)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if out_dir is not None:
        cmd.extend(["--out-dir", str(out_dir)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convenience launcher for the paper release benchmarks."
    )
    parser.add_argument(
        "--benchmark",
        choices=["newton", "chem", "grn", "all"],
        default="all",
        help="Which benchmark family to run.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results" / "paper_release_runs",
        help="Root output directory for generated result folders.",
    )
    parser.add_argument("--newton-model", default="gpt-4o-mini")
    parser.add_argument("--newton-main-url", default=None)
    parser.add_argument("--newton-pysr-iters", type=int, default=800)
    parser.add_argument("--newton-budgets", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--newton-manifest", type=Path, default=None)
    parser.add_argument("--chem-main-model", default="gpt-4o-mini")
    parser.add_argument("--chem-main-url", default=None)
    parser.add_argument("--chem-ensemble-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--chem-ensemble-url", default="http://localhost:8001/v1")
    parser.add_argument("--chem-max-per-iter", type=int, default=5)
    parser.add_argument("--chem-budgets", type=int, nargs="+", default=[40, 60, 80])
    parser.add_argument("--chem-manifest", type=Path, default=None)
    parser.add_argument("--grn-main-model", default="gpt-4o-mini")
    parser.add_argument("--grn-main-url", default=None)
    parser.add_argument("--grn-max-per-iter", type=int, default=5)
    parser.add_argument("--grn-budgets", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--grn-examples-file", type=Path, default=None)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, list[str]]] = []

    if args.benchmark in {"newton", "all"}:
        cmd = [sys.executable, str(_runner("newton"))]
        cmd.extend(["--model", args.newton_model])
        cmd.extend(["--budgets", *map(str, args.newton_budgets)])
        cmd.extend(["--pysr-iters", str(args.newton_pysr_iters)])
        if args.newton_main_url:
            cmd.extend(["--main-url", args.newton_main_url])
        if args.newton_manifest:
            cmd.extend(["--manifest", str(args.newton_manifest)])
        _append_common_args(
            cmd,
            workers=args.workers,
            limit=args.limit,
            out_dir=args.out_root / "newton",
        )
        jobs.append(("newton", cmd))

    if args.benchmark in {"chem", "all"}:
        cmd = [sys.executable, str(_runner("chem"))]
        cmd.extend(["--main-model", args.chem_main_model])
        cmd.extend(["--budgets", *map(str, args.chem_budgets)])
        cmd.extend(["--max-per-iter", str(args.chem_max_per_iter)])
        cmd.extend(["--ensemble-model", args.chem_ensemble_model])
        cmd.extend(["--ensemble-url", args.chem_ensemble_url])
        if args.chem_main_url:
            cmd.extend(["--main-url", args.chem_main_url])
        if args.chem_manifest:
            cmd.extend(["--manifest", str(args.chem_manifest)])
        _append_common_args(
            cmd,
            workers=args.workers,
            limit=args.limit,
            out_dir=args.out_root / "chem",
        )
        jobs.append(("chem", cmd))

    if args.benchmark in {"grn", "all"}:
        cmd = [sys.executable, str(_runner("grn"))]
        cmd.extend(["--main-model", args.grn_main_model])
        cmd.extend(["--budgets", *map(str, args.grn_budgets)])
        cmd.extend(["--max-per-iter", str(args.grn_max_per_iter)])
        if args.grn_main_url:
            cmd.extend(["--main-url", args.grn_main_url])
        if args.grn_examples_file:
            cmd.extend(["--examples-file", str(args.grn_examples_file)])
        _append_common_args(
            cmd,
            workers=args.workers,
            limit=args.limit,
            out_dir=args.out_root / "grn",
        )
        jobs.append(("grn", cmd))

    for name, cmd in jobs:
        print(f"[paper-release] running {name}: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
