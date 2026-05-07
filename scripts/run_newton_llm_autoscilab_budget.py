#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_paper_sweep import run_one as run_newton_one

ROOT = Path(__file__).parent.parent
DEFAULT_MANIFEST = ROOT / 'configs' / 'noise_studies' / 'newtonbench_llm_autoscilab_noise108.json'


def _load_manifest(path: Path, limit: int | None = None) -> list[dict]:
    rows = json.loads(path.read_text())
    return rows[:limit] if limit is not None else rows


def _normalize(row: dict, task: dict, budget: int) -> dict:
    return {
        'task_id': task['id'],
        'domain': task['domain'],
        'difficulty': task['difficulty'],
        'law_version': task['law_version'],
        'seed': task['seed'],
        'budget': budget,
        'noise': 0.0,
        'method': 'llm_autoscilab_newton',
        'status': row.get('status'),
        'gt_rmsle': row.get('gt_rmsle'),
        'exact_accuracy': row.get('exact_accuracy'),
        'oracle_calls': row.get('oracle_calls'),
        'llm_calls': row.get('llm_calls'),
        'duration_s': row.get('duration_s'),
        'best_equation': row.get('best_equation'),
        'law_str': row.get('law_str'),
        'error': row.get('error'),
        'result_dir': row.get('results_dir') or row.get('result_dir'),
    }


def _aggregate(rows: list[dict]) -> dict:
    completed = [r for r in rows if r.get('status') and r.get('status') != 'error']
    rmsles = [float(r['gt_rmsle']) for r in completed if isinstance(r.get('gt_rmsle'), (int, float))]
    exacts = [float(r.get('exact_accuracy') or 0.0) for r in completed]
    durations = [float(r['duration_s']) for r in completed if isinstance(r.get('duration_s'), (int, float))]
    return {
        'n_rows': len(rows),
        'n_completed': len(completed),
        'n_errors': sum(r.get('status') == 'error' for r in rows),
        'mean_gt_rmsle': mean(rmsles) if rmsles else None,
        'exact_accuracy': mean(exacts) if exacts else None,
        'mean_duration_s': mean(durations) if durations else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run NewtonBench LLM-AutoSciLab budget study on a fixed manifest.')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--budgets', type=int, nargs='+', default=[10, 20, 50])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--main-url', default=None)
    parser.add_argument('--pysr-iters', type=int, default=800)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()

    tasks = _load_manifest(args.manifest, args.limit)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or ROOT / 'results' / f'newton_llm_autoscilab_budget_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    by_budget: dict[str, dict] = {}

    print(f'[NewtonBudget] tasks={len(tasks)} workers={args.workers} model={args.model} out={out_dir}')

    for budget in args.budgets:
        budget_dir = out_dir / f'b{budget}'
        budget_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        max_per_iter = max(1, budget // 5)
        eq_every = max_per_iter
        print(f'[NewtonBudget] budget={budget} -> {budget_dir}')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_newton_one,
                    task['domain'],
                    task['difficulty'],
                    task['law_version'],
                    task['seed'],
                    args.model,
                    budget,
                    max_per_iter,
                    eq_every,
                    0.0,
                    args.pysr_iters,
                    budget_dir,
                    args.main_url,
                ): task
                for task in tasks
            }
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    raw = fut.result()
                except Exception as exc:
                    raw = {
                        'status': 'error', 'gt_rmsle': None, 'exact_accuracy': None,
                        'oracle_calls': 0, 'llm_calls': 0, 'duration_s': 0,
                        'best_equation': None, 'law_str': None,
                        'error': f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}', 'result_dir': None,
                    }
                row = _normalize(raw, task, budget)
                rows.append(row)
                print(f"[NewtonBudget] {task['id']} budget={budget}: {row['status']}")
        rows.sort(key=lambda r: (r['domain'], r['difficulty'], r['law_version'], r['seed']))
        (budget_dir / 'summary.json').write_text(json.dumps(rows, indent=2, default=str))
        agg = _aggregate(rows)
        by_budget[str(budget)] = agg
        (budget_dir / 'aggregate.json').write_text(json.dumps(agg, indent=2))
        all_rows.extend(rows)

    root_summary = {
        'benchmark': 'newtonbench', 'method': 'llm_autoscilab_newton', 'model': args.model,
        'manifest': str(args.manifest), 'budgets': args.budgets, 'workers': args.workers,
        'by_budget': by_budget,
    }
    (out_dir / 'summary.json').write_text(json.dumps(all_rows, indent=2, default=str))
    (out_dir / 'aggregate.json').write_text(json.dumps(root_summary, indent=2))
    print(f'[NewtonBudget] wrote {out_dir}')


if __name__ == '__main__':
    main()
