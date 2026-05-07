# LLM-AutoSciLab Paper Release

This release contains the minimal code needed to run the paper's three benchmark families with our method:

- `NewtonBench`
- `ActiveSciBench-Chem`
- `ActiveSciBench-GRN`

The packaged tree includes:

- `autoscilab/`: method, loops, acquisition, and oracle implementations
- `configs/`: fixed manifests used by the benchmark runners
- `NewtonBench/`: local Newton benchmark assets required by the Newton oracle
- `scripts/`: benchmark entry points plus a convenience launcher

## Environment

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required environment variables:

- `OPENAI_API_KEY`: required for the default `gpt-4o-mini` runs
- `TOGETHER_API_KEY`: only required if you use Together-hosted non-OpenAI models

Optional overrides:

- `--main-url`: point the main model at an OpenAI-compatible local or remote endpoint
- `--ensemble-url`: ChemBench ensemble endpoint, default `http://localhost:8001/v1`

## Entry Points

The simplest interface is the wrapper:

```bash
python scripts/run_all_benchmarks.py --benchmark all --workers 1 --limit 1
```

That runs a small manifest slice from all three benchmark families and writes results under:

```text
results/paper_release_runs/
```

Run a single family:

```bash
python scripts/run_all_benchmarks.py --benchmark newton --workers 1
python scripts/run_all_benchmarks.py --benchmark chem --workers 1
python scripts/run_all_benchmarks.py --benchmark grn --workers 1
```

## Direct Benchmark Runners

Newton:

```bash
python scripts/run_newton_meiv5_budget.py \
  --model gpt-4o-mini \
  --budgets 10 20 50 \
  --workers 4
```

Chem:

```bash
python scripts/run_chembench_meiv5_budget.py \
  --main-model gpt-4o-mini \
  --budgets 40 60 80 \
  --workers 4
```

GRN:

```bash
python scripts/run_grn_prompt_budget.py \
  --main-model gpt-4o-mini \
  --budgets 10 20 50 \
  --workers 4
```

## Notes

- The Newton oracle expects the bundled `NewtonBench/` directory to remain adjacent to `autoscilab/`.
- The packaged ChemBench default path runs without requiring a local ensemble server. The `--ensemble-url` flag only matters if you explicitly enable or adapt an ensemble-backed configuration.
- Results are written as JSON summaries in benchmark-specific subdirectories under `results/`.

## Smoke Test

The packaged runners were smoke-tested from this release tree by checking the CLI entry points and the wrapper launcher:

```bash
python scripts/run_newton_meiv5_budget.py --help
python scripts/run_chembench_meiv5_budget.py --help
python scripts/run_grn_prompt_budget.py --help
python scripts/run_all_benchmarks.py --help
```

For a lightweight live run, use:

```bash
python scripts/run_all_benchmarks.py --benchmark all --workers 1 --limit 1
```
