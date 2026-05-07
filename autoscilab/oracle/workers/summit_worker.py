"""
JSONL worker for Summit condition-optimization benchmarks.

Protocol:
  {"cmd":"metadata"}
  {"cmd":"run","params":{"x_1":0.2,...}}
  {"cmd":"batch","params_list":[{...}, {...}]}
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def _build_benchmark(benchmark_id: str):
    if benchmark_id == "hartmann3d":
        from summit.benchmarks.test_functions import Hartmann3D

        bench = Hartmann3D(maximize=True)
        return bench, "y", "maximize", False
    if benchmark_id == "himmelblau":
        from summit.benchmarks.test_functions import Himmelblau

        bench = Himmelblau(maximize=False)
        return bench, "y", "minimize", False
    if benchmark_id == "snar":
        from summit.benchmarks.snar import SnarBenchmark

        bench = SnarBenchmark()
        # Scalarization proxy: maximize STY / (1 + E-factor)
        return bench, "snar_scalarized", "maximize", True
    raise ValueError(f"Unsupported summit benchmark id: {benchmark_id}")


def _run_once(bench, names: list[str], params: dict[str, float], is_snar: bool):
    from summit.utils.dataset import DataSet

    row = [[float(params[n]) for n in names]]
    cond = DataSet(row, columns=names)
    res = bench.run_experiments(cond)
    latest = res.iloc[-1]
    if is_snar:
        sty = float(latest[("sty", "DATA")])
        e_factor = float(latest[("e_factor", "DATA")])
        measurement = float(sty / (1.0 + max(e_factor, 1e-12)))
        return measurement, {"sty": sty, "e_factor": e_factor}
    y = float(latest[("y", "DATA")])
    return y, {}


def main() -> None:
    benchmark_id = sys.argv[1] if len(sys.argv) > 1 else "hartmann3d"

    root = Path(__file__).resolve().parents[3]
    summit_root = root / "extern" / "summit"
    sys.path.insert(0, str(summit_root))

    bench, objective_name, direction, is_snar = _build_benchmark(benchmark_id)
    input_vars = list(bench.domain.input_variables)
    names = [v.name for v in input_vars]
    bounds = {v.name: [float(v.bounds[0]), float(v.bounds[1])] for v in input_vars}

    rng = np.random.default_rng(123)
    calib_n = 60 if is_snar else 300
    calib_vals = []
    for _ in range(calib_n):
        p = {n: float(rng.uniform(bounds[n][0], bounds[n][1])) for n in names}
        y, _ = _run_once(bench, names, p, is_snar)
        calib_vals.append(y)
    calib = np.array(calib_vals, dtype=float)
    approx_min = float(np.nanmin(calib))
    approx_max = float(np.nanmax(calib))

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            cmd = req.get("cmd")
            if cmd == "metadata":
                out = {
                    "ok": True,
                    "param_names": names,
                    "bounds": bounds,
                    "objective_name": objective_name,
                    "objective_direction": direction,
                    "approx_min": approx_min,
                    "approx_max": approx_max,
                }
            elif cmd == "run":
                y, extras = _run_once(bench, names, req.get("params", {}), is_snar)
                out = {"ok": True, "measurement": float(y), **extras}
            elif cmd == "batch":
                ys = []
                for p in req.get("params_list", []):
                    y, _ = _run_once(bench, names, p, is_snar)
                    ys.append(float(y))
                out = {"ok": True, "measurements": ys}
            else:
                out = {"ok": False, "error": f"unknown cmd: {cmd}"}
        except Exception as e:
            out = {"ok": False, "error": str(e)}

        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
