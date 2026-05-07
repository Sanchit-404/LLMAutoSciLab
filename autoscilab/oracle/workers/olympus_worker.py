"""
JSONL worker for Olympus surface optimization tasks.

Protocol:
  {"cmd":"metadata"}
  {"cmd":"run","params":{"param_0":0.2,...}}
  {"cmd":"batch","params_list":[{...}, {...}]}
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def _patch_matplotlib() -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    if hasattr(plt, "register_cmap"):
        return

    def _register_cmap(*args, **kwargs):
        cmap = kwargs.get("cmap")
        name = kwargs.get("name")
        if cmap is None:
            return
        try:
            matplotlib.colormaps.register(cmap, name=name)
        except Exception:
            pass

    plt.register_cmap = _register_cmap


def _extract_measurement(raw) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, list):
        cur = raw
        while isinstance(cur, list) and cur:
            cur = cur[0]
        try:
            return float(cur)
        except Exception:
            return float("nan")
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _run_once(surface, param_space, names: list[str], params: dict[str, float]) -> float:
    from olympus.objects import ParameterVector

    d = {n: float(params[n]) for n in names}
    vec = ParameterVector().from_dict(d, param_space)
    y = surface.run(vec)
    return _extract_measurement(y)


def main() -> None:
    surface_kind = sys.argv[1] if len(sys.argv) > 1 else "Branin"

    _patch_matplotlib()
    root = Path(__file__).resolve().parents[3]
    olympus_src = root / "extern" / "olympus" / "src"
    sys.path.insert(0, str(olympus_src))

    from olympus.surfaces import Surface

    surface = Surface(kind=surface_kind)
    param_space = surface.param_space
    names = [v.name for v in param_space]
    bounds = {v.name: [float(v.low), float(v.high)] for v in param_space}

    # Single-objective surfaces are treated as minimization.
    direction = "minimize"

    rng = np.random.default_rng(123)
    calib_vals = []
    for _ in range(400):
        params = {
            n: float(rng.uniform(bounds[n][0], bounds[n][1]))
            for n in names
        }
        calib_vals.append(_run_once(surface, param_space, names, params))
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
                    "objective_direction": direction,
                    "approx_min": approx_min,
                    "approx_max": approx_max,
                }
            elif cmd == "run":
                y = _run_once(surface, param_space, names, req.get("params", {}))
                out = {"ok": True, "measurement": float(y)}
            elif cmd == "batch":
                ys = [
                    _run_once(surface, param_space, names, p)
                    for p in req.get("params_list", [])
                ]
                out = {"ok": True, "measurements": [float(v) for v in ys]}
            else:
                out = {"ok": False, "error": f"unknown cmd: {cmd}"}
        except Exception as e:
            out = {"ok": False, "error": str(e)}

        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
