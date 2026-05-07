"""
JSONL worker for SCIGYM SBML simulations.

Protocol:
  {"cmd":"metadata"} -> control species, bounds, target species
  {"cmd":"run","params":{"S1": 0.1, ...}}
  {"cmd":"batch","params_list":[{...}, {...}]}
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from scigym.data.sbml import SBML
from scigym.data.simulator import Simulator, run_simulation


def _to_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _build_metadata(sbml: SBML) -> dict:
    init = sbml.get_initial_concentrations()
    valid_species = []
    for sid, val in init.items():
        v = _to_float(val)
        if math.isfinite(v) and v > 0:
            valid_species.append((sid, v))

    if not valid_species:
        species = sbml.get_species_ids()
        valid_species = [(sid, 1.0) for sid in species[:3]]

    control = [sid for sid, _ in valid_species[:3]]
    target = next((sid for sid, _ in valid_species if sid not in control), control[0])

    bounds = {}
    for sid, v in valid_species[:3]:
        lo = max(1e-8, 0.2 * v)
        hi = max(lo * 1.01, 5.0 * v)
        bounds[sid] = [float(lo), float(hi)]

    return {
        "control_species": control,
        "target_species": target,
        "bounds": bounds,
    }


def _run_once(sim: Simulator, meta: dict, params: dict[str, float]) -> float:
    sim.prepare_simulation()
    rr = sim._rr
    for sid in meta["control_species"]:
        if sid in params:
            rr.setInitConcentration(sid, float(params[sid]), forceRegenerate=False)
    result = run_simulation(
        rr,
        observed_species=[meta["target_species"]],
        observed_parameters=[],
        noise=0.0,
        rm_concentration_brackets=True,
        sed_simulation=sim.simulation,
    )
    series = result.result.get(meta["target_species"], [])
    if not series:
        return 1e-12
    y = float(series[-1])
    return max(1e-12, abs(y))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: scigym_worker.py <model_dir>")

    model_dir = Path(sys.argv[1])
    sbml = SBML(str(model_dir / "truth.xml"), str(model_dir / "truth.sedml"))
    sim = Simulator(sbml)
    meta = _build_metadata(sbml)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            cmd = req.get("cmd")
            if cmd == "metadata":
                out = {"ok": True, **meta}
            elif cmd == "run":
                y = _run_once(sim, meta, req.get("params", {}))
                out = {"ok": True, "measurement": y}
            elif cmd == "batch":
                ys = [_run_once(sim, meta, p) for p in req.get("params_list", [])]
                out = {"ok": True, "measurements": ys}
            else:
                out = {"ok": False, "error": f"unknown cmd: {cmd}"}
        except Exception as e:
            out = {"ok": False, "error": str(e)}

        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
