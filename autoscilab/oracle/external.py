"""
External benchmark oracles (ChemGymRL + SCIGYM) via isolated worker processes.
"""
from __future__ import annotations

import atexit
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from autoscilab.oracle.base import BaseOracle, OracleResult

_ROOT = Path(__file__).resolve().parent.parent.parent
_CHEMGYM_VENV_PY = _ROOT / "extern" / "chemgymrl" / ".venv" / "bin" / "python"
_SCIGYM_VENV_PY = _ROOT / "extern" / "SCIGYM" / ".venv" / "bin" / "python"
_OLYMPUS_VENV_PY = _ROOT / "extern" / "olympus" / ".venv" / "bin" / "python"
_SUMMIT_VENV_PY = _ROOT / "extern" / "summit" / ".venv" / "bin" / "python"
_CHEMGYM_WORKER = _ROOT / "autoscilab" / "oracle" / "workers" / "chemgym_worker.py"
_SCIGYM_WORKER = _ROOT / "autoscilab" / "oracle" / "workers" / "scigym_worker.py"
_OLYMPUS_WORKER = _ROOT / "autoscilab" / "oracle" / "workers" / "olympus_worker.py"
_SUMMIT_WORKER = _ROOT / "autoscilab" / "oracle" / "workers" / "summit_worker.py"
_SCIGYM_DATA_DIR = _ROOT / "extern" / "SCIGYM" / "data" / "small"
_CHEMGYM_DOMAIN_TO_ENV = {
    "chemgym_fictreact_bandit": "FictReactBandit-v0",
    "chemgym_fictreact_bandit_v0": "FictReactBandit-v0",
    "chemgym_fictreact_bandit_v1": "FictReactBandit-v1",
}
_OLYMPUS_DOMAIN_TO_SURFACE = {
    "olympus_branin": "Branin",
    "olympus_rosenbrock": "Rosenbrock",
}
_SUMMIT_DOMAIN_TO_BENCHMARK = {
    "summit_hartmann3d": "hartmann3d",
    "summit_himmelblau": "himmelblau",
    "summit_snar": "snar",
}


def _resolve_python(preferred: Path) -> Path:
    if preferred.exists():
        return preferred
    return Path(sys.executable)


class _JsonWorker:
    def __init__(self, python_bin: Path, script: Path, args: list[str] | None = None):
        if not python_bin.exists():
            raise FileNotFoundError(f"Missing worker python: {python_bin}")
        if not script.exists():
            raise FileNotFoundError(f"Missing worker script: {script}")
        cmd = [str(python_bin), "-u", str(script)] + (args or [])
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._proc.poll() is not None:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"Worker terminated unexpectedly. stderr:\n{err}")
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"Worker returned no response. stderr:\n{err}")
        data = json.loads(line)
        if not data.get("ok", False):
            raise RuntimeError(data.get("error", "unknown worker error"))
        return data

    def close(self) -> None:
        if not hasattr(self, "_proc"):
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()


def _compile_law(law_str: str):
    exec_globals: dict = {"np": np, "__builtins__": __builtins__}
    exec(law_str, exec_globals)
    fn = exec_globals.get("discovered_law")
    if fn is None:
        raise ValueError("No 'discovered_law' function found")
    return fn


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.abs(y_true)
    y_p = np.abs(y_pred)
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((np.log1p(y_p[mask]) - np.log1p(y_t[mask])) ** 2)))


def _evaluate_optimization_run(
    results: list[OracleResult],
    direction: str,
    metadata_key: str | None = None,
    metadata_value: str | None = None,
    approx_min: float | None = None,
    approx_max: float | None = None,
) -> dict[str, Any] | None:
    if not results:
        return None

    y = np.array([r.measurement for r in results], dtype=float)
    first = float(y[0])
    if direction == "minimize":
        best = float(np.min(y))
        improvement_ratio = float((first - best) / max(abs(first), 1e-12))
    else:
        best = float(np.max(y))
        improvement_ratio = float((best - first) / max(abs(first), 1e-12))

    out: dict[str, Any] = {
        "objective_type": "optimization",
        "objective_direction": direction,
        "best_measurement": best,
        "mean_measurement": float(np.mean(y)),
        "initial_measurement": first,
        "improvement_ratio": improvement_ratio,
        "n_evals": len(results),
    }

    if (
        approx_min is not None
        and approx_max is not None
        and np.isfinite(approx_min)
        and np.isfinite(approx_max)
        and approx_max > approx_min + 1e-12
    ):
        if direction == "minimize":
            normalized_best = (approx_max - best) / (approx_max - approx_min)
        else:
            normalized_best = (best - approx_min) / (approx_max - approx_min)
        out["normalized_best"] = float(np.clip(normalized_best, 0.0, 1.0))
        out["approx_min"] = float(approx_min)
        out["approx_max"] = float(approx_max)

    if metadata_key and metadata_value:
        out[metadata_key] = metadata_value
    return out


class ChemGymOracle(BaseOracle):
    def __init__(
        self,
        domain_id: str = "chemgym_fictreact_bandit",
        noise_level: float = 0.0,
        env_id: str = "FictReactBandit-v0",
    ):
        self._domain_id = domain_id
        self._noise_level = noise_level
        self._env_id = env_id
        self._param_names = [
            "heat_level",
            "dose_A",
            "dose_B",
            "dose_C",
            "dose_D",
            "gate_A",
            "gate_B",
            "gate_C",
            "gate_D",
        ]
        self._bounds = {p: (0.0, 1.0) for p in self._param_names}
        self._worker = _JsonWorker(_CHEMGYM_VENV_PY, _CHEMGYM_WORKER, [env_id])

    @property
    def domain(self) -> str:
        return self._domain_id

    @property
    def parameter_names(self) -> list[str]:
        return self._param_names

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return self._bounds

    @property
    def function_signature(self) -> str:
        return (
            "def discovered_law(heat_level, dose_A, dose_B, dose_C, dose_D, "
            "gate_A, gate_B, gate_C, gate_D):"
        )

    @property
    def param_description(self) -> str:
        return (
            "Action controls for FictReact bandit: normalized heat, reagent doses "
            f"(A-D), and timing gates (A-D), each in [0, 1]. Environment={self._env_id}."
        )

    @property
    def domain_tags(self) -> list[str]:
        return ["chemistry", "reaction", "nonlinear", "bandit"]

    @property
    def objective_type(self) -> str:
        return "optimization"

    @property
    def objective_direction(self) -> str:
        return "maximize"

    @property
    def objective_profile(self) -> dict[str, Any]:
        return {
            "name": "reaction_reward",
            "direction": "maximize",
            "type": "single_objective",
            "domain_family": "chem_reaction_control",
        }

    def run(self, params: dict[str, float]) -> OracleResult:
        action = [float(params[p]) for p in self._param_names]
        out = self._worker.request({"cmd": "run", "action": action})
        return OracleResult(
            params=dict(params),
            measurement=float(out["measurement"]),
            domain=self._domain_id,
            noise_level=self._noise_level,
            metadata={"reward": float(out.get("reward", 0.0)), "env_id": self._env_id},
        )

    def evaluate_law(self, law_str: str) -> dict[str, Any]:
        try:
            discovered_law = _compile_law(law_str)
        except Exception as e:
            return {"rmsle": float("nan"), "exact_accuracy": 0.0, "symbolic_equivalent": None, "error": str(e)}

        rng = np.random.default_rng(42)
        n_test = 300
        X = rng.uniform(0.0, 1.0, size=(n_test, len(self._param_names)))
        param_list = [
            {p: float(X[i, j]) for j, p in enumerate(self._param_names)}
            for i in range(n_test)
        ]
        try:
            out = self._worker.request(
                {"cmd": "batch", "actions": [[d[p] for p in self._param_names] for d in param_list]}
            )
            y_true = np.array(out["measurements"], dtype=float)
            y_pred = np.array(
                [
                    discovered_law(*[d[p] for p in self._param_names])
                    for d in param_list
                ],
                dtype=float,
            )
        except Exception as e:
            return {"rmsle": float("nan"), "exact_accuracy": 0.0, "symbolic_equivalent": None, "error": str(e)}

        rmsle = _rmsle(y_true, y_pred)
        return {
            "rmsle": rmsle,
            "exact_accuracy": 1.0 if rmsle < 0.01 else 0.0,
            "symbolic_equivalent": None,
        }

    def evaluate_run(self, results: list[OracleResult]) -> dict[str, Any] | None:
        if not results:
            return None
        measurements = np.array([r.measurement for r in results], dtype=float)
        rewards = np.array(
            [float((r.metadata or {}).get("reward", r.measurement - 2.0)) for r in results],
            dtype=float,
        )
        first = float(measurements[0])
        best = float(np.max(measurements))
        denom = max(abs(first), 1e-12)
        improvement_ratio = float((best - first) / denom)
        return {
            "objective_type": "optimization",
            "objective_direction": "maximize",
            "best_measurement": best,
            "mean_measurement": float(np.mean(measurements)),
            "initial_measurement": first,
            "improvement_ratio": improvement_ratio,
            "best_reward": float(np.max(rewards)),
            "mean_reward": float(np.mean(rewards)),
            "n_evals": len(results),
        }


def build_chemgym_oracle(domain_id: str, noise_level: float = 0.0) -> ChemGymOracle:
    if domain_id not in _CHEMGYM_DOMAIN_TO_ENV:
        valid = ", ".join(sorted(_CHEMGYM_DOMAIN_TO_ENV))
        raise ValueError(f"Unknown ChemGym domain '{domain_id}'. Valid: {valid}")
    env_id = _CHEMGYM_DOMAIN_TO_ENV[domain_id]
    return ChemGymOracle(domain_id=domain_id, noise_level=noise_level, env_id=env_id)


class SciGymOracle(BaseOracle):
    def __init__(self, domain_id: str, noise_level: float = 0.0):
        if not domain_id.startswith("scigym_"):
            raise ValueError(f"Invalid scigym domain_id: {domain_id}")
        self._domain_id = domain_id
        self._noise_level = noise_level
        self._model_id = domain_id.replace("scigym_", "", 1)
        self._model_dir = _SCIGYM_DATA_DIR / self._model_id
        if not self._model_dir.exists():
            raise FileNotFoundError(f"SCIGYM model folder not found: {self._model_dir}")
        self._worker = _JsonWorker(_SCIGYM_VENV_PY, _SCIGYM_WORKER, [str(self._model_dir)])
        meta = self._worker.request({"cmd": "metadata"})
        self._param_names = list(meta["control_species"])
        self._bounds = {k: tuple(v) for k, v in meta["bounds"].items()}
        self._target_species = meta["target_species"]

    @property
    def domain(self) -> str:
        return self._domain_id

    @property
    def parameter_names(self) -> list[str]:
        return self._param_names

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return self._bounds

    @property
    def function_signature(self) -> str:
        args = ", ".join(self._param_names)
        return f"def discovered_law({args}):"

    @property
    def param_description(self) -> str:
        return (
            f"Initial concentrations for selected species in SCIGYM model {self._model_id}. "
            f"Output is final concentration of target species {self._target_species}."
        )

    @property
    def domain_tags(self) -> list[str]:
        return ["biology", "dynamics", "time"]

    @property
    def objective_type(self) -> str:
        # In this adapter we optimize a scalar simulator response rather than
        # performing full SBML mechanism reconstruction.
        return "optimization"

    @property
    def objective_direction(self) -> str:
        return "maximize"

    @property
    def objective_profile(self) -> dict[str, Any]:
        return {
            "name": f"{self._target_species}_final_concentration",
            "direction": "maximize",
            "type": "single_objective",
            "domain_family": "systems_biology_control",
        }

    def run(self, params: dict[str, float]) -> OracleResult:
        p = {k: float(params[k]) for k in self._param_names}
        out = self._worker.request({"cmd": "run", "params": p})
        return OracleResult(
            params=p,
            measurement=float(out["measurement"]),
            domain=self._domain_id,
            noise_level=self._noise_level,
            metadata={"model_id": self._model_id, "target_species": self._target_species},
        )

    def evaluate_law(self, law_str: str) -> dict[str, Any]:
        try:
            discovered_law = _compile_law(law_str)
        except Exception as e:
            return {"rmsle": float("nan"), "exact_accuracy": 0.0, "symbolic_equivalent": None, "error": str(e)}

        rng = np.random.default_rng(42)
        n_test = 200
        param_list = []
        for _ in range(n_test):
            d = {}
            for p in self._param_names:
                lo, hi = self._bounds[p]
                d[p] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            param_list.append(d)
        try:
            out = self._worker.request({"cmd": "batch", "params_list": param_list})
            y_true = np.array(out["measurements"], dtype=float)
            y_pred = np.array(
                [
                    discovered_law(*[d[p] for p in self._param_names])
                    for d in param_list
                ],
                dtype=float,
            )
        except Exception as e:
            return {"rmsle": float("nan"), "exact_accuracy": 0.0, "symbolic_equivalent": None, "error": str(e)}

        rmsle = _rmsle(y_true, y_pred)
        return {
            "rmsle": rmsle,
            "exact_accuracy": 1.0 if rmsle < 0.01 else 0.0,
            "symbolic_equivalent": None,
        }

    def evaluate_run(self, results: list[OracleResult]) -> dict[str, Any] | None:
        if not results:
            return None
        measurements = np.array([r.measurement for r in results], dtype=float)
        first = float(measurements[0])
        best = float(np.max(measurements))
        denom = max(abs(first), 1e-12)
        improvement_ratio = float((best - first) / denom)
        return {
            "objective_type": "optimization",
            "objective_direction": "maximize",
            "best_measurement": best,
            "mean_measurement": float(np.mean(measurements)),
            "initial_measurement": first,
            "improvement_ratio": improvement_ratio,
            "target_species": self._target_species,
            "n_evals": len(results),
        }


class OlympusOracle(BaseOracle):
    def __init__(self, domain_id: str, noise_level: float = 0.0, surface_kind: str = "Branin"):
        self._domain_id = domain_id
        self._noise_level = noise_level
        self._surface_kind = surface_kind
        pybin = _resolve_python(_OLYMPUS_VENV_PY)
        self._worker = _JsonWorker(pybin, _OLYMPUS_WORKER, [surface_kind])
        meta = self._worker.request({"cmd": "metadata"})
        self._param_names = list(meta["param_names"])
        self._bounds = {k: tuple(v) for k, v in meta["bounds"].items()}
        self._direction = str(meta.get("objective_direction", "minimize"))
        self._approx_min = float(meta["approx_min"]) if meta.get("approx_min") is not None else None
        self._approx_max = float(meta["approx_max"]) if meta.get("approx_max") is not None else None

    @property
    def domain(self) -> str:
        return self._domain_id

    @property
    def parameter_names(self) -> list[str]:
        return self._param_names

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return self._bounds

    @property
    def function_signature(self) -> str:
        args = ", ".join(self._param_names)
        return f"def discovered_law({args}):"

    @property
    def param_description(self) -> str:
        return f"Olympus surface optimization task ({self._surface_kind}) over normalized condition variables."

    @property
    def domain_tags(self) -> list[str]:
        return ["optimization", "materials", "surface"]

    @property
    def objective_type(self) -> str:
        return "optimization"

    @property
    def objective_direction(self) -> str:
        return self._direction

    @property
    def objective_profile(self) -> dict[str, Any]:
        return {
            "name": f"{self._surface_kind}_objective",
            "direction": self._direction,
            "type": "single_objective",
            "domain_family": "optimization_surface",
        }

    def run(self, params: dict[str, float]) -> OracleResult:
        p = {k: float(params[k]) for k in self._param_names}
        out = self._worker.request({"cmd": "run", "params": p})
        return OracleResult(
            params=p,
            measurement=float(out["measurement"]),
            domain=self._domain_id,
            noise_level=self._noise_level,
            metadata={"surface_kind": self._surface_kind},
        )

    def evaluate_law(self, law_str: str) -> dict[str, Any]:
        try:
            discovered_law = _compile_law(law_str)
        except Exception as e:
            return {"rmsle": float("nan"), "exact_accuracy": 0.0, "symbolic_equivalent": None, "error": str(e)}

        rng = np.random.default_rng(42)
        n_test = 200
        param_list: list[dict[str, float]] = []
        for _ in range(n_test):
            d = {}
            for p in self._param_names:
                lo, hi = self._bounds[p]
                d[p] = float(rng.uniform(lo, hi))
            param_list.append(d)

        try:
            out = self._worker.request({"cmd": "batch", "params_list": param_list})
            y_true = np.array(out["measurements"], dtype=float)
            y_pred = np.array(
                [discovered_law(*[d[p] for p in self._param_names]) for d in param_list],
                dtype=float,
            )
        except Exception as e:
            return {"rmsle": float("nan"), "exact_accuracy": 0.0, "symbolic_equivalent": None, "error": str(e)}

        rmsle = _rmsle(y_true, y_pred)
        return {"rmsle": rmsle, "exact_accuracy": 1.0 if rmsle < 0.01 else 0.0, "symbolic_equivalent": None}

    def evaluate_run(self, results: list[OracleResult]) -> dict[str, Any] | None:
        return _evaluate_optimization_run(
            results=results,
            direction=self._direction,
            metadata_key="surface_kind",
            metadata_value=self._surface_kind,
            approx_min=self._approx_min,
            approx_max=self._approx_max,
        )


class SummitOracle(BaseOracle):
    def __init__(self, domain_id: str, noise_level: float = 0.0, benchmark_id: str = "hartmann3d"):
        self._domain_id = domain_id
        self._noise_level = noise_level
        self._benchmark_id = benchmark_id
        pybin = _resolve_python(_SUMMIT_VENV_PY)
        self._worker = _JsonWorker(pybin, _SUMMIT_WORKER, [benchmark_id])
        meta = self._worker.request({"cmd": "metadata"})
        self._param_names = list(meta["param_names"])
        self._bounds = {k: tuple(v) for k, v in meta["bounds"].items()}
        self._direction = str(meta.get("objective_direction", "maximize"))
        self._approx_min = float(meta["approx_min"]) if meta.get("approx_min") is not None else None
        self._approx_max = float(meta["approx_max"]) if meta.get("approx_max") is not None else None
        self._objective_name = str(meta.get("objective_name", "measurement"))

    @property
    def domain(self) -> str:
        return self._domain_id

    @property
    def parameter_names(self) -> list[str]:
        return self._param_names

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return self._bounds

    @property
    def function_signature(self) -> str:
        args = ", ".join(self._param_names)
        return f"def discovered_law({args}):"

    @property
    def param_description(self) -> str:
        return (
            "Summit condition optimization benchmark "
            f"({self._benchmark_id}) with objective '{self._objective_name}'."
        )

    @property
    def domain_tags(self) -> list[str]:
        return ["optimization", "chemistry", "conditions"]

    @property
    def objective_type(self) -> str:
        return "optimization"

    @property
    def objective_direction(self) -> str:
        return self._direction

    @property
    def objective_profile(self) -> dict[str, Any]:
        return {
            "name": self._objective_name,
            "direction": self._direction,
            "type": "single_objective",
            "domain_family": "condition_optimization",
            "benchmark": self._benchmark_id,
        }

    def run(self, params: dict[str, float]) -> OracleResult:
        p = {k: float(params[k]) for k in self._param_names}
        out = self._worker.request({"cmd": "run", "params": p})
        md = {
            "benchmark_id": self._benchmark_id,
            "objective_name": self._objective_name,
        }
        if "sty" in out:
            md["sty"] = float(out["sty"])
        if "e_factor" in out:
            md["e_factor"] = float(out["e_factor"])
        return OracleResult(
            params=p,
            measurement=float(out["measurement"]),
            domain=self._domain_id,
            noise_level=self._noise_level,
            metadata=md,
        )

    def evaluate_law(self, law_str: str) -> dict[str, Any]:
        try:
            discovered_law = _compile_law(law_str)
        except Exception as e:
            return {"rmsle": float("nan"), "exact_accuracy": 0.0, "symbolic_equivalent": None, "error": str(e)}

        rng = np.random.default_rng(42)
        n_test = 200
        param_list: list[dict[str, float]] = []
        for _ in range(n_test):
            d = {}
            for p in self._param_names:
                lo, hi = self._bounds[p]
                d[p] = float(rng.uniform(lo, hi))
            param_list.append(d)
        try:
            out = self._worker.request({"cmd": "batch", "params_list": param_list})
            y_true = np.array(out["measurements"], dtype=float)
            y_pred = np.array(
                [discovered_law(*[d[p] for p in self._param_names]) for d in param_list],
                dtype=float,
            )
        except Exception as e:
            return {"rmsle": float("nan"), "exact_accuracy": 0.0, "symbolic_equivalent": None, "error": str(e)}

        rmsle = _rmsle(y_true, y_pred)
        return {"rmsle": rmsle, "exact_accuracy": 1.0 if rmsle < 0.01 else 0.0, "symbolic_equivalent": None}

    def evaluate_run(self, results: list[OracleResult]) -> dict[str, Any] | None:
        return _evaluate_optimization_run(
            results=results,
            direction=self._direction,
            metadata_key="benchmark_id",
            metadata_value=self._benchmark_id,
            approx_min=self._approx_min,
            approx_max=self._approx_max,
        )


def build_olympus_oracle(domain_id: str, noise_level: float = 0.0) -> OlympusOracle:
    if domain_id not in _OLYMPUS_DOMAIN_TO_SURFACE:
        valid = ", ".join(sorted(_OLYMPUS_DOMAIN_TO_SURFACE))
        raise ValueError(f"Unknown Olympus domain '{domain_id}'. Valid: {valid}")
    return OlympusOracle(
        domain_id=domain_id,
        noise_level=noise_level,
        surface_kind=_OLYMPUS_DOMAIN_TO_SURFACE[domain_id],
    )


def build_summit_oracle(domain_id: str, noise_level: float = 0.0) -> SummitOracle:
    if domain_id not in _SUMMIT_DOMAIN_TO_BENCHMARK:
        valid = ", ".join(sorted(_SUMMIT_DOMAIN_TO_BENCHMARK))
        raise ValueError(f"Unknown Summit domain '{domain_id}'. Valid: {valid}")
    return SummitOracle(
        domain_id=domain_id,
        noise_level=noise_level,
        benchmark_id=_SUMMIT_DOMAIN_TO_BENCHMARK[domain_id],
    )
