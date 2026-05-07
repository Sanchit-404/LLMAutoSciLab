"""
NewtonBench oracle wrapper for all 12 physics domains.

Each domain has a different calling convention for run_experiment_for_module.
We normalize everything to: run(params: dict[str, float]) -> OracleResult.
"""
import sys
from pathlib import Path
from typing import Any

from autoscilab.oracle.base import BaseOracle, OracleResult
from autoscilab.oracle.external import (
    SciGymOracle,
    build_chemgym_oracle,
    build_olympus_oracle,
    build_summit_oracle,
)

# Path to the cloned NewtonBench repo (sibling of this package)
_DEFAULT_NB_PATH = Path(__file__).parent.parent.parent / "NewtonBench"

# Registry: domain_id → configuration
# kwarg_keys: the kwarg names the module expects (for kwargs-based modules)
# positional_args: ordered arg names when module takes positional params (m0_gravity)
# bounds: (min, max) in natural units — used for log-uniform sampling
DOMAIN_REGISTRY: dict[str, dict] = {
    "m0_gravity": {
        "module": "modules.m0_gravity",
        "param_names": ["mass1", "mass2", "distance"],
        "call_style": "positional",  # run_experiment_for_module(mass1, mass2, distance, noise_level, ...)
        "kwarg_keys": None,
        "bounds": {
            "mass1": (1.0, 1000.0),
            "mass2": (1.0, 1000.0),
            "distance": (1.0, 10.0),
        },
        "tags": ["power_law", "inverse_distance"],
    },
    "m1_coulomb_force": {
        "module": "modules.m1_coulomb_force",
        "param_names": ["q1", "q2", "distance"],
        "call_style": "kwargs",
        "kwarg_keys": ["q1", "q2", "distance"],
        "bounds": {
            "q1": (1e-6, 1e-2),
            "q2": (1e-6, 1e-2),
            "distance": (0.01, 10.0),
        },
        "tags": ["power_law", "inverse_distance"],
    },
    "m2_magnetic_force": {
        "module": "modules.m2_magnetic_force",
        "param_names": ["current1", "current2", "distance"],
        "call_style": "kwargs",
        "kwarg_keys": ["current1", "current2", "distance"],
        "bounds": {
            "current1": (0.1, 100.0),
            "current2": (0.1, 100.0),
            "distance": (0.001, 1.0),
        },
        "tags": ["power_law", "inverse_distance"],
    },
    "m3_fourier_law": {
        "module": "modules.m3_fourier_law",
        "param_names": ["k", "A", "delta_T", "d"],
        "call_style": "kwargs",
        "kwarg_keys": ["k", "A", "delta_T", "d"],
        "bounds": {
            "k": (0.1, 10.0),
            "A": (1e-4, 1e-2),
            "delta_T": (10.0, 1000.0),
            "d": (0.01, 1.0),
        },
        "tags": ["conduction", "power_law"],
    },
    "m4_snell_law": {
        "module": "modules.m4_snell_law",
        "param_names": ["refractive_index_1", "refractive_index_2", "incidence_angle"],
        "call_style": "kwargs",
        "kwarg_keys": ["refractive_index_1", "refractive_index_2", "incidence_angle"],
        "bounds": {
            "refractive_index_1": (1.0, 3.0),
            "refractive_index_2": (1.0, 3.0),
            "incidence_angle": (0.01, 1.5),  # radians, avoid exactly 0 and pi/2
        },
        "tags": ["trig", "optics"],
    },
    "m5_radioactive_decay": {
        "module": "modules.m5_radioactive_decay",
        "param_names": ["N0", "lambda_constant", "t"],
        "call_style": "kwargs",
        "kwarg_keys": ["N0", "lambda_constant", "t"],
        "bounds": {
            "N0": (100.0, 1e6),
            "lambda_constant": (0.001, 1.0),
            "t": (0.1, 100.0),
        },
        "tags": ["decay", "exponential", "time"],
    },
    "m6_underdamped_harmonic": {
        "module": "modules.m6_underdamped_harmonic",
        "param_names": ["k", "m", "b"],
        # Note: module uses k_constant, mass, b_constant as kwarg keys
        "call_style": "kwargs",
        "kwarg_keys": ["k_constant", "mass", "b_constant"],
        "bounds": {
            "k": (1.0, 100.0),
            "m": (0.1, 10.0),
            "b": (0.01, 5.0),
        },
        "tags": ["oscillatory", "damped", "trig", "time"],
    },
    "m7_malus_law": {
        "module": "modules.m7_malus_law",
        "param_names": ["I_0", "theta"],
        "call_style": "kwargs",
        "kwarg_keys": ["I_0", "theta"],
        "bounds": {
            "I_0": (0.1, 1000.0),
            "theta": (0.01, 1.57),  # 0 to ~pi/2 radians
        },
        "tags": ["trig", "optics"],
    },
    "m8_sound_speed": {
        "module": "modules.m8_sound_speed",
        "param_names": ["gamma", "T", "M"],
        "call_style": "kwargs",
        "kwarg_keys": ["adiabatic_index", "temperature", "molar_mass"],
        "bounds": {
            "gamma": (1.1, 1.8),
            "T": (200.0, 2000.0),
            "M": (0.002, 0.05),  # kg/mol
        },
        "tags": ["thermo", "power_law"],
    },
    "m9_hooke_law": {
        "module": "modules.m9_hooke_law",
        "param_names": ["x"],
        "call_style": "kwargs",
        "kwarg_keys": ["x"],
        "bounds": {
            "x": (0.001, 10.0),
        },
        "tags": ["power_law", "polynomial"],
    },
    "m10_be_distribution": {
        "module": "modules.m10_be_distribution",
        "param_names": ["omega", "T"],
        "call_style": "kwargs",
        "kwarg_keys": ["omega", "temperature"],
        "bounds": {
            "omega": (1e8, 1e14),
            "T": (100.0, 10000.0),
        },
    },
    "m11_heat_transfer": {
        "module": "modules.m11_heat_transfer",
        "param_names": ["m", "c", "delta_T"],
        "call_style": "kwargs",
        "kwarg_keys": ["m", "c", "delta_T"],
        "bounds": {
            "m": (0.01, 100.0),
            "c": (100.0, 5000.0),
            "delta_T": (0.1, 1000.0),
        },
    },
}


class NewtonBenchOracle(BaseOracle):
    def __init__(
        self,
        domain_id: str,
        difficulty: str = "easy",
        noise_level: float = 0.01,
        law_version: str | None = None,
        newtonbench_path: Path | None = None,
    ):
        if domain_id not in DOMAIN_REGISTRY:
            raise ValueError(f"Unknown domain: {domain_id}. Choose from {list(DOMAIN_REGISTRY)}")

        self._domain_id = domain_id
        self._difficulty = difficulty
        self._noise_level = noise_level
        self._cfg = DOMAIN_REGISTRY[domain_id]

        nb_path = newtonbench_path or _DEFAULT_NB_PATH
        nb_str = str(nb_path)
        if nb_str not in sys.path:
            sys.path.insert(0, nb_str)

        # Dynamically import the module
        import importlib
        mod = importlib.import_module(self._cfg["module"])
        self._run_fn = mod.run_experiment_for_module
        self._function_sig = mod.FUNCTION_SIGNATURE
        self._param_desc = mod.PARAM_DESCRIPTION

        # Import laws submodule for GT law access (bypasses nemotron judge)
        laws_mod = importlib.import_module(self._cfg["module"] + ".laws")
        self._gt_law_fn = laws_mod.get_ground_truth_law

        # Fix the law version at construction time.
        # If None, pick one randomly now and lock it in for all subsequent calls.
        # This ensures all experiments in a session use the same law.
        # Normalize integer law_version → 'v{n}' string that laws.py expects.
        if law_version is None:
            try:
                avail = mod.get_available_law_versions(difficulty)
                import random
                self._law_version = random.choice(avail)
            except Exception:
                self._law_version = "v0"  # safe fallback
        elif isinstance(law_version, int):
            self._law_version = f"v{law_version}"
        elif isinstance(law_version, str) and law_version.isdigit():
            self._law_version = f"v{law_version}"
        else:
            self._law_version = law_version

        print(f"[Oracle] {domain_id} | difficulty={difficulty} | law_version={self._law_version}")

    @property
    def domain(self) -> str:
        return self._domain_id

    @property
    def parameter_names(self) -> list[str]:
        return self._cfg["param_names"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return self._cfg["bounds"]

    @property
    def domain_tags(self) -> list[str]:
        return list(self._cfg.get("tags", []))

    @property
    def function_signature(self) -> str:
        return self._function_sig

    @property
    def param_description(self) -> str:
        return self._param_desc

    @property
    def objective_profile(self) -> dict[str, Any]:
        return {
            "name": "equation_recovery",
            "direction": "minimize_error",
            "type": "law_discovery",
            "difficulty": self._difficulty,
            "law_version": self._law_version,
        }

    def run(self, params: dict[str, float]) -> OracleResult:
        """Run one vanilla_equation experiment and return the measurement."""
        cfg = self._cfg
        call_style = cfg["call_style"]

        if call_style == "positional":
            # m0_gravity: run_experiment_for_module(mass1, mass2, distance, noise_level, ...)
            pos_vals = [params[k] for k in cfg["param_names"]]
            measurement = self._run_fn(
                *pos_vals,
                self._noise_level,
                difficulty=self._difficulty,
                system="vanilla_equation",
                law_version=self._law_version,
            )
        else:
            # All other modules: run_experiment_for_module(noise_level, difficulty=..., **kwargs)
            # Map from param_names → kwarg_keys
            kwarg_keys = cfg["kwarg_keys"]
            param_names = cfg["param_names"]
            kwargs = {kk: params[pn] for pn, kk in zip(param_names, kwarg_keys)}
            measurement = self._run_fn(
                self._noise_level,
                difficulty=self._difficulty,
                system="vanilla_equation",
                law_version=self._law_version,
                **kwargs,
            )

        return OracleResult(
            params=dict(params),
            measurement=float(measurement),
            domain=self._domain_id,
            noise_level=self._noise_level,
            metadata={"difficulty": self._difficulty, "law_version": self._law_version},
        )

    def evaluate_law(self, law_str: str) -> dict[str, Any]:
        """Evaluate a discovered law numerically via RMSLE (no LLM judge required)."""
        import numpy as np

        gt_law, _ = self._gt_law_fn(self._difficulty, self._law_version)

        # Sample 1000 log-uniform test points within the domain bounds
        rng = np.random.default_rng(42)
        param_names = self._cfg["param_names"]
        bounds = self._cfg["bounds"]
        n_test = 1000

        test_samples = {
            name: np.exp(rng.uniform(np.log(lo), np.log(hi), n_test))
            for name, (lo, hi) in bounds.items()
        }

        # Ground truth predictions (positional in param_names order)
        y_true = np.array([
            gt_law(*[test_samples[name][i] for name in param_names])
            for i in range(n_test)
        ])

        # Compile and evaluate the discovered law
        # Use a proper globals dict so trig/math functions are available inside
        # the function body (exec globals = function's __globals__).
        import math as _math
        exec_globals: dict = {"math": _math, "np": np, "__builtins__": __builtins__}
        try:
            exec(law_str, exec_globals)
            discovered_law = exec_globals.get("discovered_law")
            if discovered_law is None:
                return {"rmsle": float("nan"), "exact_accuracy": 0.0,
                        "symbolic_equivalent": None,
                        "error": "No 'discovered_law' function found in law_str"}
        except Exception as e:
            return {"rmsle": float("nan"), "exact_accuracy": 0.0,
                    "symbolic_equivalent": None,
                    "error": f"Failed to compile discovered law: {e}"}

        try:
            y_pred = np.array([
                discovered_law(*[test_samples[name][i] for name in param_names])
                for i in range(n_test)
            ])
        except Exception as e:
            return {"rmsle": float("nan"), "exact_accuracy": 0.0,
                    "symbolic_equivalent": None,
                    "error": f"Failed to evaluate discovered law: {e}"}

        # RMSLE (same formula as NewtonBench's calculate_rmsle)
        y_t = np.abs(y_true)
        y_p = np.abs(y_pred)
        mask = ~np.isnan(y_t) & ~np.isnan(y_p)
        rmsle = float(np.sqrt(np.mean((np.log1p(y_p[mask]) - np.log1p(y_t[mask])) ** 2))) if mask.any() else float("nan")

        return {
            "rmsle": rmsle,
            "exact_accuracy": 1.0 if rmsle < 0.01 else 0.0,
            "symbolic_equivalent": None,  # skipping LLM judge — no nemotron key
        }


def get_oracle(
    domain_id: str,
    difficulty: str = "easy",
    noise_level: float = 0.01,
    law_version: str | None = None,
    newtonbench_path: Path | None = None,
    grammar_mode: str = "domain_specific",
) -> BaseOracle:
    """
    Factory function for all supported oracle backends.

    grammar_mode — ChemBench only; controls what hypothesis grammar the LLM sees:
      "domain_specific" — per-domain mechanism hint (medium leakage)
      "universal"       — all possible mechanisms listed; LLM must identify from data
      "none"            — no grammar (completely blind)
    """
    if domain_id.startswith("c") and "_" in domain_id and domain_id.split("_")[0][1:].isdigit():
        from autoscilab.oracle.chembench import ChemBenchOracle
        return ChemBenchOracle(
            domain_id=domain_id,
            difficulty=difficulty,
            noise_level=noise_level,
            law_version=law_version,
            grammar_mode=grammar_mode,
        )
    if domain_id.startswith("g") and "_" in domain_id and domain_id.split("_")[0][1:].isdigit():
        from autoscilab.oracle.grnbench import GRNBenchOracle
        return GRNBenchOracle(
            domain_id=domain_id,
            difficulty=difficulty,
            noise_level=noise_level,
            law_version=law_version,
            grammar_mode=grammar_mode,
        )
    if domain_id.startswith("chemgym_"):
        return build_chemgym_oracle(domain_id, noise_level=noise_level)
    if domain_id.startswith("olympus_"):
        return build_olympus_oracle(domain_id, noise_level=noise_level)
    if domain_id.startswith("summit_"):
        return build_summit_oracle(domain_id, noise_level=noise_level)
    if domain_id.startswith("scigym_"):
        return SciGymOracle(domain_id=domain_id, noise_level=noise_level)
    return NewtonBenchOracle(
        domain_id=domain_id,
        difficulty=difficulty,
        noise_level=noise_level,
        law_version=law_version,
        newtonbench_path=newtonbench_path,
    )
