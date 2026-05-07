"""LAMMPS-based oracles for materials science benchmarks.

Two domains:
  lammps_eos — Equation of State: V (Å³/atom) → E/atom (eV)
                Ground truth: Birch-Murnaghan EOS (Cu, Cu_u3.eam potential)

  lammps_gb  — Grain Boundary Energy: tilt angle θ (°) → γ (J/m²)
               Ground truth: Read-Shockley model (Cu symmetric tilt GB)

Both oracles default to pre-cached LAMMPS data + cubic spline interpolation
(fast, no MPI dependency). Pass use_lammps=True to run live LAMMPS instead
(requires MPI + LAMMPS Python package, slow for GB: ~60-120s per angle).
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from autoscilab.oracle.base import BaseOracle, OracleResult

# ── Paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent / "data"
_EAM_FILE = _DATA_DIR / "Cu_u3.eam"

# ── EOS ground truth (fitted BM parameters from Cu_u3.eam) ────────────────────
_EOS_E0 = -3.53996       # eV/atom
_EOS_V0 = 11.81074       # Å³/atom
_EOS_B0_EV = 138.04 / 160.217662   # eV/Å³  (138.04 GPa → eV/Å³)
_EOS_B0_PRIME = 4.299

# ── GB ground truth (pre-computed LAMMPS data, angles 2–36° step 2°) ──────────
_GB_ANGLES_DEG = np.array([2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36], dtype=float)
_GB_ENERGIES_J = np.array([
    0.2745, 0.4553, 0.4580, 0.5681, 0.6767, 0.9099, 0.9570, 0.8509,
    1.0152, 0.8141, 1.0570, 1.0127, 0.9805, 1.0770, 0.9818, 1.0675,
    1.0873, 1.0888,
])
# Read-Shockley fitted parameters
_RS_E0 = 1.8943
_RS_A  = 0.4187

# LAMMPS GB bicrystal parameters (must match gb.py)
_A_LAT = 3.615
_E_BULK = -3.54
_BOX_HALF_WIDTH = 60.0
_MEASURE_WIDTH = 40.0
_Z_DEPTH = _A_LAT * 2


# ── Helper: Birch-Murnaghan EOS ────────────────────────────────────────────────
def _bm_eos(V, E0, V0, B0_eV, B0_prime):
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0_eV / 16.0) * (
        (eta - 1.0) ** 3 * B0_prime + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def _read_shockley(theta_deg, E0, A):
    theta_rad = np.radians(theta_deg)
    return E0 * theta_rad * (A - np.log(theta_rad))


# ── EOS pre-cached LAMMPS data (from eos_results.txt) ─────────────────────────
_EOS_VOLUMES = np.array([
    10.1259, 10.2867, 10.4491, 10.6132, 10.7790, 10.9466, 11.1159, 11.2869,
    11.4596, 11.6341, 11.8104, 11.9885, 12.1683, 12.3499, 12.5333, 12.7185,
    12.9056, 13.0944, 13.2851, 13.4776, 13.6720,
])
_EOS_ENERGIES = np.array([
    -3.404201, -3.431845, -3.455972, -3.476739, -3.494297, -3.508788,
    -3.520356, -3.529133, -3.535250, -3.538832, -3.540000, -3.538870,
    -3.535554, -3.530161, -3.522794, -3.513553, -3.502534, -3.489831,
    -3.475533, -3.459726, -3.442492,
])


# ── EOS Oracle ─────────────────────────────────────────────────────────────────
class LAMMPSEOSOracle(BaseOracle):
    """
    Equation of State oracle for Cu (Cu_u3.eam potential).

    Input:  V — volume per atom in Å³, range [10.0, 14.0]
    Output: energy per atom in eV (negative, deeper well = lower energy)

    Discovery target: E(V) — the Birch-Murnaghan EOS.
    Defaults to pre-cached data + cubic spline (no MPI needed).
    """

    def __init__(self, use_lammps: bool = False):
        self._use_lammps = use_lammps
        self._spline = CubicSpline(_EOS_VOLUMES, _EOS_ENERGIES, extrapolate=True)

    @property
    def domain(self) -> str:
        return "lammps_eos"

    @property
    def parameter_names(self) -> list[str]:
        return ["V"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {"V": (10.0, 14.0)}

    @property
    def function_signature(self) -> str:
        return "def discovered_law(V):"

    @property
    def param_description(self) -> str:
        return (
            "System: Copper (Cu) FCC crystal, Cu_u3.eam EAM potential\n"
            "Input:  V — volume per atom [10.0, 14.0] Å³/atom\n"
            "Output: E — potential energy per atom [eV/atom] (negative)\n"
            "Goal:   discover E(V) — the equation of state\n"
            "Hypothesis grammar:\n"
            "  The EOS typically follows a smooth well-shaped curve: "
            "a minimum at equilibrium volume V0, rising steeply at compression "
            "(V < V0) and more gently at expansion (V > V0).\n"
            "  Classical forms: Birch-Murnaghan (polynomial in (V0/V)^(2/3)), "
            "Murnaghan (power law in V/V0), Vinet (universal EOS).\n"
            "  All share: E'(V0)=0 (minimum), E''(V0)>0 (bulk modulus B0)."
        )

    def run(self, params: dict[str, float]) -> OracleResult:
        V = float(np.clip(params["V"], 10.0, 14.0))
        if self._use_lammps:
            measurement = self._run_lammps(V)
            v_out = V  # LAMMPS returns actual V; for simplicity use input V
        else:
            measurement = float(self._spline(V))
            v_out = V

        return OracleResult(
            params={"V": v_out},
            measurement=measurement,
            domain=self.domain,
            noise_level=0.0,
        )

    def _run_lammps(self, V: float) -> float:
        """Run one LAMMPS EOS point. Requires MPI + LAMMPS Python package."""
        a_lat = (4.0 * V) ** (1.0 / 3.0)
        try:
            from lammps import lammps  # type: ignore
        except ImportError as e:
            raise RuntimeError("lammps Python package not installed") from e

        orig_dir = os.getcwd()
        try:
            os.chdir(str(_DATA_DIR))
            lmp = lammps(cmdargs=["-screen", "none", "-echo", "none", "-log", "none"])
            cmds = f"""
clear
units metal
dimension 3
boundary p p p
atom_style atomic
lattice fcc {a_lat}
region sim_box block 0 4 0 4 0 4
create_box 1 sim_box
create_atoms 1 box
pair_style eam
pair_coeff * * Cu_u3.eam
variable pe_per_atom equal pe/count(all)
thermo 10
run 0
"""
            for line in cmds.split("\n"):
                if line.strip():
                    lmp.command(line.strip())
            e_atom = float(lmp.extract_variable("pe_per_atom", 0, 0))
            lmp.close()
        finally:
            os.chdir(orig_dir)
        return e_atom

    def evaluate_law(self, law_str: str) -> dict[str, Any]:
        """Score discovered law against BM-EOS ground truth on a test grid."""
        test_V = np.linspace(10.0, 14.0, 41)
        gt_E = _bm_eos(test_V, _EOS_E0, _EOS_V0, _EOS_B0_EV, _EOS_B0_PRIME)

        try:
            namespace: dict = {"math": math, "E": math.e, "pi": math.pi, "sqrt": math.sqrt, "log": math.log, "exp": math.exp, "abs": abs}
            exec(law_str, namespace)
            discovered_law = namespace["discovered_law"]
            pred_E = np.array([float(discovered_law(v)) for v in test_V])

            # RMSLE on |energy| (energies are negative)
            rmsle = float(np.sqrt(np.mean(
                (np.log1p(np.abs(pred_E)) - np.log1p(np.abs(gt_E))) ** 2
            )))
            return {"rmsle": rmsle, "exact_accuracy": rmsle < 0.05, "error": None}
        except Exception as exc:
            return {"rmsle": 999.0, "exact_accuracy": False, "error": str(exc)}


# ── GB Oracle ──────────────────────────────────────────────────────────────────
class LAMMPSGBOracle(BaseOracle):
    """
    Grain boundary energy oracle for Cu symmetric tilt bicrystal.

    Input:  theta — tilt angle in degrees, range [2.0, 36.0]
    Output: grain boundary energy in J/m²

    Uses pre-cached LAMMPS data + cubic spline interpolation so that
    evaluation is fast (the full CG minimization takes 60–120s per angle).
    Optionally set use_lammps=True to run live LAMMPS instead.
    """

    def __init__(self, use_lammps: bool = False):
        self._use_lammps = use_lammps
        # Build cubic spline interpolator over cached data
        self._spline = CubicSpline(_GB_ANGLES_DEG, _GB_ENERGIES_J, extrapolate=False)

    @property
    def domain(self) -> str:
        return "lammps_gb"

    @property
    def parameter_names(self) -> list[str]:
        return ["theta"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {"theta": (2.0, 36.0)}

    @property
    def function_signature(self) -> str:
        return "def discovered_law(theta):"

    @property
    def param_description(self) -> str:
        return (
            "System: Cu symmetric tilt grain boundary, Cu_u3.eam EAM potential\n"
            "Input:  theta — tilt misorientation angle [2.0, 36.0] degrees\n"
            "Output: gamma — grain boundary energy [J/m²] (positive)\n"
            "Goal:   discover gamma(theta) — the grain boundary energy law\n"
            "Hypothesis grammar:\n"
            "  For low-angle grain boundaries (theta < ~15°): "
            "Read-Shockley model: gamma = E0 * theta_rad * (A - ln(theta_rad)), "
            "where theta_rad = theta * pi/180.\n"
            "  This arises from a dislocation network model: "
            "energy grows with dislocation density (∝ theta) times a log factor.\n"
            "  At high angles (theta > ~15°): energy saturates and may oscillate "
            "as CSL (coincidence site lattice) orientations appear.\n"
            "  Parameters: E0 [J/m²] ~ 0.5–3, A (dimensionless) ~ 0.3–1.0."
        )

    def _run_lammps(self, theta_deg: float) -> float:
        """Run live LAMMPS bicrystal simulation. Slow (~60-120s)."""
        try:
            from lammps import lammps  # type: ignore
        except ImportError as e:
            raise RuntimeError("lammps Python package not installed") from e

        # Write data file in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.temp")
            _create_bicrystal_data(theta_deg, data_file)

            orig_dir = os.getcwd()
            try:
                os.chdir(str(_DATA_DIR))
                lmp = lammps(cmdargs=["-screen", "none", "-echo", "none", "-log", "none"])
                half = _BOX_HALF_WIDTH
                meas = _MEASURE_WIDTH
                cmds = f"""
units metal
boundary s s p
atom_style atomic
read_data {data_file}
pair_style eam
pair_coeff * * Cu_u3.eam
delete_atoms overlap 1.2 all all
region measure_box block {-meas/2} {meas/2} {-meas/2} {meas/2} INF INF
group measure_group region measure_box
compute pe_atom all pe/atom
compute pe_measure measure_group reduce sum c_pe_atom
thermo 100
min_style cg
minimize 1e-10 1e-10 5000 10000
variable n_measure equal count(measure_group)
variable area equal {meas}*(zhi-zlo)
variable e_bulk equal {_E_BULK}
variable gb_energy_eV equal (c_pe_measure-(v_n_measure*v_e_bulk))/v_area
variable gb_energy_J equal v_gb_energy_eV*16.0217657
"""
                for line in cmds.split("\n"):
                    if line.strip():
                        lmp.command(line.strip())
                energy_J = float(lmp.extract_variable("gb_energy_J", 0, 0))
                lmp.close()
            finally:
                os.chdir(orig_dir)

        return energy_J

    def run(self, params: dict[str, float]) -> OracleResult:
        theta = float(params["theta"])
        theta = np.clip(theta, 2.0, 36.0)

        if self._use_lammps:
            measurement = self._run_lammps(theta)
        else:
            val = self._spline(theta)
            if np.isnan(val):
                # Fallback: nearest-neighbor
                idx = np.argmin(np.abs(_GB_ANGLES_DEG - theta))
                val = _GB_ENERGIES_J[idx]
            measurement = float(val)

        return OracleResult(
            params={"theta": theta},
            measurement=measurement,
            domain=self.domain,
            noise_level=0.0,
        )

    def evaluate_law(self, law_str: str) -> dict[str, Any]:
        """Score discovered law against LAMMPS data points."""
        test_theta = _GB_ANGLES_DEG
        gt_gamma = _GB_ENERGIES_J

        try:
            namespace: dict = {"math": math, "E": math.e, "pi": math.pi, "sqrt": math.sqrt, "log": math.log, "exp": math.exp, "abs": abs}
            exec(law_str, namespace)
            discovered_law = namespace["discovered_law"]
            pred_gamma = np.array([float(discovered_law(t)) for t in test_theta])
            pred_gamma = np.clip(pred_gamma, 1e-8, None)

            rmsle = float(np.sqrt(np.mean(
                (np.log1p(pred_gamma) - np.log1p(gt_gamma)) ** 2
            )))
            return {"rmsle": rmsle, "exact_accuracy": rmsle < 0.15, "error": None}
        except Exception as exc:
            return {"rmsle": 999.0, "exact_accuracy": False, "error": str(exc)}


# ── Bicrystal data generator (mirrors gb.py) ──────────────────────────────────
def _create_bicrystal_data(angle_deg: float, filename: str) -> None:
    theta = np.radians(angle_deg)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * _A_LAT
    half = _BOX_HALF_WIDTH
    z_depth = _Z_DEPTH
    num_cells = int((half * 1.5) / _A_LAT) + 2
    atoms = []

    for ix in range(-num_cells, num_cells):
        for iy in range(-num_cells, num_cells):
            for iz in range(0, int(z_depth / _A_LAT) + 1):
                cell_origin = np.array([ix, iy, iz]) * _A_LAT
                for b in basis:
                    xc, yc, zc = cell_origin + b
                    x1 = xc * np.cos(-theta / 2) - yc * np.sin(-theta / 2)
                    y1 = xc * np.sin(-theta / 2) + yc * np.cos(-theta / 2)
                    if y1 < 0 and -half <= x1 <= half and -half <= y1:
                        atoms.append([x1, y1, zc])
                    x2 = xc * np.cos(theta / 2) - yc * np.sin(theta / 2)
                    y2 = xc * np.sin(theta / 2) + yc * np.cos(theta / 2)
                    if y2 >= 0 and -half <= x2 <= half and y2 <= half:
                        atoms.append([x2, y2, zc])

    with open(filename, "w") as f:
        f.write(f"LAMMPS Cu GB {angle_deg} degrees\n\n")
        f.write(f"{len(atoms)} atoms\n1 atom types\n\n")
        f.write(f"{-half} {half} xlo xhi\n")
        f.write(f"{-half} {half} ylo yhi\n")
        f.write(f"0.0 {z_depth} zlo zhi\n\n")
        f.write("Masses\n\n1 63.546\n\nAtoms\n\n")
        for i, (x, y, z) in enumerate(atoms):
            f.write(f"{i+1} 1 {x:.5f} {y:.5f} {z:.5f}\n")
