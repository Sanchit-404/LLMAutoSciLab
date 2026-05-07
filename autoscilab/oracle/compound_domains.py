"""
Compound enzyme-kinetic domains for ChemBench (c10–c99).

Each domain is a composition of:
  - substrate kinetics: mm | hill | substrate_inh | pingpong
  - inhibitor type:     none | competitive | uncompetitive | noncompetitive | product
  - temperature dep:    none | arrhenius
  - pH dep:             none | bell_curve

Everything is exported as three dicts that chembench.py merges into its own:
  COMPOUND_PARAMS      – {domain_id: {difficulty: {version: params}}}
  COMPOUND_RATE_FNS    – {domain_id: callable(p, C_A, C_I, C_B, C_P, Enz, T, pH) -> float}
  COMPOUND_REGISTRY    – {domain_id: registry_entry_dict}
  COMPOUND_CONSUME_B   – set of domain_ids that consume the B substrate (pingpong)
"""

from __future__ import annotations

import numpy as np
from typing import Callable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_R_GAS   = 8.314    # J/(mol·K)
_T_REF   = 310.0    # K  (reference temperature for Arrhenius)

# ---------------------------------------------------------------------------
# Mechanism building blocks
# ---------------------------------------------------------------------------

def _sub_mm(C_A: float, p: dict) -> float:
    return C_A / (p["Km"] + C_A)

def _sub_hill(C_A: float, p: dict) -> float:
    Kn = p["K_half"] ** p["n"]
    return C_A ** p["n"] / (Kn + C_A ** p["n"])

def _sub_sinh(C_A: float, p: dict) -> float:
    return C_A / (p["Km"] + C_A + C_A ** 2 / p["Ki_s"])

def _sub_pingpong(C_A: float, C_B: float, p: dict) -> float:
    denom = p["KmA"] * C_B + p["KmB"] * C_A + C_A * C_B
    if denom <= 0:
        return 0.0
    return C_A * C_B / denom

def _temp_arrhenius(T: float, p: dict) -> float:
    return float(np.exp(-p["Ea"] / _R_GAS * (1.0 / T - 1.0 / _T_REF)))

def _ph_bell(pH: float, p: dict) -> float:
    return 1.0 / (1.0 + 10 ** (p["pKa1"] - pH) + 10 ** (pH - p["pKa2"]))

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_compound_rate_fn(
    substrate: str,
    inhibitor: str,
    temperature: str,
    ph_dep: str,
) -> Callable:
    """
    Build a compound rate function:
        rate_fn(p, C_A, C_I, C_B, C_P, Enz, T, pH) -> float
    """

    def rate_fn(p, C_A, C_I, C_B, C_P, Enz, T, pH):
        kcat_base = p.get("kcat_ref", p.get("kcat", 5.0))

        # Temperature / pH multipliers on kcat
        t_factor  = _temp_arrhenius(T, p) if temperature == "arrhenius" else 1.0
        ph_factor = _ph_bell(pH, p)        if ph_dep == "bell_curve"    else 1.0
        kcat = kcat_base * t_factor * ph_factor

        # Substrate term (normalised, without kcat*Enz)
        if substrate == "pingpong":
            r_sub = _sub_pingpong(C_A, C_B, p)
        elif substrate == "hill":
            K_eff = p["K_half"] * (1.0 + C_I / p["Ki"]) if inhibitor == "competitive" else p["K_half"]
            r_sub = _sub_hill_eff(C_A, K_eff, p["n"])
        elif substrate in ("mm", "substrate_inh"):
            # Competitive or product inhibition modifies Km
            if inhibitor == "competitive":
                Km_app = p["Km"] * (1.0 + C_I / p["Ki"])
            elif inhibitor == "product":
                Km_app = p["Km"] * (1.0 + C_P / p["Kp"])
            else:
                Km_app = p["Km"]

            if substrate == "substrate_inh":
                Ki_s = p["Ki_s"]
                r_sub = C_A / (Km_app + C_A + C_A ** 2 / Ki_s)
            else:
                if inhibitor == "uncompetitive":
                    # denominator: Km + C_A*(1 + C_I/Ki)
                    denom = p["Km"] + C_A * (1.0 + C_I / p["Ki"])
                    r_sub = C_A / denom if denom > 0 else 0.0
                else:
                    r_sub = C_A / (Km_app + C_A) if (Km_app + C_A) > 0 else 0.0
        else:
            r_sub = 0.0

        # Noncompetitive multiplier (applies to mm and substrate_inh only for pingpong
        # we skip since pingpong doesn't have noncompetitive mode here)
        noncomp = 1.0 / (1.0 + C_I / p["Ki"]) if inhibitor == "noncompetitive" else 1.0

        return kcat * Enz * r_sub * noncomp

    return rate_fn


def _sub_hill_eff(C_A: float, K_eff: float, n: float) -> float:
    Kn = K_eff ** n
    return C_A ** n / (Kn + C_A ** n) if (Kn + C_A ** n) > 0 else 0.0

# ---------------------------------------------------------------------------
# Domain spec list  (domain_id, substrate, inhibitor, temperature, ph_dep)
# ---------------------------------------------------------------------------
# c10–c99 = 90 new domains (10 existing c0–c9 stay unchanged)

COMPOUND_DOMAIN_SPECS: list[tuple[str, str, str, str, str]] = [
    # ── MM + modifiers (c10–c22) ──────────────────────────────────────────
    ("c10_mm_competitive_arrhenius",          "mm", "competitive",    "arrhenius", "none"),
    ("c11_mm_competitive_ph",                 "mm", "competitive",    "none",      "bell_curve"),
    ("c12_mm_uncompetitive_arrhenius",        "mm", "uncompetitive",  "arrhenius", "none"),
    ("c13_mm_uncompetitive_ph",               "mm", "uncompetitive",  "none",      "bell_curve"),
    ("c14_mm_noncompetitive_arrhenius",       "mm", "noncompetitive", "arrhenius", "none"),
    ("c15_mm_noncompetitive_ph",              "mm", "noncompetitive", "none",      "bell_curve"),
    ("c16_mm_product_arrhenius",              "mm", "product",        "arrhenius", "none"),
    ("c17_mm_product_ph",                     "mm", "product",        "none",      "bell_curve"),
    ("c18_mm_arrhenius_ph",                   "mm", "none",           "arrhenius", "bell_curve"),
    ("c19_mm_competitive_arrhenius_ph",       "mm", "competitive",    "arrhenius", "bell_curve"),
    ("c20_mm_uncompetitive_arrhenius_ph",     "mm", "uncompetitive",  "arrhenius", "bell_curve"),
    ("c21_mm_noncompetitive_arrhenius_ph",    "mm", "noncompetitive", "arrhenius", "bell_curve"),
    ("c22_mm_product_arrhenius_ph",           "mm", "product",        "arrhenius", "bell_curve"),
    # ── Pingpong + modifiers (c23–c32) ───────────────────────────────────
    ("c23_pingpong_arrhenius",                "pingpong", "none",          "arrhenius", "none"),
    ("c24_pingpong_ph",                       "pingpong", "none",          "none",      "bell_curve"),
    ("c25_pingpong_competitive",              "pingpong", "competitive",   "none",      "none"),
    ("c26_pingpong_noncompetitive",           "pingpong", "noncompetitive","none",      "none"),
    ("c27_pingpong_competitive_arrhenius",    "pingpong", "competitive",   "arrhenius", "none"),
    ("c28_pingpong_arrhenius_ph",             "pingpong", "none",          "arrhenius", "bell_curve"),
    ("c29_pingpong_noncompetitive_arrhenius", "pingpong", "noncompetitive","arrhenius", "none"),
    ("c30_pingpong_competitive_ph",           "pingpong", "competitive",   "none",      "bell_curve"),
    ("c31_pingpong_noncompetitive_ph",        "pingpong", "noncompetitive","none",      "bell_curve"),
    ("c32_pingpong_competitive_arrhenius_ph", "pingpong", "competitive",   "arrhenius", "bell_curve"),
    # ── Hill + modifiers (c33–c47) ────────────────────────────────────────
    ("c33_hill_competitive",                  "hill", "competitive",    "none",      "none"),
    ("c34_hill_uncompetitive",                "hill", "uncompetitive",  "none",      "none"),
    ("c35_hill_noncompetitive",               "hill", "noncompetitive", "none",      "none"),
    ("c36_hill_product",                      "hill", "product",        "none",      "none"),
    ("c37_hill_arrhenius",                    "hill", "none",           "arrhenius", "none"),
    ("c38_hill_ph",                           "hill", "none",           "none",      "bell_curve"),
    ("c39_hill_competitive_arrhenius",        "hill", "competitive",    "arrhenius", "none"),
    ("c40_hill_competitive_ph",               "hill", "competitive",    "none",      "bell_curve"),
    ("c41_hill_noncompetitive_arrhenius",     "hill", "noncompetitive", "arrhenius", "none"),
    ("c42_hill_noncompetitive_ph",            "hill", "noncompetitive", "none",      "bell_curve"),
    ("c43_hill_arrhenius_ph",                 "hill", "none",           "arrhenius", "bell_curve"),
    ("c44_hill_competitive_arrhenius_ph",     "hill", "competitive",    "arrhenius", "bell_curve"),
    ("c45_hill_noncompetitive_arrhenius_ph",  "hill", "noncompetitive", "arrhenius", "bell_curve"),
    ("c46_hill_uncompetitive_arrhenius",      "hill", "uncompetitive",  "arrhenius", "none"),
    ("c47_hill_product_arrhenius",            "hill", "product",        "arrhenius", "none"),
    # ── Substrate inhibition + modifiers (c48–c64) ───────────────────────
    ("c48_sinh_competitive",                  "substrate_inh", "competitive",    "none",      "none"),
    ("c49_sinh_uncompetitive",                "substrate_inh", "uncompetitive",  "none",      "none"),
    ("c50_sinh_noncompetitive",               "substrate_inh", "noncompetitive", "none",      "none"),
    ("c51_sinh_product",                      "substrate_inh", "product",        "none",      "none"),
    ("c52_sinh_arrhenius",                    "substrate_inh", "none",           "arrhenius", "none"),
    ("c53_sinh_ph",                           "substrate_inh", "none",           "none",      "bell_curve"),
    ("c54_sinh_competitive_arrhenius",        "substrate_inh", "competitive",    "arrhenius", "none"),
    ("c55_sinh_competitive_ph",               "substrate_inh", "competitive",    "none",      "bell_curve"),
    ("c56_sinh_noncompetitive_arrhenius",     "substrate_inh", "noncompetitive", "arrhenius", "none"),
    ("c57_sinh_noncompetitive_ph",            "substrate_inh", "noncompetitive", "none",      "bell_curve"),
    ("c58_sinh_arrhenius_ph",                 "substrate_inh", "none",           "arrhenius", "bell_curve"),
    ("c59_sinh_competitive_arrhenius_ph",     "substrate_inh", "competitive",    "arrhenius", "bell_curve"),
    ("c60_sinh_noncompetitive_arrhenius_ph",  "substrate_inh", "noncompetitive", "arrhenius", "bell_curve"),
    ("c61_sinh_product_arrhenius",            "substrate_inh", "product",        "arrhenius", "none"),
    ("c62_sinh_product_ph",                   "substrate_inh", "product",        "none",      "bell_curve"),
    ("c63_sinh_uncompetitive_arrhenius",      "substrate_inh", "uncompetitive",  "arrhenius", "none"),
    ("c64_sinh_uncompetitive_ph",             "substrate_inh", "uncompetitive",  "none",      "bell_curve"),
]

assert len(COMPOUND_DOMAIN_SPECS) == 55, f"Expected 55 compound domains, got {len(COMPOUND_DOMAIN_SPECS)}"

# ---------------------------------------------------------------------------
# Parameter ranges per difficulty
# ---------------------------------------------------------------------------
_PARAM_RANGES: dict[str, dict] = {
    "easy": {
        "kcat":     (3.0,  8.0),
        "Km":       (0.5,  5.0),
        "Ki":       (0.5,  3.0),
        "Ea":       (40000, 60000),
        "K_half":   (1.0,  5.0),
        "n":        (1.5,  2.5),
        "Ki_s":     (20.0, 100.0),
        "pKa_gap":  (2.5,  4.0),
        "pKa_ctr":  (6.5,  7.5),
        "KmA":      (0.5,  3.0),
        "KmB":      (1.0,  5.0),
        "Kp":       (2.0,  10.0),
    },
    "medium": {
        "kcat":     (2.0,  12.0),
        "Km":       (0.1,  10.0),
        "Ki":       (0.2,  5.0),
        "Ea":       (55000, 72000),
        "K_half":   (0.5,  10.0),
        "n":        (1.5,  3.5),
        "Ki_s":     (5.0,  200.0),
        "pKa_gap":  (1.5,  4.0),
        "pKa_ctr":  (6.0,  8.0),
        "KmA":      (0.2,  8.0),
        "KmB":      (0.5,  20.0),
        "Kp":       (0.5,  20.0),
    },
    "hard": {
        "kcat":     (1.0,  15.0),
        "Km":       (0.01, 50.0),
        "Ki":       (0.05, 0.3),
        "Ea":       (70000, 85000),
        "K_half":   (0.1,  50.0),
        "n":        (2.0,  6.0),
        "Ki_s":     (1.0,  500.0),
        "pKa_gap":  (0.8,  2.5),
        "pKa_ctr":  (5.5,  8.5),
        "KmA":      (0.05, 50.0),
        "KmB":      (0.05, 50.0),
        "Kp":       (0.05, 0.5),
    },
}

_N_VERSIONS = 6  # v0–v5


def _generate_params(
    domain_id: str,
    substrate: str,
    inhibitor: str,
    temperature: str,
    ph_dep: str,
) -> dict:
    """Return {difficulty: {version: params}} for v0–v5."""
    result: dict = {}
    for difficulty, ranges in _PARAM_RANGES.items():
        seed_key = f"{domain_id}|{difficulty}"
        seed_int = int.from_bytes(
            __import__("hashlib").sha256(seed_key.encode()).digest()[:4], "little"
        )
        rng = np.random.default_rng(seed_int)
        versions: dict = {}
        for vi in range(_N_VERSIONS):
            p: dict = {}
            # kcat (or kcat_ref for Arrhenius)
            kcat_key = "kcat_ref" if temperature == "arrhenius" else "kcat"
            p[kcat_key] = float(rng.uniform(*ranges["kcat"]))

            # Substrate params
            if substrate in ("mm", "substrate_inh"):
                p["Km"] = float(rng.uniform(*ranges["Km"]))
            if substrate == "substrate_inh":
                p["Ki_s"] = float(rng.uniform(*ranges["Ki_s"]))
            if substrate == "hill":
                p["K_half"] = float(rng.uniform(*ranges["K_half"]))
                p["n"]      = float(rng.uniform(*ranges["n"]))
            if substrate == "pingpong":
                p["KmA"] = float(rng.uniform(*ranges["KmA"]))
                p["KmB"] = float(rng.uniform(*ranges["KmB"]))

            # Inhibitor params
            if inhibitor in ("competitive", "uncompetitive", "noncompetitive"):
                p["Ki"] = float(rng.uniform(*ranges["Ki"]))
            if inhibitor == "product":
                if "Km" not in p:
                    p["Km"] = float(rng.uniform(*ranges["Km"]))
                p["Kp"] = float(rng.uniform(*ranges["Kp"]))

            # Temperature
            if temperature == "arrhenius":
                p["Ea"] = float(rng.uniform(*ranges["Ea"]))

            # pH
            if ph_dep == "bell_curve":
                ctr = float(rng.uniform(*ranges["pKa_ctr"]))
                gap = float(rng.uniform(*ranges["pKa_gap"]))
                p["pKa1"] = ctr - gap / 2.0
                p["pKa2"] = ctr + gap / 2.0

            versions[f"v{vi}"] = p
        result[difficulty] = versions
    return result


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _build_equation_str(substrate: str, inhibitor: str, temperature: str, ph_dep: str) -> str:
    """Human-readable rate equation string."""
    # Substrate term
    if substrate == "mm":
        sub = "kcat·Enz·C_A/(Km·(...)·+·C_A)"
    elif substrate == "hill":
        sub = "kcat·Enz·C_A^n/(K_half^n·(...)·+·C_A^n)"
    elif substrate == "substrate_inh":
        sub = "kcat·Enz·C_A/(Km·(...)·+·C_A·+·C_A²/Ki_s)"
    elif substrate == "pingpong":
        sub = "kcat·Enz·C_A·C_B/(KmA·C_B·+·KmB·C_A·+·C_A·C_B)"
    else:
        sub = "kcat·Enz·..."

    mods = []
    if temperature == "arrhenius":
        mods.append("kcat=kcat_ref·exp(-Ea/R·(1/T-1/T_ref))")
    if ph_dep == "bell_curve":
        mods.append("kcat×=1/(1+10^(pKa1-pH)+10^(pH-pKa2))")
    if inhibitor == "competitive":
        mods.append("Km_app=Km·(1+C_I/Ki)")
    elif inhibitor == "uncompetitive":
        mods.append("denom+=C_A·C_I/Ki")
    elif inhibitor == "noncompetitive":
        mods.append("×1/(1+C_I/Ki)")
    elif inhibitor == "product":
        mods.append("Km_app=Km·(1+C_P/Kp)")

    eq = sub
    if mods:
        eq += "  [" + "; ".join(mods) + "]"
    return eq


def _relevant_vars(substrate: str, inhibitor: str, temperature: str, ph_dep: str) -> list[str]:
    vs = ["C_A", "Enz"]
    if substrate == "pingpong":
        vs.append("C_B")
    if inhibitor in ("competitive", "uncompetitive", "noncompetitive"):
        vs.append("C_I")
    if inhibitor == "product":
        vs.append("C_P")
    if temperature == "arrhenius":
        vs.append("T")
    if ph_dep == "bell_curve":
        vs.append("pH")
    return vs


def _build_tags(substrate: str, inhibitor: str, temperature: str, ph_dep: str) -> list[str]:
    tags = [substrate]
    if inhibitor != "none":
        tags.append(inhibitor)
    if temperature != "none":
        tags.append(temperature)
    if ph_dep != "none":
        tags.append(ph_dep)
    return tags


def _build_schema(substrate: str, inhibitor: str, temperature: str, ph_dep: str) -> str:
    parts = ["A → P catalyzed by enzyme Enz"]
    if substrate == "pingpong":
        parts = ["A + B → P catalyzed by enzyme Enz (bisubstrate pingpong)"]
    mods = []
    if inhibitor != "none":
        mods.append(f"inhibitor C_I ({inhibitor})")
    if inhibitor == "product":
        mods.append("product feedback (C_P inhibits)")
    if temperature == "arrhenius":
        mods.append("Arrhenius temperature modulation")
    if ph_dep == "bell_curve":
        mods.append("pH-dependent activity (bell curve)")
    if mods:
        parts.append("with " + "; ".join(mods))
    return " ".join(parts)


def _build_grammar(substrate: str, inhibitor: str, temperature: str, ph_dep: str) -> str:
    lines = []
    if substrate == "mm":
        lines.append("C_A: MM saturation C_A/(Km+C_A)")
    elif substrate == "hill":
        lines.append("C_A: Hill cooperative C_A^n/(K_half^n+C_A^n)")
    elif substrate == "substrate_inh":
        lines.append("C_A: substrate inhibition C_A/(Km+C_A+C_A²/Ki_s)")
    elif substrate == "pingpong":
        lines.append("C_A,C_B: bisubstrate pingpong C_A·C_B/(KmA·C_B+KmB·C_A+C_A·C_B)")

    if inhibitor == "competitive":
        lines.append("C_I: competitive — raises apparent Km")
    elif inhibitor == "uncompetitive":
        lines.append("C_I: uncompetitive — reduces apparent Vmax at high C_A")
    elif inhibitor == "noncompetitive":
        lines.append("C_I: noncompetitive — reduces Vmax at all C_A")
    elif inhibitor == "product":
        lines.append("C_P: product feedback inhibition — raises apparent Km ∝ C_P/Kp")
    else:
        lines.append("C_I: no inhibitor effect; C_P: no product feedback")

    if temperature == "arrhenius":
        lines.append("T: Arrhenius exp(-Ea/R·(1/T-1/T_ref))")
    else:
        lines.append("T: no primary temperature dependence")

    if ph_dep == "bell_curve":
        lines.append("pH: bell curve ionization with pKa1 and pKa2")
    else:
        lines.append("pH: no primary pH dependence")

    lines.append("Enz: linear scaling (r ∝ Enz)")
    return "\n".join(lines)


def get_compound_law_str(domain_id: str, params: dict) -> str:
    """Return a Python 'discovered_law' function string for a compound domain."""
    spec = next((s for s in COMPOUND_DOMAIN_SPECS if s[0] == domain_id), None)
    if spec is None:
        raise ValueError(f"Unknown compound domain: {domain_id}")
    _, substrate, inhibitor, temperature, ph_dep = spec
    p = params

    sig = "def discovered_law(C_A, C_I, C_B, C_P, Enz, T, pH):"
    lines = [sig, "    import numpy as np"]

    # kcat with optional Arrhenius and pH modifiers
    kcat_val = p.get("kcat_ref", p.get("kcat", 5.0))
    if temperature == "arrhenius":
        lines.append(f"    kcat = {kcat_val} * np.exp(-{p['Ea']} / 8.314 * (1/T - 1/310.0))")
    else:
        lines.append(f"    kcat = {kcat_val}")

    if ph_dep == "bell_curve":
        lines.append(f"    kcat = kcat / (1 + 10**({p['pKa1']} - pH) + 10**(pH - {p['pKa2']}))")

    # Inhibition modifiers on Km
    if inhibitor == "competitive":
        if substrate in ("mm", "substrate_inh"):
            lines.append(f"    Km_app = {p['Km']} * (1 + C_I / {p['Ki']})")
        elif substrate == "hill":
            lines.append(f"    K_eff = {p['K_half']} * (1 + C_I / {p['Ki']})")
    elif inhibitor == "product":
        lines.append(f"    Km_app = {p['Km']} * (1 + C_P / {p['Kp']})")
    elif substrate in ("mm", "substrate_inh"):
        lines.append(f"    Km_app = {p['Km']}")

    # Substrate term
    if substrate == "mm":
        if inhibitor == "uncompetitive":
            lines.append(f"    r_sub = C_A / ({p['Km']} + C_A * (1 + C_I / {p['Ki']}))")
        else:
            lines.append("    r_sub = C_A / (Km_app + C_A)")
    elif substrate == "substrate_inh":
        if inhibitor == "uncompetitive":
            lines.append(f"    r_sub = C_A / ({p['Km']} + C_A * (1 + C_I / {p['Ki']}) + C_A**2 / {p['Ki_s']})")
        else:
            lines.append(f"    r_sub = C_A / (Km_app + C_A + C_A**2 / {p['Ki_s']})")
    elif substrate == "hill":
        if inhibitor == "competitive":
            lines.append(f"    r_sub = C_A**{p['n']} / (K_eff**{p['n']} + C_A**{p['n']})")
        else:
            lines.append(f"    r_sub = C_A**{p['n']} / ({p['K_half']}**{p['n']} + C_A**{p['n']})")
    elif substrate == "pingpong":
        lines.append(f"    denom = {p['KmA']} * C_B + {p['KmB']} * C_A + C_A * C_B")
        lines.append("    r_sub = C_A * C_B / denom if denom > 0 else 0.0")

    # Noncompetitive multiplier
    if inhibitor == "noncompetitive":
        lines.append(f"    noncomp = 1.0 / (1 + C_I / {p['Ki']})")
        lines.append("    return kcat * Enz * r_sub * noncomp")
    else:
        lines.append("    return kcat * Enz * r_sub")

    return "\n".join(lines)


def _infer_bo_failure(inhibitor: str, temperature: str, ph_dep: str) -> str:
    modes = []
    if inhibitor in ("competitive", "noncompetitive"):
        modes.append("anti_alignment_inhibitor")
    elif inhibitor == "uncompetitive":
        modes.append("regime_collapse_uncompetitive")
    elif inhibitor == "product":
        modes.append("anti_alignment_product_strict")
    if temperature == "arrhenius":
        modes.append("anti_alignment_temperature")
    if ph_dep == "bell_curve":
        modes.append("anti_alignment_ph")
    return "; ".join(modes) if modes else "none_expected"


def _make_registry_entry(domain_id: str, substrate: str, inhibitor: str,
                          temperature: str, ph_dep: str) -> dict:
    return {
        "equation":          _build_equation_str(substrate, inhibitor, temperature, ph_dep),
        "relevant_vars":     _relevant_vars(substrate, inhibitor, temperature, ph_dep),
        "tags":              _build_tags(substrate, inhibitor, temperature, ph_dep),
        "bo_failure_mode":   _infer_bo_failure(inhibitor, temperature, ph_dep),
        "reaction_schema":   _build_schema(substrate, inhibitor, temperature, ph_dep),
        "hypothesis_grammar": _build_grammar(substrate, inhibitor, temperature, ph_dep),
    }


# ===========================================================================
# NOVEL / EXOTIC MECHANISMS  (c65–c99)
# ===========================================================================
# Each novel mechanism introduces a functional form that is OUTSIDE the
# standard {MM, Hill, substrate-inhibition, ping-pong} × {competitive,
# uncompetitive, noncompetitive, product} × {Arrhenius, pH-bell} library.
# Citations are given for each mechanism family.
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared helpers for novel mechanisms
# ---------------------------------------------------------------------------

def _get_kcat(p: dict, T: float, pH: float, temp: str, ph_dep: str) -> float:
    kcat = p.get("kcat_ref", p.get("kcat", 5.0))
    if temp == "arrhenius":
        kcat *= _temp_arrhenius(T, p)
    if ph_dep == "bell_curve":
        kcat *= _ph_bell(pH, p)
    if ph_dep == "monotonic_alkaline":
        # Single pKa: activity increases with pH (alkaline optimum)
        # ref: Dixon & Webb (1979) Enzymes, 3rd ed., Academic Press
        kcat /= (1.0 + 10.0 ** (p["pKa"] - pH))
    return kcat


# ---------------------------------------------------------------------------
# Novel rate functions — one per mechanism family
# ---------------------------------------------------------------------------

def _novel_ordered_bi_bi(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Ordered Sequential Bisubstrate (Ordered Bi Bi).
    Both substrates must bind in order (A first, then B) before catalysis.
    Denominator has an extra KiA·KmB constant term vs ping-pong.
    Ref: Cleland, W.W. (1963). Biochim Biophys Acta 67:104-137.
    Real examples: lactate dehydrogenase, alcohol dehydrogenase.
    """
    kcat = _get_kcat(p, T, pH, temp, ph)
    KiA, KmA, KmB = p["KiA"], p["KmA"], p["KmB"]
    if inh == "competitive":
        KiA_app = KiA * (1.0 + C_I / p["Ki"])
    else:
        KiA_app = KiA
    denom = KiA_app * KmB + KmB * C_A + KmA * C_B + C_A * C_B
    r = kcat * Enz * C_A * C_B / max(denom, 1e-12)
    if inh == "noncompetitive":
        r /= (1.0 + C_I / p["Ki"])
    return r


def _novel_reversible_mm(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Reversible Michaelis-Menten (Haldane equation).
    Near-equilibrium reactions where C_P drives reverse reaction.
    r = (Vf·C_A/KmA − Vr·C_P/KmP) / (1 + C_A/KmA + C_P/KmP)
    Ref: Haldane, J.B.S. (1930). Enzymes. Longmans, Green & Co., London.
    Real examples: triose phosphate isomerase, phosphoglycerate mutase.
    """
    t_f = _temp_arrhenius(T, p) if temp == "arrhenius" else 1.0
    ph_f = _ph_bell(pH, p) if ph == "bell_curve" else 1.0
    Vf = p["kcat_f"] * Enz * t_f * ph_f
    Vr = p["kcat_r"] * Enz * t_f * ph_f
    KmA = p["KmA"] * (1.0 + C_I / p["Ki"]) if inh == "competitive" else p["KmA"]
    KmP = p["KmP"]
    numer = Vf * C_A / KmA - Vr * C_P / KmP
    denom = 1.0 + C_A / KmA + C_P / KmP
    return numer / max(denom, 1e-12)


def _novel_allosteric_act(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Allosteric Activation (MWC model, simplified).
    C_I is REPURPOSED as activator (positive allostery): rate increases
    with C_I instead of decreasing. Km_app falls as activator binds.
    r = kcat·Enz·C_A/(Km+C_A) · C_I/(Kact+C_I)
    Ref: Monod, J., Wyman, J., Changeux, J.P. (1965). J Mol Biol 12(1):88-118.
    Real examples: CTP synthetase (activated by GTP), pyruvate kinase (FBP).
    """
    kcat = _get_kcat(p, T, pH, temp, ph)
    Km_app = p["Km"] * (1.0 + C_P / p["Ki_inh"]) if inh == "product_feedback" else p["Km"]
    sub_term = C_A / (Km_app + C_A)
    act_term = C_I / (p["Kact"] + C_I)      # C_I = activator here
    return kcat * Enz * sub_term * act_term


def _novel_anticoop_hill(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Anti-cooperative binding (Hill equation with n < 1).
    Binding of one substrate reduces affinity for subsequent molecules.
    Gives a concave saturation curve — initial slope steeper than MM.
    r = kcat·Enz·C_A^n / (K_half^n + C_A^n),  n ∈ (0.3, 0.8)
    Ref: Koshland, D.E., Nemethy, G., Filmer, D. (1966). Biochemistry 5(1):365-385.
    Real examples: some bacterial dehydrogenases, glutamate synthase.
    """
    kcat = _get_kcat(p, T, pH, temp, ph)
    n = p["n"]          # n < 1 (anti-cooperative)
    K_half = p["K_half"]
    if inh == "competitive":
        K_half = K_half * (1.0 + C_I / p["Ki"])
    elif inh == "noncompetitive":
        kcat /= (1.0 + C_I / p["Ki"])
    Kn = K_half ** n
    r_sub = C_A ** n / (Kn + C_A ** n) if (Kn + C_A ** n) > 0 else 0.0
    return kcat * Enz * r_sub


def _novel_fractal(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Fractal / Anomalous Kinetics.
    In crowded or spatially heterogeneous environments the effective
    rate constant is a power law in concentration (non-integer exponent).
    r = k · Enz · C_A^alpha,  alpha ∈ (0.3, 0.9)
    Ref: Kopelman, R. (1988). Science 241(4873):1620-1626.
    Also: Savageau, M.A. (1976). Biochemical Systems Analysis. Addison-Wesley.
    Real context: intracellular enzymes in crowded cytoplasm, membrane enzymes.
    """
    kcat = _get_kcat(p, T, pH, temp, ph)
    alpha = p["alpha"]
    r = kcat * Enz * (C_A ** alpha)
    if inh == "competitive":
        # Inhibitor attenuates the effective rate constant
        r /= (1.0 + (C_I / p["Ki"]) ** p.get("beta", 1.0))
    return r


def _novel_mixed_inh(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Mixed (General) Inhibition.
    Inhibitor affects both Km (like competitive) and Vmax (like noncompetitive)
    simultaneously. Pure competitive and noncompetitive are special cases.
    r = kcat·Enz·C_A / (Km·(1+C_I/Ki) + C_A·(1+C_I/Ki_prime))
    Ref: Segel, I.H. (1975). Enzyme Kinetics. Wiley-Interscience.
    Real examples: most real-world inhibitors exhibit mixed character.
    """
    kcat = _get_kcat(p, T, pH, temp, ph)
    Ki       = p["Ki"]
    Ki_prime = p["Ki_prime"]    # Ki_prime > Ki → more competitive than noncompetitive
    denom = p["Km"] * (1.0 + C_I / Ki) + C_A * (1.0 + C_I / Ki_prime)
    return kcat * Enz * C_A / max(denom, 1e-12)


def _novel_coop_inh(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Cooperative (Hill-type) Inhibition.
    Inhibitor binds allosteric site cooperatively — sigmoid inhibition curve,
    not the hyperbolic curve of classical noncompetitive inhibition.
    r = kcat·Enz·C_A/(Km+C_A) · 1/(1+(C_I/Ki)^n_inh)
    Ref: Monod, Wyman, Changeux (1965) J Mol Biol 12:88  [MWC allosteric]
    Real examples: ATCase (inhibited by CTP cooperatively), many regulatory enzymes.
    """
    kcat = _get_kcat(p, T, pH, temp, ph)
    n_inh = p["n_inh"]          # Hill exponent for inhibitor (> 1)
    sub_term = C_A / (p["Km"] + C_A)
    inh_term = 1.0 / (1.0 + (C_I / p["Ki"]) ** n_inh)
    return kcat * Enz * sub_term * inh_term


def _novel_monotonic_ph(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Monotonic pH Dependence (single ionisation pKa).
    Unlike the bell-curve (two pKas), some enzymes have a single ionisable
    group whose protonation either activates or inactivates the enzyme.
    r = kcat·Enz·C_A/(Km+C_A) / (1 + 10^(pKa − pH))  [alkaline optimum]
    Ref: Dixon, M., Webb, E.C. (1979). Enzymes, 3rd ed. Academic Press.
    Real examples: ribonuclease A (acid optimum ~pH 7), urease (alkaline).
    """
    kcat = p.get("kcat_ref", p.get("kcat", 5.0))
    if temp == "arrhenius":
        kcat *= _temp_arrhenius(T, p)
    # Monotonic: single pKa
    kcat /= (1.0 + 10.0 ** (p["pKa"] - pH))   # alkaline optimum (activity ↑ with pH)
    Km_app = p["Km"] * (1.0 + C_I / p["Ki"]) if inh == "competitive" else p["Km"]
    return kcat * Enz * C_A / (Km_app + C_A)


def _novel_metal_act(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Metal Ion Activation (metalloenzyme cooperative cofactor binding).
    C_B = metal ion cofactor (not consumed, unlike bisubstrate ping-pong).
    Metal binds cooperatively (Hill-type) to the enzyme and is required
    for activity. Rate is the product of substrate and cofactor saturation.
    r = kcat·Enz·C_A/(KmA+C_A) · C_B^n_met/(Km_met^n_met+C_B^n_met)
    Ref: Maret, W. (2016). Int J Mol Sci 17(1):66.
    Real examples: carbonic anhydrase (Zn²⁺), kinases (Mg²⁺), nitrogenase (Mo/Fe).
    """
    kcat = _get_kcat(p, T, pH, temp, ph)
    n_met = p["n_met"]
    sub_term  = C_A / (p["KmA"] + C_A)
    Km_met_n  = p["Km_met"] ** n_met
    CB_n      = C_B ** n_met
    cof_term  = CB_n / (Km_met_n + CB_n) if (Km_met_n + CB_n) > 0 else 0.0
    if inh == "competitive":
        sub_term = C_A / (p["KmA"] * (1.0 + C_I / p["Ki"]) + C_A)
    return kcat * Enz * sub_term * cof_term


def _novel_product_act(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Product Activation (Autocatalysis / Positive Feedback).
    The reaction product activates the enzyme — a positive feedback loop.
    At low C_P: rate ≈ standard MM. Rate increases monotonically with C_P.
    r = kcat·Enz·C_A/(Km+C_A) · (Kthresh + C_P) / Kthresh
    Ref: Goldbeter, A. (1997). Biochemical Oscillations and Cellular Rhythms.
         Cambridge University Press. (Chapter 2, autocatalytic enzyme cycles)
    Real examples: trypsinogen → trypsin (trypsin catalyses its own activation),
                   caspase cascade, prion propagation kinetics.
    """
    kcat = _get_kcat(p, T, pH, temp, ph)
    sub_term = C_A / (p["Km"] + C_A)
    act_term = (p["Kthresh"] + C_P) / p["Kthresh"]   # C_P activates
    if inh == "competitive":
        # C_I competes with substrate (still both effects present)
        sub_term = C_A / (p["Km"] * (1.0 + C_I / p["Ki"]) + C_A)
    return kcat * Enz * sub_term * act_term


def _novel_two_sub_inh(p, C_A, C_I, C_B, C_P, Enz, T, pH, inh, temp, ph):
    """
    Dual-substrate Inhibition (both C_I and C_P inhibit simultaneously).
    C_I raises apparent Km (competitive). C_P reduces apparent Vmax (separate
    noncompetitive-style binding site). Independent inhibition sites.
    r = kcat·Enz·C_A / ((Km·(1+C_I/Ki) + C_A) · (1+C_P/Kp))
    Ref: Segel (1975); also Fromm, H.J. (1979). Methods Enzymol 63:42-53.
    Real examples: many phosphatases and kinases where ADP (product) and a
                   second metabolite both inhibit the enzyme independently.
    """
    kcat = _get_kcat(p, T, pH, temp, ph)
    Km_app = p["Km"] * (1.0 + C_I / p["Ki"])
    vmax_factor = 1.0 + C_P / p["Kp"]
    return kcat * Enz * C_A / max((Km_app + C_A) * vmax_factor, 1e-12)


# ---------------------------------------------------------------------------
# Novel domain registry (domain_id, rate_fn, extra_inh, temp, ph_dep)
# ---------------------------------------------------------------------------
# Column meanings:
#   novel_type  – string key mapping to a _novel_* function above
#   extra_inh   – secondary modifier applied on top of the core novel mechanism
#                 ("none" | "competitive" | "noncompetitive" | "product_feedback")
#   temp        – "none" | "arrhenius"
#   ph_dep      – "none" | "bell_curve" | "monotonic_alkaline"
# ---------------------------------------------------------------------------

_NOVEL_FN_MAP = {
    "ordered_bi_bi":  _novel_ordered_bi_bi,
    "reversible_mm":  _novel_reversible_mm,
    "allosteric_act": _novel_allosteric_act,
    "anticoop_hill":  _novel_anticoop_hill,
    "fractal":        _novel_fractal,
    "mixed_inh":      _novel_mixed_inh,
    "coop_inh":       _novel_coop_inh,
    "monotonic_ph":   _novel_monotonic_ph,
    "metal_act":      _novel_metal_act,
    "product_act":    _novel_product_act,
    "two_sub_inh":    _novel_two_sub_inh,
}

# (domain_id, novel_type, extra_inh, temp, ph_dep)
NOVEL_DOMAIN_SPECS: list[tuple] = [
    # ── Core novel mechanisms (c65–c75) — single novel mechanism, no modifiers ──
    ("c65_ordered_bi_bi",             "ordered_bi_bi",  "none",             "none",      "none"),
    ("c66_reversible_mm",             "reversible_mm",  "none",             "none",      "none"),
    ("c67_allosteric_act",            "allosteric_act", "none",             "none",      "none"),
    ("c68_anticoop_hill",             "anticoop_hill",  "none",             "none",      "none"),
    ("c69_fractal_kinetics",          "fractal",        "none",             "none",      "none"),
    ("c70_mixed_inhibition",          "mixed_inh",      "none",             "none",      "none"),
    ("c71_coop_inhibition",           "coop_inh",       "none",             "none",      "none"),
    ("c72_monotonic_ph",              "monotonic_ph",   "none",             "none",      "none"),
    ("c73_metal_activation",          "metal_act",      "none",             "none",      "none"),
    ("c74_product_activation",        "product_act",    "none",             "none",      "none"),
    ("c75_two_substrate_inhibition",  "two_sub_inh",    "none",             "none",      "none"),

    # ── Novel + Arrhenius temperature dependence (c76–c82) ──────────────────
    ("c76_ordered_bi_bi_arrhenius",   "ordered_bi_bi",  "none",             "arrhenius", "none"),
    ("c77_reversible_mm_arrhenius",   "reversible_mm",  "none",             "arrhenius", "none"),
    ("c78_allosteric_act_arrhenius",  "allosteric_act", "none",             "arrhenius", "none"),
    ("c79_anticoop_arrhenius",        "anticoop_hill",  "none",             "arrhenius", "none"),
    ("c80_mixed_inh_arrhenius",       "mixed_inh",      "none",             "arrhenius", "none"),
    ("c81_coop_inh_arrhenius",        "coop_inh",       "none",             "arrhenius", "none"),
    ("c82_metal_act_arrhenius",       "metal_act",      "none",             "arrhenius", "none"),

    # ── Novel + pH bell curve (c83–c88) ─────────────────────────────────────
    ("c83_ordered_bi_bi_ph",          "ordered_bi_bi",  "none",             "none",      "bell_curve"),
    ("c84_reversible_mm_ph",          "reversible_mm",  "none",             "none",      "bell_curve"),
    ("c85_allosteric_act_ph",         "allosteric_act", "none",             "none",      "bell_curve"),
    ("c86_anticoop_ph",               "anticoop_hill",  "none",             "none",      "bell_curve"),
    ("c87_coop_inh_ph",               "coop_inh",       "none",             "none",      "bell_curve"),
    ("c88_product_act_ph",            "product_act",    "none",             "none",      "bell_curve"),

    # ── Novel + secondary inhibitor/modifier (c89–c94) ──────────────────────
    ("c89_ordered_bi_bi_competitive", "ordered_bi_bi",  "competitive",      "none",      "none"),
    ("c90_ordered_bi_bi_noncomp",     "ordered_bi_bi",  "noncompetitive",   "none",      "none"),
    ("c91_reversible_mm_competitive", "reversible_mm",  "competitive",      "none",      "none"),
    ("c92_allosteric_act_feedback",   "allosteric_act", "product_feedback", "none",      "none"),
    ("c93_fractal_competitive",       "fractal",        "competitive",      "none",      "none"),
    ("c94_metal_act_competitive",     "metal_act",      "competitive",      "none",      "none"),

    # ── Full compound: novel + Arrhenius + pH or inhibitor (c95–c99) ────────
    ("c95_ordered_bi_bi_arr_ph",      "ordered_bi_bi",  "none",             "arrhenius", "bell_curve"),
    ("c96_reversible_mm_arr_ph",      "reversible_mm",  "none",             "arrhenius", "bell_curve"),
    ("c97_allosteric_act_arr_ph",     "allosteric_act", "none",             "arrhenius", "bell_curve"),
    ("c98_two_sub_inh_arrhenius",     "two_sub_inh",    "none",             "arrhenius", "none"),
    ("c99_metal_act_arr_ph",          "metal_act",      "none",             "arrhenius", "bell_curve"),
]

assert len(NOVEL_DOMAIN_SPECS) == 35, f"Expected 35 novel domains, got {len(NOVEL_DOMAIN_SPECS)}"


# ---------------------------------------------------------------------------
# Novel parameter generation
# ---------------------------------------------------------------------------

_NOVEL_PARAM_TEMPLATES: dict[str, dict] = {
    # Each entry: {difficulty: {param: (lo, hi)}}
    "ordered_bi_bi": {
        "easy":   {"kcat": (3,8),   "KiA": (0.5,3),  "KmA": (0.5,3),  "KmB": (1,5),   "Ki": (0.5,3),  "Ea": (40000,60000), "pKa1": (5.8,6.5), "pKa2": (7.5,8.2)},
        "medium": {"kcat": (2,12),  "KiA": (0.2,6),  "KmA": (0.2,8),  "KmB": (0.5,10),"Ki": (0.2,5),  "Ea": (55000,72000), "pKa1": (5.5,6.8), "pKa2": (7.2,8.5)},
        "hard":   {"kcat": (1,15),  "KiA": (0.05,15),"KmA": (0.05,20),"KmB": (0.1,20),"Ki": (0.05,10),"Ea": (70000,85000), "pKa1": (5.0,7.0), "pKa2": (7.0,9.0)},
    },
    "reversible_mm": {
        "easy":   {"kcat_f": (3,8),  "kcat_r": (0.3,1.5), "KmA": (0.5,3), "KmP": (1,8),  "Ki": (0.5,3),  "Ea": (40000,60000), "pKa1": (5.8,6.5), "pKa2": (7.5,8.2)},
        "medium": {"kcat_f": (2,12), "kcat_r": (0.2,3),   "KmA": (0.2,8), "KmP": (0.5,15),"Ki": (0.2,5), "Ea": (55000,72000), "pKa1": (5.5,6.8), "pKa2": (7.2,8.5)},
        "hard":   {"kcat_f": (1,15), "kcat_r": (0.1,5),   "KmA": (0.05,20),"KmP":(0.1,30),"Ki": (0.05,10),"Ea":(70000,85000), "pKa1": (5.0,7.0), "pKa2": (7.0,9.0)},
    },
    "allosteric_act": {
        "easy":   {"kcat": (3,8),  "Km": (0.5,5),  "Kact": (1,5),   "Ki_inh": (2,10), "Ea": (40000,60000), "pKa1": (5.8,6.5), "pKa2": (7.5,8.2)},
        "medium": {"kcat": (2,12), "Km": (0.2,10), "Kact": (0.5,10),"Ki_inh": (0.5,20),"Ea": (55000,72000), "pKa1": (5.5,6.8), "pKa2": (7.2,8.5)},
        "hard":   {"kcat": (1,15), "Km": (0.05,30),"Kact": (0.1,20),"Ki_inh": (0.1,50),"Ea": (70000,85000), "pKa1": (5.0,7.0), "pKa2": (7.0,9.0)},
    },
    "anticoop_hill": {
        "easy":   {"kcat": (3,8),  "K_half": (1,5),  "n": (0.4,0.7),  "Ki": (0.5,3),  "Ea": (40000,60000), "pKa1": (5.8,6.5), "pKa2": (7.5,8.2)},
        "medium": {"kcat": (2,12), "K_half": (0.5,10),"n": (0.35,0.75),"Ki": (0.2,5), "Ea": (55000,72000), "pKa1": (5.5,6.8), "pKa2": (7.2,8.5)},
        "hard":   {"kcat": (1,15), "K_half": (0.1,30),"n": (0.3,0.8),  "Ki": (0.05,10),"Ea": (70000,85000), "pKa1": (5.0,7.0), "pKa2": (7.0,9.0)},
    },
    "fractal": {
        "easy":   {"kcat": (3,8),  "alpha": (0.4,0.75), "Ki": (0.5,3),  "beta": (0.8,1.5), "Ea": (40000,60000), "pKa1": (5.8,6.5), "pKa2": (7.5,8.2)},
        "medium": {"kcat": (2,12), "alpha": (0.35,0.8), "Ki": (0.2,5),  "beta": (0.6,2.0), "Ea": (55000,72000), "pKa1": (5.5,6.8), "pKa2": (7.2,8.5)},
        "hard":   {"kcat": (1,15), "alpha": (0.3,0.9),  "Ki": (0.05,10),"beta": (0.5,2.5), "Ea": (70000,85000), "pKa1": (5.0,7.0), "pKa2": (7.0,9.0)},
    },
    "mixed_inh": {
        "easy":   {"kcat": (3,8),  "Km": (0.5,5),  "Ki": (0.5,3),  "Ki_prime": (3,15), "Ea": (40000,60000), "pKa1": (5.8,6.5), "pKa2": (7.5,8.2)},
        "medium": {"kcat": (2,12), "Km": (0.2,10), "Ki": (0.2,5),  "Ki_prime": (1,30), "Ea": (55000,72000), "pKa1": (5.5,6.8), "pKa2": (7.2,8.5)},
        "hard":   {"kcat": (1,15), "Km": (0.05,30),"Ki": (0.05,10),"Ki_prime": (0.5,50),"Ea": (70000,85000), "pKa1": (5.0,7.0), "pKa2": (7.0,9.0)},
    },
    "coop_inh": {
        "easy":   {"kcat": (3,8),  "Km": (0.5,5),  "Ki": (1,5),   "n_inh": (1.8,3.0), "Ea": (40000,60000), "pKa1": (5.8,6.5), "pKa2": (7.5,8.2)},
        "medium": {"kcat": (2,12), "Km": (0.2,10), "Ki": (0.5,10),"n_inh": (1.5,4.0), "Ea": (55000,72000), "pKa1": (5.5,6.8), "pKa2": (7.2,8.5)},
        "hard":   {"kcat": (1,15), "Km": (0.05,30),"Ki": (0.1,20),"n_inh": (1.3,5.0), "Ea": (70000,85000), "pKa1": (5.0,7.0), "pKa2": (7.0,9.0)},
    },
    "monotonic_ph": {
        "easy":   {"kcat": (3,8),  "Km": (0.5,5),  "Ki": (0.5,3),  "pKa": (6.5,7.5),  "Ea": (40000,60000)},
        "medium": {"kcat": (2,12), "Km": (0.2,10), "Ki": (0.2,5),  "pKa": (6.0,8.0),  "Ea": (55000,72000)},
        "hard":   {"kcat": (1,15), "Km": (0.05,30),"Ki": (0.05,10),"pKa": (5.5,8.5),  "Ea": (70000,85000)},
    },
    "metal_act": {
        "easy":   {"kcat": (3,8),  "KmA": (0.5,5),  "Km_met": (0.5,5),  "n_met": (1.5,2.5), "Ki": (0.5,3),  "Ea": (40000,60000), "pKa1": (5.8,6.5), "pKa2": (7.5,8.2)},
        "medium": {"kcat": (2,12), "KmA": (0.2,10), "Km_met": (0.2,10), "n_met": (1.3,3.0), "Ki": (0.2,5), "Ea": (55000,72000), "pKa1": (5.5,6.8), "pKa2": (7.2,8.5)},
        "hard":   {"kcat": (1,15), "KmA": (0.05,30),"Km_met": (0.05,20),"n_met": (1.2,4.0), "Ki": (0.05,10),"Ea": (70000,85000), "pKa1": (5.0,7.0), "pKa2": (7.0,9.0)},
    },
    "product_act": {
        "easy":   {"kcat": (3,8),  "Km": (0.5,5),  "Kthresh": (2,10),  "Ki": (0.5,3),  "Ea": (40000,60000), "pKa1": (5.8,6.5), "pKa2": (7.5,8.2)},
        "medium": {"kcat": (2,12), "Km": (0.2,10), "Kthresh": (0.5,20),"Ki": (0.2,5),  "Ea": (55000,72000), "pKa1": (5.5,6.8), "pKa2": (7.2,8.5)},
        "hard":   {"kcat": (1,15), "Km": (0.05,30),"Kthresh": (0.1,50),"Ki": (0.05,10),"Ea": (70000,85000), "pKa1": (5.0,7.0), "pKa2": (7.0,9.0)},
    },
    "two_sub_inh": {
        "easy":   {"kcat": (3,8),  "Km": (0.5,5),  "Ki": (0.5,3),  "Kp": (2,10),  "Ea": (40000,60000)},
        "medium": {"kcat": (2,12), "Km": (0.2,10), "Ki": (0.2,5),  "Kp": (0.5,20),"Ea": (55000,72000)},
        "hard":   {"kcat": (1,15), "Km": (0.05,30),"Ki": (0.05,10),"Kp": (0.1,50),"Ea": (70000,85000)},
    },
}

_NOVEL_REQUIRED_PARAMS: dict[str, set] = {
    "ordered_bi_bi":  {"kcat", "KiA", "KmA", "KmB"},
    "reversible_mm":  {"kcat_f", "kcat_r", "KmA", "KmP"},
    "allosteric_act": {"kcat", "Km", "Kact", "Ki_inh"},
    "anticoop_hill":  {"kcat", "K_half", "n"},
    "fractal":        {"kcat", "alpha"},
    "mixed_inh":      {"kcat", "Km", "Ki", "Ki_prime"},
    "coop_inh":       {"kcat", "Km", "Ki", "n_inh"},
    "monotonic_ph":   {"kcat", "Km", "pKa"},
    "metal_act":      {"kcat", "KmA", "Km_met", "n_met"},
    "product_act":    {"kcat", "Km", "Kthresh"},
    "two_sub_inh":    {"kcat", "Km", "Ki", "Kp"},
}

_NOVEL_OPTIONAL_PARAMS: dict[str, dict] = {
    # param_name: needed_when
    "Ki":      ("competitive", "noncompetitive", "Ki"),          # used by many
    "beta":    ("fractal",),                                      # fractal inhibition exponent
    "Ea":      ("arrhenius",),
    "pKa1":    ("bell_curve",),
    "pKa2":    ("bell_curve",),
    "pKa":     ("monotonic_alkaline",),
}


def _generate_novel_params(domain_id: str, novel_type: str,
                            extra_inh: str, temp: str, ph_dep: str) -> dict:
    """Return {difficulty: {version: params}} for v0–v5."""
    template = _NOVEL_PARAM_TEMPLATES[novel_type]
    result: dict = {}
    for difficulty, ranges in template.items():
        seed_key = f"novel|{domain_id}|{difficulty}"
        seed_int = int.from_bytes(
            __import__("hashlib").sha256(seed_key.encode()).digest()[:4], "little"
        )
        rng = np.random.default_rng(seed_int)
        versions: dict = {}
        for vi in range(6):
            p: dict = {}
            # Sample all params in the template for this difficulty
            for param, (lo, hi) in ranges.items():
                p[param] = float(rng.uniform(lo, hi))
            # Rename kcat → kcat_ref when Arrhenius is active
            if temp == "arrhenius" and "kcat" in p:
                p["kcat_ref"] = p.pop("kcat")
            # Extra pKa for bell_curve if not already present from template
            if ph_dep == "bell_curve" and "pKa1" not in p:
                ctr = float(rng.uniform(6.5, 7.5))
                gap = float(rng.uniform(2.5, 4.0))
                p["pKa1"] = ctr - gap / 2.0
                p["pKa2"] = ctr + gap / 2.0
            versions[f"v{vi}"] = p
        result[difficulty] = versions
    return result


# ---------------------------------------------------------------------------
# Novel law string generation
# ---------------------------------------------------------------------------

def _novel_law_str(domain_id: str, novel_type: str, extra_inh: str,
                   temp: str, ph_dep: str, params: dict) -> str:
    """Return a Python 'discovered_law' function string for a novel domain."""
    p = params
    lines = [
        "def discovered_law(C_A, C_I, C_B, C_P, Enz, T, pH):",
        "    import numpy as np",
    ]

    # kcat line
    kcat_val = p.get("kcat_ref", p.get("kcat_f", p.get("kcat", 5.0)))
    kcat_name = "kcat_f" if novel_type == "reversible_mm" else "kcat"
    if temp == "arrhenius":
        lines.append(f"    {kcat_name} = {kcat_val} * np.exp(-{p['Ea']} / 8.314 * (1/T - 1/310.0))")
    else:
        lines.append(f"    {kcat_name} = {kcat_val}")
    if ph_dep == "bell_curve":
        lines.append(f"    {kcat_name} = {kcat_name} / (1 + 10**({p['pKa1']:.4f} - pH) + 10**(pH - {p['pKa2']:.4f}))")

    # Mechanism body
    if novel_type == "ordered_bi_bi":
        KiA_expr = f"{p['KiA']:.4f}" if extra_inh != "competitive" else f"{p['KiA']:.4f} * (1 + C_I / {p['Ki']:.4f})"
        lines.append(f"    denom = {KiA_expr} * {p['KmB']:.4f} + {p['KmB']:.4f} * C_A + {p['KmA']:.4f} * C_B + C_A * C_B")
        r_expr = f"    r = kcat * Enz * C_A * C_B / denom"
        if extra_inh == "noncompetitive":
            r_expr += f" / (1 + C_I / {p['Ki']:.4f})"
        lines.append(r_expr)
        lines.append("    return r")

    elif novel_type == "reversible_mm":
        kr_val = p.get("kcat_r", 1.0)
        if temp == "arrhenius":
            lines.append(f"    kcat_r = {kr_val} * np.exp(-{p['Ea']} / 8.314 * (1/T - 1/310.0))")
        else:
            lines.append(f"    kcat_r = {kr_val}")
        if ph_dep == "bell_curve":
            lines.append(f"    kcat_r = kcat_r / (1 + 10**({p['pKa1']:.4f} - pH) + 10**(pH - {p['pKa2']:.4f}))")
        KmA_expr = f"{p['KmA']:.4f} * (1 + C_I / {p['Ki']:.4f})" if extra_inh == "competitive" else f"{p['KmA']:.4f}"
        lines.append(f"    Vf = kcat_f * Enz;  Vr = kcat_r * Enz")
        lines.append(f"    KmA_app = {KmA_expr}")
        lines.append(f"    numer = Vf * C_A / KmA_app - Vr * C_P / {p['KmP']:.4f}")
        lines.append(f"    denom = 1 + C_A / KmA_app + C_P / {p['KmP']:.4f}")
        lines.append("    return numer / max(denom, 1e-12)")

    elif novel_type == "allosteric_act":
        Km_expr = f"{p['Km']:.4f} * (1 + C_P / {p['Ki_inh']:.4f})" if extra_inh == "product_feedback" else f"{p['Km']:.4f}"
        lines.append(f"    sub_term = C_A / ({Km_expr} + C_A)")
        lines.append(f"    act_term = C_I / ({p['Kact']:.4f} + C_I)   # C_I = activator")
        lines.append(f"    return kcat * Enz * sub_term * act_term")

    elif novel_type == "anticoop_hill":
        K_expr = f"{p['K_half']:.4f} * (1 + C_I / {p['Ki']:.4f})" if extra_inh == "competitive" else f"{p['K_half']:.4f}"
        mult = f" / (1 + C_I / {p['Ki']:.4f})" if extra_inh == "noncompetitive" else ""
        lines.append(f"    K_eff = {K_expr}")
        lines.append(f"    n = {p['n']:.4f}   # n < 1: anti-cooperative")
        lines.append(f"    r_sub = C_A**n / (K_eff**n + C_A**n)")
        lines.append(f"    return kcat{mult} * Enz * r_sub")

    elif novel_type == "fractal":
        alpha = p["alpha"]
        inh_str = ""
        if extra_inh == "competitive":
            beta = p.get("beta", 1.0)
            inh_str = f" / (1 + (C_I / {p['Ki']:.4f})**{beta:.4f})"
        lines.append(f"    return kcat * Enz * C_A**{alpha:.4f}{inh_str}")

    elif novel_type == "mixed_inh":
        lines.append(f"    denom = {p['Km']:.4f} * (1 + C_I / {p['Ki']:.4f}) + C_A * (1 + C_I / {p['Ki_prime']:.4f})")
        lines.append(f"    return kcat * Enz * C_A / max(denom, 1e-12)")

    elif novel_type == "coop_inh":
        lines.append(f"    sub_term = C_A / ({p['Km']:.4f} + C_A)")
        lines.append(f"    inh_term = 1.0 / (1 + (C_I / {p['Ki']:.4f})**{p['n_inh']:.4f})")
        lines.append(f"    return kcat * Enz * sub_term * inh_term")

    elif novel_type == "monotonic_ph":
        # kcat already incorporates pKa above via _novel_monotonic_ph,
        # but for the law string we need to write it inline
        lines[-1] = f"    kcat = {kcat_val} / (1 + 10**({p['pKa']:.4f} - pH))"
        if temp == "arrhenius":
            lines[-1] = (f"    kcat = {kcat_val} * np.exp(-{p['Ea']} / 8.314 * (1/T - 1/310.0))"
                         f" / (1 + 10**({p['pKa']:.4f} - pH))")
        Km_expr = f"{p['Km']:.4f} * (1 + C_I / {p['Ki']:.4f})" if extra_inh == "competitive" else f"{p['Km']:.4f}"
        lines.append(f"    return kcat * Enz * C_A / ({Km_expr} + C_A)")

    elif novel_type == "metal_act":
        n_met = p["n_met"]
        Km_expr = f"{p['KmA']:.4f} * (1 + C_I / {p['Ki']:.4f})" if extra_inh == "competitive" else f"{p['KmA']:.4f}"
        lines.append(f"    sub_term  = C_A / ({Km_expr} + C_A)")
        lines.append(f"    cof_term  = C_B**{n_met:.4f} / ({p['Km_met']:.4f}**{n_met:.4f} + C_B**{n_met:.4f})")
        lines.append(f"    return kcat * Enz * sub_term * cof_term")

    elif novel_type == "product_act":
        Km_expr = f"{p['Km']:.4f} * (1 + C_I / {p['Ki']:.4f})" if extra_inh == "competitive" else f"{p['Km']:.4f}"
        lines.append(f"    sub_term = C_A / ({Km_expr} + C_A)")
        lines.append(f"    act_term = ({p['Kthresh']:.4f} + C_P) / {p['Kthresh']:.4f}  # C_P activates")
        lines.append(f"    return kcat * Enz * sub_term * act_term")

    elif novel_type == "two_sub_inh":
        lines.append(f"    Km_app      = {p['Km']:.4f} * (1 + C_I / {p['Ki']:.4f})")
        lines.append(f"    vmax_factor = 1 + C_P / {p['Kp']:.4f}")
        lines.append(f"    return kcat * Enz * C_A / max((Km_app + C_A) * vmax_factor, 1e-12)")

    else:
        lines.append("    raise NotImplementedError")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Novel registry entry builder
# ---------------------------------------------------------------------------

_NOVEL_DESCRIPTIONS = {
    "ordered_bi_bi":  ("Ordered Sequential Bisubstrate (Ordered Bi Bi)",
                       "Cleland (1963) Biochim Biophys Acta 67:104",
                       ["C_A", "C_B", "Enz"]),
    "reversible_mm":  ("Reversible Michaelis-Menten (Haldane equation)",
                       "Haldane (1930) Enzymes. Longmans, Green & Co.",
                       ["C_A", "C_P", "Enz"]),
    "allosteric_act": ("Allosteric Activation (MWC simplified, C_I = activator)",
                       "Monod, Wyman, Changeux (1965) J Mol Biol 12:88",
                       ["C_A", "C_I", "Enz"]),
    "anticoop_hill":  ("Anti-cooperative Hill kinetics (n < 1)",
                       "Koshland, Nemethy, Filmer (1966) Biochemistry 5:365",
                       ["C_A", "Enz"]),
    "fractal":        ("Fractal / Anomalous Kinetics (power-law, non-integer exponent)",
                       "Kopelman (1988) Science 241:1620; Savageau (1976)",
                       ["C_A", "Enz"]),
    "mixed_inh":      ("Mixed (General) Inhibition (Ki ≠ Ki_prime)",
                       "Segel (1975) Enzyme Kinetics. Wiley-Interscience.",
                       ["C_A", "C_I", "Enz"]),
    "coop_inh":       ("Cooperative (Hill-type) Inhibition",
                       "Monod, Wyman, Changeux (1965) J Mol Biol 12:88",
                       ["C_A", "C_I", "Enz"]),
    "monotonic_ph":   ("Monotonic pH Dependence (single ionisation pKa)",
                       "Dixon & Webb (1979) Enzymes, 3rd ed. Academic Press.",
                       ["C_A", "Enz"]),
    "metal_act":      ("Metal Ion Cooperative Activation (metalloenzyme)",
                       "Maret (2016) Int J Mol Sci 17:66",
                       ["C_A", "C_B", "Enz"]),
    "product_act":    ("Product Activation / Autocatalysis (positive feedback)",
                       "Goldbeter (1997) Biochemical Oscillations. Cambridge UP.",
                       ["C_A", "C_P", "Enz"]),
    "two_sub_inh":    ("Dual-Substrate Inhibition (C_I competitive, C_P noncompetitive)",
                       "Segel (1975); Fromm (1979) Methods Enzymol 63:42",
                       ["C_A", "C_I", "C_P", "Enz"]),
}


def _make_novel_registry_entry(domain_id: str, novel_type: str,
                                extra_inh: str, temp: str, ph_dep: str) -> dict:
    desc, citation, base_vars = _NOVEL_DESCRIPTIONS[novel_type]
    relevant = list(base_vars)
    if extra_inh in ("competitive", "noncompetitive") and "C_I" not in relevant:
        relevant.append("C_I")
    if extra_inh == "product_feedback" and "C_P" not in relevant:
        relevant.append("C_P")
    if temp == "arrhenius":
        relevant.append("T")
    if ph_dep in ("bell_curve", "monotonic_alkaline"):
        relevant.append("pH")

    tags = [novel_type]
    if extra_inh != "none":
        tags.append(extra_inh)
    if temp != "none":
        tags.append(temp)
    if ph_dep != "none":
        tags.append(ph_dep)

    return {
        "equation":           f"{desc}",
        "citation":           citation,
        "relevant_vars":      relevant,
        "tags":               tags,
        "bo_failure_mode":    "novel_mechanism_outside_library",
        "reaction_schema":    f"Novel: {desc}",
        "hypothesis_grammar": f"Novel mechanism — {desc}\nCitation: {citation}",
        "is_novel":           True,
    }


# ---------------------------------------------------------------------------
# Build exports
# ---------------------------------------------------------------------------

COMPOUND_PARAMS: dict = {}
COMPOUND_RATE_FNS: dict = {}
COMPOUND_REGISTRY: dict = {}
COMPOUND_CONSUME_B: set = set()

for _spec in COMPOUND_DOMAIN_SPECS:
    _did, _sub, _inh, _temp, _ph = _spec
    COMPOUND_PARAMS[_did]   = _generate_params(_did, _sub, _inh, _temp, _ph)
    COMPOUND_RATE_FNS[_did] = make_compound_rate_fn(_sub, _inh, _temp, _ph)
    COMPOUND_REGISTRY[_did] = _make_registry_entry(_did, _sub, _inh, _temp, _ph)
    if _sub == "pingpong":
        COMPOUND_CONSUME_B.add(_did)

# Novel domains
for _nspec in NOVEL_DOMAIN_SPECS:
    _did, _ntype, _einh, _temp, _ph = _nspec
    _fn = _NOVEL_FN_MAP[_ntype]

    COMPOUND_PARAMS[_did]   = _generate_novel_params(_did, _ntype, _einh, _temp, _ph)

    # Wrap the novel function into a closure matching (p, C_A, C_I, C_B, C_P, Enz, T, pH)
    def _make_novel_closure(fn, einh, temp, ph):
        def _novel_rate(p, C_A, C_I, C_B, C_P, Enz, T, pH):
            return fn(p, C_A, C_I, C_B, C_P, Enz, T, pH, einh, temp, ph)
        return _novel_rate

    COMPOUND_RATE_FNS[_did] = _make_novel_closure(_fn, _einh, _temp, _ph)
    COMPOUND_REGISTRY[_did] = _make_novel_registry_entry(_did, _ntype, _einh, _temp, _ph)
    # metal_act and ordered_bi_bi use C_B (not consumed, but required)
    if _ntype in ("metal_act",):
        COMPOUND_CONSUME_B.add(_did)   # treated as active (varied) even if not consumed


# ---------------------------------------------------------------------------
# Law string accessor (works for both compound and novel domains)
# ---------------------------------------------------------------------------

def get_domain_law_str(domain_id: str, params: dict) -> str:
    """Return 'discovered_law' Python function string for any c10–c99 domain."""
    # Novel domain?
    nspec = next((s for s in NOVEL_DOMAIN_SPECS if s[0] == domain_id), None)
    if nspec is not None:
        _, ntype, einh, temp, ph = nspec
        return _novel_law_str(domain_id, ntype, einh, temp, ph, params)
    # Compound domain
    return get_compound_law_str(domain_id, params)
