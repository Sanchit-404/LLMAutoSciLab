"""
ChemBench oracle — enzyme kinetics benchmark for hypothesis-driven discovery.

10 domains designed so BO's acquisition function (EI/UCB) systematically fails:
- Anti-alignment: mechanism-revealing experiments are in low-rate regions EI avoids
- Regime collapse: BO-preferred regions generate data that mimics a simpler mechanism
- Peak invisible: EI clusters near rate maximum, misses shape-revealing tails

All domains share the same 7-input / 6-output interface. The oracle always returns
all 6 outputs; irrelevance is emergent from patterns (e.g. C_B_t == C_B_0 means B
is inert), not hardcoded hiding.

Discovery target: rate law r0 = f(C_A, C_I, C_B, C_P, Enz, T, pH)
Function signature for discovered law: def discovered_law(C_A, C_I, C_B, C_P, Enz, T, pH):
"""
from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

from autoscilab.oracle.base import BaseOracle, OracleResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_R_GAS = 8.314      # J / (mol·K)
_T_REF = 310.0      # K  (reference temperature for Arrhenius domains)
_T_FIXED = 2.0      # min (fixed reaction time for mass-balance outputs)

# All 7 input variables and their bounds
CHEM_INPUT_VARS = ["C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH"]

CHEM_INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "C_A": (0.01, 100.0),   # mM  — substrate A
    "C_I": (0.0,  50.0),    # mM  — inhibitor   (log-uniform not suitable; lower=0)
    "C_B": (0.01, 100.0),   # mM  — substrate B
    "C_P": (0.0,  20.0),    # mM  — initial product
    "Enz": (0.01, 10.0),    # mg/mL — enzyme loading
    "T":   (278.0, 368.0),  # K
    "pH":  (4.0,  10.0),
}

# Variables sampled log-uniformly; rest are sampled linearly
CHEM_LOG_VARS = {"C_A", "C_B", "Enz"}

CHEM_OUTPUT_VARS = ["r0", "C_A_t", "C_B_t", "C_P_t", "C_I_t", "conversion"]

CHEM_RMSLE_THRESHOLD = 0.05   # vs 0.01 for NewtonBench physics

# Universal enzyme kinetics grammar — domain-agnostic biochemistry knowledge.
# Lists ALL possible mechanism families for each variable without revealing
# which one is active in any given domain.  Use grammar_mode="universal" to
# give the LLM this as fair background knowledge (vs "domain_specific" which
# names the active mechanism directly, and "none" which gives nothing).
CHEM_UNIVERSAL_GRAMMAR = (
    "Enzyme kinetic mechanism families to consider for each variable:\n"
    "  Substrate C_A: linear | MM saturation C_A/(Km+C_A) | "
    "Hill cooperative C_A^n/(K^n+C_A^n) | "
    "substrate inhibition (bell curve) C_A/(Km+C_A+C_A^2/Ki)\n"
    "  Inhibitor C_I: none (irrelevant) | "
    "competitive — raises apparent Km, inhibition vanishes at high C_A | "
    "uncompetitive — reduces apparent Vmax, no effect at low C_A | "
    "noncompetitive (mixed) — reduces Vmax proportionally at all C_A\n"
    "  Temperature T: none (irrelevant) | "
    "Arrhenius exp(-Ea/R*(1/T-1/T_ref)) — exponential in 1/T, "
    "large effect over 278-368 K range\n"
    "  pH: none (irrelevant) | "
    "bell curve — two ionization pKa values define pH optimum; "
    "rate drops sharply below pKa1 or above pKa2\n"
    "  Second substrate C_B: irrelevant | "
    "required — bisubstrate pingpong C_A*C_B/(KmA*C_B+KmB*C_A+C_A*C_B)\n"
    "  Product C_P: irrelevant | "
    "inhibitory feedback — raises apparent Km proportional to C_P/Kp\n"
    "  Enzyme Enz: always linear scaling of rate (r proportional to Enz)\n"
    "\n"
    "How to EXPERIMENTALLY DISCRIMINATE between mechanism candidates:\n"
    "  Inhibition type (C_I relevance + structural form):\n"
    "    Step 1 — Is C_I relevant at all? Run C_I=0 vs C_I at maximum. "
    "If no rate change, C_I is irrelevant; stop here.\n"
    "    Step 2 — If relevant, sweep C_A from well below Km to well above Km "
    "at both C_I=0 and C_I high (5-10x estimated Ki):\n"
    "      Competitive:    inhibition shrinks as C_A increases "
    "(ratio r(C_I=0)/r(C_I=high) approaches 1 at high C_A)\n"
    "      Uncompetitive:  inhibition absent when C_A << Km; appears only at high C_A\n"
    "      Noncompetitive: inhibition is constant across the full C_A range\n"
    "  Product inhibition:\n"
    "    Pre-load C_P at maximum (e.g. 15-20 mM) vs C_P=0 while varying C_A. "
    "If rate drops proportionally at all C_A values, product inhibition is present.\n"
    "  Bisubstrate kinetics:\n"
    "    Test C_B at very low (near 0) AND very high values. "
    "Rate must approach zero as C_B → 0 for a true bisubstrate mechanism. "
    "If rate is unchanged by C_B, it is irrelevant.\n"
    "  Temperature dependence:\n"
    "    Test at both 278 K and 368 K; if rate changes less than 2-3x, T is likely irrelevant. "
    "True Arrhenius gives >10x change across this range, and log(rate) is linear in 1/T. "
    "A T coefficient that implies activation energy outside 20-80 kJ/mol is likely spurious fitting.\n"
    "  Substrate inhibition vs MM saturation:\n"
    "    Test C_A at values far above the apparent saturation point. "
    "If rate DECREASES past a peak, substrate inhibition is present; if it plateaus, MM is correct.\n"
    "  Hill cooperativity vs MM:\n"
    "    Compare the sigmoid sharpness: test at C_A = 0.1*K_half and C_A = 10*K_half. "
    "Hill gives a much steeper rise than MM over this range (n>1 compresses the transition).\n"
    "  pH bell curve:\n"
    "    Test at extreme pH values (4-5 and 9-10) as well as near the optimum. "
    "Bell curve rates drop sharply near the pKa values; flat response means pH is irrelevant."
)

# Function signature all discovered laws must match
CHEM_FUNCTION_SIGNATURE = "def discovered_law(C_A, C_I, C_B, C_P, Enz, T, pH):"

# ---------------------------------------------------------------------------
# Law parameter tables
# Each domain: difficulty → version → {param: value}
# ---------------------------------------------------------------------------

_PARAMS: dict[str, dict] = {

    "c0_michaelis_menten": {
        # r0 = kcat * Enz * C_A / (Km + C_A)
        "easy":   {"v0": {"kcat": 5.0,  "Km": 1.0},
                   "v1": {"kcat": 8.0,  "Km": 0.2},
                   "v2": {"kcat": 3.0,  "Km": 10.0}},
        "medium": {"v0": {"kcat": 5.0,  "Km": 0.1},
                   "v1": {"kcat": 10.0, "Km": 50.0},
                   "v2": {"kcat": 2.0,  "Km": 5.0}},
        "hard":   {"v0": {"kcat": 5.0,  "Km": 0.01},
                   "v1": {"kcat": 15.0, "Km": 80.0},
                   "v2": {"kcat": 1.0,  "Km": 0.5}},
    },

    "c1_competitive_inhibition": {
        # r0 = kcat * Enz * C_A / (Km * (1 + C_I/Ki) + C_A)
        "easy":   {"v0": {"kcat": 5.0,  "Km": 2.0,  "Ki": 1.0},
                   "v1": {"kcat": 5.0,  "Km": 1.0,  "Ki": 5.0},
                   "v2": {"kcat": 5.0,  "Km": 5.0,  "Ki": 0.5}},
        "medium": {"v0": {"kcat": 5.0,  "Km": 2.0,  "Ki": 0.3},
                   "v1": {"kcat": 8.0,  "Km": 0.5,  "Ki": 2.0},
                   "v2": {"kcat": 3.0,  "Km": 10.0, "Ki": 0.1}},
        "hard":   {"v0": {"kcat": 5.0,  "Km": 0.1,  "Ki": 0.05},
                   "v1": {"kcat": 10.0, "Km": 50.0, "Ki": 20.0},
                   "v2": {"kcat": 2.0,  "Km": 1.0,  "Ki": 0.01}},
    },

    "c2_product_inhibition": {
        # r0 = kcat * Enz * C_A / (Km * (1 + C_P/Kp) + C_A)
        "easy":   {"v0": {"kcat": 5.0,  "Km": 1.0,  "Kp": 5.0},
                   "v1": {"kcat": 5.0,  "Km": 0.5,  "Kp": 2.0},
                   "v2": {"kcat": 5.0,  "Km": 5.0,  "Kp": 10.0}},
        "medium": {"v0": {"kcat": 5.0,  "Km": 1.0,  "Kp": 0.5},
                   "v1": {"kcat": 8.0,  "Km": 3.0,  "Kp": 1.0},
                   "v2": {"kcat": 3.0,  "Km": 0.2,  "Kp": 0.2}},
        "hard":   {"v0": {"kcat": 5.0,  "Km": 0.5,  "Kp": 0.1},
                   "v1": {"kcat": 10.0, "Km": 20.0, "Kp": 50.0},
                   "v2": {"kcat": 2.0,  "Km": 0.05, "Kp": 0.05}},
    },

    "c3_arrhenius_temperature": {
        # r0 = kcat_ref * exp(-Ea/R * (1/T - 1/T_ref)) * Enz * C_A / (Km + C_A)
        "easy":   {"v0": {"kcat_ref": 5.0,  "Ea": 50000, "Km": 1.0},
                   "v1": {"kcat_ref": 3.0,  "Ea": 45000, "Km": 0.5},
                   "v2": {"kcat_ref": 8.0,  "Ea": 55000, "Km": 5.0}},
        "medium": {"v0": {"kcat_ref": 5.0,  "Ea": 65000, "Km": 1.0},
                   "v1": {"kcat_ref": 10.0, "Ea": 70000, "Km": 2.0},
                   "v2": {"kcat_ref": 2.0,  "Ea": 60000, "Km": 10.0}},
        "hard":   {"v0": {"kcat_ref": 5.0,  "Ea": 80000, "Km": 0.1},
                   "v1": {"kcat_ref": 15.0, "Ea": 85000, "Km": 50.0},
                   "v2": {"kcat_ref": 1.0,  "Ea": 55000, "Km": 0.02}},
    },

    "c4_ph_activity": {
        # r0 = kcat * Enz * C_A / ((Km + C_A) * (1 + 10^(pKa1-pH) + 10^(pH-pKa2)))
        "easy":   {"v0": {"kcat": 10.0, "Km": 1.0,  "pKa1": 5.5, "pKa2": 8.5},
                   "v1": {"kcat": 8.0,  "Km": 0.5,  "pKa1": 6.0, "pKa2": 8.0},
                   "v2": {"kcat": 8.0,  "Km": 5.0,  "pKa1": 5.0, "pKa2": 9.0}},
        "medium": {"v0": {"kcat": 10.0, "Km": 1.0,  "pKa1": 6.5, "pKa2": 7.5},
                   "v1": {"kcat": 6.0,  "Km": 2.0,  "pKa1": 5.8, "pKa2": 8.2},
                   "v2": {"kcat": 12.0, "Km": 0.2,  "pKa1": 6.0, "pKa2": 7.0}},
        "hard":   {"v0": {"kcat": 10.0, "Km": 0.1,  "pKa1": 4.5, "pKa2": 9.5},
                   "v1": {"kcat": 20.0, "Km": 20.0, "pKa1": 6.8, "pKa2": 7.2},
                   "v2": {"kcat": 3.0,  "Km": 0.5,  "pKa1": 5.5, "pKa2": 6.5}},
    },

    "c5_pingpong_bisubstrate": {
        # r0 = kcat * Enz * C_A * C_B / (KmA * C_B + KmB * C_A + C_A * C_B)
        # At high C_B >> KmB: reduces to MM in C_A — EI regime collapse target
        "easy":   {"v0": {"kcat": 5.0,  "KmA": 1.0,  "KmB": 2.0},
                   "v1": {"kcat": 5.0,  "KmA": 0.2,  "KmB": 0.5},
                   "v2": {"kcat": 5.0,  "KmA": 5.0,  "KmB": 10.0}},
        "medium": {"v0": {"kcat": 5.0,  "KmA": 0.5,  "KmB": 5.0},
                   "v1": {"kcat": 8.0,  "KmA": 10.0, "KmB": 1.0},
                   "v2": {"kcat": 3.0,  "KmA": 0.1,  "KmB": 20.0}},
        "hard":   {"v0": {"kcat": 5.0,  "KmA": 0.05, "KmB": 50.0},
                   "v1": {"kcat": 10.0, "KmA": 50.0, "KmB": 0.05},
                   "v2": {"kcat": 2.0,  "KmA": 2.0,  "KmB": 0.1}},
    },

    "c6_uncompetitive_inhibition": {
        # r0 = kcat * Enz * C_A / (Km + C_A * (1 + C_I/Ki))
        # At low C_A << Km: inhibitor has NO effect (distinguishes from competitive)
        "easy":   {"v0": {"kcat": 5.0,  "Km": 2.0,  "Ki": 1.0},
                   "v1": {"kcat": 5.0,  "Km": 1.0,  "Ki": 5.0},
                   "v2": {"kcat": 5.0,  "Km": 5.0,  "Ki": 0.5}},
        "medium": {"v0": {"kcat": 5.0,  "Km": 2.0,  "Ki": 0.3},
                   "v1": {"kcat": 8.0,  "Km": 0.5,  "Ki": 2.0},
                   "v2": {"kcat": 3.0,  "Km": 10.0, "Ki": 0.1}},
        "hard":   {"v0": {"kcat": 5.0,  "Km": 0.1,  "Ki": 0.05},
                   "v1": {"kcat": 10.0, "Km": 50.0, "Ki": 20.0},
                   "v2": {"kcat": 2.0,  "Km": 1.0,  "Ki": 0.01}},
    },

    "c7_substrate_inhibition": {
        # r0 = kcat * Enz * C_A / (Km + C_A + C_A**2 / Ki)
        # Bell curve peak at C_A_opt = sqrt(Km * Ki) — EI clusters near peak
        "easy":   {"v0": {"kcat": 8.0,  "Km": 0.5,  "Ki": 50.0},
                   "v1": {"kcat": 8.0,  "Km": 1.0,  "Ki": 20.0},
                   "v2": {"kcat": 8.0,  "Km": 2.0,  "Ki": 100.0}},
        "medium": {"v0": {"kcat": 8.0,  "Km": 0.5,  "Ki": 5.0},
                   "v1": {"kcat": 5.0,  "Km": 0.2,  "Ki": 1.0},
                   "v2": {"kcat": 10.0, "Km": 5.0,  "Ki": 200.0}},
        "hard":   {"v0": {"kcat": 8.0,  "Km": 0.1,  "Ki": 0.5},
                   "v1": {"kcat": 15.0, "Km": 10.0, "Ki": 500.0},
                   "v2": {"kcat": 3.0,  "Km": 0.02, "Ki": 0.1}},
    },

    "c8_hill_cooperativity": {
        # r0 = kcat * Enz * C_A**n / (K_half**n + C_A**n)
        # Sharp sigmoidal — EI concentrates in transition zone, misses plateaus
        "easy":   {"v0": {"kcat": 5.0,  "K_half": 2.0,  "n": 2.0},
                   "v1": {"kcat": 5.0,  "K_half": 1.0,  "n": 3.0},
                   "v2": {"kcat": 5.0,  "K_half": 5.0,  "n": 1.5}},
        "medium": {"v0": {"kcat": 5.0,  "K_half": 2.0,  "n": 4.0},
                   "v1": {"kcat": 8.0,  "K_half": 0.5,  "n": 2.5},
                   "v2": {"kcat": 3.0,  "K_half": 10.0, "n": 3.0}},
        "hard":   {"v0": {"kcat": 5.0,  "K_half": 0.1,  "n": 5.0},
                   "v1": {"kcat": 10.0, "K_half": 50.0, "n": 2.0},
                   "v2": {"kcat": 2.0,  "K_half": 1.0,  "n": 6.0}},
    },

    "c9_noncompetitive_inhibition": {
        # r0 = kcat * Enz * C_A / ((1 + C_I/Ki) * (Km + C_A))
        # Inhibitor reduces rate at ALL C_A values — distinguishes from c1 and c6
        "easy":   {"v0": {"kcat": 5.0,  "Km": 2.0,  "Ki": 1.0},
                   "v1": {"kcat": 5.0,  "Km": 1.0,  "Ki": 5.0},
                   "v2": {"kcat": 5.0,  "Km": 5.0,  "Ki": 0.5}},
        "medium": {"v0": {"kcat": 5.0,  "Km": 2.0,  "Ki": 0.3},
                   "v1": {"kcat": 8.0,  "Km": 0.5,  "Ki": 2.0},
                   "v2": {"kcat": 3.0,  "Km": 10.0, "Ki": 0.1}},
        "hard":   {"v0": {"kcat": 5.0,  "Km": 0.1,  "Ki": 0.05},
                   "v1": {"kcat": 10.0, "Km": 50.0, "Ki": 20.0},
                   "v2": {"kcat": 2.0,  "Km": 1.0,  "Ki": 0.01}},
    },
}

# ---------------------------------------------------------------------------
# Domain registry — metadata, BO failure mode, task description
# ---------------------------------------------------------------------------

CHEM_DOMAIN_REGISTRY: dict[str, dict] = {
    "c0_michaelis_menten": {
        "equation":        "r0 = kcat * Enz * C_A / (Km + C_A)",
        "relevant_vars":   ["C_A", "Enz"],
        "tags":            ["saturation"],
        "bo_failure_mode": "none",
        "reaction_schema": "A → P  (catalyzed by E)",
        "hypothesis_grammar": (
            "Substrate C_A: saturation (Michaelis-Menten) | "
            "Inhibitor C_I: none | Temperature: none | pH: none | "
            "Second substrate C_B: irrelevant | Product C_P: irrelevant | "
            "Enzyme E: linear scaling"
        ),
    },
    "c1_competitive_inhibition": {
        "equation":        "r0 = kcat * Enz * C_A / (Km * (1 + C_I/Ki) + C_A)",
        "relevant_vars":   ["C_A", "C_I", "Enz"],
        "tags":            ["saturation", "competitive_inhibition"],
        "bo_failure_mode": "anti_alignment",
        "reaction_schema": "A → P  (catalyzed by E; I competes with A for active site)",
        "hypothesis_grammar": (
            "Substrate C_A: saturation (MM) | "
            "Inhibitor C_I: competitive (raises apparent Km; effect vanishes at high C_A) | "
            "Temperature: none | pH: none | "
            "Second substrate C_B: irrelevant | Product C_P: irrelevant | "
            "Enzyme E: linear scaling"
        ),
    },
    "c2_product_inhibition": {
        "equation":        "r0 = kcat * Enz * C_A / (Km * (1 + C_P/Kp) + C_A)",
        "relevant_vars":   ["C_A", "C_P", "Enz"],
        "tags":            ["saturation", "product_inhibition"],
        "bo_failure_mode": "anti_alignment",
        "reaction_schema": "A → P  (catalyzed by E; product P feeds back to inhibit)",
        "hypothesis_grammar": (
            "Substrate C_A: saturation (MM) | "
            "Inhibitor C_I: irrelevant | "
            "Product C_P: inhibitory feedback (raises apparent Km) | "
            "Temperature: none | pH: none | "
            "Second substrate C_B: irrelevant | "
            "Enzyme E: linear scaling"
        ),
    },
    "c3_arrhenius_temperature": {
        "equation":        "r0 = kcat_ref * exp(-Ea/R * (1/T - 1/T_ref)) * Enz * C_A / (Km + C_A)",
        "relevant_vars":   ["C_A", "T", "Enz"],
        "tags":            ["saturation", "arrhenius"],
        "bo_failure_mode": "anti_alignment",
        "reaction_schema": "A → P  (catalyzed by E; rate is strongly temperature-dependent)",
        "hypothesis_grammar": (
            "Substrate C_A: saturation (MM) | "
            "Temperature T: Arrhenius (exponential in 1/T) | "
            "Inhibitor C_I: none | pH: none | "
            "Second substrate C_B: irrelevant | Product C_P: irrelevant | "
            "Enzyme E: linear scaling"
        ),
    },
    "c4_ph_activity": {
        "equation":        "r0 = kcat * Enz * C_A / ((Km + C_A) * (1 + 10^(pKa1-pH) + 10^(pH-pKa2)))",
        "relevant_vars":   ["C_A", "pH", "Enz"],
        "tags":            ["saturation", "ph_ionization"],
        "bo_failure_mode": "anti_alignment",
        "reaction_schema": "A → P  (catalyzed by E; activity depends on ionization state of active site)",
        "hypothesis_grammar": (
            "Substrate C_A: saturation (MM) | "
            "pH: bell-curve activity (two pKa values define optimal pH) | "
            "Temperature: none | Inhibitor C_I: none | "
            "Second substrate C_B: irrelevant | Product C_P: irrelevant | "
            "Enzyme E: linear scaling"
        ),
    },
    "c5_pingpong_bisubstrate": {
        "equation":        "r0 = kcat * Enz * C_A * C_B / (KmA * C_B + KmB * C_A + C_A * C_B)",
        "relevant_vars":   ["C_A", "C_B", "Enz"],
        "tags":            ["bisubstrate", "pingpong"],
        "bo_failure_mode": "regime_collapse",
        "reaction_schema": "A + B → P  (catalyzed by E; both substrates required)",
        "hypothesis_grammar": (
            "Substrate C_A: required | "
            "Second substrate C_B: required (pingpong bisubstrate mechanism) | "
            "Inhibitor C_I: none | Product C_P: irrelevant | "
            "Temperature: none | pH: none | "
            "Enzyme E: linear scaling"
        ),
    },
    "c6_uncompetitive_inhibition": {
        "equation":        "r0 = kcat * Enz * C_A / (Km + C_A * (1 + C_I/Ki))",
        "relevant_vars":   ["C_A", "C_I", "Enz"],
        "tags":            ["saturation", "uncompetitive_inhibition"],
        "bo_failure_mode": "regime_collapse",
        "reaction_schema": "A → P  (catalyzed by E; I binds only to enzyme-substrate complex)",
        "hypothesis_grammar": (
            "Substrate C_A: saturation (MM) | "
            "Inhibitor C_I: uncompetitive (reduces apparent Vmax at high C_A; no effect at low C_A) | "
            "Temperature: none | pH: none | "
            "Second substrate C_B: irrelevant | Product C_P: irrelevant | "
            "Enzyme E: linear scaling"
        ),
    },
    "c7_substrate_inhibition": {
        "equation":        "r0 = kcat * Enz * C_A / (Km + C_A + C_A**2 / Ki)",
        "relevant_vars":   ["C_A", "Enz"],
        "tags":            ["substrate_inhibition", "bell_curve"],
        "bo_failure_mode": "peak_invisible",
        "reaction_schema": "A → P  (catalyzed by E; excess substrate inhibits — bell-curve rate)",
        "hypothesis_grammar": (
            "Substrate C_A: substrate inhibition bell curve (peak at C_A_opt = sqrt(Km*Ki)) | "
            "Inhibitor C_I: none | Temperature: none | pH: none | "
            "Second substrate C_B: irrelevant | Product C_P: irrelevant | "
            "Enzyme E: linear scaling"
        ),
    },
    "c8_hill_cooperativity": {
        "equation":        "r0 = kcat * Enz * C_A**n / (K_half**n + C_A**n)",
        "relevant_vars":   ["C_A", "Enz"],
        "tags":            ["hill", "cooperative", "sigmoidal"],
        "bo_failure_mode": "peak_invisible",
        "reaction_schema": "A → P  (catalyzed by E; cooperative binding — sigmoidal rate)",
        "hypothesis_grammar": (
            "Substrate C_A: Hill cooperative (sigmoidal, n > 1) | "
            "Inhibitor C_I: none | Temperature: none | pH: none | "
            "Second substrate C_B: irrelevant | Product C_P: irrelevant | "
            "Enzyme E: linear scaling"
        ),
    },
    "c9_noncompetitive_inhibition": {
        "equation":        "r0 = kcat * Enz * C_A / ((1 + C_I/Ki) * (Km + C_A))",
        "relevant_vars":   ["C_A", "C_I", "Enz"],
        "tags":            ["saturation", "noncompetitive_inhibition"],
        "bo_failure_mode": "anti_alignment",
        "reaction_schema": "A → P  (catalyzed by E; I reduces Vmax at all substrate levels)",
        "hypothesis_grammar": (
            "Substrate C_A: saturation (MM) | "
            "Inhibitor C_I: noncompetitive (reduces Vmax proportionally at all C_A values) | "
            "Temperature: none | pH: none | "
            "Second substrate C_B: irrelevant | Product C_P: irrelevant | "
            "Enzyme E: linear scaling"
        ),
    },
}

# ---------------------------------------------------------------------------
# Primary rate law functions
# Each returns r0_primary (no noise, no secondary effects)
# Accept all 7 inputs for uniformity; irrelevant ones ignored
# ---------------------------------------------------------------------------

def _rate_c0(p: dict, C_A, C_I, C_B, C_P, Enz, T, pH) -> float:
    return p["kcat"] * Enz * C_A / (p["Km"] + C_A)

def _rate_c1(p: dict, C_A, C_I, C_B, C_P, Enz, T, pH) -> float:
    return p["kcat"] * Enz * C_A / (p["Km"] * (1.0 + C_I / p["Ki"]) + C_A)

def _rate_c2(p: dict, C_A, C_I, C_B, C_P, Enz, T, pH) -> float:
    return p["kcat"] * Enz * C_A / (p["Km"] * (1.0 + C_P / p["Kp"]) + C_A)

def _rate_c3(p: dict, C_A, C_I, C_B, C_P, Enz, T, pH) -> float:
    arr = np.exp(-p["Ea"] / _R_GAS * (1.0 / T - 1.0 / _T_REF))
    return p["kcat_ref"] * arr * Enz * C_A / (p["Km"] + C_A)

def _rate_c4(p: dict, C_A, C_I, C_B, C_P, Enz, T, pH) -> float:
    ion = 1.0 + 10.0 ** (p["pKa1"] - pH) + 10.0 ** (pH - p["pKa2"])
    return p["kcat"] * Enz * C_A / ((p["Km"] + C_A) * ion)

def _rate_c5(p: dict, C_A, C_I, C_B, C_P, Enz, T, pH) -> float:
    denom = p["KmA"] * C_B + p["KmB"] * C_A + C_A * C_B
    return p["kcat"] * Enz * C_A * C_B / denom if denom > 0 else 0.0

def _rate_c6(p: dict, C_A, C_I, C_B, C_P, Enz, T, pH) -> float:
    return p["kcat"] * Enz * C_A / (p["Km"] + C_A * (1.0 + C_I / p["Ki"]))

def _rate_c7(p: dict, C_A, C_I, C_B, C_P, Enz, T, pH) -> float:
    return p["kcat"] * Enz * C_A / (p["Km"] + C_A + C_A ** 2 / p["Ki"])

def _rate_c8(p: dict, C_A, C_I, C_B, C_P, Enz, T, pH) -> float:
    n = p["n"]
    Kn = p["K_half"] ** n
    CAn = C_A ** n
    return p["kcat"] * Enz * CAn / (Kn + CAn)

def _rate_c9(p: dict, C_A, C_I, C_B, C_P, Enz, T, pH) -> float:
    return p["kcat"] * Enz * C_A / ((1.0 + C_I / p["Ki"]) * (p["Km"] + C_A))


_RATE_FNS = {
    "c0_michaelis_menten":       _rate_c0,
    "c1_competitive_inhibition": _rate_c1,
    "c2_product_inhibition":     _rate_c2,
    "c3_arrhenius_temperature":  _rate_c3,
    "c4_ph_activity":            _rate_c4,
    "c5_pingpong_bisubstrate":   _rate_c5,
    "c6_uncompetitive_inhibition": _rate_c6,
    "c7_substrate_inhibition":   _rate_c7,
    "c8_hill_cooperativity":     _rate_c8,
    "c9_noncompetitive_inhibition": _rate_c9,
}

# Domains where C_B is consumed (stoichiometric bisubstrate)
_CONSUME_B = {"c5_pingpong_bisubstrate"}

# ---------------------------------------------------------------------------
# Merge compound domains (c10–c99)
# ---------------------------------------------------------------------------
try:
    from autoscilab.oracle.compound_domains import (
        COMPOUND_PARAMS,
        COMPOUND_RATE_FNS,
        COMPOUND_REGISTRY,
        COMPOUND_CONSUME_B,
    )
    _PARAMS            = {**_PARAMS, **COMPOUND_PARAMS}
    _RATE_FNS          = {**_RATE_FNS, **COMPOUND_RATE_FNS}
    CHEM_DOMAIN_REGISTRY = {**CHEM_DOMAIN_REGISTRY, **COMPOUND_REGISTRY}
    _CONSUME_B         = _CONSUME_B | COMPOUND_CONSUME_B
except ImportError as _e:
    import warnings
    warnings.warn(f"compound_domains not loaded: {_e}")

# ---------------------------------------------------------------------------
# Benchmark domain filter
# ---------------------------------------------------------------------------
# Domains excluded from the standard benchmark due to structural complexity:
#   - pH bell curve (two pKa constants): requires discovering 10^(pKa-pH) + 10^(pH-pKa2)
#   - Haldane reversible MM: numerator subtraction (Vf*C_A/KmA - Vr*C_P/KmP)
#   - >5 free parameters
# These are preserved in CHEM_DOMAIN_REGISTRY for research use but excluded
# from CHEM_BENCHMARK_DOMAINS (the standard 54-domain fair benchmark).

CHEM_EXCLUDED_DOMAINS: set[str] = {
    # pH bell curve (double ionisation) — 30 domains
    "c4_ph_activity",
    "c11_mm_competitive_ph", "c13_mm_uncompetitive_ph", "c15_mm_noncompetitive_ph",
    "c17_mm_product_ph", "c18_mm_arrhenius_ph",
    "c19_mm_competitive_arrhenius_ph", "c20_mm_uncompetitive_arrhenius_ph",
    "c21_mm_noncompetitive_arrhenius_ph", "c22_mm_product_arrhenius_ph",
    "c24_pingpong_ph", "c28_pingpong_arrhenius_ph",
    "c30_pingpong_competitive_ph", "c31_pingpong_noncompetitive_ph",
    "c32_pingpong_competitive_arrhenius_ph",
    "c38_hill_ph", "c40_hill_competitive_ph", "c42_hill_noncompetitive_ph",
    "c43_hill_arrhenius_ph", "c44_hill_competitive_arrhenius_ph",
    "c45_hill_noncompetitive_arrhenius_ph",
    "c53_sinh_ph", "c55_sinh_competitive_ph", "c57_sinh_noncompetitive_ph",
    "c58_sinh_arrhenius_ph", "c59_sinh_competitive_arrhenius_ph",
    "c60_sinh_noncompetitive_arrhenius_ph", "c62_sinh_product_ph", "c64_sinh_uncompetitive_ph",
    "c83_ordered_bi_bi_ph", "c84_reversible_mm_ph", "c85_allosteric_act_ph",
    "c86_anticoop_ph", "c87_coop_inh_ph", "c88_product_act_ph",
    "c95_ordered_bi_bi_arr_ph", "c96_reversible_mm_arr_ph",
    "c97_allosteric_act_arr_ph", "c99_metal_act_arr_ph",
    # Haldane reversible MM (numerator subtraction) — 3 additional
    "c66_reversible_mm", "c77_reversible_mm_arrhenius", "c91_reversible_mm_competitive",
    # >5 free parameters — 1 additional
    "c82_metal_act_arrhenius",
}

# The 54 fair benchmark domains (all of CHEM_DOMAIN_REGISTRY minus excluded)
CHEM_BENCHMARK_DOMAINS: list[str] = sorted(
    d for d in CHEM_DOMAIN_REGISTRY if d not in CHEM_EXCLUDED_DOMAINS
)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _secondary_effects(T: float, pH: float) -> float:
    """
    Small background T/pH effects present in all enzymes (realistic biology).
    Maximum ±5% combined — well below RMSLE threshold of 0.05.
    These appear in training data but evaluate_law compares against the clean
    primary law, so correct primary-mechanism discovery always passes.
    """
    # Background Arrhenius: Ea_bg = 1000 J/mol → ±4% over 278–368 K range
    t_factor = np.exp(-1000.0 / _R_GAS * (1.0 / T - 1.0 / _T_REF))
    # Background pH: max −2% at pH extremes (4 or 10)
    ph_factor = 1.0 - 0.0025 * (pH - 7.0) ** 2
    return float(t_factor * ph_factor)


def _deterministic_noise(
    domain: str, difficulty: str, law_version: str,
    inputs: dict[str, float], noise_level: float
) -> float:
    """SHA-256-seeded noise — reproducible for the same (domain, inputs) tuple."""
    if noise_level == 0.0:
        return 0.0
    seed_str = (
        f"{domain}|{difficulty}|{law_version}|"
        + "|".join(f"{k}={v:.8g}" for k, v in sorted(inputs.items()))
    )
    digest = hashlib.sha256(seed_str.encode()).digest()
    seed_int = struct.unpack("<Q", digest[:8])[0] % (2 ** 32)
    return float(np.random.default_rng(seed_int).normal(0.0, noise_level))


def _compute_outputs(
    r0_true: float,
    C_A: float, C_I: float, C_B: float, C_P: float,
    consume_B: bool,
) -> dict[str, float]:
    """Mass-balance outputs from true (noiseless) r0 and fixed reaction time."""
    consumed_A = r0_true * _T_FIXED
    C_A_t = max(C_A - consumed_A, 0.0)
    C_P_t = C_P + (C_A - C_A_t)           # product accumulates from both sources
    C_B_t = max(C_B - (C_A - C_A_t), 0.0) if consume_B else C_B
    C_I_t = C_I                             # inhibitors are not consumed
    conversion = (C_A - C_A_t) / C_A if C_A > 1e-12 else 0.0
    return {
        "C_A_t": C_A_t,
        "C_B_t": C_B_t,
        "C_P_t": C_P_t,
        "C_I_t": C_I_t,
        "conversion": conversion,
    }


# ---------------------------------------------------------------------------
# ChemBenchOracle
# ---------------------------------------------------------------------------

class ChemBenchOracle(BaseOracle):
    """
    Oracle for a single ChemBench domain.

    Parameters
    ----------
    domain_id   : one of CHEM_DOMAIN_REGISTRY keys (e.g. 'c1_competitive_inhibition')
    difficulty  : 'easy' | 'medium' | 'hard'
    noise_level : multiplicative noise std (0.01 = 1%)
    law_version : 'v0' | 'v1' | 'v2' | None (random at construction)
    """

    def __init__(
        self,
        domain_id: str,
        difficulty: str = "easy",
        noise_level: float = 0.01,
        law_version: str | None = None,
        grammar_mode: str = "domain_specific",
    ):
        """
        grammar_mode controls what hypothesis grammar the LLM sees via param_description:
          "domain_specific" — per-domain grammar names the active mechanism (medium leakage)
          "universal"       — all possible mechanisms listed; LLM identifies from data (fair)
          "none"            — no grammar section at all (completely blind)
        """
        if domain_id not in CHEM_DOMAIN_REGISTRY:
            raise ValueError(
                f"Unknown ChemBench domain: {domain_id!r}. "
                f"Choose from: {sorted(CHEM_DOMAIN_REGISTRY)}"
            )
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"difficulty must be 'easy'|'medium'|'hard', got {difficulty!r}")

        self._domain_id = domain_id
        self._difficulty = difficulty
        self._noise_level = noise_level
        self._cfg = CHEM_DOMAIN_REGISTRY[domain_id]
        self._params_table = _PARAMS[domain_id]
        self._rate_fn = _RATE_FNS[domain_id]
        self._consume_B = domain_id in _CONSUME_B
        self._grammar_mode = grammar_mode

        # Fix law version at construction time
        available = list(self._params_table[difficulty].keys())
        if law_version is None:
            import random
            self._law_version = random.choice(available)
        elif law_version in available:
            self._law_version = law_version
        else:
            raise ValueError(
                f"law_version {law_version!r} not available for "
                f"{domain_id}/{difficulty}. Available: {available}"
            )

        self._law_params = self._params_table[difficulty][self._law_version]
        print(
            f"[ChemBenchOracle] {domain_id} | {difficulty} | {self._law_version} | "
            f"noise={noise_level}"
        )

    # --- BaseOracle properties -----------------------------------------------

    @property
    def domain(self) -> str:
        return self._domain_id

    @property
    def parameter_names(self) -> list[str]:
        return CHEM_INPUT_VARS

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(CHEM_INPUT_BOUNDS)

    @property
    def domain_tags(self) -> list[str]:
        return list(self._cfg.get("tags", []))

    @property
    def function_signature(self) -> str:
        return CHEM_FUNCTION_SIGNATURE

    @property
    def param_description(self) -> str:
        schema = self._cfg["reaction_schema"]
        if self._grammar_mode == "universal":
            grammar_text: str | None = CHEM_UNIVERSAL_GRAMMAR
        elif self._grammar_mode == "none":
            grammar_text = None
        else:  # "domain_specific" (default)
            grammar_text = self._cfg["hypothesis_grammar"]
        # When grammar is "none" or "universal", hide the domain-specific reaction
        # schema (which names the active mechanism) and replace with a generic one.
        if self._grammar_mode in ("none", "universal"):
            reaction_line = "Reaction: enzyme-catalyzed biochemical reaction (mechanism unknown)"
        else:
            reaction_line = f"Reaction: {schema}"
        lines = [
            reaction_line,
            "Inputs: C_A [0.01–100 mM], C_I [0–50 mM], C_B [0.01–100 mM], "
            "C_P [0–20 mM], Enz [0.01–10 mg/mL], T [278–368 K], pH [4–10]",
            f"Time: t={_T_FIXED} min (fixed)",
            "Observables: r0 [mM/min], C_A(t), C_B(t), C_P(t), C_I(t), conversion",
        ]
        if grammar_text is not None:
            lines.append(f"Hypothesis grammar: {grammar_text}")
        return "\n".join(lines)

    @property
    def objective_profile(self) -> dict[str, Any]:
        return {
            "name": "equation_recovery",
            "direction": "minimize_error",
            "type": "law_discovery",
            "difficulty": self._difficulty,
            "law_version": self._law_version,
        }

    # --- Oracle run ----------------------------------------------------------

    def run(self, params: dict[str, float]) -> OracleResult:
        """
        Run one experiment.

        params must contain all 7 keys: C_A, C_I, C_B, C_P, Enz, T, pH.
        Returns OracleResult where measurement = r0 (observed, with noise and
        secondary effects). All 6 outputs are in metadata['outputs'].
        """
        C_A  = float(params["C_A"])
        C_I  = float(params["C_I"])
        C_B  = float(params["C_B"])
        C_P  = float(params["C_P"])
        Enz  = float(params["Enz"])
        T    = float(params["T"])
        pH   = float(params["pH"])

        # Primary rate (clean, exact, for given law version)
        r0_primary = self._rate_fn(
            self._law_params, C_A, C_I, C_B, C_P, Enz, T, pH
        )

        # Small secondary effects (realistic biology noise floor)
        r0_true = r0_primary * _secondary_effects(T, pH)

        # Deterministic multiplicative noise
        noise = _deterministic_noise(
            self._domain_id, self._difficulty, self._law_version,
            dict(params), self._noise_level
        )
        r0_obs = max(r0_true * (1.0 + noise), 0.0)

        # Mass-balance outputs
        outputs = _compute_outputs(r0_true, C_A, C_I, C_B, C_P, self._consume_B)
        outputs["r0"] = r0_obs

        return OracleResult(
            params=dict(params),
            measurement=r0_obs,          # scalar used by GP / AL
            domain=self._domain_id,
            noise_level=self._noise_level,
            metadata={
                "difficulty":   self._difficulty,
                "law_version":  self._law_version,
                "outputs":      outputs,   # full 6-output vector for LLM
            },
        )

    # --- Law evaluation ------------------------------------------------------

    def evaluate_law(self, law_str: str) -> dict[str, Any]:
        """
        Evaluate a discovered law against the clean primary rate law.

        Secondary effects and noise are NOT included in y_true so that a
        correctly-discovered primary law always achieves RMSLE ≈ 0.
        RMSLE threshold for 'exact' is CHEM_RMSLE_THRESHOLD = 0.05.
        """
        rng = np.random.default_rng(42)
        n_test = 1000

        # Sample test inputs (relevant vars broadly; others at defaults/random)
        test: dict[str, np.ndarray] = {}
        for var, (lo, hi) in CHEM_INPUT_BOUNDS.items():
            if var in CHEM_LOG_VARS and lo > 0:
                test[var] = np.exp(rng.uniform(np.log(lo), np.log(hi), n_test))
            else:
                # Linear, but avoid exactly 0 for C_I and C_P
                test[var] = rng.uniform(lo, hi, n_test)

        # Ground truth: primary law only (no secondary effects, no noise)
        p = self._law_params
        y_true = np.array([
            self._rate_fn(
                p,
                test["C_A"][i], test["C_I"][i], test["C_B"][i], test["C_P"][i],
                test["Enz"][i],   test["T"][i],   test["pH"][i],
            )
            for i in range(n_test)
        ])

        # Compile discovered law
        import math as _math
        exec_globals: dict = {"np": np, "math": _math, "__builtins__": __builtins__}
        try:
            exec(law_str, exec_globals)
            discovered_law = exec_globals.get("discovered_law")
            if discovered_law is None:
                return {
                    "rmsle": float("nan"), "exact_accuracy": 0.0,
                    "error": "No 'discovered_law' function found in law_str",
                }
        except Exception as exc:
            return {
                "rmsle": float("nan"), "exact_accuracy": 0.0,
                "error": f"Compile error: {exc}",
            }

        # Evaluate discovered law
        try:
            y_pred = np.array([
                discovered_law(
                    test["C_A"][i], test["C_I"][i], test["C_B"][i], test["C_P"][i],
                    test["Enz"][i],   test["T"][i],   test["pH"][i],
                )
                for i in range(n_test)
            ])
        except Exception as exc:
            return {
                "rmsle": float("nan"), "exact_accuracy": 0.0,
                "error": f"Evaluation error: {exc}",
            }

        # RMSLE (same formula as NewtonBench)
        yt = np.abs(y_true)
        yp = np.abs(y_pred)
        mask = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp >= 0)
        if not mask.any():
            return {"rmsle": float("nan"), "exact_accuracy": 0.0, "error": "All values invalid"}

        rmsle = float(np.sqrt(np.mean(
            (np.log1p(yp[mask]) - np.log1p(yt[mask])) ** 2
        )))
        exact = 1.0 if rmsle < CHEM_RMSLE_THRESHOLD else 0.0

        return {
            "rmsle":          rmsle,
            "exact_accuracy": exact,
            "law_params":     p,
            "error":          None,
        }

    def get_ground_truth_law_str(self) -> str:
        """Return a Python function string for the exact ground truth law."""
        p = self._law_params
        d = self._domain_id
        sig = "def discovered_law(C_A, C_I, C_B, C_P, Enz, T, pH):"
        if d == "c0_michaelis_menten":
            return f"{sig}\n    return {p['kcat']} * Enz * C_A / ({p['Km']} + C_A)"
        if d == "c1_competitive_inhibition":
            return (f"{sig}\n    return {p['kcat']} * Enz * C_A "
                    f"/ ({p['Km']} * (1 + C_I / {p['Ki']}) + C_A)")
        if d == "c2_product_inhibition":
            return (f"{sig}\n    return {p['kcat']} * Enz * C_A "
                    f"/ ({p['Km']} * (1 + C_P / {p['Kp']}) + C_A)")
        if d == "c3_arrhenius_temperature":
            return (f"{sig}\n    import numpy as np\n"
                    f"    return {p['kcat_ref']} * np.exp(-{p['Ea']} / 8.314 "
                    f"* (1/T - 1/310.0)) * Enz * C_A / ({p['Km']} + C_A)")
        if d == "c4_ph_activity":
            return (f"{sig}\n    ion = 1 + 10**({p['pKa1']} - pH) + 10**(pH - {p['pKa2']})\n"
                    f"    return {p['kcat']} * Enz * C_A / (({p['Km']} + C_A) * ion)")
        if d == "c5_pingpong_bisubstrate":
            return (f"{sig}\n    return {p['kcat']} * Enz * C_A * C_B "
                    f"/ ({p['KmA']} * C_B + {p['KmB']} * C_A + C_A * C_B)")
        if d == "c6_uncompetitive_inhibition":
            return (f"{sig}\n    return {p['kcat']} * Enz * C_A "
                    f"/ ({p['Km']} + C_A * (1 + C_I / {p['Ki']}))")
        if d == "c7_substrate_inhibition":
            return (f"{sig}\n    return {p['kcat']} * Enz * C_A "
                    f"/ ({p['Km']} + C_A + C_A**2 / {p['Ki']})")
        if d == "c8_hill_cooperativity":
            return (f"{sig}\n    import numpy as np\n"
                    f"    return {p['kcat']} * Enz * C_A**{p['n']} "
                    f"/ ({p['K_half']}**{p['n']} + C_A**{p['n']})")
        if d == "c9_noncompetitive_inhibition":
            return (f"{sig}\n    return {p['kcat']} * Enz * C_A "
                    f"/ ((1 + C_I / {p['Ki']}) * ({p['Km']} + C_A))")
        # Compound + novel domains (c10–c99): unified law string generator
        try:
            from autoscilab.oracle.compound_domains import get_domain_law_str
            return get_domain_law_str(d, p)
        except Exception as exc:
            raise ValueError(f"No ground truth template for {d}: {exc}") from exc

    def get_available_law_versions(self) -> list[str]:
        return list(self._params_table[self._difficulty].keys())
