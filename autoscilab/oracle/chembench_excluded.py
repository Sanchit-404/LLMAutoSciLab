# ChemBench Excluded Domains
# 43 domains excluded from the standard 57-domain benchmark.
# These are preserved in CHEM_DOMAIN_REGISTRY for research use.
# To re-include any domain, remove it from CHEM_EXCLUDED_DOMAINS in chembench.py.

# ---------------------------------------------------------------------------
# Reason 1: pH bell curve (double ionisation — two pKa constants)
# GT uses 1/(1 + 10^(pKa1-pH) + 10^(pH-pKa2))
# Requires discovering two independent ionisation constants from data alone.
# ---------------------------------------------------------------------------
PH_BELL_DOMAINS = [
    "c4_ph_activity",
    # MM + pH
    "c11_mm_competitive_ph", "c13_mm_uncompetitive_ph", "c15_mm_noncompetitive_ph",
    "c17_mm_product_ph", "c18_mm_arrhenius_ph",
    "c19_mm_competitive_arrhenius_ph", "c20_mm_uncompetitive_arrhenius_ph",
    "c21_mm_noncompetitive_arrhenius_ph", "c22_mm_product_arrhenius_ph",
    # Pingpong + pH
    "c24_pingpong_ph", "c28_pingpong_arrhenius_ph",
    "c30_pingpong_competitive_ph", "c31_pingpong_noncompetitive_ph",
    "c32_pingpong_competitive_arrhenius_ph",
    # Hill + pH
    "c38_hill_ph", "c40_hill_competitive_ph", "c42_hill_noncompetitive_ph",
    "c43_hill_arrhenius_ph", "c44_hill_competitive_arrhenius_ph",
    "c45_hill_noncompetitive_arrhenius_ph",
    # Substrate inhibition + pH
    "c53_sinh_ph", "c55_sinh_competitive_ph", "c57_sinh_noncompetitive_ph",
    "c58_sinh_arrhenius_ph", "c59_sinh_competitive_arrhenius_ph",
    "c60_sinh_noncompetitive_arrhenius_ph", "c62_sinh_product_ph", "c64_sinh_uncompetitive_ph",
    # Novel + pH
    "c83_ordered_bi_bi_ph", "c84_reversible_mm_ph", "c85_allosteric_act_ph",
    "c86_anticoop_ph", "c87_coop_inh_ph", "c88_product_act_ph",
    "c95_ordered_bi_bi_arr_ph", "c96_reversible_mm_arr_ph",
    "c97_allosteric_act_arr_ph", "c99_metal_act_arr_ph",
]

# ---------------------------------------------------------------------------
# Reason 2: Haldane reversible Michaelis-Menten (numerator subtraction)
# GT: (Vf*C_A/KmA - Vr*C_P/KmP) / (1 + C_A/KmA + C_P/KmP)
# Rate can be zero or negative near equilibrium; C_P appears with opposite
# signs in numerator vs denominator — structurally unlike all other domains.
# ---------------------------------------------------------------------------
HALDANE_DOMAINS = [
    "c66_reversible_mm",
    "c77_reversible_mm_arrhenius",
    "c84_reversible_mm_ph",       # also in PH_BELL_DOMAINS
    "c91_reversible_mm_competitive",
    "c96_reversible_mm_arr_ph",   # also in PH_BELL_DOMAINS
]

# ---------------------------------------------------------------------------
# Reason 3: >5 free parameters
# ---------------------------------------------------------------------------
HIGH_PARAM_DOMAINS = [
    "c82_metal_act_arrhenius",    # 6 params: kcat_ref, KmA, Km_met, n_met, Ea, Ki
]

ALL_EXCLUDED = sorted(set(PH_BELL_DOMAINS + HALDANE_DOMAINS + HIGH_PARAM_DOMAINS))
