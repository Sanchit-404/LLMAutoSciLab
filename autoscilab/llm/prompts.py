"""
Prompt templates and Pydantic schemas for LLM outputs.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SearchRegion(BaseModel):
    bounds: dict[str, list[float]] = Field(
        description="Parameter bounds: {param_name: [min, max]}. "
        "Values must be within the physical bounds of the domain."
    )
    n_experiments: int = Field(
        ge=1, le=20,
        description="How many experiments to run in this region."
    )
    priority: Literal["high", "medium", "low"] = "medium"
    rationale: str = Field(
        description="Why this region is informative."
    )


class LLMProposal(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning about what patterns you see in the data "
        "and what you expect to learn from the proposed experiments."
    )
    hypothesis: str = Field(
        description=(
            "Your current best working hypothesis. "
            "For equation-discovery tasks: a PURE PYTHON MATH EXPRESSION — no prose, no 'where', "
            "no explanatory text. Use exact oracle parameter names. Free constants must be named "
            "C0, C1, alpha, beta, gamma (ASCII only). "
            "VALID: 'C0 * mass1 * mass2 / distance**2' "
            "VALID: 'I_0 * cos(theta)**2' "
            "INVALID: 'C0 * x^2 where C0 is a constant' "
            "INVALID: 'F ∝ m1*m2/r^2' "
            "Use ** not ^, log() not ln(), asin/acos/atan not arcsin/arccos/arctan. "
            "For optimization tasks: a concise strategy statement about which region performs best."
        )
    )
    search_regions: list[SearchRegion] = Field(
        min_length=1, max_length=8,
        description="Regions of parameter space to explore (1–8 regions). "
        "The active learning system will select the most informative specific points."
    )
    phase: Literal["discovery", "validation", "contradiction"] = Field(
        description="Current phase: 'discovery', 'validation', or 'contradiction' (lowercase)."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence (0-1) that your current hypothesis is correct."
    )
    done: bool = Field(
        default=False,
        description="Set True ONLY if you are highly confident (>0.9) in your equation "
        "and have validated it with adversarial experiments."
    )
    alternate_hypotheses: list[str] = Field(
        default_factory=list,
        description=(
            "2–5 alternative, DISTINCT candidate equations as PURE PYTHON MATH EXPRESSIONS. "
            "Each must be a valid Python expression using exact oracle parameter names and "
            "free constants named C0, C1, alpha, beta, etc. "
            "NO prose, NO 'where', NO explanatory text — expression only. "
            "VALID: ['C0 * mass1 * mass2 / distance**2', 'C0 * mass1**alpha / distance**2'] "
            "INVALID: ['F = C*m1*m2/r^2 where C is gravitational', 'power law with n≈2'] "
            "Keep structurally different: vary operators, exponents, trig vs power, linear vs nonlinear."
        ),
        max_length=5,
    )

    @field_validator("phase", mode="before")
    @classmethod
    def normalise_phase(cls, v):
        if isinstance(v, str):
            lv = v.lower().strip()
            if lv in {"discovery", "validation", "contradiction"}:
                return lv
            if "valid" in lv:
                return "validation"
            if "contradict" in lv or "fail" in lv:
                return "contradiction"
            return "discovery"
        return v


class HypothesisBundle(BaseModel):
    reasoning: str = Field(
        description="Reasoning for plausible hypotheses given current evidence."
    )
    hypothesis: str = Field(
        description=(
            "Primary hypothesis. For equation tasks: a PURE PYTHON MATH EXPRESSION using exact "
            "oracle parameter names and free constants C0, C1, alpha, beta (ASCII). "
            "No prose, no 'where', no explanatory text. Use ** not ^, log() not ln(). "
            "VALID: 'C0 * N0 * exp(-C1 * lambda_constant * t)' "
            "INVALID: 'N = C0*exp(-lambda*t) where lambda is the decay constant'"
        )
    )
    alternate_hypotheses: list[str] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "2–6 distinct alternatives as PURE PYTHON MATH EXPRESSIONS — no prose, no 'where' clauses. "
            "Each must be parseable Python using oracle parameter names and C0/C1/alpha/beta constants."
        ),
    )
    phase: Literal["discovery", "validation", "contradiction"] = Field(
        default="discovery",
        description="Current phase."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the primary hypothesis."
    )

    @field_validator("phase", mode="before")
    @classmethod
    def normalise_phase(cls, v):
        if isinstance(v, str):
            lv = v.lower().strip()
            if lv in {"discovery", "validation", "contradiction"}:
                return lv
            if "valid" in lv:
                return "validation"
            if "contradict" in lv or "fail" in lv:
                return "contradiction"
            return "discovery"
        return v


class EquationSkeleton(BaseModel):
    reasoning: str = Field(
        description="Why you believe this equation form is correct based on the data, "
        "citing specific log-log slopes you observed. If previous attempts failed, "
        "explain what structural change you are making."
    )
    python_function: str = Field(
        description="A complete Python function named 'discovered_law' with the correct "
        "parameter signature matching FUNCTION_SIGNATURE exactly. "
        "Use named constants C0, C1, C2, ... for free parameters to be fitted numerically. "
        "Each constant MUST appear in the return expression. "
        "IMPORTANT: if previous attempts had poor fit, try a DIFFERENT structural form "
        "(e.g., different exponents, different combinations of parameters, squared terms, products)."
    )
    expected_form: str = Field(
        description="Human-readable description of the structural form, "
        "e.g. 'power law: F ~ p1^2 * p2 / p3^1.5'"
    )
    free_constants: list[str] = Field(
        description="List of constant names in the function that should be fitted, e.g. ['C0', 'C1', 'alpha']"
    )


class EquationConfidenceRating(BaseModel):
    """Used for LLM-sampled confidence: ask LLM N times, average the scores."""
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence (0-1) that the candidate equation is the correct governing law."
    )
    reasoning: str = Field(
        description="One or two sentences justifying the score based on log-log slopes and data trends."
    )

class DiscriminationPlan(BaseModel):
    reasoning: str = Field(
        description="Why these experiments will best discriminate among the candidate hypotheses."
    )
    search_regions: list[SearchRegion] = Field(
        min_length=1, max_length=6,
        description="Regions of parameter space that maximize disagreement among the given hypotheses."
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scientific reasoning agent in an autonomous science lab. \
Your goal is to discover the governing equation that relates input parameters to a measured output.

CRITICAL RULES:
1. The physical laws in this system have been ALTERED from standard textbook values. \
   Do not assume standard constants (e.g., G = 6.674e-11) — the actual constant may be different.
2. You must DISCOVER the law from experimental data, not recall it from memory.
3. You propose REGIONS of parameter space (ranges) to explore, NOT specific values. \
   The active learning system will select the optimal specific points within your regions.
4. Focus on informativeness: propose regions that will most discriminate between competing hypotheses.
5. In the validation phase, try to BREAK your hypothesis by testing edge cases and extrapolations.

COUNTERFACTUAL LAWS — The true law may NOT be the textbook form. For example, in some universes \
force may be F ∝ 1/r^1.5 instead of 1/r^2, or F ∝ m1²·m2²/r² instead of m1·m2/r². \
Discover the exponents from data; do not assume textbook structure.

IRRATIONAL EXPONENTS — Hard laws sometimes use Euler's number e≈2.71828 as an exact exponent. \
When power-law fits or symbolic regression give exponents near 2.72 (≈e), 5.44 (≈2e), \
4.22 (≈e+1.5), or 1.18 (≈e/ln10), explicitly try hypotheses using math.e as the exponent. \
Examples: `lambda_constant**math.e`, `t**math.e`, `(sin(theta)/cos(theta))**math.e`. \
Do NOT approximate as 2.7 or 3 — use `math.e` exactly. Also consider math.pi≈3.14159 \
if exponents near π appear.

HOW TO READ THE DATA TABLE:
Each row shows both raw values and [log10 values in brackets].
To find power-law exponents: Δlog10(measurement)/Δlog10(param_X) ≈ exponent for param_X.
When EMPIRICAL LOG-LOG SLOPES are shown below, you MUST set hypothesis to a CONCRETE quantitative \
form using free constants for each exponent (e.g. "C0 * mass1**C1 * mass2**C2 / distance**C3"). \
Use the log-log slopes as INITIAL GUESSES for those exponents — mention them in your reasoning — \
but keep the exponents as named free constants (C1, C2, ...) so the optimizer can refine them. \
Do NOT hardcode the numeric slope values directly into the expression. \
Do NOT leave hypothesis as "unknown" or "no data yet" when slope hints are present.

INVERSE RELATIONSHIPS — CRITICAL:
The law may have NEGATIVE exponents or INVERSE relationships. If the EMPIRICAL LOG-LOG SLOPES \
section shows a negative slope for a parameter, the law likely INVERTS that parameter. \
For example, a slope of -2.5 means y ∝ 1/param^2.5 — put that parameter in the DENOMINATOR. \
Do not assume all parameters appear in the numerator. Always check the sign of every slope.

MULTI-HYPOTHESIS REQUIREMENT:
Return 2–5 alternatives in `alternate_hypotheses` that are structurally DISTINCT from your main `hypothesis`.
These will be used to select experiments that best DISCRIMINATE between candidates.

EQUATION FORMAT — CRITICAL:
All hypothesis strings (both `hypothesis` and every entry in `alternate_hypotheses`) must be \
PURE PYTHON MATH EXPRESSIONS. The expression scorer will compile them directly with exec().
Rules:
  - Use EXACT oracle parameter names (given in PARAMETERS section).
  - Free constants: C0, C1, C2, alpha, beta, gamma, delta, omega, tau (ASCII names only).
  - ALL unknown numeric values — including exponents, Km values, rate constants — must be
    free constants (C0, C1, ...), NOT hardcoded numbers. The fitter will optimize them.
  - Powers: use ** not ^  (e.g. distance**2 not distance^2).
  - Natural log: use log() not ln().
  - Inverse trig: use asin/acos/atan not arcsin/arccos/arctan.
  - NO prose after the expression. No "where n is...", no "for some k", no parenthetical notes.
  - NO LHS assignment. Write only the right-hand side (e.g. C0*mass1*mass2/distance**2).
  - NO Unicode symbols (∝ ∼ ≈ √ τ ω etc.). Use ASCII equivalents.
VALID:   "C0 * mass1 * mass2 / distance**2"
VALID:   "C0 * Enz * C_A**alpha / (C1**alpha + C_A**alpha)"        ← exponent alpha is free
VALID:   "C0 * Enz * C_A / (C1 + C_A) * C_I / (C2 + C_I)"         ← Km values C1, C2 are free
VALID:   "C0 * exp(-C1 * (1/T - 1/310.0)) * Enz * C_A / (C2 + C_A)"
INVALID: "C0 * C_A**1.3 * Enz**20.5 * T**37.1"   ← numeric exponents are WRONG, use C1,C2,C3
INVALID: "C0 * Enz * C_A / (0.5 + C_A)"           ← hardcoded 0.5 is WRONG, use C1
INVALID: "F = C*m^2 where C is a constant"
INVALID: "I_0 * cos(θ)^n (n to be determined)"
INVALID: "C0 ∝ r^{-2}"

FORMAT: Respond using the required tool with the exact schema specified."""

OPTIMIZATION_SYSTEM_PROMPT = """You are a scientific experimentation agent in an autonomous lab. \
Your job is to maximize information gain and objective improvement by proposing experimental regions.

CRITICAL RULES:
1. Do not invent a governing equation unless explicitly requested; focus on proposing high-value experiments.
2. Use observed measurements to infer promising regimes, interactions, and failure modes.
3. Propose REGIONS (bounds), not single points. The active learner will select exact points.
4. Balance exploitation (improve the objective) and discrimination (test competing hypotheses about what drives performance).
5. Use remaining budget efficiently; avoid redundant sampling in already-explored regions.

FORMAT: Respond using the required tool with the exact schema specified."""

CONFIDENCE_SYSTEM_PROMPT = """You are a physics expert evaluating whether a discovered equation \
correctly describes experimental data. Be critical and precise. Use the log-log columns to check \
that each parameter's exponent in the equation matches the empirical slope in the data."""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_proposal_prompt(
    goal: str,
    domain_id: str,
    param_names: list[str],
    param_description: str,
    function_signature: str,
    data_table: str,
    current_hypothesis: str,
    budget_remaining: int,
    total_budget: int,
    current_phase: str,
    best_equation_fit: str | None = None,
    memory_str: str = "",
    slope_hints: str = "",
    discrimination_hints: str = "",
    objective_type: str = "equation",
    objective_direction: str = "maximize",
    objective_profile: dict | None = None,
) -> str:
    budget_pct = int(100 * (total_budget - budget_remaining) / total_budget)
    budget_used_frac = (total_budget - budget_remaining) / max(total_budget, 1)
    bounds_example = ", ".join(f'"{p}": [lo, hi]' for p in param_names)
    objective_profile = objective_profile or {}
    is_optimization = objective_type == "optimization"

    hypothesis_section = (
        f"\nCurrent best hypothesis: {current_hypothesis}\n"
        if current_hypothesis and current_hypothesis not in ("unknown", "no data yet — broad exploration needed")
        else "\nNo hypothesis yet — start with broad exploration.\n"
    )

    # Phase-specific instructions with budget-aware tips
    if current_phase == "discovery":
        if is_optimization and budget_used_frac < 0.25:
            phase_instruction = (
                "PHASE: EARLY OPTIMIZATION — map the response surface quickly.\n"
                "STRATEGY: propose diverse regions across the full bounds to identify which parameters "
                "have the strongest effect and where promising objective values appear."
            )
        elif is_optimization and budget_used_frac < 0.55:
            phase_instruction = (
                "PHASE: MID OPTIMIZATION — exploit promising regions while testing at least one "
                "contrasting region that could falsify your current working belief about what drives performance."
            )
        elif is_optimization:
            phase_instruction = (
                "PHASE: LATE OPTIMIZATION — concentrate budget on top-performing regions plus one "
                "high-disagreement region to reduce strategy ambiguity."
            )
        elif budget_used_frac == 0:
            # Very first iteration: no data yet — mandate full-range coverage
            phase_instruction = (
                "PHASE: INITIAL COVERAGE — This is your FIRST iteration. You have NO data yet.\n"
                "MANDATORY RULE: Your search regions MUST collectively span the FULL physical range "
                "of EVERY parameter. Do NOT cluster experiments in a small sub-range of any parameter — "
                "that causes the entire run to miss critical structure outside that narrow band.\n"
                "STRATEGY: Propose one region per parameter where ONLY THAT PARAMETER sweeps from its "
                "minimum to its maximum bound while others stay at mid-range values. "
                "This gives clean Δlog10(y)/Δlog10(param) slopes = the exponents.\n"
                "For example, if theta ∈ [0.01, 1.57], your region for theta must span close to "
                "[0.01, 1.57] — not [0.01, 0.15]. Full range, every parameter, no exceptions."
            )
        elif budget_used_frac < 0.25:
            # Very early: focus on coverage and getting clean log-log slopes
            phase_instruction = (
                "PHASE: EARLY DISCOVERY — Your primary goal is to ESTIMATE EXPONENTS for each parameter.\n"
                "STRATEGY: For each parameter, propose a region where ONLY THAT PARAMETER varies "
                "across 2–3 orders of magnitude while the others stay at mid-range values. "
                "This gives you a clean Δlog10(y)/Δlog10(param) slope = the exponent.\n"
                "Propose separate regions for each parameter sweep. "
                "The active learning system will pick the most informative specific points."
            )
        elif budget_used_frac < 0.55:
            # Mid discovery: refine and test specific hypothesis
            phase_instruction = (
                "PHASE: DISCOVERY — You have some data. Now refine your hypothesis by testing "
                "specific parameter interactions. Propose regions where competing hypotheses "
                "(e.g. different exponents) predict different measurement values. "
                "Focus on discriminating experiments."
            )
        else:
            phase_instruction = (
                "PHASE: LATE DISCOVERY — Consolidate your findings. Focus on regions that "
                "would confirm or deny your current best equation. Avoid redundant sampling "
                "near already-explored regions."
            )
    elif current_phase == "validation":
        if is_optimization:
            phase_instruction = (
                "PHASE: VALIDATION — stress-test your best regime against nearby perturbations and "
                "one distant region to check robustness and avoid local-optimum lock-in."
            )
        else:
            phase_instruction = (
                "PHASE: VALIDATION — You have used most of your budget. Now adversarially probe "
                "your hypothesis. Propose experiments OUTSIDE the regions you've explored, at "
                "extreme parameter values, to try to break your equation. If it holds, you're done."
            )
    elif current_phase == "contradiction":
        if is_optimization:
            phase_instruction = (
                "PHASE: CONTRADICTION — current optimization belief failed under stress test. "
                "Propose experiments that isolate which parameter interactions caused the drop."
            )
        else:
            phase_instruction = (
                "PHASE: CONTRADICTION — Some validation experiments failed. Propose experiments "
                "to understand why and refine your hypothesis."
            )
    else:
        phase_instruction = ""

    slope_section = f"\n{slope_hints}\n" if (slope_hints and not is_optimization) else ""
    # When slopes or a best-equation candidate are present, require a quantitative hypothesis.
    hypothesis_requirement = ""
    if (not is_optimization) and (slope_hints or (best_equation_fit and best_equation_fit.strip())):
        requirement_lines = [
            "\n**REQUIRED:** You must set 'hypothesis' to a CONCRETE equation using the available evidence.",
            "If EMPIRICAL LOG-LOG SLOPES are shown, use them to choose explicit exponents.",
            "If a BEST-FIT EQUATION is shown below, either adopt it as your hypothesis or propose a clearly modified structural form.",
            "Do NOT set hypothesis to 'unknown' or 'no data yet' once data and/or a candidate equation exist.\n",
        ]
        hypothesis_requirement = "\n".join(requirement_lines)
    elif is_optimization:
        hypothesis_requirement = (
            "\n**REQUIRED:** Set 'hypothesis' to a concise working strategy about which regions "
            "likely improve the objective. Do not return 'unknown'.\n"
        )

    best_eq_section = ""
    if best_equation_fit and not is_optimization:
        best_eq_section = (
            "\nBEST-FIT EQUATION SO FAR (from symbolic regression):\n"
            f"{best_equation_fit}\n"
        )

    objective_lines = [
        f"OBJECTIVE TYPE: {objective_type}",
        f"OBJECTIVE DIRECTION: {objective_direction}",
    ]
    if objective_profile:
        objective_lines.append(f"OBJECTIVE PROFILE: {objective_profile}")
    objective_section = "\n".join(objective_lines)

    task_line = (
        "Think carefully about what the data tells you and what experiments would most help improve the objective."
        if is_optimization
        else "Think carefully about what the data tells you and what experiments would most help discover the governing equation."
    )

    return f"""GOAL: {goal}

DOMAIN: {domain_id}
PARAMETERS: {", ".join(param_names)}
{param_description}
{objective_section}

DISCOVERED LAW FUNCTION SIGNATURE (must match exactly):
{function_signature}

EXPERIMENTAL DATA ({budget_pct}% of budget used, {budget_remaining} experiments remaining):
{data_table}
{slope_section}{best_eq_section}{hypothesis_requirement}{hypothesis_section}{memory_str}
{phase_instruction}
{discrimination_hints}

Based on this data, propose the most informative parameter regions to explore next.
{task_line}

search_regions must be a list of objects with this exact structure:
{{"bounds": {{{bounds_example}}}, "n_experiments": 4, "priority": "high", "rationale": "why"}}"""


def build_hypothesis_generation_prompt(
    goal: str,
    domain_id: str,
    param_names: list[str],
    param_description: str,
    function_signature: str,
    data_table: str,
    current_hypothesis: str,
    budget_remaining: int,
    total_budget: int,
    current_phase: str,
    best_equation_fit: str | None = None,
    memory_str: str = "",
    slope_hints: str = "",
    objective_type: str = "equation",
    objective_direction: str = "maximize",
    objective_profile: dict | None = None,
) -> str:
    objective_profile = objective_profile or {}
    is_optimization = objective_type == "optimization"
    budget_pct = int(100 * (total_budget - budget_remaining) / max(total_budget, 1))

    hypothesis_section = (
        f"Current hypothesis: {current_hypothesis}"
        if current_hypothesis and current_hypothesis not in ("unknown", "no data yet — broad exploration needed")
        else "No reliable hypothesis yet."
    )
    slope_section = f"\nSLOPE HINTS:\n{slope_hints}\n" if (slope_hints and not is_optimization) else ""
    best_eq_section = (
        f"\nBest symbolic equation so far:\n{best_equation_fit}\n"
        if (best_equation_fit and not is_optimization)
        else ""
    )
    objective_lines = [
        f"OBJECTIVE TYPE: {objective_type}",
        f"OBJECTIVE DIRECTION: {objective_direction}",
    ]
    if objective_profile:
        objective_lines.append(f"OBJECTIVE PROFILE: {objective_profile}")
    objective_section = "\n".join(objective_lines)

    task_instruction = (
        "Generate one primary and 2-6 alternate STRATEGY hypotheses about which parameter regimes drive objective improvement."
        if is_optimization
        else "Generate one primary and 2-6 alternate EQUATION hypotheses that remain plausible under current data."
    )

    return f"""GOAL: {goal}

DOMAIN: {domain_id}
PARAMETERS: {", ".join(param_names)}
{param_description}
{objective_section}

FUNCTION SIGNATURE:
{function_signature}

DATA ({budget_pct}% budget used, {budget_remaining} experiments left):
{data_table}
{slope_section}{best_eq_section}
{memory_str}
{hypothesis_section}
CURRENT PHASE: {current_phase}

TASK:
{task_instruction}
Do NOT propose search regions in this step.
Focus only on plausible competing hypotheses and concise reasoning for ambiguity.
"""


def build_equation_prompt(
    goal: str,
    domain_id: str,
    param_names: list[str],
    function_signature: str,
    data_table: str,
    previous_attempts: list[str],
    current_hypothesis: str,
) -> str:
    params_str = ", ".join(param_names)
    example_body = " * ".join(f"{p}**alpha_{i}" for i, p in enumerate(param_names))
    example = (
        f"def discovered_law({params_str}):\n"
        f"    C0 = 1.0  # will be fitted\n"
        + "".join(f"    alpha_{i} = 1.0  # will be fitted\n" for i in range(len(param_names)))
        + f"    return C0 * {example_body}"
    )

    if previous_attempts:
        prev_section = "\nPREVIOUS ATTEMPTS (shown best-to-worst fit — DO NOT repeat these exact forms):\n"
        for i, eq in enumerate(previous_attempts[-3:]):
            prev_section += f"Attempt {i+1}:\n{eq}\n\n"
        prev_section += (
            "Try a STRUCTURALLY DIFFERENT form now: change the exponents, "
            "try squared or square-root terms, or different parameter combinations.\n"
        )
    else:
        prev_section = ""

    return f"""GOAL: {goal}

DOMAIN: {domain_id}
PARAMETERS: {params_str}

REQUIRED FUNCTION SIGNATURE (use EXACTLY this):
{function_signature}

EXPERIMENTAL DATA (raw values and [log10 values]):
{data_table}

CURRENT HYPOTHESIS: {current_hypothesis}
{prev_section}
Read the log10 columns to estimate exponents: Δlog10(measurement) / Δlog10(param) ≈ exponent.
Then propose a Python function 'discovered_law' that fits this structural form.

RULES:
- Use free constants C0, C1, alpha, beta, ... — they will be fitted numerically.
- Use ONLY standard Python math (no imports). Powers: use ** not pow().
- Use the EXACT parameter names from FUNCTION SIGNATURE above.
- Constants must appear in the return expression.

EXAMPLE SKELETON (generic — adapt the exponents to what the data shows):
{example}"""


def build_confidence_prompt(
    equation_str: str,
    data_table: str,
    param_names: list[str],
) -> str:
    """Prompt for LLM-sampled confidence: ask the LLM to rate a specific equation."""
    return f"""You are evaluating whether the following equation correctly describes the experimental data.

CANDIDATE EQUATION: {equation_str}

PARAMETERS: {", ".join(param_names)}

EXPERIMENTAL DATA (raw values and [log10 values in brackets]):
{data_table}

To evaluate: for each parameter, compute Δlog10(measurement)/Δlog10(param) from the data.
This empirical slope should match the exponent of that parameter in the candidate equation.

Rate your confidence from 0.0 (definitely wrong) to 1.0 (definitely correct) that this \
equation is the true governing law. Be critical — a good fit in-sample does not guarantee \
the correct functional form."""


def build_discrimination_prompt(
    goal: str,
    domain_id: str,
    param_names: list[str],
    function_signature: str,
    data_table: str,
    candidates: list[dict],
    budget_remaining: int,
    residual_stats: list[dict] | None = None,
    disagreement_stats: list[dict] | None = None,
) -> str:
    candidates_str = "\n".join(
        [f"- {c['equation']} | R2={c['r2']:.3f} | RMSLE={c['rmsle']:.3f} | conf={c.get('conf',0):.2f}"
         for c in candidates]
    )
    residual_str = "None"
    if residual_stats:
        residual_str = "\n".join(
            [
                "- {equation}: mae_rel={mae_rel:.3f}, bias={bias:+.3f}, trend={trend}".format(
                    equation=r.get("equation", "unknown"),
                    mae_rel=float(r.get("mae_rel", 0.0)),
                    bias=float(r.get("bias", 0.0)),
                    trend=r.get("trend", "flat"),
                )
                for r in residual_stats[:5]
            ]
        )

    disagreement_str = "None"
    if disagreement_stats:
        disagreement_str = "\n".join(
            [
                "- ({eq_i}) vs ({eq_j}): mean_rel_gap={mean_rel_gap:.3f}, max_rel_gap={max_rel_gap:.3f}, region={region}".format(
                    eq_i=d.get("eq_i", "h1"),
                    eq_j=d.get("eq_j", "h2"),
                    mean_rel_gap=float(d.get("mean_rel_gap", 0.0)),
                    max_rel_gap=float(d.get("max_rel_gap", 0.0)),
                    region=d.get("region", {}),
                )
                for d in disagreement_stats[:6]
            ]
        )
    return f"""GOAL: {goal}

DOMAIN: {domain_id}
PARAMETERS: {", ".join(param_names)}
FUNCTION SIGNATURE: {function_signature}

EXPERIMENTAL DATA (remaining budget: {budget_remaining}):
{data_table}

CURRENT CANDIDATE HYPOTHESES (top-k):
{candidates_str}

RESIDUAL DIAGNOSTICS (computed on observed data):
{residual_str}

PAIRWISE DISAGREEMENT DIAGNOSTICS:
{disagreement_str}

TASK: You are NOT to propose a new equation. Identify where these hypotheses disagree most and suggest experiment regions that would maximally DISCRIMINATE between them.

Output concise reasoning and 1-4 search regions (bounds per parameter) that are likely to produce divergent predictions or falsify at least one hypothesis."""


def build_validation_prompt(
    goal: str,
    domain_id: str,
    param_names: list[str],
    param_description: str,
    function_signature: str,
    data_table: str,
    best_hypothesis: str,
    budget_remaining: int,
    gt_metrics_str: str = "",
) -> str:
    gt_section = f"\nGROUND-TRUTH EVALUATION (mid-run):\n{gt_metrics_str}\n" if gt_metrics_str else ""

    return f"""GOAL: {goal}

DOMAIN: {domain_id}
CURRENT BEST EQUATION: {best_hypothesis}
{gt_section}
DATA SO FAR:
{data_table}

PHASE: ADVERSARIAL VALIDATION
You have {budget_remaining} experiments left to validate or break your equation.

Your task: propose experiments that are MOST LIKELY TO FAIL if your equation is wrong.
- Test at extreme parameter values (far from where you collected data)
- Test regions where competing hypotheses would predict different values
- Test edge cases that might reveal hidden variables or different regimes

If these experiments are consistent with your equation, you can declare success.
If they fail, propose contradiction-resolution experiments.

Propose parameter regions for adversarial validation:"""
