# NewtonBench: Benchmark structure

## Task count: 324

The benchmark has **324 tasks** formed by:

- **12 domains** (physics modules): `m0_gravity`, `m1_coulomb_force`, `m2_magnetic_force`, `m3_fourier_law`, `m4_snell_law`, `m5_radioactive_decay`, `m6_underdamped_harmonic`, `m7_malus_law`, `m8_sound_speed`, `m9_hooke_law`, `m10_be_distribution`, `m11_heat_transfer`
- **3 equation difficulties**: `easy`, `medium`, `hard` (complexity of the *target law*)
- **3 law versions** (per difficulty, where defined): `v0`, `v1`, `v2` — different counterfactual forms (e.g. different exponents or which parameters appear)
- **2 agent backends**: `vanilla_agent` (LLM only), `code_assisted_agent` (LLM + Python execution)
- **1 system** (for “vanilla equation” evaluation): `vanilla_equation` — the agent gets inputs/outputs and submits a single equation (other systems: simple/complex add more apparatus)

Rough count: 12 × 3 × 3 × 2 = 216 configs; some modules may have fewer law versions → 324 total tasks.

---

## Two dimensions of “difficulty”

1. **Equation difficulty** (`easy` / `medium` / `hard`)  
   - Controls *structural complexity* of the ground-truth law (e.g. more parameters, non-integer exponents, sums of terms).
2. **Model system** (`vanilla_equation` / `simple_system` / `complex_system`)  
   - Controls *how* the agent interacts with the world (e.g. direct force measurement vs orbital motion vs more complex apparatus).  
   - “Vanilla equation” = direct input → output; others add dynamics or extra observables.

---

## Per-domain law layout (example: m0_gravity)

Each module has a **LAW_REGISTRY** in `modules/<module>/laws.py`:

```
difficulty: easy   | medium | hard
─────────────────────────────────────
v0  |  one law     |  one law  |  one law
v1  |  one law     |  one law  |  one law
v2  |  one law     |  one law  |  one law
```

- **m0_gravity** (force between two masses):
  - **easy v0:** F = C·m1·m2/r^1.5  
  - **easy v1:** F = C·m1/r^2 (mass2 irrelevant)  
  - **easy v2:** F = C·(m1²·m2²)/r^2  
  - **medium v0:** F = C·(m1·m2)²/r^1.5  
  - **medium v1:** F = C·m1/r^2.6  
  - **medium v2:** F = C·(m1²·m2²)·r^2  
  - **hard v0:** F = C·(m1+m2)²/r^1.5  
  - **hard v1:** F = C·m1^1.3/r^2.6  
  - **hard v2:** F = C·(m1²+m2²)·r^2  

So for each (domain, difficulty) you get 3 *law versions* (v0, v1, v2); each is one “task” when combined with agent type and system.

---

## Evaluation flow

1. **NewtonBench standalone** (`run_all_evaluations.py`):  
   For each (model, module, agent_backend, difficulty, law_version, system, noise): run N **trials**. Each trial: LLM runs experiments (via `<run_experiment>`) and submits `<final_law>`. Compare to ground truth → **RMSLE**, **exact_accuracy**, **symbolic_equivalent** (LLM judge).

2. **AutoSciLab pipeline** (our `scripts/run.py`):  
   Single discovery run per (domain, difficulty, law_version, budget). Oracle = NewtonBench module; LLM proposes regions → AL selects points → PySR fits equation. Compare best equation to ground truth → **GT RMSLE**, **exact_accuracy**.

---

## Results layout (NewtonBench)

```
evaluation_results/
  {model_name}/           e.g. gpt-4o-mini
    {module}/             e.g. m0_gravity
      {agent_backend}/    vanilla_agent | code_assisted_agent
        {difficulty}/     easy | medium | hard
          {law_version}/  v0 | v1 | v2
            {system}_noise{level}_v{run}/
              trials/
                trial0.json, trial1.json, ...
              aggregated_results.json
```

---

## Summary

| Dimension        | Options                                      |
|-----------------|-----------------------------------------------|
| Domains         | 12 (m0–m11)                                  |
| Equation diff   | easy, medium, hard                           |
| Law version     | v0, v1, v2 (per difficulty, module-dependent) |
| Agent           | vanilla_agent, code_assisted_agent           |
| System          | vanilla_equation, (simple/complex)           |
| Noise           | e.g. 0.0, 0.01                               |

**324 tasks** = all combinations used in the benchmark (some modules may have fewer versions).
