from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from autoscilab.al.gp_model import GPSurrogate
from autoscilab.al.selector import ALSelector
from autoscilab.data.store import ExperimentStore
from autoscilab.grn_mei_v5_experiments.graph_search import (
    GraphHypothesisFit,
    GraphHypothesisLineage,
    build_hypothesis_lineages,
    select_committee_from_lineages,
)
from autoscilab.grn_mei_v5_experiments.schemas import (
    GraphTranslationProposal,
    HypothesisReasoningProposal,
    SingleGraphTranslationProposal,
)
from autoscilab.llm.client import LLMClient
from autoscilab.llm.memory import LLMMemory
from autoscilab.oracle.base import BaseOracle
from autoscilab.oracle.grnbench import GRNBenchOracle, GRN_INPUT_VARS


REASONING_SYSTEM_PROMPT = """You are a scientific reasoning agent for mechanistic gene regulatory network discovery.

Goal:
- infer the hidden signed regulatory graph from perturbation-response experiments
- propose multiple competing mechanism hypotheses in natural language
- propose the next experimental regions that best discriminate them

Rules:
1. Produce 2-5 distinct natural-language hypotheses.
2. Hypotheses should describe causal mechanisms over nodes signal, A, B, C, R.
3. Do not output graph edges here. Focus on reasoning and competing stories.
4. Search regions must stay within bounds and should target informative perturbations, not generic broad sweeps.
5. Every search region MUST specify bounds for all 5 parameters exactly:
   signal, pert_A, pert_B, pert_C, pert_R.
6. Prefer hypotheses that are falsifiable by perturbation.
"""


TRANSLATION_SYSTEM_PROMPT = """You convert natural-language GRN hypotheses into canonical signed graphs.

Rules:
1. Use only nodes: signal, A, B, C, R.
2. signal may be a source but never a destination.
3. Use only signed edges with sign +1 or -1.
4. No self-loops and no duplicate directed edges.
5. Each translated graph must contain a path from signal to C.
6. Keep the graph sparse and faithful to the original hypothesis, without adding speculative structure unless required.
7. Return one graph translation per hypothesis_id.
8. IMPORTANT: signal is source-only. Never use signal as a destination node.
9. Valid edge example: {"src":"signal","dst":"A","sign":1}
10. Invalid edge example: {"src":"A","dst":"signal","sign":1}
11. Before you finalize a graph, verify that following directed edges from signal reaches C in that same graph.
12. If a hypothesis does not mention C explicitly, you must still add the minimal faithful directed chain needed so signal can reach C.
13. Invalid graph examples include:
    - signal -> A, B -> C  (invalid: no directed path from signal to C)
    - signal -> A, A -> B  (invalid: path stops before C)
14. Valid path examples include:
    - signal -> A, A -> C
    - signal -> A, A -> B, B -> C
    - signal -> A, A -> B, B -> R, R -> C
15. Return only graphs that satisfy the path constraint. Do not return a partial or uncertain graph.
"""


@dataclass
class GRNMEIGraphConfig:
    domain_id: str
    goal: str = "Discover the hidden GRN mechanism from perturbation-response experiments."
    difficulty: str = "easy"
    law_version: str = "v0"
    noise_level: float = 0.0
    budget: int = 20
    max_experiments_per_iter: int = 4
    min_data_for_graph_fit: int = 8
    acquisition: str = "ei"
    llm_model: str = "gpt-4o-mini"
    max_completion_tokens: int = 4096
    candidate_pool_size: int = 512
    top_k_graphs: int = 5
    restarts: int = 3
    max_edges: int = 6
    max_indegree: int = 3
    max_graph_edit_distance: int = 1
    per_depth_candidate_cap: int | None = 48
    diversity_weight: float = 0.15
    state_weight: float = 0.5
    use_internal_state: bool = True
    complexity_penalty: float = 0.0
    bic_penalty_scale: float = 0.0
    max_llm_retry_attempts: int = 3
    experiment_mode: str = "baseline"
    results_dir: Path = Path("results/")
    seed: int = 0


@dataclass
class GRNMEIGraphResult:
    domain: str
    best_graph: dict[str, Any] | None
    final_evaluation: dict[str, Any] | None
    n_oracle_calls: int
    n_llm_calls: int
    duration_seconds: float
    history: list[dict] = field(default_factory=list)
    status: str = ""


def _sample_log_uniform(
    bounds: dict[str, tuple[float, float]],
    rng: np.random.Generator,
    n: int,
) -> list[dict[str, float]]:
    pts = []
    for _ in range(n):
        p: dict[str, float] = {}
        for var in GRN_INPUT_VARS:
            lo, hi = bounds[var]
            p[var] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        pts.append(p)
    return pts


def _clip_region(
    region_bounds: dict[str, list[float]],
    oracle_bounds: dict[str, tuple[float, float]],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for param in GRN_INPUT_VARS:
        lo, hi = region_bounds.get(param, list(oracle_bounds[param]))
        olo, ohi = oracle_bounds[param]
        out[param] = [float(max(olo, min(lo, ohi))), float(max(olo, min(hi, ohi)))]
        if out[param][0] > out[param][1]:
            out[param] = [float(olo), float(ohi)]
    return out


def _store_to_table(store: ExperimentStore, max_rows: int = 20) -> str:
    rows = store.get_all()[-max_rows:]
    if not rows:
        return "No experiments yet."
    header = "signal | pert_A | pert_B | pert_C | pert_R | reporter | A | B | C | R"
    lines = [header]
    for row in rows:
        panel = row.metadata.get("marker_panel") or {}
        vals = [
            f"{row.params['signal']:.4g}",
            f"{row.params['pert_A']:.4g}",
            f"{row.params['pert_B']:.4g}",
            f"{row.params['pert_C']:.4g}",
            f"{row.params['pert_R']:.4g}",
            f"{float(panel.get('reporter', row.measurement)):.6g}",
            f"{float(panel.get('A', row.metadata.get('states', {}).get('A', float('nan')))):.6g}",
            f"{float(panel.get('B', row.metadata.get('states', {}).get('B', float('nan')))):.6g}",
            f"{float(panel.get('C', row.metadata.get('states', {}).get('C', float('nan')))):.6g}",
            f"{float(panel.get('R', row.metadata.get('states', {}).get('R', float('nan')))):.6g}",
        ]
        lines.append(" | ".join(vals))
    return "\n".join(lines)


def _lineage_summary(oracle: GRNBenchOracle, lineages: list[GraphHypothesisLineage]) -> str:
    if not lineages:
        return "No fitted graph lineages yet."
    lines = []
    for lineage in lineages:
        exact_eval = oracle.evaluate_graph(lineage.exact_graph)
        best_eval = oracle.evaluate_graph(lineage.best_fit.graph)
        lines.append(
            f"[{lineage.hypothesis_id}] {lineage.hypothesis_text} | "
            f"exact={lineage.exact_fit.summary} edge_f1={exact_eval.get('edge_f1', 0.0):.3f} | "
            f"best={lineage.best_fit.summary} edge_f1={best_eval.get('edge_f1', 0.0):.3f} | "
            f"drift={lineage.drift_steps} ({lineage.drift_summary})"
        )
    return "\n".join(lines)


def _build_reasoning_prompt(
    cfg: GRNMEIGraphConfig,
    oracle: GRNBenchOracle,
    store: ExperimentStore,
    current_hypothesis: str,
    lineage_summary: str,
    budget_remaining: int,
) -> str:
    extra = ""
    if cfg.experiment_mode == "prompt":
        extra = (
            "HYPOTHESIS DIVERSITY REQUIREMENT:\n"
            "- Include at least one direct shortcut explanation (for example, signal/A acts on C directly).\n"
            "- Include at least one mediated explanation with an intermediate node between signal and C.\n"
            "- Include at least one feedback or repressor-based explanation when plausible from the data.\n"
            "- Explicitly name which perturbations would discriminate direct vs mediated vs feedback explanations.\n\n"
        )
    return (
        f"GOAL:\n{cfg.goal}\n\n"
        f"DOMAIN:\n{oracle.param_description}\n\n"
        f"CURRENT BEST HYPOTHESIS:\n{current_hypothesis}\n\n"
        f"RECENT DATA:\n{_store_to_table(store, max_rows=25)}\n\n"
        f"HYPOTHESIS FIT SUMMARY:\n{lineage_summary}\n\n"
        f"{extra}"
        f"BUDGET REMAINING: {budget_remaining}\n"
        f"MAX EXPERIMENTS THIS ITERATION: {cfg.max_experiments_per_iter}\n\n"
        "Return:\n"
        "- 2-5 natural-language mechanism hypotheses with stable ids like h1, h2, h3\n"
        "- a primary_hypothesis_id naming the current best one\n"
        "- 1-4 search regions for the next experiments\n"
        "- confidence and done flag\n"
    )


def _build_translation_prompt(
    hypotheses: list[dict[str, Any]],
) -> str:
    lines = [
        "Translate each natural-language hypothesis into one signed graph.",
        "Hypotheses:",
    ]
    for hypothesis in hypotheses:
        lines.append(f"- {hypothesis['hypothesis_id']}: {hypothesis['text']}")
    lines.append("")
    lines.append("Return one graph translation per hypothesis_id.")
    lines.append("Fill this schema exactly:")
    lines.append('{')
    lines.append('  "translations": [')
    lines.append('    {')
    lines.append('      "hypothesis_id": "h1",')
    lines.append('      "rationale": "why this graph matches the hypothesis",')
    lines.append('      "assumptions": ["optional assumption"],')
    lines.append('      "edges": [')
    lines.append('        {"src":"signal","dst":"A","sign":1},')
    lines.append('        {"src":"A","dst":"C","sign":-1}')
    lines.append('      ]')
    lines.append('    }')
    lines.append('  ]')
    lines.append('}')
    lines.append("dst must be one of A, B, C, R. Never use dst='signal'.")
    lines.append("Every translated graph MUST contain a directed path from signal to C.")
    lines.append("Before returning, check each translation by tracing a path from signal until C is reached.")
    lines.append("If a graph does not reach C, it is invalid and must be rewritten before you answer.")
    lines.append("Examples of valid paths:")
    lines.append('- signal -> A -> C')
    lines.append('- signal -> A -> B -> C')
    lines.append('- signal -> A -> B -> R -> C')
    lines.append("Examples of invalid graphs:")
    lines.append('- signal -> A and B -> C (invalid: disconnected from C)')
    lines.append('- signal -> A -> B (invalid: no edge into C)')
    return "\n".join(lines)


def _build_single_translation_prompt(
    hypothesis: dict[str, Any],
) -> str:
    return (
        "Translate this single natural-language hypothesis into one signed graph.\n\n"
        f'hypothesis_id: {hypothesis["hypothesis_id"]}\n'
        f'text: {hypothesis["text"]}\n\n'
        "Fill this schema exactly:\n"
        "{\n"
        '  "translation": {\n'
        f'    "hypothesis_id": "{hypothesis["hypothesis_id"]}",\n'
        '    "rationale": "why this graph matches the hypothesis",\n'
        '    "assumptions": ["optional assumption"],\n'
        '    "edges": [\n'
        '      {"src":"signal","dst":"A","sign":1},\n'
        '      {"src":"A","dst":"C","sign":1}\n'
        "    ]\n"
        "  }\n"
        "}\n"
        "Requirements:\n"
        "- dst must be one of A, B, C, R. Never use dst='signal'.\n"
        "- The graph must contain a directed path from signal to C.\n"
        "- Use at most 6 edges.\n"
        "- If the hypothesis does not mention C explicitly, add the minimal faithful chain needed so signal reaches C.\n"
    )


def _sample_region(bounds: dict[str, list[float]], n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.zeros((n, len(GRN_INPUT_VARS)), dtype=float)
    for j, param in enumerate(GRN_INPUT_VARS):
        lo, hi = bounds[param]
        X[:, j] = np.exp(rng.uniform(np.log(max(lo, 1e-9)), np.log(max(hi, lo * 1.001)), size=n))
    return X


def _probe_candidates(region_bounds: dict[str, list[float]]) -> np.ndarray:
    mids = {param: float(np.sqrt(bounds[0] * bounds[1])) for param, bounds in region_bounds.items()}
    los = {param: float(bounds[0]) for param, bounds in region_bounds.items()}
    his = {param: float(bounds[1]) for param, bounds in region_bounds.items()}
    candidates: list[dict[str, float]] = []
    base = {param: mids[param] for param in GRN_INPUT_VARS}
    candidates.append(base)
    signal_levels = [los["signal"], mids["signal"], his["signal"]]
    for signal in signal_levels:
        for node in ("pert_A", "pert_B", "pert_C", "pert_R"):
            low = dict(base)
            low["signal"] = signal
            low[node] = los[node]
            candidates.append(low)
            high = dict(base)
            high["signal"] = signal
            high[node] = his[node]
            candidates.append(high)
    pair_tests = [
        ("pert_A", "pert_B"),
        ("pert_A", "pert_C"),
        ("pert_B", "pert_C"),
        ("pert_R", "pert_C"),
    ]
    for a, b in pair_tests:
        pt = dict(base)
        pt["signal"] = his["signal"]
        pt[a] = los[a]
        pt[b] = his[b]
        candidates.append(pt)
        pt = dict(base)
        pt["signal"] = his["signal"]
        pt[a] = his[a]
        pt[b] = los[b]
        candidates.append(pt)
    X = np.array([[cand[param] for param in GRN_INPUT_VARS] for cand in candidates], dtype=float)
    return np.unique(X, axis=0)


def _select_graph_disagreement_points(
    fits: list[GraphHypothesisFit],
    region_bounds: dict[str, list[float]],
    n_points: int,
    candidate_pool_size: int,
    diversity_weight: float,
    rng: np.random.Generator,
    use_probe_candidates: bool = False,
) -> list[dict[str, float]]:
    X = _sample_region(region_bounds, candidate_pool_size, rng)
    if use_probe_candidates:
        probes = _probe_candidates(region_bounds)
        X = np.vstack([X, probes])
    pred_logs = []
    for fit in fits:
        y_hat = np.maximum(fit.predict(X), 1e-9)
        pred_logs.append(np.log(y_hat))
    scores = np.var(np.vstack(pred_logs), axis=0)
    order = np.argsort(scores)[::-1]
    chosen = [int(order[0])] if len(order) else []
    if len(chosen) < n_points:
        pmin = X.min(axis=0)
        pmax = X.max(axis=0) + 1e-12
        normed = (X - pmin) / (pmax - pmin)
        while len(chosen) < min(n_points, len(X)):
            best_idx = None
            best_score = -1e18
            for idx in order:
                idx = int(idx)
                if idx in chosen:
                    continue
                min_dist = min(float(np.linalg.norm(normed[idx] - normed[c])) for c in chosen)
                score = float(scores[idx]) + diversity_weight * min_dist
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            chosen.append(best_idx)
    return [{param: float(X[i, j]) for j, param in enumerate(GRN_INPUT_VARS)} for i in chosen]


class GRNMEIGraphLoop:
    def __init__(
        self,
        config: GRNMEIGraphConfig,
        api_key: str | None = None,
        oracle: BaseOracle | None = None,
        base_url: str | None = None,
    ):
        self._cfg = config
        self._oracle = oracle or GRNBenchOracle(
            domain_id=config.domain_id,
            difficulty=config.difficulty,
            law_version=config.law_version,
            noise_level=config.noise_level,
            grammar_mode="none",
        )
        self._store = ExperimentStore(config.domain_id, self._oracle.parameter_names)
        self._gp = GPSurrogate(use_log_y=True)
        self._selector = ALSelector(self._oracle, self._gp)
        self._memory = LLMMemory()
        self._client = LLMClient(
            model=config.llm_model,
            api_key=api_key,
            base_url=base_url,
            max_completion_tokens=config.max_completion_tokens,
        )
        self._rng = np.random.default_rng(config.seed)
        self._llm_calls = 0
        self._current_hypothesis = "No graph hypothesis yet."
        self._history: list[dict[str, Any]] = []

    def _reason_hypotheses(
        self,
        lineages: list[GraphHypothesisLineage],
        extra_instruction: str | None = None,
    ) -> HypothesisReasoningProposal:
        prompt = _build_reasoning_prompt(
            self._cfg,
            self._oracle,
            self._store,
            self._current_hypothesis,
            _lineage_summary(self._oracle, lineages),
            self._cfg.budget - len(self._store),
        )
        if extra_instruction:
            prompt += f"\nADDITIONAL CORRECTION:\n{extra_instruction}\n"
        self._llm_calls += 1
        return self._client.complete_json(
            messages=[{"role": "user", "content": prompt}],
            system=REASONING_SYSTEM_PROMPT,
            schema=HypothesisReasoningProposal,
            tool_name="propose_grn_hypotheses",
        )

    def _translate_hypotheses(
        self,
        reasoning: HypothesisReasoningProposal,
        extra_instruction: str | None = None,
    ) -> GraphTranslationProposal:
        translations = []
        errors: list[str] = []
        for hyp in reasoning.hypotheses:
            prompt = _build_single_translation_prompt(hyp.model_dump())
            if extra_instruction:
                prompt += f"\n\nADDITIONAL CORRECTION:\n{extra_instruction}\n"
            last_error = None
            for _attempt in range(1, self._cfg.max_llm_retry_attempts + 1):
                self._llm_calls += 1
                try:
                    payload = self._client.complete_json(
                        messages=[{"role": "user", "content": prompt}],
                        system=TRANSLATION_SYSTEM_PROMPT,
                        schema=SingleGraphTranslationProposal,
                        tool_name="translate_single_hypothesis_to_graph",
                    )
                    translations.append(payload.translation)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = str(exc)
                    prompt += (
                        "\n\nCORRECTION:\n"
                        f"The previous translation was invalid: {last_error}\n"
                        "Return exactly one valid translation for this same hypothesis_id."
                    )
            if last_error is not None:
                errors.append(f"{hyp.hypothesis_id}: {last_error}")
        if len(translations) < 2:
            raise RuntimeError(
                "Failed to translate enough valid graph hypotheses. "
                + " | ".join(errors[:5])
            )
        return GraphTranslationProposal(translations=translations)

    def _build_lineages(self, reasoning: HypothesisReasoningProposal, translation: GraphTranslationProposal) -> list[GraphHypothesisLineage]:
        hyp_map = {hyp.hypothesis_id: hyp for hyp in reasoning.hypotheses}
        structured = []
        for item in translation.translations:
            if item.hypothesis_id not in hyp_map:
                continue
            structured.append(
                {
                    "hypothesis_id": item.hypothesis_id,
                    "hypothesis_text": hyp_map[item.hypothesis_id].text,
                    "translation_rationale": item.rationale,
                    "translation_assumptions": item.assumptions,
                    "edges": [edge.model_dump() for edge in item.edges],
                }
            )
        if not structured:
            return []
        valid_structured: list[dict[str, Any]] = []
        for item in structured:
            try:
                build_hypothesis_lineages(
                    self._store,
                    [item],
                    restarts=self._cfg.restarts,
                    max_edges=self._cfg.max_edges,
                    max_indegree=self._cfg.max_indegree,
                    max_edit_distance=self._cfg.max_graph_edit_distance,
                    per_depth_candidate_cap=self._cfg.per_depth_candidate_cap,
                    complexity_penalty=self._cfg.complexity_penalty,
                    bic_penalty_scale=self._cfg.bic_penalty_scale,
                    state_weight=self._cfg.state_weight,
                    use_internal_state=self._cfg.use_internal_state,
                )
                valid_structured.append(item)
            except Exception:
                continue
        if not valid_structured:
            return []
        return build_hypothesis_lineages(
            self._store,
            valid_structured,
            restarts=self._cfg.restarts,
            max_edges=self._cfg.max_edges,
            max_indegree=self._cfg.max_indegree,
            max_edit_distance=self._cfg.max_graph_edit_distance,
            per_depth_candidate_cap=self._cfg.per_depth_candidate_cap,
            complexity_penalty=self._cfg.complexity_penalty,
            bic_penalty_scale=self._cfg.bic_penalty_scale,
            state_weight=self._cfg.state_weight,
            use_internal_state=self._cfg.use_internal_state,
        )

    def _reason_translate_until_valid(
        self,
        last_lineages: list[GraphHypothesisLineage],
    ) -> tuple[HypothesisReasoningProposal, GraphTranslationProposal, list[GraphHypothesisLineage], list[GraphHypothesisFit]]:
        correction: str | None = None
        for attempt in range(1, self._cfg.max_llm_retry_attempts + 1):
            reasoning = self._reason_hypotheses(last_lineages, extra_instruction=correction)
            primary_map = {hyp.hypothesis_id: hyp.text for hyp in reasoning.hypotheses}
            self._current_hypothesis = primary_map.get(reasoning.primary_hypothesis_id, self._current_hypothesis)
            print(f"[GRN-MEI-Graph] LLM hypothesis: {self._current_hypothesis}")

            translation = self._translate_hypotheses(reasoning, extra_instruction=correction)
            if len(self._store) < self._cfg.min_data_for_graph_fit:
                return reasoning, translation, [], []

            lineages = self._build_lineages(reasoning, translation)
            committee = select_committee_from_lineages(lineages, self._cfg.top_k_graphs)
            if len(lineages) >= 2 and len(committee) >= 2:
                return reasoning, translation, lineages, committee

            correction = (
                "Previous output did not produce at least 2 valid graph hypotheses after translation/fitting. "
                "Return at least 2 structurally distinct, valid, sparse hypotheses with explicit directed paths from signal to C. "
                "Discard any idea that requires signal as a destination or leaves C disconnected."
            )
            print(f"[GRN-MEI-Graph] retry reasoning/translation attempt={attempt} valid_lineages={len(lineages)} committee={len(committee)}")

        raise RuntimeError("Failed to obtain at least 2 valid graph hypotheses after LLM retries.")

    def _run_region(self, region_bounds: dict[str, list[float]], n_points: int, fits: list[GraphHypothesisFit]) -> list[dict[str, float]]:
        clipped = _clip_region(region_bounds, self._oracle.parameter_bounds)
        if len(fits) < 2:
            return self._selector.select(
                store=self._store,
                region_bounds=clipped,
                n_points=n_points,
                acquisition=self._cfg.acquisition,
            )
        return _select_graph_disagreement_points(
            fits=fits,
            region_bounds=clipped,
            n_points=n_points,
            candidate_pool_size=self._cfg.candidate_pool_size,
            diversity_weight=self._cfg.diversity_weight,
            rng=self._rng,
            use_probe_candidates=self._cfg.experiment_mode == "acquisition",
        )

    def _select_points_with_llm_retries(
        self,
        reasoning: HypothesisReasoningProposal,
        lineages: list[GraphHypothesisLineage],
        committee: list[GraphHypothesisFit],
    ) -> tuple[HypothesisReasoningProposal, list[dict[str, float]]]:
        if len(committee) < 2:
            remaining = self._cfg.budget - len(self._store)
            total_requested = sum(max(1, region.n_experiments) for region in reasoning.search_regions)
            batch_cap = min(self._cfg.max_experiments_per_iter, remaining)
            selected_points: list[dict[str, float]] = []
            for region in reasoning.search_regions:
                if len(selected_points) >= batch_cap:
                    break
                alloc = max(1, int(round(batch_cap * region.n_experiments / max(total_requested, 1))))
                alloc = min(alloc, batch_cap - len(selected_points))
                points = self._run_region(region.bounds, alloc, committee)
                selected_points.extend(points[:alloc])
            if selected_points:
                return reasoning, selected_points[:batch_cap]
            raise RuntimeError("Failed to obtain AL points from LLM-proposed regions before graph fitting.")

        current_reasoning = reasoning
        correction: str | None = None
        for attempt in range(1, self._cfg.max_llm_retry_attempts + 1):
            remaining = self._cfg.budget - len(self._store)
            total_requested = sum(max(1, region.n_experiments) for region in current_reasoning.search_regions)
            batch_cap = min(self._cfg.max_experiments_per_iter, remaining)
            selected_points: list[dict[str, float]] = []
            for region in current_reasoning.search_regions:
                if len(selected_points) >= batch_cap:
                    break
                alloc = max(1, int(round(batch_cap * region.n_experiments / max(total_requested, 1))))
                alloc = min(alloc, batch_cap - len(selected_points))
                points = self._run_region(region.bounds, alloc, committee)
                selected_points.extend(points[:alloc])
            if selected_points:
                return current_reasoning, selected_points[:batch_cap]
            correction = (
                "Previous search regions did not yield usable graph-disagreement experiments. "
                "Propose narrower, more discriminative regions that separate the current graph hypotheses."
            )
            print(f"[GRN-MEI-Graph] retry experiment proposal attempt={attempt}")
            current_reasoning = self._reason_hypotheses(lineages, extra_instruction=correction)
        raise RuntimeError("Failed to obtain graph-based experiment points after LLM retries.")

    def run(self) -> GRNMEIGraphResult:
        out_dir = Path(self._cfg.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        history_path = out_dir / "history.json"
        print(
            f"[GRN-MEI-Graph] {self._cfg.domain_id} | budget={self._cfg.budget} "
            f"noise={self._cfg.noise_level} model={self._cfg.llm_model}"
        )

        init_n = min(self._cfg.max_experiments_per_iter, self._cfg.budget)
        for point in _sample_log_uniform(self._oracle.parameter_bounds, self._rng, init_n):
            self._store.add(self._oracle.run(point))

        iteration = 0
        last_lineages: list[GraphHypothesisLineage] = []
        last_committee: list[GraphHypothesisFit] = []
        while len(self._store) < self._cfg.budget:
            iteration += 1
            print(f"[GRN-MEI-Graph] iter={iteration} n={len(self._store)}/{self._cfg.budget}")
            reasoning, translation, lineages, committee = self._reason_translate_until_valid(last_lineages)
            if lineages:
                last_lineages = lineages
                last_committee = committee
                print(
                    f"[GRN-MEI-Graph] best-lineage "
                    f"{lineages[0].hypothesis_id} exact={lineages[0].exact_fit.summary} "
                    f"best={lineages[0].best_fit.summary} drift={lineages[0].drift_steps}"
                )
            reasoning, selected_points = self._select_points_with_llm_retries(reasoning, lineages, committee)
            batch_cap = min(self._cfg.max_experiments_per_iter, self._cfg.budget - len(self._store))

            for point in selected_points[:batch_cap]:
                if len(self._store) >= self._cfg.budget:
                    break
                self._store.add(self._oracle.run(point))

            hist_row = {
                "iteration": iteration,
                "n_experiments": len(self._store),
                "primary_hypothesis_id": reasoning.primary_hypothesis_id,
                "natural_language_hypotheses": [hyp.model_dump() for hyp in reasoning.hypotheses],
                "translated_graphs": [item.model_dump() for item in translation.translations],
                "llm_reasoning": reasoning.reasoning,
                "confidence": reasoning.confidence,
                "search_regions": [region.model_dump() for region in reasoning.search_regions],
                "selected_points": selected_points[:batch_cap],
                "graph_lineages": [
                    {
                        "hypothesis_id": lineage.hypothesis_id,
                        "hypothesis_text": lineage.hypothesis_text,
                        "translation_rationale": lineage.translation_rationale,
                        "translation_assumptions": lineage.translation_assumptions,
                        "exact_graph": lineage.exact_graph,
                        "exact_fit": {
                            "summary": lineage.exact_fit.summary,
                            "train_rmsle": lineage.exact_fit.train_rmsle,
                            "reporter_rmsle": lineage.exact_fit.reporter_rmsle,
                            "state_rmsle": lineage.exact_fit.state_rmsle,
                            "score": lineage.exact_fit.score,
                            "graph_eval": self._oracle.evaluate_graph(lineage.exact_graph),
                        },
                        "best_graph": lineage.best_fit.graph,
                        "best_fit": {
                            "summary": lineage.best_fit.summary,
                            "train_rmsle": lineage.best_fit.train_rmsle,
                            "reporter_rmsle": lineage.best_fit.reporter_rmsle,
                            "state_rmsle": lineage.best_fit.state_rmsle,
                            "score": lineage.best_fit.score,
                            "graph_eval": self._oracle.evaluate_graph(lineage.best_fit.graph),
                        },
                        "drift_steps": lineage.drift_steps,
                        "drift_summary": lineage.drift_summary,
                    }
                    for lineage in lineages
                ],
                "committee_graphs": [
                    {
                        "edges": fit.graph["edges"],
                        "summary": fit.summary,
                        "train_rmsle": fit.train_rmsle,
                        "reporter_rmsle": fit.reporter_rmsle,
                        "state_rmsle": fit.state_rmsle,
                        "score": fit.score,
                        "graph_eval": self._oracle.evaluate_graph(fit.graph),
                    }
                    for fit in committee
                ],
            }
            self._history.append(hist_row)
            history_path.write_text(json.dumps(self._history, indent=2))

        if not last_lineages:
            _reasoning, _translation, last_lineages, last_committee = self._reason_translate_until_valid([])

        best_graph = last_committee[0].graph if last_committee else None
        final_eval = self._oracle.evaluate_graph(best_graph) if best_graph else None
        self._store.save(out_dir / "experiments.json")
        summary = {
            "domain": self._cfg.domain_id,
            "difficulty": self._cfg.difficulty,
            "law_version": self._cfg.law_version,
            "budget": self._cfg.budget,
            "n_oracle_calls": len(self._store),
            "n_llm_calls": self._llm_calls,
            "duration_s": time.time() - t0,
            "best_graph": best_graph,
            "final_graph_eval": final_eval,
            "final_hypothesis": self._current_hypothesis,
            "graph_lineages": [
                {
                    "hypothesis_id": lineage.hypothesis_id,
                    "hypothesis_text": lineage.hypothesis_text,
                    "exact_graph": lineage.exact_graph,
                    "exact_fit": {
                        "summary": lineage.exact_fit.summary,
                        "train_rmsle": lineage.exact_fit.train_rmsle,
                        "reporter_rmsle": lineage.exact_fit.reporter_rmsle,
                        "state_rmsle": lineage.exact_fit.state_rmsle,
                        "score": lineage.exact_fit.score,
                        "graph_eval": self._oracle.evaluate_graph(lineage.exact_graph),
                    },
                    "best_graph": lineage.best_fit.graph,
                    "best_fit": {
                        "summary": lineage.best_fit.summary,
                        "train_rmsle": lineage.best_fit.train_rmsle,
                        "reporter_rmsle": lineage.best_fit.reporter_rmsle,
                        "state_rmsle": lineage.best_fit.state_rmsle,
                        "score": lineage.best_fit.score,
                        "graph_eval": self._oracle.evaluate_graph(lineage.best_fit.graph),
                    },
                    "drift_steps": lineage.drift_steps,
                    "drift_summary": lineage.drift_summary,
                }
                for lineage in last_lineages
            ],
            "committee_graphs": [
                {
                    "edges": fit.graph["edges"],
                    "summary": fit.summary,
                    "train_rmsle": fit.train_rmsle,
                    "reporter_rmsle": fit.reporter_rmsle,
                    "state_rmsle": fit.state_rmsle,
                    "score": fit.score,
                    "graph_eval": self._oracle.evaluate_graph(fit.graph),
                }
                for fit in last_committee
            ],
            "status": "completed" if final_eval else "no_graph",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return GRNMEIGraphResult(
            domain=self._cfg.domain_id,
            best_graph=best_graph,
            final_evaluation=final_eval,
            n_oracle_calls=len(self._store),
            n_llm_calls=self._llm_calls,
            duration_seconds=summary["duration_s"],
            history=self._history,
            status=summary["status"],
        )
