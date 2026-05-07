"""
LLMMemory: persistent per-run memory of discovered equations.

Injected into every LLM proposal prompt so the LLM can see what PySR has
already found and update its hypothesis accordingly — closing the feedback loop
between the equation learner and the experiment-proposal agent.

gt_rmsle is populated mid-run when the oracle's evaluate_law() is called on
high-confidence fits (R² > 0.95), providing the LLM with ground-truth quality
signal to guide further exploration or confirm termination.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class EquationRecord:
    iteration: int
    n_data: int
    equation: str       # sympy string form
    r2: float
    rmsle: float
    bootstrap_conf: float
    gt_rmsle: float | None = None   # ground-truth RMSLE from evaluate_law() mid-run


@dataclass
class NegativeEvidenceRecord:
    iteration: int
    hypothesis: str
    evidence: str
    gt_rmsle: float | None = None


@dataclass
class ValidatedHypothesisRecord:
    """A hypothesis validated by discriminating experiments — not necessarily a PySR equation."""
    iteration: int
    hypothesis: str        # raw hypothesis string from LLM
    fit_rmsle: float       # RMSLE on the discriminating data
    n_data: int            # number of points used to score it


@dataclass
class HypothesisScoreRecord:
    """Archive of all scored hypotheses across the run."""
    iteration: int
    hypothesis: str
    canonical_hypothesis: str
    fit_rmsle: float
    n_data: int
    status: Literal["validated", "failed", "uncertain", "unscored"]


@dataclass
class ConsultantAdviceRecord:
    """Advice from the strong-model consultant at a fixed trigger point."""
    call_number: int    # 1, 2, or 3
    trigger:     str    # "patterns" | "residuals" | "final"
    advice:      str


class LLMMemory:
    """
    Accumulates the history of equations discovered by PySR across iterations.
    Produces a compact text block that is injected into every LLM proposal prompt.
    """

    def __init__(self) -> None:
        self._records: list[EquationRecord] = []
        self._negative_evidence: list[NegativeEvidenceRecord] = []
        self._validated_hypotheses: list[ValidatedHypothesisRecord] = []
        self._hypothesis_score_archive: list[HypothesisScoreRecord] = []
        self._hypothesis_stats: dict[str, dict[str, float | int | str]] = {}
        self._consultant_advice: list[ConsultantAdviceRecord] = []

    def _canonicalise_hypothesis(self, hypothesis: str) -> str:
        """Normalize equation-like or prose hypothesis for stable dedupe keys."""
        import re

        h = (hypothesis or "").strip().lower()
        if not h:
            return ""
        # Strip leading assignment (e.g., "i = ...")
        m = re.match(r"^[a-z_]\w*\s*=\s*", h)
        if m:
            h = h[m.end():]
        # Normalize exponent operator and whitespace.
        h = h.replace("^", "**")
        h = re.sub(r"\s+", " ", h).strip()
        return h

    def add_hypothesis_score(
        self,
        iteration: int,
        hypothesis: str,
        fit_rmsle: float,
        n_data: int,
        status: Literal["validated", "failed", "uncertain", "unscored"],
    ) -> None:
        """Persist every scored hypothesis and aggregate stats by canonical form."""
        hyp = (hypothesis or "").strip()
        if not hyp:
            return
        canon = self._canonicalise_hypothesis(hyp)
        if not canon:
            return

        self._hypothesis_score_archive.append(
            HypothesisScoreRecord(
                iteration=iteration,
                hypothesis=hyp,
                canonical_hypothesis=canon,
                fit_rmsle=float(fit_rmsle),
                n_data=int(n_data),
                status=status,
            )
        )

        stats = self._hypothesis_stats.get(
            canon,
            {
                "example": hyp,
                "validated_count": 0,
                "failed_count": 0,
                "uncertain_count": 0,
                "unscored_count": 0,
                "best_rmsle": float("inf"),
                "last_rmsle": float("inf"),
                "last_iter": 0,
            },
        )
        if status == "validated":
            stats["validated_count"] = int(stats["validated_count"]) + 1
            stats["best_rmsle"] = min(float(stats["best_rmsle"]), float(fit_rmsle))
        elif status == "failed":
            stats["failed_count"] = int(stats["failed_count"]) + 1
            stats["best_rmsle"] = min(float(stats["best_rmsle"]), float(fit_rmsle))
        elif status == "uncertain":
            stats["uncertain_count"] = int(stats["uncertain_count"]) + 1
            stats["best_rmsle"] = min(float(stats["best_rmsle"]), float(fit_rmsle))
        else:
            stats["unscored_count"] = int(stats["unscored_count"]) + 1

        stats["last_rmsle"] = float(fit_rmsle)
        stats["last_iter"] = int(iteration)
        self._hypothesis_stats[canon] = stats

    def should_avoid_hypothesis(self, hypothesis: str, min_failed: int = 2) -> bool:
        """
        Suppress repeatedly falsified hypotheses unless they have validated support.
        """
        canon = self._canonicalise_hypothesis(hypothesis)
        if not canon:
            return False
        stats = self._hypothesis_stats.get(canon)
        if not stats:
            return False
        failed = int(stats.get("failed_count", 0))
        validated = int(stats.get("validated_count", 0))
        return failed >= min_failed and validated == 0

    def validated_support(self, hypothesis: str) -> int:
        """How many times this (canonical) hypothesis has been validated."""
        canon = self._canonicalise_hypothesis(hypothesis)
        if not canon:
            return 0
        stats = self._hypothesis_stats.get(canon)
        if not stats:
            return 0
        return int(stats.get("validated_count", 0))

    def best_hypothesis_stats(self, top_k: int = 10) -> list[dict[str, float | int | str]]:
        """Top canonical hypothesis stats, ranked by validation support then best RMSLE."""
        stats = list(self._hypothesis_stats.values())
        stats.sort(
            key=lambda s: (
                -int(s.get("validated_count", 0)),
                float(s.get("best_rmsle", float("inf"))),
                int(s.get("failed_count", 0)),
            )
        )
        return stats[:top_k]

    def add(
        self,
        iteration: int,
        n_data: int,
        equation: str,
        r2: float,
        rmsle: float,
        bootstrap_conf: float,
        gt_rmsle: float | None = None,
    ) -> None:
        self._records.append(
            EquationRecord(iteration, n_data, equation, r2, rmsle, bootstrap_conf,
                           gt_rmsle=gt_rmsle)
        )

    def best(self) -> EquationRecord | None:
        if not self._records:
            return None
        return max(self._records, key=lambda r: r.r2)

    def best_gt_rmsle(self) -> float | None:
        """Return the best (lowest) GT RMSLE seen so far, or None if none available."""
        records_with_gt = [r for r in self._records if r.gt_rmsle is not None]
        if not records_with_gt:
            return None
        return min(r.gt_rmsle for r in records_with_gt)

    def top_k(self, k: int = 3) -> list[EquationRecord]:
        """Return top-k equations by R²."""
        return sorted(self._records, key=lambda r: r.r2, reverse=True)[:k]

    def add_validated_hypothesis(
        self,
        iteration: int,
        hypothesis: str,
        fit_rmsle: float,
        n_data: int,
    ) -> None:
        """Record a hypothesis that survived discriminating experiments."""
        hyp = (hypothesis or "").strip()
        if not hyp:
            return
        # Avoid exact duplicates (keep better-scoring one if already present)
        for i, existing in enumerate(self._validated_hypotheses):
            if existing.hypothesis == hyp:
                if fit_rmsle < existing.fit_rmsle:
                    self._validated_hypotheses[i] = ValidatedHypothesisRecord(
                        iteration=iteration, hypothesis=hyp,
                        fit_rmsle=fit_rmsle, n_data=n_data,
                    )
                return
        self._validated_hypotheses.append(
            ValidatedHypothesisRecord(
                iteration=iteration, hypothesis=hyp,
                fit_rmsle=fit_rmsle, n_data=n_data,
            )
        )
        self.add_hypothesis_score(
            iteration=iteration,
            hypothesis=hyp,
            fit_rmsle=fit_rmsle,
            n_data=n_data,
            status="validated",
        )

    def get_validated_hypotheses(self, top_k: int = 5) -> list[ValidatedHypothesisRecord]:
        """Return the top-k validated hypotheses ordered by fit quality."""
        return sorted(self._validated_hypotheses, key=lambda r: r.fit_rmsle)[:top_k]

    def add_consultant_advice(
        self,
        call_number: int,
        trigger: str,
        advice: str,
    ) -> None:
        """Store advice from the strong-model consultant for injection into prompts."""
        if not advice or not advice.strip():
            return
        self._consultant_advice.append(
            ConsultantAdviceRecord(
                call_number=call_number,
                trigger=trigger,
                advice=advice.strip(),
            )
        )

    def add_negative_evidence(
        self,
        iteration: int,
        hypothesis: str,
        evidence: str,
        gt_rmsle: float | None = None,
    ) -> None:
        hyp = (hypothesis or "").strip()
        ev = (evidence or "").strip()
        if not hyp or not ev:
            return
        self._negative_evidence.append(
            NegativeEvidenceRecord(
                iteration=iteration,
                hypothesis=hyp,
                evidence=ev,
                gt_rmsle=gt_rmsle,
            )
        )
        if gt_rmsle is not None:
            self.add_hypothesis_score(
                iteration=iteration,
                hypothesis=hyp,
                fit_rmsle=float(gt_rmsle),
                n_data=0,
                status="failed",
            )

    def to_prompt_str(self) -> str:
        """
        Compact summary injected into the LLM proposal prompt.
        Forces the LLM to acknowledge PySR's finding and update its hypothesis.
        """
        has_content = bool(
            self._records or self._validated_hypotheses
            or self._negative_evidence or self._consultant_advice
        )
        if not has_content:
            return ""

        lines: list[str] = []
        if not self._records:
            # Still emit validated/negative sections even without PySR output
            validated = self.get_validated_hypotheses(top_k=4)
            if validated:
                lines += [
                    "",
                    "=== VALIDATED HYPOTHESIS CANDIDATES (survived discriminating experiments) ===",
                ]
                for rec in validated:
                    lines.append(
                        f"- [iter {rec.iteration}, n={rec.n_data}] {rec.hypothesis} "
                        f"| fit_RMSLE={rec.fit_rmsle:.4f}"
                    )
                lines += [
                    "INSTRUCTION: These structural forms were empirically consistent with data collected",
                    "specifically to distinguish competing hypotheses. Use them as seeds for your next",
                    "hypothesis. Prefer variants or refinements of the survivors over new untested forms.",
                    "=== END VALIDATED CANDIDATES ===",
                    "",
                ]
            if self._negative_evidence:
                lines += ["=== NEGATIVE EVIDENCE (falsified/poor hypotheses) ==="]
                for rec in self._negative_evidence:
                    gt = f" | GT_RMSLE={rec.gt_rmsle:.4f}" if rec.gt_rmsle is not None else ""
                    lines.append(f"- [iter {rec.iteration}] Reject/de-prioritize: {rec.hypothesis}{gt}")
                    lines.append(f"  Evidence: {rec.evidence}")
                lines += [
                    "INSTRUCTION: Avoid proposing experiments that only reconfirm these rejected forms.",
                    "=== END NEGATIVE EVIDENCE ===", "",
                ]
            if self._hypothesis_stats:
                lines += [
                    "=== HYPOTHESIS SCOREBOARD (ALL SCORED FORMS) ===",
                ]
                for stats in sorted(
                    self._hypothesis_stats.values(),
                    key=lambda s: (-int(s.get("validated_count", 0)), int(s.get("failed_count", 0))),
                ):
                    lines.append(
                        f"- {stats.get('example', '')} | "
                        f"validated={int(stats.get('validated_count', 0))} "
                        f"failed={int(stats.get('failed_count', 0))} "
                        f"uncertain={int(stats.get('uncertain_count', 0))} "
                        f"best_rmsle={float(stats.get('best_rmsle', float('inf'))):.4f}"
                    )
                lines += [
                    "INSTRUCTION: Do not repeat hypotheses with repeated failures unless new evidence strongly contradicts those failures.",
                    "=== END HYPOTHESIS SCOREBOARD ===",
                    "",
                ]
            if self._consultant_advice:
                lines += ["=== SCIENTIFIC ADVISOR CONSULTATIONS ==="]
                for rec in self._consultant_advice:
                    lines.append(f"[Consultation {rec.call_number}/3 — {rec.trigger.upper()}]")
                    for advice_line in rec.advice.splitlines():
                        lines.append(f"  {advice_line}")
                lines += [
                    "",
                    "INSTRUCTION: Treat the advisor's analysis as high-quality domain knowledge.",
                    "Update your hypothesis and experimental strategy to reflect its findings.",
                    "=== END ADVISOR CONSULTATIONS ===",
                    "",
                ]
            return "\n".join(lines)

        best = self.best()
        lines = [
            "",
            "=== EQUATION DISCOVERY HISTORY (from symbolic regression) ===",
        ]
        for r in self._records:
            marker = "⭐" if r is best else "  "
            gt_str = f" | GT_RMSLE={r.gt_rmsle:.5f}" if r.gt_rmsle is not None else ""
            lines.append(
                f"{marker} [iter {r.iteration}, n={r.n_data}] "
                f"{r.equation} | R²={r.r2:.4f} | RMSLE={r.rmsle:.4f} | "
                f"conf={r.bootstrap_conf:.2f}{gt_str}"
            )

        lines += [
            "",
            f"BEST EQUATION FOUND SO FAR: {best.equation}",
            f"  R²={best.r2:.4f}  RMSLE={best.rmsle:.4f}  bootstrap_confidence={best.bootstrap_conf:.2f}",
        ]

        best_gt = self.best_gt_rmsle()
        if best_gt is not None:
            quality = "✓ EXCELLENT" if best_gt < 0.01 else ("≈ GOOD" if best_gt < 0.1 else "✗ POOR")
            lines.append(
                f"  Ground-truth RMSLE (mid-run eval): {best_gt:.5f}  {quality}"
            )
            if best_gt < 0.05:
                lines += [
                    "",
                    "⚠ STRONG MATCH WITH GROUND TRUTH DETECTED.",
                    "Your next proposal should VALIDATE this equation at extreme parameter",
                    "values rather than continuing broad exploration.",
                ]

        lines += [
            "",
            "INSTRUCTION: Update your hypothesis field to match this equation exactly",
            "(or explain clearly in your reasoning why you think it is wrong).",
            "Do NOT keep repeating a vague 'k * x^n' hypothesis if a specific equation",
            "has already been found with high R² and bootstrap confidence.",
            "=== END EQUATION HISTORY ===",
            "",
        ]

        validated = self.get_validated_hypotheses(top_k=4)
        if validated:
            lines += [
                "=== VALIDATED HYPOTHESIS CANDIDATES (survived discriminating experiments) ===",
            ]
            for rec in validated:
                lines.append(
                    f"- [iter {rec.iteration}, n={rec.n_data}] {rec.hypothesis} "
                    f"| fit_RMSLE={rec.fit_rmsle:.4f}"
                )
            lines += [
                "INSTRUCTION: These structural forms were empirically consistent with data collected",
                "specifically to distinguish competing hypotheses. Use them as seeds for your next",
                "hypothesis. Prefer variants or refinements of the survivors over new untested forms.",
                "=== END VALIDATED CANDIDATES ===",
                "",
            ]

        if self._negative_evidence:
            lines += [
                "=== NEGATIVE EVIDENCE (falsified/poor hypotheses) ===",
            ]
            for rec in self._negative_evidence:
                gt = f" | GT_RMSLE={rec.gt_rmsle:.4f}" if rec.gt_rmsle is not None else ""
                lines.append(
                    f"- [iter {rec.iteration}] Reject/de-prioritize: {rec.hypothesis}{gt}"
                )
                lines.append(f"  Evidence: {rec.evidence}")
            lines += [
                "INSTRUCTION: Avoid proposing experiments that only reconfirm these rejected forms.",
                "Prioritize regions likely to separate currently plausible alternatives.",
                "=== END NEGATIVE EVIDENCE ===",
                "",
            ]
        if self._hypothesis_stats:
            lines += [
                "=== HYPOTHESIS SCOREBOARD (ALL SCORED FORMS) ===",
            ]
            for stats in sorted(
                self._hypothesis_stats.values(),
                key=lambda s: (-int(s.get("validated_count", 0)), int(s.get("failed_count", 0))),
            ):
                lines.append(
                    f"- {stats.get('example', '')} | "
                    f"validated={int(stats.get('validated_count', 0))} "
                    f"failed={int(stats.get('failed_count', 0))} "
                    f"uncertain={int(stats.get('uncertain_count', 0))} "
                    f"best_rmsle={float(stats.get('best_rmsle', float('inf'))):.4f}"
                )
            lines += [
                "INSTRUCTION: Do not repeat hypotheses with repeated failures unless new evidence strongly contradicts those failures.",
                "=== END HYPOTHESIS SCOREBOARD ===",
                "",
            ]

        if self._consultant_advice:
            lines += ["=== SCIENTIFIC ADVISOR CONSULTATIONS ==="]
            for rec in self._consultant_advice:
                lines.append(
                    f"[Consultation {rec.call_number}/3 — {rec.trigger.upper()}]"
                )
                # Indent each line of the advice for readability
                for advice_line in rec.advice.splitlines():
                    lines.append(f"  {advice_line}")
            lines += [
                "",
                "INSTRUCTION: Treat the advisor's analysis as high-quality domain knowledge.",
                "Update your hypothesis and experimental strategy to reflect its findings.",
                "=== END ADVISOR CONSULTATIONS ===",
                "",
            ]

        return "\n".join(lines)


class NoOpMemory(LLMMemory):
    """No-op memory used for ablations."""

    def __init__(self) -> None:
        pass

    def add_hypothesis_score(self, *args, **kwargs) -> None:
        return None

    def should_avoid_hypothesis(self, *args, **kwargs) -> bool:
        return False

    def validated_support(self, *args, **kwargs) -> int:
        return 0

    def best_hypothesis_stats(self, *args, **kwargs) -> list[dict[str, float | int | str]]:
        return []

    def add(self, *args, **kwargs) -> None:
        return None

    def best(self):
        return None

    def best_gt_rmsle(self) -> float | None:
        return None

    def top_k(self, *args, **kwargs):
        return []

    def add_validated_hypothesis(self, *args, **kwargs) -> None:
        return None

    def get_validated_hypotheses(self, *args, **kwargs):
        return []

    def add_consultant_advice(self, *args, **kwargs) -> None:
        return None

    def add_negative_evidence(self, *args, **kwargs) -> None:
        return None

    def add_observation(self, *args, **kwargs) -> None:
        return None

    def to_prompt_str(self) -> str:
        return ""
