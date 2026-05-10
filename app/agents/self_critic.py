from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class CritiqueResult:
    improved_response: str
    flagged_error: bool
    confidence: float
    issues: list[str]


class SelfCritic:
    """Post-generation self-critique module for consistency, contradictions, and completeness."""

    def _logical_consistency_check(self, response: str) -> list[str]:
        issues: list[str] = []
        if response.count("therefore") > 0 and response.count("because") == 0:
            issues.append("inference_without_explicit_support")
        if len(response.split()) < 6:
            issues.append("insufficient_explanation_length")
        return issues

    def _contradiction_check(self, response: str) -> list[str]:
        issues: list[str] = []
        lower = response.lower()
        contradiction_pairs = [
            ("always", "never"),
            ("must", "must not"),
            ("is true", "is false"),
        ]
        for a, b in contradiction_pairs:
            if a in lower and b in lower:
                issues.append(f"contradiction:{a}|{b}")
        return issues

    def _completeness_check(self, response: str) -> list[str]:
        issues: list[str] = []
        has_step_markers = bool(re.search(r"\b(1\.|2\.|step|first|second|finally)\b", response.lower()))
        if not has_step_markers:
            issues.append("missing_structured_steps")
        if "TODO" in response:
            issues.append("contains_placeholder")
        return issues

    def _score(self, issues: list[str]) -> float:
        penalties = {
            "inference_without_explicit_support": 0.15,
            "insufficient_explanation_length": 0.2,
            "missing_structured_steps": 0.1,
            "contains_placeholder": 0.3,
        }
        base = 1.0
        score = base
        for issue in issues:
            score -= penalties.get(issue, 0.25 if issue.startswith("contradiction:") else 0.1)
        return round(max(0.0, min(1.0, score)), 3)

    def critique(self, generated_response: str) -> CritiqueResult:
        issues = []
        issues.extend(self._logical_consistency_check(generated_response))
        issues.extend(self._contradiction_check(generated_response))
        issues.extend(self._completeness_check(generated_response))

        confidence = self._score(issues)
        flagged_error = any(issue.startswith("contradiction:") for issue in issues) or confidence < 0.45

        improved = generated_response
        if "missing_structured_steps" in issues:
            improved = f"Step 1: Clarify the goal.\nStep 2: Present reasoning with evidence.\nStep 3: Provide final recommendation.\n\n{improved}"
        if "inference_without_explicit_support" in issues:
            improved += "\n\nAdded support: include explicit evidence before final inference."
        if "contains_placeholder" in issues:
            improved = improved.replace("TODO", "[resolved detail required]")

        return CritiqueResult(
            improved_response=improved,
            flagged_error=flagged_error,
            confidence=confidence,
            issues=issues,
        )
