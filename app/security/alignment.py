from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(slots=True)
class AlignmentResult:
    allowed: bool
    refused: bool
    risk_labels: list[str]
    response: str
    safe_alternatives: list[str] = field(default_factory=list)
    confidence: float = 1.0


class AlignmentLayer:
    """Safety alignment layer with harmful-output checks, misinformation checks, and refusal fallback."""

    def __init__(self) -> None:
        self.harmful_patterns = [
            r"\b(build|make)\s+(a\s+)?bomb\b",
            r"\bkill\s+someone\b",
            r"\bmalware\b",
            r"\bddos\b",
        ]
        self.unsafe_instruction_patterns = [
            r"\bdisable\s+safety\b",
            r"\bbypass\s+security\b",
            r"\bsteal\s+password\b",
            r"\bexploit\s+vulnerability\b",
        ]
        self.misinformation_patterns = [
            r"\bearth\s+is\s+flat\b",
            r"\bvaccines\s+cause\s+autism\b",
            r"\b5g\s+causes\s+disease\b",
        ]

    def _detect(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

    def _safe_alternatives(self, labels: list[str]) -> list[str]:
        alternatives = []
        if "harmful_outputs" in labels or "unsafe_instructions" in labels:
            alternatives.extend(
                [
                    "I can help with legal safety practices and risk prevention.",
                    "I can provide incident response, defense, and secure design guidance.",
                ]
            )
        if "misinformation" in labels:
            alternatives.extend(
                [
                    "I can provide evidence-based information from reputable public health and scientific sources.",
                    "I can help verify claims with citations and uncertainty labels.",
                ]
            )
        if not alternatives:
            alternatives.append("I can help with a safer, policy-compliant version of your request.")
        return alternatives

    def _refusal_message(self, labels: list[str]) -> str:
        return (
            "I can’t assist with that request because it may cause harm, spread misinformation, "
            "or provide unsafe instructions."
        )

    def evaluate(self, generated_response: str) -> AlignmentResult:
        labels: list[str] = []
        if self._detect(generated_response, self.harmful_patterns):
            labels.append("harmful_outputs")
        if self._detect(generated_response, self.misinformation_patterns):
            labels.append("misinformation")
        if self._detect(generated_response, self.unsafe_instruction_patterns):
            labels.append("unsafe_instructions")

        if labels:
            safe_alts = self._safe_alternatives(labels)
            return AlignmentResult(
                allowed=False,
                refused=True,
                risk_labels=labels,
                response=self._refusal_message(labels),
                safe_alternatives=safe_alts,
                confidence=0.15,
            )

        return AlignmentResult(
            allowed=True,
            refused=False,
            risk_labels=[],
            response=generated_response,
            safe_alternatives=[],
            confidence=0.95,
        )
