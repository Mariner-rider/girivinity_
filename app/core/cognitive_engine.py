from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ThoughtChain:
    query: str
    steps: list[str] = field(default_factory=list)
    sub_problems: list[str] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    reasoning_type: str = "direct"


REASONING_TRIGGERS = {
    "multi_step": [
        "how",
        "why",
        "explain",
        "compare",
        "difference",
        "analyze",
        "evaluate",
        "what would happen",
        "if",
        "because",
        "therefore",
        "calculate",
        "solve",
    ],
    "creative": [
        "write",
        "create",
        "design",
        "imagine",
        "story",
        "poem",
        "suggest",
        "idea",
        "generate",
    ],
    "factual": [
        "what is",
        "who is",
        "when",
        "where",
        "define",
        "list",
        "name",
        "which",
    ],
    "technical": [
        "code",
        "implement",
        "debug",
        "error",
        "function",
        "algorithm",
        "kernel",
        "cuda",
        "python",
        "fix",
    ],
}


class CognitiveEngine:
    """Structured reasoning before synthesis."""

    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        ce = cfg.get("cognitive_engine", {})
        self.max_sub_problems = int(ce.get("max_sub_problems", 4))
        self.confidence_threshold = float(ce.get("confidence_threshold", 0.6))
        self.verbose_reasoning = bool(ce.get("verbose_reasoning", False))

    def think(self, query: str, context: str) -> ThoughtChain:
        reasoning_type = self._classify_reasoning(query)
        sub_problems = self._decompose(query, reasoning_type)
        steps = self._build_thought_steps(query, sub_problems, context, reasoning_type)
        conclusion = self._synthesise_conclusion(steps)
        confidence = self._score_confidence(steps, context, reasoning_type)

        chain = ThoughtChain(
            query=query,
            steps=steps,
            sub_problems=sub_problems,
            conclusion=conclusion,
            confidence=confidence,
            reasoning_type=reasoning_type,
        )

        logger.info(
            "CognitiveEngine: type=%s sub_problems=%d confidence=%.2f",
            reasoning_type,
            len(sub_problems),
            confidence,
        )
        return chain

    def build_enriched_prompt(self, chain: ThoughtChain, base_prompt: str) -> str:
        if chain.reasoning_type == "factual" and chain.confidence > 0.8:
            return base_prompt

        reasoning_block = "\n".join(
            [
                "Let me think through this carefully:",
                *[f"Step {i + 1}: {step}" for i, step in enumerate(chain.steps)],
                f"Conclusion: {chain.conclusion}",
                "",
            ]
        )
        return base_prompt.replace("Question:", f"{reasoning_block}\nQuestion:")

    def _classify_reasoning(self, query: str) -> str:
        q = query.lower()
        scores: dict[str, int] = {}
        for rtype, keywords in REASONING_TRIGGERS.items():
            scores[rtype] = sum(1 for kw in keywords if kw in q)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "direct"

    def _decompose(self, query: str, reasoning_type: str) -> list[str]:
        if reasoning_type in ("factual", "direct"):
            return []
        if len(query.split()) < 8:
            return []

        sub_problems = []

        if "and" in query.lower():
            parts = re.split(r"\band\b", query, flags=re.IGNORECASE)
            sub_problems.extend([p.strip() for p in parts if len(p.strip()) > 5])

        if "compare" in query.lower() or "difference" in query.lower():
            entities = re.findall(r"\b[A-Z][a-z]+\b|\b\w+(?:\s+\w+)?\b", query)
            if len(entities) >= 2:
                sub_problems.append(f"Understand: {entities[0]}")
                sub_problems.append(f"Understand: {entities[1]}")
                sub_problems.append("Compare the two")

        if reasoning_type == "technical":
            sub_problems.extend(
                [
                    "Understand what is being asked",
                    "Identify constraints and requirements",
                    "Design the solution approach",
                    "Verify correctness",
                ]
            )

        return sub_problems[: self.max_sub_problems]

    def _build_thought_steps(
        self,
        query: str,
        sub_problems: list[str],
        context: str,
        reasoning_type: str,
    ) -> list[str]:
        steps = [f"The question asks about: {query[:100]}"]

        if context:
            steps.append(f"Available context covers: {context[:150].strip()}...")

        for sp in sub_problems:
            steps.append(f"Breaking down: {sp}")

        if reasoning_type == "technical":
            steps.append("Checking for edge cases and constraints")
        elif reasoning_type == "multi_step":
            steps.append("Connecting the logical chain of reasoning")
        elif reasoning_type == "creative":
            steps.append("Considering structure, tone, and creativity")

        return steps

    def _synthesise_conclusion(self, steps: list[str]) -> str:
        if not steps:
            return "Proceeding with direct answer"
        return (
            f"Based on {len(steps)} reasoning steps, "
            "I can now form a well-structured answer"
        )

    def _score_confidence(self, steps: list[str], context: str, reasoning_type: str) -> float:
        base = 0.5
        if context:
            base += 0.2
        if len(steps) >= 3:
            base += 0.1
        if reasoning_type == "factual":
            base += 0.1
        return min(1.0, base)
