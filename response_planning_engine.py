"""ResponsePlanningEngine — chooses response structure before generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ResponseFormat(Enum):
    PROSE = "prose"
    BULLET_POINTS = "bullet_points"
    CODE = "code"
    TABLE = "table"
    STEP_BY_STEP = "step_by_step"
    MIXED = "mixed"


class ResponseLength(Enum):
    ONE_SENTENCE = "one_sentence"
    SHORT = "short"
    MEDIUM = "medium"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass(frozen=True, slots=True)
class ResponsePlan:
    format: ResponseFormat
    length: ResponseLength
    tone: str
    sections: list[str]
    use_examples: bool
    use_analogies: bool
    use_code_blocks: bool

    def to_directive(self) -> str:
        parts = [f"Format your response as: {self.format.value}", f"Length: {self.length.value}", f"Tone: {self.tone}"]
        if self.sections:
            parts.append(f"Include these sections: {', '.join(self.sections)}")
        if self.use_examples:
            parts.append("Include a concrete example.")
        if self.use_analogies:
            parts.append("Use an analogy to explain the core concept.")
        if self.use_code_blocks:
            parts.append("Include code blocks where appropriate.")
        return "\n".join(parts)


class ResponsePlanningEngine:
    def plan(self, task: str, user_model: Any = None, sentiment: dict[str, Any] | None = None) -> ResponsePlan:
        task_lower = (task or "").lower()
        expertise = float(getattr(user_model, "expertise_level", 3.0)) if user_model else 3.0
        emotion = (sentiment or {}).get("dominant_emotion", "neutral")

        fmt = self._determine_format(task_lower)
        length = self._determine_length(task_lower, len((task or "").split()))
        tone = self._determine_tone(expertise, emotion)

        return ResponsePlan(
            format=fmt,
            length=length,
            tone=tone,
            sections=self._determine_sections(task_lower, fmt),
            use_examples=expertise <= 3.5 or "example" in task_lower,
            use_analogies=expertise <= 2.5,
            use_code_blocks=fmt in (ResponseFormat.CODE, ResponseFormat.MIXED),
        )

    def _determine_format(self, task_lower: str) -> ResponseFormat:
        if any(word in task_lower for word in ("code", "script", "function", "implement", "write a")):
            return ResponseFormat.CODE
        if any(word in task_lower for word in ("compare", "difference", "vs", "table")):
            return ResponseFormat.TABLE
        if any(word in task_lower for word in ("steps", "how to", "guide", "tutorial", "process")):
            return ResponseFormat.STEP_BY_STEP
        if any(word in task_lower for word in ("list", "what are", "give me", "examples of")):
            return ResponseFormat.BULLET_POINTS
        return ResponseFormat.PROSE

    def _determine_length(self, task_lower: str, word_count: int) -> ResponseLength:
        if "one sentence" in task_lower:
            return ResponseLength.ONE_SENTENCE
        if "comprehensive" in task_lower or "detailed" in task_lower or "full" in task_lower:
            return ResponseLength.COMPREHENSIVE
        if word_count < 5:
            return ResponseLength.SHORT
        if word_count < 15:
            return ResponseLength.MEDIUM
        return ResponseLength.DETAILED

    def _determine_tone(self, expertise: float, emotion: str) -> str:
        if emotion in ("frustration", "anger"):
            return "calm_and_supportive"
        if emotion == "curiosity":
            return "enthusiastic_and_educational"
        if expertise >= 4.0:
            return "technical_and_concise"
        if expertise <= 2.0:
            return "simple_and_encouraging"
        return "balanced"

    def _determine_sections(self, task_lower: str, fmt: ResponseFormat) -> list[str]:
        if fmt == ResponseFormat.STEP_BY_STEP:
            return ["Overview", "Steps", "Expected Outcome"]
        if "explain" in task_lower or "what is" in task_lower:
            return ["Definition", "How it works", "Example"]
        if "compare" in task_lower:
            return ["Overview", "Key Differences", "Recommendation"]
        if "security" in task_lower or "threat" in task_lower:
            return ["Threat Assessment", "Impact", "Mitigations"]
        return []
