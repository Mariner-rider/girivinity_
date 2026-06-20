"""
InstructionFollowingEngine — parses complex multi-step instructions and converts
natural-language requests into an ordered task graph for the AgentController.

Handles:
  - Multi-part instructions: "First do X, then do Y, finally do Z"
  - Conditional instructions: "If the CVE is critical, also check Y"
  - Parallel instructions: "Simultaneously analyse A and B"
  - Clarification requests: detects ambiguous instructions and generates a
    clarifying question rather than guessing
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum


class StepType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass(slots=True)
class InstructionStep:
    step_id: str
    instruction: str
    step_type: StepType
    depends_on: list[str] = field(default_factory=list)
    condition: str = ""


@dataclass(slots=True)
class InstructionPlan:
    original: str
    steps: list[InstructionStep]
    is_ambiguous: bool = False
    clarifying_question: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "original": self.original,
                "steps": [
                    {
                        "id": step.step_id,
                        "instruction": step.instruction,
                        "type": step.step_type.value,
                        "depends_on": step.depends_on,
                        "condition": step.condition,
                    }
                    for step in self.steps
                ],
                "is_ambiguous": self.is_ambiguous,
                "clarifying_question": self.clarifying_question,
            },
            ensure_ascii=False,
        )


class InstructionFollowingEngine:
    SEQUENCE_MARKERS = ("first", "second", "third", "then", "next", "after that", "finally", "lastly")
    PARALLEL_MARKERS = ("simultaneously", "at the same time", "in parallel", "also", "and also")
    CONDITIONAL_MARKERS = ("if", "when", "only if", "unless", "in case")
    AMBIGUITY_SIGNALS = ("something", "stuff", "things", "etc", "and so on", "whatever")

    def parse(self, instruction: str) -> InstructionPlan:
        instruction = (instruction or "").strip()
        if not instruction:
            return InstructionPlan("", [], True, "What would you like me to do?")

        lowered = instruction.lower()
        if any(re.search(rf"\b{re.escape(signal)}\b", lowered) for signal in self.AMBIGUITY_SIGNALS):
            return InstructionPlan(instruction, [], True, self._generate_clarifying_question(instruction))

        return InstructionPlan(original=instruction, steps=self._extract_steps(instruction))

    def _extract_steps(self, instruction: str) -> list[InstructionStep]:
        marker_pattern = "|".join(re.escape(marker) for marker in self.SEQUENCE_MARKERS)
        parts = [part.strip(" ,.;") for part in re.split(rf"\b(?:{marker_pattern})\b", instruction, flags=re.I)]
        parts = [part for part in parts if part]
        if not parts:
            parts = [instruction]

        steps: list[InstructionStep] = []
        previous_sequential: str | None = None
        for index, part in enumerate(parts, start=1):
            step_type = StepType.SEQUENTIAL
            condition = ""
            normalized = part.lower()

            conditional = self._parse_condition(part)
            if conditional:
                condition, part = conditional
                step_type = StepType.CONDITIONAL
                normalized = part.lower()

            if any(marker in normalized for marker in self.PARALLEL_MARKERS):
                step_type = StepType.PARALLEL if step_type != StepType.CONDITIONAL else step_type

            step_id = f"step_{index}"
            depends_on = [previous_sequential] if previous_sequential and step_type == StepType.SEQUENTIAL else []
            steps.append(InstructionStep(step_id, part.strip(" ,.;"), step_type, depends_on, condition))
            if step_type == StepType.SEQUENTIAL:
                previous_sequential = step_id

        return steps

    def _parse_condition(self, text: str) -> tuple[str, str] | None:
        match = re.match(r"\s*(if|when|only if|unless|in case)\s+(.+?)(?:,|\bthen\b)\s+(.+)$", text, re.I)
        if not match:
            return None
        keyword, condition, action = match.groups()
        return f"{keyword.lower()} {condition.strip()}", action.strip()

    def _generate_clarifying_question(self, instruction: str) -> str:
        excerpt = instruction[:100].replace("\n", " ")
        return f"Your instruction contains vague terms. Could you be more specific about what you mean in: '{excerpt}'?"
