"""Instruction-following engine for strict prompt adherence and constraint-compliant output."""

from __future__ import annotations

import re
from dataclasses import dataclass


class ResponseGenerator:
    def generate(self, prompt: str, requirements: list[str]) -> str:
        req_text = "\n".join(f"- {r}" for r in requirements)
        return f"Response for prompt: {prompt}\nRequirements addressed:\n{req_text}"


@dataclass(slots=True)
class ValidationResult:
    compliant: bool
    missing_requirements: list[str]


@dataclass(slots=True)
class InstructionResponse:
    response: str
    compliant: bool
    attempts: int
    missing_requirements: list[str]


class InstructionFollowingEngine:
    def __init__(self, generator: ResponseGenerator | None = None, max_attempts: int = 2) -> None:
        self.generator = generator or ResponseGenerator()
        self.max_attempts = max_attempts

    def parse_constraints(self, prompt: str) -> list[str]:
        constraints = []
        for line in prompt.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("-") or line.lower().startswith("must") or line.lower().startswith("constraint"):
                constraints.append(line.lstrip("- ").strip())
        return constraints

    def extract_requirements(self, prompt: str) -> list[str]:
        constraints = self.parse_constraints(prompt)
        # fallback heuristic when explicit list is absent
        if not constraints:
            verbs = re.findall(r"\b(include|ensure|return|output|provide|use|avoid)\b\s+([^.,;\n]+)", prompt.lower())
            constraints = [f"{v[0]} {v[1].strip()}" for v in verbs]
        return constraints

    def validate_output(self, output: str, requirements: list[str]) -> ValidationResult:
        lower = output.lower()
        missing = []
        for req in requirements:
            req_tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", req.lower()) if len(t) > 2][:3]
            if req_tokens and not all(token in lower for token in req_tokens):
                missing.append(req)
        return ValidationResult(compliant=len(missing) == 0, missing_requirements=missing)

    def run(self, prompt: str) -> InstructionResponse:
        requirements = self.extract_requirements(prompt)
        if not requirements:
            output = self.generator.generate(prompt, [])
            return InstructionResponse(response=output, compliant=True, attempts=1, missing_requirements=[])

        current_output = ""
        validation = ValidationResult(compliant=False, missing_requirements=requirements)
        for attempt in range(1, self.max_attempts + 1):
            current_output = self.generator.generate(prompt, requirements)
            validation = self.validate_output(current_output, requirements)
            if validation.compliant:
                return InstructionResponse(
                    response=current_output,
                    compliant=True,
                    attempts=attempt,
                    missing_requirements=[],
                )

            # If mismatch -> regenerate with explicit missing requirement emphasis.
            requirements = [*requirements, *[f"MUST include: {r}" for r in validation.missing_requirements]]

        return InstructionResponse(
            response=current_output,
            compliant=False,
            attempts=self.max_attempts,
            missing_requirements=validation.missing_requirements,
        )
