"""Response planning system to produce structured response blueprints."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(slots=True)
class ResponseBlueprint:
    query_type: str
    response_format: str
    sections: list[dict]

    def to_json(self) -> str:
        return json.dumps(
            {
                "query_type": self.query_type,
                "response_format": self.response_format,
                "sections": self.sections,
            },
            ensure_ascii=False,
        )


class ResponsePlanningSystem:
    FORMATS = {"explanation", "code", "report", "step-by-step"}

    def analyze_query_type(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["code", "implement", "function", "script", "bug"]):
            return "technical_implementation"
        if any(k in q for k in ["report", "summary", "analysis", "findings"]):
            return "analytical_reporting"
        if any(k in q for k in ["how to", "steps", "guide", "walkthrough"]):
            return "procedural_guidance"
        return "conceptual_explanation"

    def select_response_format(self, query_type: str, query: str) -> str:
        q = query.lower()
        if query_type == "technical_implementation":
            return "code"
        if query_type == "analytical_reporting":
            return "report"
        if query_type == "procedural_guidance":
            return "step-by-step"
        if any(k in q for k in ["explain", "what is", "why"]):
            return "explanation"
        return "explanation"

    def structure_answer(self, response_format: str, query: str) -> list[dict]:
        base_intro = {"title": "Context", "instruction": f"Address user query: {query.strip()}"}
        if response_format == "code":
            return [
                base_intro,
                {"title": "Implementation", "instruction": "Provide complete, runnable code."},
                {"title": "Validation", "instruction": "Include test/verification steps."},
            ]
        if response_format == "report":
            return [
                base_intro,
                {"title": "Executive Summary", "instruction": "Summarize key findings."},
                {"title": "Evidence", "instruction": "Present data points and analysis."},
                {"title": "Recommendations", "instruction": "Provide actionable recommendations."},
            ]
        if response_format == "step-by-step":
            return [
                base_intro,
                {"title": "Steps", "instruction": "Provide numbered step-by-step instructions."},
                {"title": "Pitfalls", "instruction": "List common mistakes and mitigations."},
            ]
        return [
            base_intro,
            {"title": "Explanation", "instruction": "Explain concept clearly with examples."},
            {"title": "Takeaways", "instruction": "Provide concise key takeaways."},
        ]

    def build_blueprint(self, query: str) -> ResponseBlueprint:
        query_type = self.analyze_query_type(query)
        response_format = self.select_response_format(query_type, query)
        sections = self.structure_answer(response_format, query)
        return ResponseBlueprint(query_type=query_type, response_format=response_format, sections=sections)
