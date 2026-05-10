from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(slots=True)
class ReasoningPlan:
    parsed_query: dict
    intent: str
    sub_tasks: list[str]
    execution_plan: list[dict]

    def to_json(self) -> str:
        return json.dumps(
            {
                "parsed_query": self.parsed_query,
                "intent": self.intent,
                "sub_tasks": self.sub_tasks,
                "execution_plan": self.execution_plan,
            },
            ensure_ascii=False,
        )


class ReasoningPlanner:
    def parse_query(self, query: str) -> dict:
        cleaned = query.strip()
        tokens = re.findall(r"[a-zA-Z0-9_]+", cleaned.lower())
        return {
            "original": cleaned,
            "tokens": tokens,
            "length": len(tokens),
            "has_question": "?" in cleaned,
        }

    def identify_intent(self, parsed_query: dict) -> str:
        tokens = set(parsed_query.get("tokens", []))
        if {"build", "create", "generate"} & tokens:
            return "creation"
        if {"analyze", "compare", "evaluate"} & tokens:
            return "analysis"
        if {"fix", "debug", "resolve"} & tokens:
            return "troubleshooting"
        return "general_assistance"

    def decompose_sub_tasks(self, parsed_query: dict, intent: str) -> list[str]:
        original = parsed_query.get("original", "")
        base = [f"Understand user goal: {original}"]
        if intent == "creation":
            base += [
                "Collect required constraints and outputs",
                "Design implementation approach",
                "Prepare deliverable artifacts",
            ]
        elif intent == "analysis":
            base += [
                "Gather relevant data points",
                "Evaluate evidence and tradeoffs",
                "Summarize findings with rationale",
            ]
        elif intent == "troubleshooting":
            base += [
                "Reproduce issue context",
                "Identify root cause hypotheses",
                "Propose and validate fixes",
            ]
        else:
            base += ["Identify useful next actions", "Provide concise actionable response"]
        return base

    def create_execution_plan(self, sub_tasks: list[str]) -> list[dict]:
        return [{"step": i + 1, "task": task, "status": "pending"} for i, task in enumerate(sub_tasks)]

    def build_plan(self, query: str) -> ReasoningPlan:
        parsed = self.parse_query(query)
        intent = self.identify_intent(parsed)
        sub_tasks = self.decompose_sub_tasks(parsed, intent)
        execution_plan = self.create_execution_plan(sub_tasks)
        return ReasoningPlan(
            parsed_query=parsed,
            intent=intent,
            sub_tasks=sub_tasks,
            execution_plan=execution_plan,
        )
