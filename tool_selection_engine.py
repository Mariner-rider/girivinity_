"""Tool selection engine to choose optimal execution path per user query."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class SelectionResult:
    decision: str
    confidence: float
    reason: str


class ToolSelectionEngine:
    OPTIONS = {"LLM only", "RAG", "API call", "agent workflow"}

    def _score_llm_only(self, query: str) -> float:
        simple_tokens = {"write", "summarize", "explain", "translate", "rephrase"}
        tokens = set(re.findall(r"[a-zA-Z]+", query.lower()))
        score = 0.5 + (0.3 if tokens & simple_tokens else 0.0)
        if any(k in query.lower() for k in ["latest", "current", "real-time", "price"]):
            score -= 0.35
        return max(0.0, min(1.0, score))

    def _score_rag(self, query: str) -> float:
        if any(k in query.lower() for k in ["document", "policy", "manual", "knowledge base", "cite"]):
            return 0.9
        if any(k in query.lower() for k in ["according to", "from the data", "what does our"]):
            return 0.78
        return 0.3

    def _score_api_call(self, query: str) -> float:
        if any(k in query.lower() for k in ["weather", "stock", "price", "news", "score", "exchange rate"]):
            return 0.92
        if any(k in query.lower() for k in ["today", "latest", "current"]):
            return 0.72
        return 0.25

    def _score_agent_workflow(self, query: str) -> float:
        if any(k in query.lower() for k in ["plan", "workflow", "multi-step", "analyze and build", "orchestrate"]):
            return 0.9
        if len(query.split()) > 22:
            return 0.7
        return 0.28

    def select(self, user_query: str) -> SelectionResult:
        scores = {
            "LLM only": self._score_llm_only(user_query),
            "RAG": self._score_rag(user_query),
            "API call": self._score_api_call(user_query),
            "agent workflow": self._score_agent_workflow(user_query),
        }
        decision = max(scores, key=scores.get)
        confidence = round(scores[decision], 3)
        reason = f"selected={decision}; scores={{{', '.join(f'{k}:{v:.2f}' for k, v in scores.items())}}}"
        return SelectionResult(decision=decision, confidence=confidence, reason=reason)
