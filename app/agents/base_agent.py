"""
BaseAgent — all agents inherit from this.

Contract:
  1. system_prompt() — defines agent's role and output format
  2. build_prompt(task, memory, mailbox, context, user_model, episodes) → str
  3. parse_response(raw: str) → dict  (JSON extraction with fallback)
  4. run(...) → AgentResult  (calls LLM, computes real confidence from entropy)
"""

from __future__ import annotations

import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_controller import AgentResult


class BaseAgent(ABC):
    def __init__(
        self,
        llm_engine: Any,
        rag_engine: Any,
        theory_of_mind: Any | None = None,
        episodic_memory: Any | None = None,
    ) -> None:
        self.llm = llm_engine
        self.rag = rag_engine
        self.tom = theory_of_mind
        self.episodic = episodic_memory

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def system_prompt(self, user_model: Any | None = None) -> str:
        ...

    @abstractmethod
    def build_prompt(
        self,
        task: str,
        memory: Any,
        mailbox: list[Any],
        rag_results: list[dict[str, Any]],
        user_model: Any | None,
        episodes: list[Any],
    ) -> str:
        ...

    @abstractmethod
    def parse_response(self, raw: str) -> dict[str, Any]:
        ...

    def run(
        self,
        task: str,
        memory: Any,
        mailbox: list[Any],
        scratchpad: list[str],
        user_id: str | None = None,
        sentiment: dict[str, Any] | None = None,
    ) -> "AgentResult":
        from agent_controller import AgentMessage, AgentResult
        from app.cognition.episodic_memory import Episode

        rag_results = self.rag.query(task, top_k=6) if self.rag else []
        user_key = user_id or "anon"
        user_model = self.tom.update(user_key, task, sentiment or {}) if self.tom else None
        episodes = self.episodic.recall(user_key, task) if self.episodic else []

        prompt = self.build_prompt(task, memory, mailbox, rag_results, user_model, episodes)
        scratchpad.append(f"[{self.name}] prompt built, {len(prompt)} chars")

        raw = self._generate_text(prompt)
        parsed = self.parse_response(raw)

        entropy = 1.0
        confidence = float(parsed.get("confidence", 0.5) or 0.5)
        if hasattr(self.llm, "get_token_entropy"):
            try:
                entropy = float(self.llm.get_token_entropy(prompt))
                confidence = round(1.0 / (1.0 + entropy), 3)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                confidence = round(confidence, 3)

        scratchpad.append(f"[{self.name}] confidence={confidence}, entropy={entropy:.3f}")

        if self.episodic and user_id:
            self.episodic.store(
                Episode(
                    user_id=user_id,
                    timestamp=time.time(),
                    query_summary=task[:100],
                    key_facts=self._as_list(parsed.get("findings", parsed.get("key_facts", []))),
                    emotion=sentiment.get("dominant_emotion", "neutral") if sentiment else "neutral",
                    topics=self._extract_topics(task),
                    episode_id=str(uuid.uuid4()),
                )
            )

        mailbox.append(AgentMessage(self.name, "next", raw[:500]))
        return AgentResult(
            agent_name=self.name,
            output=str(parsed.get("answer", parsed.get("analysis", parsed.get("summary", raw[:500])))),
            confidence=confidence,
            citations=[r["source"] for r in rag_results if r.get("source")],
        )

    def _generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        result = self.llm.generate(prompt, max_tokens=max_tokens, stream=False)
        return str(getattr(result, "text", result))

    def _safe_json_parse(self, text: str) -> dict[str, Any]:
        """Try to extract JSON from LLM output. Falls back to raw text."""
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"raw": text}
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return parsed if isinstance(parsed, dict) else {"raw": text}
            except json.JSONDecodeError:
                pass
        return {"raw": text}

    def _format_rag_context(self, rag_results: list[dict[str, Any]]) -> str:
        if not rag_results:
            return "No relevant context retrieved."
        parts = ["## Retrieved Context"]
        for i, result in enumerate(rag_results, 1):
            parts.append(
                f"[{i}] (score={result.get('score', 0):.2f}, "
                f"source={result.get('source', 'unknown')})\n{result.get('text', '')[:400]}"
            )
        return "\n\n".join(parts)

    def _format_user_context(self, user_model: Any | None) -> str:
        if self.tom and user_model and hasattr(self.tom, "to_system_prompt_addon"):
            return self.tom.to_system_prompt_addon(user_model)
        return "## User Context\nNo personalised user model available."

    def _format_episodes(self, episodes: list[Any]) -> str:
        if self.episodic and hasattr(self.episodic, "to_context_string"):
            return self.episodic.to_context_string(episodes)
        if not episodes:
            return ""
        return "\n".join(str(episode) for episode in episodes)

    def _extract_topics(self, text: str) -> list[str]:
        words = re.findall(r"[a-z]+", text.lower())
        stopwords = {"this", "that", "with", "from", "have", "they", "what", "when", "where", "which"}
        return [word for word in words if len(word) > 4 and word not in stopwords]

    def _as_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value:
            return [str(value)]
        return []
