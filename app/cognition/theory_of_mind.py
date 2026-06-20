"""
TheoryOfMindEngine — models the user's mental state from conversation history.

Tracks per-turn:
  - Inferred expertise level (1–5 scale, auto-detected from vocabulary + question type)
  - Inferred emotional state (from SentimentEngine)
  - Inferred intent (informational, task, emotional, adversarial)
  - Knowledge gaps (topics user seems confused about)
  - Communication preferences (detail level, example style, formality)

Outputs a UserMentalModel that agents inject into their system prompts:
  "The user appears to be an expert in Python but novice in ML.
   They seem frustrated. They prefer concise answers with code examples.
   They have asked about 'gradient descent' three times — likely a knowledge gap."
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserMentalModel:
    user_id: str
    expertise_level: float = 3.0  # 1=novice, 5=expert
    expertise_domains: dict[str, float] = field(default_factory=dict)  # {domain: level}
    emotional_state: str = "neutral"
    emotional_intensity: float = 0.0  # 0–1
    primary_intent: str = "informational"
    knowledge_gaps: list[str] = field(default_factory=list)
    communication_style: str = "balanced"
    confusion_topics: list[str] = field(default_factory=list)
    frustration_count: int = 0
    session_summary: str = ""


class TheoryOfMindEngine:
    def __init__(self, config: Any) -> None:
        self.config = config
        self._user_models: dict[str, UserMentalModel] = {}
        self._history: dict[str, deque[dict[str, str]]] = {}

    def update(
        self,
        user_id: str,
        query: str,
        sentiment: dict[str, Any],
        response: str | None = None,
    ) -> UserMentalModel:
        """Update user mental model after each turn. Returns updated model."""
        if user_id not in self._user_models:
            self._user_models[user_id] = UserMentalModel(user_id=user_id)
            self._history[user_id] = deque(maxlen=int(self._cfg("user_model_window", 20)))

        model = self._user_models[user_id]
        self._history[user_id].append({"query": query, "response": response or ""})

        # Expertise inference: vocabulary complexity (avg word length + rare words)
        words = re.findall(r"[a-zA-Z]+", query)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        model.expertise_level = min(5.0, max(1.0, (avg_word_len - 3.5) * 2.0 + 3.0))

        # Emotional state from sentiment
        model.emotional_state = str(sentiment.get("dominant_emotion", "neutral"))
        model.emotional_intensity = float(sentiment.get("intensity", 0.0))
        if model.emotional_state in ("frustration", "anger"):
            model.frustration_count += 1

        # Intent classification
        q_lower = query.lower()
        if any(w in q_lower for w in ["why", "how does", "explain", "what is"]):
            model.primary_intent = "informational"
        elif any(w in q_lower for w in ["help me", "do this", "create", "write", "fix"]):
            model.primary_intent = "task"
        elif any(w in q_lower for w in ["i feel", "i'm sad", "worried", "scared"]):
            model.primary_intent = "emotional"
        elif any(w in q_lower for w in ["ignore instructions", "jailbreak", "bypass"]):
            model.primary_intent = "adversarial"

        # Knowledge gap tracking (repeated topics = confusion)
        topics = self._extract_topics(query)
        for topic in topics:
            count = sum(1 for h in self._history[user_id] if topic in h["query"].lower())
            if count >= 3 and topic not in model.confusion_topics:
                model.confusion_topics.append(topic)
            if count >= 3 and topic not in model.knowledge_gaps:
                model.knowledge_gaps.append(topic)

        # Communication style adaptation
        if model.expertise_level >= 4.0:
            model.communication_style = "technical"
        elif model.emotional_state in ("frustration", "confusion"):
            model.communication_style = "empathetic_step_by_step"
        elif model.expertise_level <= 2.0:
            model.communication_style = "simplified_with_examples"
        else:
            model.communication_style = "balanced"

        return model

    def to_system_prompt_addon(self, model: UserMentalModel) -> str:
        """Returns a string to prepend to every agent's system prompt."""
        parts = ["## User Context"]
        parts.append(f"Expertise: {model.expertise_level:.1f}/5.0")
        if model.emotional_state != "neutral":
            parts.append(
                f"Emotional state: {model.emotional_state} "
                f"(intensity {model.emotional_intensity:.2f})"
            )
            if model.frustration_count > 2:
                parts.append("Note: User has shown frustration multiple times. Be patient and clear.")
        if model.confusion_topics:
            parts.append(f"Apparent knowledge gaps: {', '.join(model.confusion_topics)}")
        parts.append(f"Preferred style: {model.communication_style}")
        parts.append(f"Current intent: {model.primary_intent}")
        return "\n".join(parts)

    def _extract_topics(self, text: str) -> list[str]:
        """Simple noun-phrase extraction for topic tracking."""
        words = re.findall(r"[a-z]+", text.lower())
        # Filter to meaningful words (> 4 chars, not stopwords)
        stopwords = {"this", "that", "with", "from", "have", "they", "what", "when", "where", "which"}
        return [w for w in words if len(w) > 4 and w not in stopwords]

    def _cfg(self, name: str, default: Any) -> Any:
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return getattr(self.config, name, default)
