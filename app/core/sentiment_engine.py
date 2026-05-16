from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SentimentProfile:
    user_id: str
    query: str
    emotion: str
    intensity: float
    tone: str
    language_mix: str
    urgency: float
    expertise_signal: str
    response_style: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


EMOTION_LEXICON = {
    "frustrated": [
        "not working",
        "doesn't work",
        "broken",
        "failed",
        "error",
        "wrong",
        "stupid",
        "useless",
        "hate",
        "terrible",
        "why",
        "again",
        "still",
        "nothing works",
        "kuch nahi",
        "bakwaas",
    ],
    "confused": [
        "don't understand",
        "confused",
        "unclear",
        "what does",
        "how does",
        "what is",
        "explain",
        "samjhao",
        "kya hai",
        "pata nahi",
        "not sure",
        "lost",
    ],
    "excited": [
        "amazing",
        "great",
        "awesome",
        "love",
        "fantastic",
        "incredible",
        "best",
        "brilliant",
        "mast",
        "zabardast",
        "wow",
        "superb",
    ],
    "urgent": [
        "urgent",
        "asap",
        "immediately",
        "now",
        "quickly",
        "fast",
        "jaldi",
        "abhi",
        "deadline",
        "emergency",
    ],
    "curious": [
        "curious",
        "wondering",
        "interested",
        "tell me more",
        "how",
        "why",
        "what if",
        "bataiye",
        "janana chahta",
    ],
    "sad": [
        "sad",
        "depressed",
        "unhappy",
        "upset",
        "worried",
        "anxious",
        "scared",
        "dara",
        "pareshan",
    ],
}

EXPERTISE_SIGNALS = {
    "expert": [
        "implementation",
        "algorithm",
        "complexity",
        "latency",
        "throughput",
        "architecture",
        "kernel",
        "gradient",
        "backpropagation",
        "optimization",
        "hyperparameter",
    ],
    "intermediate": [
        "function",
        "class",
        "variable",
        "loop",
        "database",
        "api",
        "server",
        "model",
        "training",
    ],
    "beginner": [
        "how to",
        "what is",
        "simple",
        "basic",
        "start",
        "learn",
        "beginner",
        "newbie",
        "first time",
    ],
}

RESPONSE_STYLE_MAP = {
    ("frustrated", "expert"): "technical_empathetic",
    ("frustrated", "beginner"): "simple_encouraging",
    ("confused", "expert"): "detailed_technical",
    ("confused", "beginner"): "simple_stepwise",
    ("excited", "expert"): "technical_enthusiastic",
    ("excited", "beginner"): "encouraging_simple",
    ("urgent", "expert"): "concise_technical",
    ("urgent", "beginner"): "concise_simple",
    ("curious", "expert"): "deep_technical",
    ("curious", "beginner"): "engaging_simple",
    ("neutral", "expert"): "technical",
    ("neutral", "beginner"): "clear_simple",
    ("neutral", "intermediate"): "balanced",
}


class SentimentEngine:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        self.store_history = cfg.get("sentiment_engine", {}).get("store_history", True)

    def analyse(self, query: str, user_id: str) -> SentimentProfile:
        emotion = self._detect_emotion(query)
        intensity = self._measure_intensity(query, emotion)
        tone = self._detect_tone(query)
        language = self._detect_language(query)
        urgency = self._detect_urgency(query)
        expertise = self._detect_expertise(query)
        style = self._determine_response_style(emotion, expertise)

        profile = SentimentProfile(
            user_id=user_id,
            query=query,
            emotion=emotion,
            intensity=intensity,
            tone=tone,
            language_mix=language,
            urgency=urgency,
            expertise_signal=expertise,
            response_style=style,
        )

        if self.store_history:
            self._store(profile)

        logger.info(
            "Sentiment: user=%s emotion=%s expertise=%s style=%s",
            user_id,
            emotion,
            expertise,
            style,
        )
        return profile

    def get_style_instruction(self, profile: SentimentProfile) -> str:
        instructions = {
            "simple_encouraging": "Respond in simple, friendly language. Encourage the user. Use short sentences. Avoid jargon.",
            "technical_empathetic": "Respond with technical precision but acknowledge the difficulty. Be direct and solution-focused.",
            "detailed_technical": "Provide a thorough technical explanation with examples, edge cases, and best practices.",
            "simple_stepwise": "Break the answer into numbered steps. Use simple language. Give one example.",
            "concise_technical": "Be concise. Lead with the answer. No preamble. Technical precision required.",
            "concise_simple": "Answer in 2-3 sentences maximum. Simple language only.",
            "deep_technical": "Go deep. Include theory, implementation, trade-offs, and advanced considerations.",
            "engaging_simple": "Be enthusiastic and encouraging. Explain simply with a good analogy.",
            "technical": "Standard technical response with accuracy and completeness.",
            "clear_simple": "Clear, accurate, appropriately detailed response.",
            "balanced": "Balance technical detail with accessibility. Include an example.",
            "technical_enthusiastic": "Match their energy. Go deep technically and highlight what makes this interesting.",
        }

        base = instructions.get(profile.response_style, "Respond helpfully and accurately.")

        if profile.language_mix == "hindi":
            base += " Respond in Hindi (Devanagari script)."
        elif profile.language_mix == "hinglish":
            base += " Mix English and Hindi naturally in your response."

        if profile.urgency > 0.7:
            base += " Lead with the most important information first."

        return base

    def _detect_emotion(self, query: str) -> str:
        q = query.lower()
        scores: dict[str, int] = {}
        for emotion, keywords in EMOTION_LEXICON.items():
            scores[emotion] = sum(1 for kw in keywords if kw in q)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "neutral"

    def _measure_intensity(self, query: str, emotion: str) -> float:
        caps_ratio = sum(1 for c in query if c.isupper()) / max(len(query), 1)
        exclamations = query.count("!") + query.count("?")
        repeat_chars = len(re.findall(r"(.)\1{2,}", query))
        raw = caps_ratio * 0.4 + min(exclamations * 0.15, 0.4) + repeat_chars * 0.1
        return min(1.0, round(raw, 3))

    def _detect_tone(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["please", "could you", "would you", "kindly"]):
            return "formal"
        if any(w in q for w in ["def ", "class ", "import ", "sudo ", "git "]):
            return "technical"
        if any(w in q for w in ["feel", "sad", "happy", "scared", "worried"]):
            return "emotional"
        return "casual"

    def _detect_language(self, query: str) -> str:
        hindi_chars = len(re.findall(r"[\u0900-\u097F]", query))
        total = len(query.strip())
        if hindi_chars == 0:
            return "english"
        ratio = hindi_chars / max(total, 1)
        return "hindi" if ratio > 0.5 else "hinglish"

    def _detect_urgency(self, query: str) -> float:
        q = query.lower()
        urgent_words = EMOTION_LEXICON["urgent"]
        hits = sum(1 for w in urgent_words if w in q)
        return min(1.0, hits * 0.35)

    def _detect_expertise(self, query: str) -> str:
        q = query.lower()
        for level, keywords in EXPERTISE_SIGNALS.items():
            if sum(1 for kw in keywords if kw in q) >= 2:
                return level
        return "intermediate"

    def _determine_response_style(self, emotion: str, expertise: str) -> str:
        return RESPONSE_STYLE_MAP.get(
            (emotion, expertise),
            RESPONSE_STYLE_MAP.get(("neutral", expertise), "balanced"),
        )

    def _store(self, profile: SentimentProfile) -> None:
        try:
            from app.core import db

            db.execute(
                """
                INSERT INTO sentiment_history
                    (user_id, query_hash, emotion, intensity,
                     tone, language_mix, urgency,
                     expertise_signal, response_style, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    profile.user_id,
                    str(hash(profile.query))[:16],
                    profile.emotion,
                    profile.intensity,
                    profile.tone,
                    profile.language_mix,
                    profile.urgency,
                    profile.expertise_signal,
                    profile.response_style,
                    profile.timestamp,
                ),
            )
        except Exception as exc:
            logger.warning("Sentiment store failed: %s", exc)
