from __future__ import annotations

import re
from dataclasses import dataclass

from app.security.policy import secure_operation


@dataclass(slots=True)
class AdaptationPlan:
    response_depth: str
    tone: str
    examples: str


@dataclass(slots=True)
class ProfileResult:
    user_level: str
    vocabulary_score: float
    sentence_complexity_score: float
    domain_signal_score: float
    adaptation: AdaptationPlan


class UserProfiler:
    _ADVANCED_TERMS = {
        "quantization", "faiss", "embedding", "orchestration", "latency", "throughput",
        "backpropagation", "regularization", "distributed", "tokenization", "retrieval",
    }
    _INTERMEDIATE_TERMS = {
        "api", "model", "database", "vector", "pipeline", "memory", "context", "prompt",
    }

    _DOMAIN_TERMS = {
        "ml": {"model", "embedding", "tokenization", "backpropagation", "regularization"},
        "infra": {"latency", "throughput", "distributed", "orchestration", "pipeline"},
        "retrieval": {"faiss", "vector", "retrieval", "context", "memory"},
    }

    def _sentence_complexity(self, prompt: str) -> float:
        sentences = [s.strip() for s in re.split(r"[.!?]+", prompt) if s.strip()]
        if not sentences:
            return 0.0
        clause_markers = len(re.findall(r"\b(which|that|although|however|because|therefore)\b", prompt.lower()))
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        return round(min(1.0, (avg_sentence_len / 25.0) + (clause_markers / 8.0)), 3)

    def _domain_signal(self, tokens: list[str]) -> float:
        unique = set(tokens)
        domain_hits = sum(len(unique & terms) for terms in self._DOMAIN_TERMS.values())
        return round(min(1.0, domain_hits / 8.0), 3)

    def _adaptation_for_level(self, level: str) -> AdaptationPlan:
        if level == "expert":
            return AdaptationPlan(
                response_depth="deep",
                tone="technical and concise",
                examples="advanced and domain-specific",
            )
        if level == "intermediate":
            return AdaptationPlan(
                response_depth="moderate",
                tone="clear and practical",
                examples="mixed conceptual and practical",
            )
        return AdaptationPlan(
            response_depth="foundational",
            tone="friendly and guided",
            examples="simple analogies and step-by-step",
        )

    @secure_operation("profiling.profile_user")
    def profile(self, prompt: str) -> ProfileResult:
        tokens = re.findall(r"[a-zA-Z_]+", prompt.lower())
        if not tokens:
            return ProfileResult(
                user_level="beginner",
                vocabulary_score=0.0,
                sentence_complexity_score=0.0,
                domain_signal_score=0.0,
                adaptation=self._adaptation_for_level("beginner"),
            )

        unique = set(tokens)
        advanced_hits = len(unique & self._ADVANCED_TERMS)
        intermediate_hits = len(unique & self._INTERMEDIATE_TERMS)
        avg_word_len = sum(len(t) for t in tokens) / len(tokens)

        vocabulary_score = round(
            advanced_hits * 1.0 + intermediate_hits * 0.5 + avg_word_len / 10.0,
            3,
        )
        sentence_complexity_score = self._sentence_complexity(prompt)
        domain_signal_score = self._domain_signal(tokens)

        combined_score = round(
            (vocabulary_score / 4.0) * 0.45 + sentence_complexity_score * 0.25 + domain_signal_score * 0.30,
            3,
        )

        if combined_score >= (18 / 25) or advanced_hits >= 3:
            level = "expert"
        elif combined_score >= 0.42 or intermediate_hits >= 2:
            level = "intermediate"
        else:
            level = "beginner"

        return ProfileResult(
            user_level=level,
            vocabulary_score=vocabulary_score,
            sentence_complexity_score=sentence_complexity_score,
            domain_signal_score=domain_signal_score,
            adaptation=self._adaptation_for_level(level),
        )
