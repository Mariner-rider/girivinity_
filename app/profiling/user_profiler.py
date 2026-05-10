from __future__ import annotations

import re
from dataclasses import dataclass

from app.security.policy import secure_operation


@dataclass(slots=True)
class ProfileResult:
    user_level: str
    vocabulary_score: float


class UserProfiler:
    _ADVANCED_TERMS = {
        "quantization", "faiss", "embedding", "orchestration", "latency", "throughput",
        "backpropagation", "regularization", "distributed", "tokenization", "retrieval",
    }
    _INTERMEDIATE_TERMS = {
        "api", "model", "database", "vector", "pipeline", "memory", "context", "prompt",
    }

    @secure_operation("profiling.profile_user")
    def profile(self, prompt: str) -> ProfileResult:
        tokens = re.findall(r"[a-zA-Z_]+", prompt.lower())
        if not tokens:
            return ProfileResult(user_level="beginner", vocabulary_score=0.0)

        unique = set(tokens)
        advanced_hits = len(unique & self._ADVANCED_TERMS)
        intermediate_hits = len(unique & self._INTERMEDIATE_TERMS)
        avg_word_len = sum(len(t) for t in tokens) / len(tokens)

        vocabulary_score = round(
            advanced_hits * 1.0 + intermediate_hits * 0.5 + avg_word_len / 10.0,
            3,
        )

        if vocabulary_score >= 3.0 or advanced_hits >= 3:
            level = "expert"
        elif vocabulary_score >= 1.8 or intermediate_hits >= 2:
            level = "intermediate"
        else:
            level = "beginner"

        return ProfileResult(user_level=level, vocabulary_score=vocabulary_score)
