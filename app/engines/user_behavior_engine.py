"""Privacy-compliant user behavior tracking and embedding engine."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class UserInteraction:
    user_id: str
    query: str
    clicked_topics: list[str] = field(default_factory=list)
    dwell_seconds: float = 0.0
    timestamp: float = 0.0


@dataclass(slots=True)
class UserProfile:
    anonymized_user_id: str
    interests: list[str]
    recent_queries: list[str]
    interaction_patterns: dict
    embedding: list[float]


class PrivacyGuard:
    def __init__(self, require_consent: bool = True) -> None:
        self.require_consent = require_consent
        self.consents: set[str] = set()

    def grant_consent(self, user_id: str) -> None:
        self.consents.add(user_id)

    def check(self, user_id: str) -> None:
        if self.require_consent and user_id not in self.consents:
            raise PermissionError("Tracking blocked: missing user consent.")

    def anonymize(self, user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


class UserBehaviorTrackingSystem:
    def __init__(self, privacy_guard: PrivacyGuard | None = None, embedding_dim: int = 16) -> None:
        self.privacy_guard = privacy_guard or PrivacyGuard()
        self.embedding_dim = embedding_dim
        self._events: dict[str, list[UserInteraction]] = defaultdict(list)

    def track_interaction(self, event: UserInteraction) -> None:
        self.privacy_guard.check(event.user_id)
        sanitized_query = re.sub(r"\s+", " ", event.query.strip())
        self._events[event.user_id].append(
            UserInteraction(
                user_id=event.user_id,
                query=sanitized_query,
                clicked_topics=list(event.clicked_topics),
                dwell_seconds=max(0.0, float(event.dwell_seconds)),
                timestamp=event.timestamp,
            )
        )

    def _extract_interests(self, events: list[UserInteraction], top_k: int = 5) -> list[str]:
        topic_counter = Counter()
        for ev in events:
            topic_counter.update([t.lower() for t in ev.clicked_topics if t.strip()])
        return [topic for topic, _ in topic_counter.most_common(top_k)]

    def _interaction_patterns(self, events: list[UserInteraction]) -> dict:
        if not events:
            return {"avg_dwell_seconds": 0.0, "query_frequency": 0, "topic_diversity": 0}
        avg_dwell = sum(ev.dwell_seconds for ev in events) / len(events)
        all_topics = {t.lower() for ev in events for t in ev.clicked_topics}
        return {
            "avg_dwell_seconds": round(avg_dwell, 3),
            "query_frequency": len(events),
            "topic_diversity": len(all_topics),
        }

    def generate_user_embedding(self, user_id: str) -> list[float]:
        self.privacy_guard.check(user_id)
        events = self._events.get(user_id, [])
        vec = np.zeros((self.embedding_dim,), dtype="float32")
        for ev in events:
            seed = int(hashlib.sha256(ev.query.encode("utf-8")).hexdigest(), 16) % (10**8)
            rng = np.random.default_rng(seed)
            vec += rng.normal(0, 1, size=(self.embedding_dim,)).astype("float32")
            vec += float(ev.dwell_seconds) * 0.01
        norm = math.sqrt(float(np.dot(vec, vec)))
        if norm > 0:
            vec = vec / norm
        return vec.astype("float32").tolist()

    def build_user_profile(self, user_id: str) -> UserProfile:
        self.privacy_guard.check(user_id)
        events = self._events.get(user_id, [])
        interests = self._extract_interests(events)
        recent_queries = [ev.query for ev in events[-10:]]
        patterns = self._interaction_patterns(events)
        embedding = self.generate_user_embedding(user_id)
        return UserProfile(
            anonymized_user_id=self.privacy_guard.anonymize(user_id),
            interests=interests,
            recent_queries=recent_queries,
            interaction_patterns=patterns,
            embedding=embedding,
        )


class RecommendationSystem:
    def recommend(self, profile: UserProfile, candidates: list[dict], top_k: int = 3) -> list[dict]:
        ranked = []
        user_interests = set(profile.interests)
        for item in candidates:
            tags = set(t.lower() for t in item.get("tags", []))
            overlap = len(user_interests & tags)
            score = overlap + (0.1 if profile.interaction_patterns.get("avg_dwell_seconds", 0) > 10 else 0)
            ranked.append((score, item))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked[:top_k]]


class AdTargetingSystem:
    def target_segments(self, profile: UserProfile) -> list[str]:
        segments = []
        if "finance" in profile.interests:
            segments.append("fintech_intenders")
        if "ai" in profile.interests or "machine learning" in profile.interests:
            segments.append("ai_professionals")
        if profile.interaction_patterns.get("avg_dwell_seconds", 0) > 20:
            segments.append("high_engagement_users")
        return segments or ["broad_audience"]
