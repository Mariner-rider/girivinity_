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

@dataclass(slots=True)
class CognitiveProfile:
    anonymized_user_id: str
    learning_style: str
    knowledge_graph: dict
    interaction_rhythm: str
    preferred_response_format: str
    expertise: dict


class CognitiveProfileBuilder:
    def __init__(self, tracker: UserBehaviorTrackingSystem, theory_of_mind_engine=None) -> None:
        self.tracker = tracker
        self.theory_of_mind = theory_of_mind_engine

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "CognitiveProfileBuilder":
        guard = PrivacyGuard(require_consent=False)
        return cls(UserBehaviorTrackingSystem(guard))

    def build(self, user_id: str) -> CognitiveProfile:
        profile = self.tracker.build_user_profile(user_id)
        events = self.tracker._events.get(user_id, [])
        text = " ".join(ev.query for ev in events)
        lower = text.lower()
        learning_style = "visual" if any(w in lower for w in ["diagram", "show", "visual"]) else "example-driven" if "example" in lower or "code" in lower else "textual"
        preferred = "code" if any(w in lower for w in ["python", "code", "function"]) else "bullet points" if any("?" in ev.query and len(ev.query) < 80 for ev in events) else "prose"
        rhythm = "deep explorations" if profile.interaction_patterns.get("avg_dwell_seconds", 0) > 30 else "quick queries"
        known = profile.interests
        gaps = [topic for topic in ["security", "python", "architecture", "rag"] if topic not in known]
        expertise = {}
        if self.theory_of_mind and hasattr(self.theory_of_mind, "infer"):
            try: expertise = self.theory_of_mind.infer(text).__dict__
            except Exception: expertise = {}
        return CognitiveProfile(profile.anonymized_user_id, learning_style, {"known_topics": known, "knowledge_gaps": gaps}, rhythm, preferred, expertise)


class PersonalisationEngine:
    def __init__(self, profile_builder: CognitiveProfileBuilder) -> None:
        self.profile_builder = profile_builder

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "PersonalisationEngine":
        return cls(CognitiveProfileBuilder.from_config(path))

    def personalize_prompt(self, base_prompt: str, user_id: str) -> str:
        p = self.profile_builder.build(user_id)
        instructions = [f"User learning style: {p.learning_style}.", f"Preferred format: {p.preferred_response_format}.", f"Interaction rhythm: {p.interaction_rhythm}."]
        if "python" in p.knowledge_graph.get("known_topics", []): instructions.append("Skip beginner Python explanations unless requested.")
        if p.learning_style == "visual": instructions.append("Use compact ASCII diagrams where helpful.")
        if "security" in p.knowledge_graph.get("known_topics", []): instructions.append("Pre-load cybersecurity context and be precise about risk.")
        return base_prompt.rstrip() + "\n\nPersonalisation:\n" + "\n".join(f"- {i}" for i in instructions)

    def apply_to_agent(self, agent, user_id: str):
        if hasattr(agent, "system_prompt"):
            agent.system_prompt = self.personalize_prompt(agent.system_prompt, user_id)
        return agent


class FeedbackHarvester:
    def __init__(self, db_path: str = "data/user_feedback.sqlite3") -> None:
        import sqlite3
        self.db_path = db_path
        Path = __import__('pathlib').Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, interaction_id TEXT, user_id TEXT, rating INTEGER, correction TEXT, signal_type TEXT, dwell_seconds REAL, follow_up_confusion INTEGER, repeated_question INTEGER, created_at REAL)")
            db.commit()

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "FeedbackHarvester":
        return cls()

    def collect_explicit(self, interaction_id: str, rating: int, correction: str | None = None, user_id: str | None = None) -> None:
        self._insert(interaction_id, user_id, rating, correction, "explicit", 0.0, False, False)

    def collect_implicit(self, interaction_id: str, user_id: str | None = None, dwell_seconds: float = 0.0, follow_up_question: str = "", previous_question: str = "") -> None:
        confusion = bool(re.search(r"confused|don't understand|what do you mean|clarify", follow_up_question, re.I))
        repeated = bool(previous_question and follow_up_question and self._similar(previous_question, follow_up_question) > (3 / 4))
        rating = -1 if confusion or repeated else 1 if dwell_seconds > 20 else 0
        self._insert(interaction_id, user_id, rating, None, "implicit", dwell_seconds, confusion, repeated)

    def _insert(self, interaction_id, user_id, rating, correction, signal_type, dwell, confusion, repeated):
        import sqlite3, time
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT INTO feedback(interaction_id,user_id,rating,correction,signal_type,dwell_seconds,follow_up_confusion,repeated_question,created_at) VALUES(?,?,?,?,?,?,?,?,?)", (interaction_id, user_id, int(rating), correction, signal_type, float(dwell), int(confusion), int(repeated), time.time()))
            db.commit()

    @staticmethod
    def _similar(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / max(1, len(sa | sb))
