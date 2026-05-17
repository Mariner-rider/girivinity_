from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class UserModel:
    user_id: str
    inferred_role: str = "general"
    primary_language: str = "english"
    expertise_level: str = "intermediate"
    dominant_emotion: str = "neutral"
    topics_of_interest: list[str] = field(default_factory=list)
    interaction_count: int = 0
    avg_query_length: float = 0.0
    preferred_response_style: str = "balanced"
    session_count: int = 0
    last_seen: str = ""
    trust_score: float = 0.5


ROLE_SIGNALS = {
    "developer": ["code", "function", "bug", "deploy", "git", "api", "debug", "python", "javascript", "docker"],
    "researcher": ["research", "paper", "study", "hypothesis", "data", "analysis", "literature", "citation", "methodology"],
    "student": ["homework", "exam", "learn", "explain", "concept", "subject", "class", "assignment", "understand"],
    "doctor": ["patient", "diagnosis", "treatment", "medicine", "symptom", "clinical", "hospital", "drug", "dose"],
    "lawyer": ["legal", "law", "court", "section", "ipc", "act", "judgment", "rights", "contract", "clause"],
    "engineer": ["circuit", "design", "system", "mechanical", "electrical", "structural", "hardware", "sensor"],
    "business": ["revenue", "market", "strategy", "customer", "profit", "startup", "investment", "product"],
}


class SocialEngine:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        self.profiles_dir = Path(cfg.get("social_engine", {}).get("profiles_dir", "data/user_profiles"))
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create(self, user_id: str) -> UserModel:
        profile_path = self.profiles_dir / f"{user_id}.json"
        if profile_path.exists():
            try:
                data = json.loads(profile_path.read_text(encoding="utf-8"))
                return UserModel(**data)
            except Exception:
                pass
        return UserModel(user_id=user_id)

    def update(self, user_id: str, query: str, sentiment_profile) -> UserModel:
        model = self.get_or_create(user_id)

        model.interaction_count += 1
        model.last_seen = datetime.now(timezone.utc).isoformat()
        model.avg_query_length = (
            (model.avg_query_length * (model.interaction_count - 1) + len(query.split()))
            / model.interaction_count
        )

        detected_role = self._infer_role(query)
        if detected_role != "general":
            model.inferred_role = detected_role

        model.primary_language = sentiment_profile.language_mix
        model.expertise_level = sentiment_profile.expertise_signal

        topic = self._extract_topic(query)
        if topic and topic not in model.topics_of_interest:
            model.topics_of_interest.append(topic)
            model.topics_of_interest = model.topics_of_interest[-20:]

        model.preferred_response_style = sentiment_profile.response_style
        model.trust_score = min(1.0, model.trust_score + 0.01 * model.interaction_count)

        self._save(model)
        return model

    def get_context_injection(self, model: UserModel) -> str:
        parts = []
        if model.inferred_role != "general":
            parts.append(f"This user appears to be a {model.inferred_role}.")
        if model.topics_of_interest:
            recent = model.topics_of_interest[-3:]
            parts.append(f"They have shown interest in: {', '.join(recent)}.")
        if model.interaction_count > 10:
            parts.append(
                f"This is a returning user ({model.interaction_count} interactions). "
                f"They prefer {model.preferred_response_style} style responses."
            )
        if model.expertise_level == "expert":
            parts.append("Use expert-level terminology.")
        elif model.expertise_level == "beginner":
            parts.append("Use simple language and give examples.")
        return " ".join(parts)

    def _infer_role(self, query: str) -> str:
        q = query.lower()
        scores: dict[str, int] = {}
        for role, keywords in ROLE_SIGNALS.items():
            scores[role] = sum(1 for kw in keywords if kw in q)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] >= 2 else "general"

    def _extract_topic(self, query: str) -> str | None:
        words = [
            w
            for w in query.lower().split()
            if len(w) > 4
            and w
            not in {
                "what",
                "when",
                "where",
                "which",
                "would",
                "could",
                "should",
                "about",
                "their",
                "there",
                "these",
                "those",
                "please",
                "using",
                "write",
            }
        ]
        return words[0] if words else None

    def _save(self, model: UserModel) -> None:
        path = self.profiles_dir / f"{model.user_id}.json"
        path.write_text(json.dumps(asdict(model), indent=2), encoding="utf-8")
