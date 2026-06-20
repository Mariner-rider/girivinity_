"""
EpisodicMemory — per-user long-term memory stored in FAISS with time decay.

Each memory is a structured event:
  {user_id, timestamp, query_summary, key_facts, outcomes, emotion, topics}

Retrieval: semantic search + recency weighting.
  score = cosine_similarity * (1 - decay_factor * days_ago / decay_days)

Use cases:
  - "Last time you asked about X..." (contextual recall)
  - "You mentioned your project deadline is Friday..." (cross-session context)
  - Personalised examples based on user's past interests
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Episode:
    user_id: str
    timestamp: float
    query_summary: str
    key_facts: list[str]
    emotion: str
    topics: list[str]
    episode_id: str


class _Vector(list):
    def reshape(self, rows: int, cols: int) -> list[list[float]]:
        return [list(self)]


class _InMemoryIndex:
    def __init__(self) -> None:
        self.vectors: list[list[float]] = []

    def add(self, vector: Any) -> None:
        self.vectors.append(list(vector[0]))

    def search(self, query_vector: Any, k: int) -> tuple[list[list[float]], list[list[int]]]:
        if not self.vectors:
            return [[0.0]], [[-1]]
        query = list(query_vector[0])
        scored = [(sum(q * v for q, v in zip(query, vector, strict=False)), idx) for idx, vector in enumerate(self.vectors)]
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:k]
        return [[score for score, _ in top]], [[idx for _, idx in top]]


class EpisodicMemory:
    def __init__(self, config: Any, embedder: Any) -> None:
        self.config = config
        self.embedder = embedder
        self._index = self._build_index()
        self._episodes: list[Episode] = []
        self._user_episode_ids: dict[str, list[int]] = {}  # user_id → list of index positions

    def store(self, episode: Episode) -> None:
        text = f"{episode.query_summary} {' '.join(episode.key_facts)} {' '.join(episode.topics)}"
        emb = self._encode(text)
        idx = len(self._episodes)
        self._episodes.append(episode)
        self._index.add(emb.reshape(1, -1))
        self._user_episode_ids.setdefault(episode.user_id, []).append(idx)

    def recall(self, user_id: str, query: str, top_k: int = 5) -> list[Episode]:
        """Retrieve relevant episodes for this user, with recency weighting."""
        if user_id not in self._user_episode_ids:
            return []
        user_indices = set(self._user_episode_ids[user_id])
        query_emb = self._encode(query)
        k = min(top_k * 10, max(len(self._episodes), 1))  # over-fetch then filter
        scores, indices = self._index.search(query_emb.reshape(1, -1), k)
        now = time.time()
        decay_days = float(self._cfg("episode_decay_days", 90))
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx not in user_indices:
                continue
            ep = self._episodes[idx]
            days_ago = (now - ep.timestamp) / 86400
            decay = math.exp(-days_ago / max(decay_days, 1))
            final_score = float(score) * decay
            results.append((final_score, ep))
        results.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in results[:top_k]]

    def to_context_string(self, episodes: list[Episode]) -> str:
        if not episodes:
            return ""
        parts = ["## Relevant past interactions with this user:"]
        for ep in episodes:
            parts.append(f"- [{ep.topics}] {ep.query_summary} (emotion: {ep.emotion})")
        return "\n".join(parts)

    def summarise_episode(self, query: str, response: str, sentiment: dict[str, Any], topics: list[str]) -> str:
        """Create a compact summary string for storing as an episode."""
        words = query.split()
        return " ".join(words[:20]) + ("..." if len(words) > 20 else "")

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(episode)) for episode in self._episodes)

    def _build_index(self) -> Any:
        if importlib.util.find_spec("faiss") is None:
            return _InMemoryIndex()
        faiss = importlib.import_module("faiss")
        embedding_dim = int(self._cfg("embedding_dim", 384))
        return faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine after L2 norm

    def _encode(self, text: str) -> Any:
        emb = self.embedder.encode(text, normalize_embeddings=True)
        if hasattr(emb, "astype"):
            return emb.astype("float32")
        return _Vector(float(value) for value in emb)

    def _cfg(self, name: str, default: Any) -> Any:
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return getattr(self.config, name, default)
