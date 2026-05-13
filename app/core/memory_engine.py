from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    node_id: str
    user_id: str
    content: str
    node_type: str
    importance: float
    access_count: int = 0
    created_at: str = ""
    last_accessed: str = ""
    related_nodes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryEngine:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        me = cfg.get("memory_engine", {})
        chroma_path = cfg["rag"]["chroma_path"]
        self.max_memories_per_user = int(me.get("max_memories_per_user", 1000))
        self.recall_top_k = int(me.get("recall_top_k", 5))
        self.importance_threshold = float(me.get("importance_threshold", 0.3))
        import chromadb

        client = chromadb.PersistentClient(path=chroma_path)
        self.collection = client.get_or_create_collection(
            "user_memories", metadata={"hnsw:space": "cosine"}
        )

    def remember(self, user_id: str, query: str, response: str) -> list[MemoryNode]:
        nodes = self._extract_nodes(user_id, query, response)
        for node in nodes:
            if node.importance >= self.importance_threshold:
                self._store_node(node)
        return nodes

    def recall(self, user_id: str, query: str) -> list[MemoryNode]:
        from app.core.query_router import get_embedder

        embedder = get_embedder()
        vec = embedder.encode(f"{user_id} {query}").tolist()

        try:
            results = self.collection.query(
                query_embeddings=[vec],
                n_results=self.recall_top_k,
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("Memory recall failed: %s", exc)
            return []

        nodes = []
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []

        for i, doc in enumerate(docs):
            score = max(0.0, 1.0 - dists[i] / 2.0) if dists else 0.5
            if score >= 0.4:
                meta = metas[i] if i < len(metas) else {}
                nodes.append(
                    MemoryNode(
                        node_id=meta.get("node_id", ""),
                        user_id=user_id,
                        content=doc,
                        node_type=meta.get("node_type", "fact"),
                        importance=float(meta.get("importance", 0.5)),
                        access_count=int(meta.get("access_count", 0)) + 1,
                        created_at=meta.get("created_at", ""),
                        last_accessed=datetime.now(timezone.utc).isoformat(),
                    )
                )

        self._increment_access(nodes)
        return nodes

    def build_memory_context(self, nodes: list[MemoryNode]) -> str:
        if not nodes:
            return ""
        lines = ["[Relevant memories about this user:]"]
        for node in sorted(nodes, key=lambda n: n.importance, reverse=True):
            lines.append(f"- {node.content}")
        return "\n".join(lines)

    def remember_async(self, user_id: str, query: str, response: str) -> None:
        threading.Thread(
            target=self.remember,
            args=(user_id, query, response),
            daemon=True,
        ).start()

    def _extract_nodes(self, user_id: str, query: str, response: str) -> list[MemoryNode]:
        nodes = []
        now = datetime.now(timezone.utc).isoformat()

        facts = self._extract_facts(query + " " + response)
        for fact in facts[:5]:
            nodes.append(
                MemoryNode(
                    node_id=self._make_id(user_id, fact),
                    user_id=user_id,
                    content=fact,
                    node_type="fact",
                    importance=self._score_importance(fact, query),
                    created_at=now,
                    last_accessed=now,
                )
            )

        topic = self._extract_main_topic(query)
        if topic:
            nodes.append(
                MemoryNode(
                    node_id=self._make_id(user_id, f"topic:{topic}"),
                    user_id=user_id,
                    content=f"User asked about: {topic}",
                    node_type="topic",
                    importance=0.6,
                    created_at=now,
                    last_accessed=now,
                )
            )

        preferences = self._detect_preferences(query)
        for pref in preferences:
            nodes.append(
                MemoryNode(
                    node_id=self._make_id(user_id, f"pref:{pref}"),
                    user_id=user_id,
                    content=f"User preference: {pref}",
                    node_type="preference",
                    importance=0.7,
                    created_at=now,
                    last_accessed=now,
                )
            )

        return nodes

    def _store_node(self, node: MemoryNode) -> None:
        from app.core.query_router import get_embedder

        embedder = get_embedder()
        vec = embedder.encode(f"{node.user_id} {node.content}").tolist()
        try:
            self.collection.upsert(
                ids=[node.node_id],
                embeddings=[vec],
                documents=[node.content],
                metadatas=[
                    {
                        "user_id": node.user_id,
                        "node_id": node.node_id,
                        "node_type": node.node_type,
                        "importance": node.importance,
                        "access_count": node.access_count,
                        "created_at": node.created_at,
                        "last_accessed": node.last_accessed,
                    }
                ],
            )
        except Exception as exc:
            logger.warning("Memory store failed: %s", exc)

    def _increment_access(self, nodes: list[MemoryNode]) -> None:
        for node in nodes:
            try:
                self.collection.upsert(
                    ids=[node.node_id],
                    metadatas=[
                        {
                            "access_count": node.access_count,
                            "last_accessed": node.last_accessed,
                        }
                    ],
                )
            except Exception:
                pass

    def _extract_facts(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if 20 < len(s.strip()) < 200][:5]

    def _extract_main_topic(self, query: str) -> str | None:
        words = [
            w
            for w in query.lower().split()
            if len(w) > 5
            and w
            not in {
                "please",
                "could",
                "would",
                "should",
                "about",
                "their",
                "there",
                "which",
                "using",
            }
        ]
        return words[0] if words else None

    def _detect_preferences(self, query: str) -> list[str]:
        prefs = []
        q = query.lower()
        if "prefer" in q or "like" in q or "want" in q:
            prefs.append(f"expressed preference in: {query[:80]}")
        if "always" in q or "never" in q:
            prefs.append(f"stated rule: {query[:80]}")
        return prefs

    def _score_importance(self, fact: str, query: str) -> float:
        query_words = set(query.lower().split())
        fact_words = set(fact.lower().split())
        overlap = len(query_words & fact_words) / max(len(query_words), 1)
        length_bonus = min(0.2, len(fact) / 500)
        return min(1.0, round(0.4 + overlap * 0.4 + length_bonus, 3))

    def _make_id(self, user_id: str, content: str) -> str:
        import hashlib

        return hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:24]
