from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer

from app.security.policy import SecurityGuard, secure_operation


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class RedisClient(Protocol):
    def lpush(self, key: str, value: str) -> int:
        ...

    def ltrim(self, key: str, start: int, stop: int) -> None:
        ...

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        ...


class Neo4jClient(Protocol):
    def add_relationship(self, source: int, target: int, relation: str) -> None:
        ...

    def neighbors(self, node_id: int) -> list[dict]:
        ...


class InMemoryRedis:
    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    def lpush(self, key: str, value: str) -> int:
        self._store.setdefault(key, []).insert(0, value)
        return len(self._store[key])

    def ltrim(self, key: str, start: int, stop: int) -> None:
        if key not in self._store:
            return
        self._store[key] = self._store[key][start : stop + 1]

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        values = self._store.get(key, [])
        end = None if stop == -1 else stop + 1
        return values[start:end]


class InMemoryNeo4j:
    def __init__(self) -> None:
        self._edges: dict[int, list[dict]] = {}

    def add_relationship(self, source: int, target: int, relation: str) -> None:
        self._edges.setdefault(source, []).append({"target": target, "relation": relation})

    def neighbors(self, node_id: int) -> list[dict]:
        return list(self._edges.get(node_id, []))


class HFTextEmbedder:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with np.errstate(all="ignore"):
            outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state.detach().cpu().numpy()
        attention_mask = encoded["attention_mask"].unsqueeze(-1).detach().cpu().numpy()
        summed = np.sum(token_embeddings * attention_mask, axis=1)
        counts = np.clip(np.sum(attention_mask, axis=1), a_min=1e-9, a_max=None)
        vectors = summed / counts
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.clip(norms, 1e-12, None)).astype("float32")


@dataclass(slots=True)
class MemoryRecord:
    memory_id: int
    text: str
    metadata: dict


class MemorySystem:
    def __init__(
        self,
        embedding_dim: int = 384,
        short_term_limit: int = 50,
        embedder: Embedder | None = None,
        redis_client: RedisClient | None = None,
        neo4j_client: Neo4jClient | None = None,
        security_guard: SecurityGuard | None = None,
    ) -> None:
        self.short_term_limit = short_term_limit
        self.short_term = deque(maxlen=short_term_limit)
        self.embedder = embedder or HFTextEmbedder()
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.records: list[MemoryRecord] = []
        self.redis = redis_client or InMemoryRedis()
        self.graph = neo4j_client or InMemoryNeo4j()
        self.security_guard = security_guard or SecurityGuard()

    @secure_operation("memory.store")
    def store_memory(self, text: str, metadata: dict | None = None) -> int:
        self.security_guard.validate_prompt(text)
        metadata = metadata or {}
        vector = self.embedder.encode([text])
        self.index.add(vector)

        memory_id = len(self.records)
        record = MemoryRecord(memory_id=memory_id, text=text, metadata=metadata)
        self.records.append(record)
        self.short_term.append(record)

        self.redis.lpush("memory:short_term", str(memory_id))
        self.redis.ltrim("memory:short_term", 0, self.short_term_limit - 1)
        return memory_id

    @secure_operation("memory.retrieve")
    def retrieve_memory(self, memory_id: int) -> MemoryRecord:
        return self.records[memory_id]

    @secure_operation("memory.retrieve_context")
    def retrieve_context(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        self.security_guard.validate_prompt(query)
        if not self.records:
            return []

        start = time.perf_counter()
        query_vector = self.embedder.encode([query])
        scores, ids = self.index.search(query_vector, min(top_k, len(self.records)))
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > 100:
            pass

        recency_ids = [int(memory_id) for memory_id in self.redis.lrange("memory:short_term", 0, top_k - 1)]
        boosted = set(recency_ids)
        ranked_ids = [int(idx) for idx in ids[0] if idx != -1]
        position = {memory_id: i for i, memory_id in enumerate(ranked_ids)}
        ranked_ids.sort(key=lambda memory_id: (memory_id not in boosted, position[memory_id]))

        results: list[MemoryRecord] = []
        for memory_id in ranked_ids:
            results.append(self.records[memory_id])
        return results

    @secure_operation("memory.graph_relationship_builder")
    def graph_relationship_builder(self, source_id: int, target_id: int, relation: str) -> None:
        if source_id >= len(self.records) or target_id >= len(self.records):
            raise IndexError("Invalid memory ids provided for graph linkage")
        self.graph.add_relationship(source_id, target_id, relation)

    @secure_operation("memory.similarity_search")
    def similarity_search(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        return self.retrieve_context(query=query, top_k=top_k)
