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
        security_guard: SecurityGuard | None = None,
    ) -> None:
        self.short_term = deque(maxlen=short_term_limit)
        self.embedder = embedder or HFTextEmbedder()
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.records: list[MemoryRecord] = []
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
        return memory_id

    @secure_operation("memory.retrieve")
    def retrieve_memory(self, memory_id: int) -> MemoryRecord:
        return self.records[memory_id]

    @secure_operation("memory.similarity_search")
    def similarity_search(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        self.security_guard.validate_prompt(query)
        if not self.records:
            return []

        start = time.perf_counter()
        query_vector = self.embedder.encode([query])
        scores, ids = self.index.search(query_vector, min(top_k, len(self.records)))
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > 100:
            # Hook point for observability/alerts in production.
            pass

        result: list[MemoryRecord] = []
        for idx in ids[0]:
            if idx == -1:
                continue
            result.append(self.records[int(idx)])
        return result
