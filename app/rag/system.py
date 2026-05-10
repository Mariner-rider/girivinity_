from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.profiling.user_profiler import UserProfiler
from app.security.policy import SecurityGuard, secure_operation


@dataclass(slots=True)
class RetrievedChunk:
    document_id: str
    text: str
    score: float
    metadata: dict


@dataclass(slots=True)
class RAGResponse:
    answer: str
    sources: list[dict]
    citations: list[str]
    confidence: float
    context: str


class QueryEmbedder(Protocol):
    def encode(self, text: str) -> np.ndarray:
        ...


class VectorSearcher(Protocol):
    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class RAGSystem:
    def __init__(
        self,
        embedder: QueryEmbedder,
        searcher: VectorSearcher,
        generator: Generator,
        profiler: UserProfiler | None = None,
        security_guard: SecurityGuard | None = None,
    ) -> None:
        self.embedder = embedder
        self.searcher = searcher
        self.generator = generator
        self.profiler = profiler or UserProfiler()
        self.security_guard = security_guard or SecurityGuard()

    def build_context(self, chunks: list[RetrievedChunk], max_chars: int = 2500) -> str:
        blocks: list[str] = []
        current_len = 0
        for chunk in chunks:
            block = f"[source:{chunk.document_id}] {chunk.text.strip()}"
            if not block.strip():
                continue
            if current_len + len(block) > max_chars:
                break
            blocks.append(block)
            current_len += len(block)
        return "\n\n".join(blocks)

    def _build_prompt(self, query: str, context: str, user_level: str) -> str:
        return (
            "You are a retrieval-augmented assistant. Use only the provided context. "
            "If unsure, say so.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            f"User level: {user_level}. Adjust explanation depth accordingly.\n"
            "Answer with concise bullet points and cite sources like [source:ID]."
        )

    def _confidence(self, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        top_scores = [max(0.0, min(1.0, float(chunk.score))) for chunk in chunks[:3]]
        return round(sum(top_scores) / len(top_scores), 3)

    @secure_operation("rag.retrieve")
    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        self.security_guard.validate_prompt(query)
        query_vector = self.embedder.encode(query)
        return self.searcher.search(query_vector, top_k=top_k)

    @secure_operation("rag.generate")
    def generate(self, query: str, top_k: int = 4) -> RAGResponse:
        profile = self.profiler.profile(query)

        # 1) Query embedding + 2) vector search(top-k) occur inside retrieve().
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return RAGResponse(
                answer="insufficient information",
                sources=[],
                citations=[],
                confidence=0.0,
                context="",
            )

        # 3) context assembly
        context = self.build_context(chunks)

        sources = [
            {
                "document_id": chunk.document_id,
                "score": round(float(chunk.score), 4),
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        citations = [f"source:{chunk.document_id}" for chunk in chunks]

        self.security_guard.require_grounding(sources=sources, context=context)
        prompt = self._build_prompt(query, context, profile.user_level)

        # 4) grounded response
        answer = self.generator.generate(prompt)

        response = RAGResponse(
            answer=answer,
            sources=sources,
            citations=citations,
            confidence=self._confidence(chunks),
            context=context,
        )
        return response
