"""RAG package exports with lazy imports for optional retrieval dependencies."""

from __future__ import annotations

__all__ = ["RAGSystem", "RAGResponse", "RetrievedChunk", "RAGEngine", "RAGConfig"]


def __getattr__(name: str):
    if name in {"RAGSystem", "RAGResponse", "RetrievedChunk"}:
        from app.rag import system

        return getattr(system, name)
    if name in {"RAGEngine", "RAGConfig"}:
        from app.rag import rag_engine

        return getattr(rag_engine, name)
    raise AttributeError(f"module 'app.rag' has no attribute {name!r}")
