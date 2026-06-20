"""RAG package exports with lazy imports for optional vector dependencies."""

__all__ = ["RAGEngine", "RAGSystem", "RAGResponse", "RetrievedChunk"]


def __getattr__(name: str):
    if name == "RAGEngine":
        from app.rag.rag_engine import RAGEngine

        return RAGEngine
    if name in {"RAGSystem", "RAGResponse", "RetrievedChunk"}:
        from app.rag.system import RAGResponse, RAGSystem, RetrievedChunk

        exports = {
            "RAGSystem": RAGSystem,
            "RAGResponse": RAGResponse,
            "RetrievedChunk": RetrievedChunk,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
