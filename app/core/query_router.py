from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import threading
from pathlib import Path
from typing import Any

import yaml


def _optional_module(module_name: str) -> Any | None:
    if module_name in sys.modules:
        return sys.modules[module_name]
    if importlib.util.find_spec(module_name) is None:
        return None
    return importlib.import_module(module_name)


chromadb = _optional_module("chromadb")
_sentence_transformers = _optional_module("sentence_transformers")
SentenceTransformer = getattr(_sentence_transformers, "SentenceTransformer", None)

logger = logging.getLogger(__name__)

# Shared singleton embedder — loaded once, reused everywhere
_EMBEDDER: SentenceTransformer | None = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            if SentenceTransformer is None:
                raise ModuleNotFoundError("sentence-transformers is required")
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def reset_model_cache() -> None:
    global _EMBEDDER
    _EMBEDDER = None


class QueryRouter:
    @classmethod
    def reset_model_cache(cls) -> None:
        reset_model_cache()

    def __init__(self) -> None:
        if chromadb is None:
            raise ModuleNotFoundError("chromadb is required")
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        chroma_path = cfg["rag"]["chroma_path"]  # "data/chroma"
        self.threshold: float = 0.72
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection("girivinity_knowledge")

    def route(self, query: str) -> dict:
        embedder = get_embedder()
        encoded = embedder.encode(query)
        query_vec = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        # ChromaDB returns L2 distances — convert to 0-1 similarity score
        scores = [max(0.0, 1.0 - (d / 2.0)) for d in distances]

        if scores and scores[0] >= self.threshold:
            top = [
                {"text": docs[i], "score": scores[i], "meta": results["metadatas"][0][i]}
                for i in range(min(3, len(docs)))
            ]
            return {
                "source": "knowledge_base",
                "chunks": top,
                "context_string": self._build_context(top),
                "trigger_web": False,
                "confidence": scores[0],
                "urls": [],
            }

        # KB miss → go to web
        from app.core.web_intelligence import WebIntelligence

        web = WebIntelligence().search(query)

        raw = web.get("raw_chunks", [])
        if raw:
            threading.Thread(
                target=self._queue_training,
                args=(query, raw),
                daemon=True,
            ).start()

        chunks = web.get("answer_chunks", [])
        return {
            "source": "web" if chunks else "none",
            "chunks": chunks,
            "context_string": self._build_context(chunks),
            "trigger_web": bool(chunks),
            "confidence": chunks[0].get("score", 0.5) if chunks else 0.0,
            "urls": [s["url"] for s in web.get("sources", [])],
            "error": web.get("error"),
        }

    def _build_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return ""
        lines = [f"[{i + 1}] {c.get('text', '')}" for i, c in enumerate(chunks[:3])]
        return "Context:\n" + "\n".join(lines)

    def _queue_training(self, query: str, chunks: list[dict]) -> None:
        try:
            from app.core.self_trainer import SelfTrainer

            SelfTrainer().queue(query=query, chunks=chunks)
        except Exception as exc:
            logger.warning("Training queue failed: %s", exc)
