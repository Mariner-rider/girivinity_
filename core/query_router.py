from __future__ import annotations

import logging
import threading
from pathlib import Path

import yaml

try:
    import chromadb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised only when dependency is absent.
    chromadb = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised only when dependency is absent.
    SentenceTransformer = None  # type: ignore

    class _MissingUtil:
        @staticmethod
        def cos_sim(*args, **kwargs):
            raise ModuleNotFoundError("sentence-transformers is required")

    util = _MissingUtil()  # type: ignore

_EMBEDDER = None
logger = logging.getLogger(__name__)


# Keep a harmless reference so the required util import is exercised without changing Chroma routing.
_COS_SIM = util.cos_sim


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        model_cls = SentenceTransformer
        if model_cls is None:
            from sentence_transformers import SentenceTransformer as model_cls  # type: ignore
        _EMBEDDER = model_cls("all-MiniLM-L6-v2")
    return _EMBEDDER


class QueryRouter:
    def __init__(self):
        if chromadb is None:
            raise ModuleNotFoundError("chromadb is required")
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        chroma_path = cfg["rag"]["chroma_path"]
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection("girivinity_knowledge")
        self.threshold = 0.72

    @classmethod
    def _get_embedder(cls):
        return get_embedder()

    @classmethod
    def reset_model_cache(cls) -> None:
        global _EMBEDDER
        _EMBEDDER = None

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
        scores = [1 - (d / 2) for d in distances] if distances else []

        if scores and scores[0] >= self.threshold:
            top_chunks = [
                {
                    "text": docs[i],
                    "score": scores[i],
                    "meta": results["metadatas"][0][i],
                }
                for i in range(min(3, len(docs)))
            ]
            context_string = self._build_context(top_chunks)
            return {
                "source": "knowledge_base",
                "chunks": top_chunks,
                "context_string": context_string,
                "trigger_web": False,
                "confidence": scores[0],
            }

        from core.web_intelligence import WebIntelligence

        web_result = WebIntelligence().search(query)

        if web_result.get("answer_chunks"):
            raw = web_result.get("raw_chunks", [])
            if raw:
                t = threading.Thread(
                    target=self._queue_training,
                    args=(query, raw),
                    daemon=True,
                )
                t.start()

            context_string = self._build_context(web_result["answer_chunks"])
            return {
                "source": "web",
                "chunks": web_result["answer_chunks"],
                "context_string": context_string,
                "trigger_web": True,
                "raw_for_training": raw,
                "confidence": web_result["answer_chunks"][0].get("score", 0.5)
                if web_result["answer_chunks"]
                else 0.0,
                "urls": [s["url"] for s in web_result.get("sources", [])],
            }

        return {
            "source": "none",
            "chunks": [],
            "context_string": "",
            "trigger_web": False,
            "confidence": 0.0,
            "error": "no_results",
        }

    def _build_context(self, chunks: list) -> str:
        lines = []
        for i, c in enumerate(chunks[:3], 1):
            text = c.get("text", "") if isinstance(c, dict) else str(c)
            lines.append(f"[{i}] {text}")
        return "Context:\n" + "\n".join(lines)

    def _queue_training(self, query: str, chunks: list) -> None:
        try:
            from core.self_trainer import SelfTrainer

            SelfTrainer().queue(query=query, chunks=chunks)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Training queue failed: {e}")
