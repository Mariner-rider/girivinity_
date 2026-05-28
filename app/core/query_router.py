from __future__ import annotations

import logging
import threading
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EMBEDDER: SentenceTransformer | None = None
_LOCK = threading.Lock()


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    with _LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


class QueryRouter:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        chroma_path: str = cfg["rag"]["chroma_path"]
        self.threshold: float = 0.72
        client = chromadb.PersistentClient(path=chroma_path)
        self.collection = client.get_or_create_collection("girivinity_knowledge")

    def route(self, query: str) -> dict:
        embedder = get_embedder()
        vec = embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[vec],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
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
        return "Context:\n" + "\n".join(f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(chunks[:3]))

    def _queue_training(self, query: str, chunks: list[dict]) -> None:
        try:
            from app.core.self_trainer import SelfTrainer

            SelfTrainer().queue(query=query, chunks=chunks)
        except Exception as exc:
            logger.warning("Training queue failed: %s", exc)
        try:
            from app.core.skill_forge import SkillForge

            urls = [c.get("url", "") for c in chunks if c.get("url")]
            SkillForge().generate_async(
                topic=query, chunks=chunks, urls=urls
            )
        except Exception as exc:
            logger.warning("SkillForge async failed: %s", exc)
