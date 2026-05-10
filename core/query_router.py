from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml


class QueryRouter:
    """Primary intelligence router for grounding every user query.

    The router first attempts to ground the request in the local Girivinity
    knowledge base. Only when local confidence is insufficient does it fall back
    to live web intelligence and asynchronously queue raw web chunks for future
    training.
    """

    _embedder: ClassVar[Any | None] = None
    _embedder_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        config_path: str | Path = "config.yaml",
        threshold: float = 0.72,
        collection_name: str = "girivinity_knowledge",
    ) -> None:
        self.config_path = Path(config_path)
        self.threshold = threshold
        self.collection_name = collection_name
        self._collection = self._load_collection()

    @classmethod
    def _get_embedder(cls):
        if cls._embedder is None:
            with cls._embedder_lock:
                if cls._embedder is None:
                    sentence_transformers = importlib.import_module("sentence_transformers")
                    cls._embedder = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
        return cls._embedder

    @classmethod
    def reset_model_cache(cls) -> None:
        """Test/helper hook for clearing the singleton embedding model."""
        with cls._embedder_lock:
            cls._embedder = None

    def route(self, query: str) -> dict[str, Any]:
        query_embedding = self._embed_query(query)
        kb_result = self._search_knowledge_base(query_embedding)
        if kb_result["score"] > self.threshold:
            chunks = kb_result["chunks"]
            urls = kb_result["urls"]
            return self._final_response(
                source="kb",
                chunks=chunks,
                urls=urls,
                confidence=kb_result["score"],
                trigger_web=False,
            )

        web_result = self._search_web(query)
        chunks = web_result["chunks"]
        urls = web_result["urls"]
        raw_for_training = web_result["raw_for_training"]
        self._queue_training_async(query, raw_for_training)
        return self._final_response(
            source="web",
            chunks=chunks,
            urls=urls,
            confidence=web_result["confidence"],
            trigger_web=True,
            raw_for_training=raw_for_training,
        )

    def _embed_query(self, query: str) -> list[float]:
        vector = self._get_embedder().encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vector, dtype=np.float32).reshape(-1).tolist()

    def _load_collection(self):
        chromadb = importlib.import_module("chromadb")
        chroma_path = self._rag_chroma_path()
        if chroma_path:
            client = chromadb.PersistentClient(path=chroma_path)
        else:
            client = chromadb.Client()
        return client.get_or_create_collection(name=self.collection_name)

    def _rag_chroma_path(self) -> str | None:
        if not self.config_path.exists():
            return None
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        rag = raw.get("rag") or {}
        modules_rag = (raw.get("modules") or {}).get("rag") or {}
        path = rag.get("chroma_path") or modules_rag.get("chroma_path")
        return str(path) if path else None

    def _search_knowledge_base(self, query_embedding: list[float]) -> dict[str, Any]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )
        docs = [str(doc) for doc in (result.get("documents") or [[]])[0] if str(doc).strip()]
        metadatas = [dict(item or {}) for item in (result.get("metadatas") or [[]])[0]]
        distances = list((result.get("distances") or [[]])[0])
        scores = [self._score_from_distance(distance) for distance in distances]
        top_score = scores[0] if scores else 0.0
        urls = self._urls_from_metadata(metadatas[: len(docs)])
        return {"chunks": docs, "urls": urls, "score": top_score}

    def _search_web(self, query: str) -> dict[str, Any]:
        from core.web_intelligence import WebIntelligence

        result = WebIntelligence(query).search()
        raw_chunks = list(result.get("raw_chunks") or [])
        answer_chunks = result.get("answer_chunks") or result.get("chunks") or []
        chunks = [self._chunk_text(chunk) for chunk in answer_chunks]
        if not chunks:
            chunks = [self._chunk_text(chunk) for chunk in raw_chunks[:3]]
        sources = [dict(source) for source in (result.get("sources") or []) if source.get("url")]
        urls = [str(source["url"]) for source in sources]
        confidence = max([float(source.get("score") or 0.0) for source in sources], default=0.0)
        return {
            "chunks": chunks,
            "urls": urls,
            "raw_for_training": raw_chunks or answer_chunks,
            "confidence": max(0.0, min(1.0, confidence)),
        }

    def _queue_training_async(self, query: str, raw_for_training: list[Any]) -> None:
        def queue() -> None:
            from core.self_trainer import SelfTrainer

            SelfTrainer().queue(query=query, chunks=raw_for_training)

        thread = threading.Thread(target=queue, daemon=True, name="query-router-self-training")
        thread.start()

    def _final_response(
        self,
        *,
        source: str,
        chunks: list[str],
        urls: list[str],
        confidence: float,
        trigger_web: bool,
        raw_for_training: list[Any] | None = None,
    ) -> dict[str, Any]:
        top_chunks = chunks[:3]
        context_lines = ["Context:"]
        context_lines.extend(f"[{index}] {chunk}" for index, chunk in enumerate(top_chunks, start=1))
        context_lines.append("")
        context_lines.append("Sources: " + (", ".join(urls) if urls else "none"))
        response = {
            "context_string": "\n".join(context_lines),
            "source": source,
            "chunks": chunks,
            "urls": urls,
            "confidence": max(0.0, min(1.0, confidence)),
            "trigger_web": trigger_web,
        }
        if raw_for_training is not None:
            response["raw_for_training"] = raw_for_training
        return response

    def _score_from_distance(self, distance: Any) -> float:
        value = float(distance)
        if 0.0 <= value <= 2.0:
            return max(0.0, min(1.0, 1.0 - value))
        return 1.0 / (1.0 + max(0.0, value))

    def _urls_from_metadata(self, metadatas: list[dict[str, Any]]) -> list[str]:
        urls: list[str] = []
        for metadata in metadatas:
            for key in ("url", "source_url", "source"):
                value = metadata.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    urls.append(value)
                    break
        return urls

    def _chunk_text(self, chunk: Any) -> str:
        if isinstance(chunk, dict):
            return str(chunk.get("text") or chunk.get("chunk") or "")
        return str(chunk)
