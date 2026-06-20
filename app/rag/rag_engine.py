"""Hybrid dense/sparse RAG retrieval engine with CrossEncoder reranking."""

from __future__ import annotations

import logging
import pickle
import re
import threading
import uuid
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Hybrid retrieval: FAISS (dense) + BM25 (sparse) combined with RRF.

    Dense retrieval: SentenceTransformer -> FAISS IndexFlatIP
    Sparse retrieval: BM25Okapi from rank-bm25
    Fusion: RRF score = Σ 1/(k + rank_i) where k=60

    Metadata filtering supports exact key matches plus ``date_after`` against common
    document date fields (``date``, ``created_at``, ``published_at``, ``timestamp``).
    """

    def __init__(self, config: Any):
        self.config = config
        self.embedding_model_name = self._config_get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.reranker_model_name = self._config_get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.rrf_k = int(self._config_get("rrf_k", 60))
        self._lock = threading.RLock()
        self._embedder: Any | None = None
        self._reranker: Any | None = None
        self._dense_index: Any | None = None
        self._bm25: Any | None = None
        self._embedding_dim: int | None = None
        self._documents: list[dict[str, Any]] = []
        self._deleted_doc_ids: set[str] = set()

    def _config_get(self, key: str, default: Any) -> Any:
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _get_reranker(self) -> Any:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _encode(self, texts: list[str]) -> Any:
        import numpy as np

        embeddings = self._get_embedder().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return embeddings / norms

    def _rebuild_indexes_locked(self) -> None:
        active_docs = [doc for doc in self._documents if doc["doc_id"] not in self._deleted_doc_ids]
        if not active_docs:
            self._dense_index = None
            self._bm25 = None
            self._embedding_dim = None
            return

        import faiss
        from rank_bm25 import BM25Okapi

        embeddings = self._encode([doc["text"] for doc in active_docs])
        self._embedding_dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(self._embedding_dim)
        index.add(embeddings)
        self._dense_index = index
        self._bm25 = BM25Okapi([self._tokenize(doc["text"]) for doc in active_docs])

    def _active_documents_locked(self) -> list[dict[str, Any]]:
        return [doc for doc in self._documents if doc["doc_id"] not in self._deleted_doc_ids]

    def add(self, text: str, source: str = "", metadata: dict | None = None) -> str:
        """Embed and store a document. Returns doc_id. Thread-safe."""
        doc_id = str(uuid.uuid4())
        document = {
            "doc_id": doc_id,
            "text": text,
            "source": source,
            "metadata": dict(metadata or {}),
        }
        with self._lock:
            self._documents.append(document)
            self._rebuild_indexes_locked()
        return doc_id

    def add_batch(self, docs: list[dict]) -> list[str]:
        """Batch insert. Each dict: {text, source?, metadata?}. Returns doc_ids."""
        doc_ids: list[str] = []
        with self._lock:
            for item in docs:
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                self._documents.append(
                    {
                        "doc_id": doc_id,
                        "text": item["text"],
                        "source": item.get("source", ""),
                        "metadata": dict(item.get("metadata") or {}),
                    }
                )
            self._rebuild_indexes_locked()
        return doc_ids

    def _dense_rank_locked(self, query: str, candidate_count: int) -> list[str]:
        if self._dense_index is None:
            return []
        query_embedding = self._encode([query])
        _, indices = self._dense_index.search(query_embedding, candidate_count)
        active_docs = self._active_documents_locked()
        return [active_docs[int(index)]["doc_id"] for index in indices[0] if int(index) >= 0]

    def _sparse_rank_locked(self, query: str, candidate_count: int) -> list[str]:
        if self._bm25 is None:
            return []
        import numpy as np

        scores = self._bm25.get_scores(self._tokenize(query))
        order = np.argsort(-scores)[:candidate_count]
        active_docs = self._active_documents_locked()
        return [active_docs[int(index)]["doc_id"] for index in order if float(scores[int(index)]) > 0.0]

    def _rrf_fuse(self, rankings: list[list[str]]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))
        return fused

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            except ValueError:
                return None
        return None

    def _matches_metadata(self, doc: dict[str, Any], filters: dict | None) -> bool:
        if not filters:
            return True
        metadata = doc.get("metadata") or {}
        for key, expected in filters.items():
            if key == "date_after":
                threshold = self._parse_date(expected)
                actual = None
                for date_key in ("date", "created_at", "published_at", "timestamp"):
                    actual = self._parse_date(metadata.get(date_key))
                    if actual is not None:
                        break
                if threshold is not None and (actual is None or actual < threshold):
                    return False
            elif metadata.get(key) != expected:
                return False
        return True

    def query(
        self,
        query: str,
        top_k: int = 8,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """Run dense retrieval, BM25 retrieval, RRF fusion, metadata filtering, and reranking."""
        if not query.strip() or top_k <= 0:
            return []

        with self._lock:
            active_docs = self._active_documents_locked()
            if not active_docs:
                return []
            candidate_count = min(len(active_docs), max(top_k * 3, 20))
            dense_ids = self._dense_rank_locked(query, candidate_count)
            sparse_ids = self._sparse_rank_locked(query, candidate_count)
            fused = self._rrf_fuse([dense_ids, sparse_ids])
            docs_by_id = {doc["doc_id"]: doc for doc in active_docs}
            candidates = [
                (doc_id, score)
                for doc_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)
                if self._matches_metadata(docs_by_id[doc_id], filter_metadata)
            ]
            rerank_candidates = candidates[:20]
            tail = candidates[20:]

        if rerank_candidates:
            pairs = [(query, docs_by_id[doc_id]["text"]) for doc_id, _ in rerank_candidates]
            rerank_scores = self._get_reranker().predict(pairs)
            scored = [(doc_id, float(score)) for score, (doc_id, _) in zip(rerank_scores, rerank_candidates, strict=False)]
            scored.extend(tail)
            scored.sort(key=lambda item: item[1], reverse=True)
        else:
            scored = []

        results: list[dict[str, Any]] = []
        for doc_id, score in scored[:top_k]:
            doc = docs_by_id[doc_id]
            results.append(
                {
                    "doc_id": doc_id,
                    "text": doc["text"],
                    "source": doc.get("source", ""),
                    "score": float(score),
                    "metadata": dict(doc.get("metadata") or {}),
                }
            )
        return results

    def delete(self, doc_ids: list[str]) -> int:
        """Delete documents by id and return the number of newly deleted documents."""
        with self._lock:
            existing = {doc["doc_id"] for doc in self._documents}
            to_delete = [doc_id for doc_id in doc_ids if doc_id in existing and doc_id not in self._deleted_doc_ids]
            self._deleted_doc_ids.update(to_delete)
            if to_delete:
                self._rebuild_indexes_locked()
            return len(to_delete)

    def save(self, path: str) -> None:
        """Persist documents and deletion markers. Models and indexes are rebuilt on load."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "documents": self._documents,
                "deleted_doc_ids": list(self._deleted_doc_ids),
                "embedding_model_name": self.embedding_model_name,
                "reranker_model_name": self.reranker_model_name,
                "rrf_k": self.rrf_k,
            }
        with destination.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: str) -> None:
        """Load documents and rebuild dense/sparse indexes."""
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        with self._lock:
            self._documents = list(payload.get("documents", []))
            self._deleted_doc_ids = set(payload.get("deleted_doc_ids", []))
            self.embedding_model_name = payload.get("embedding_model_name", self.embedding_model_name)
            self.reranker_model_name = payload.get("reranker_model_name", self.reranker_model_name)
            self.rrf_k = int(payload.get("rrf_k", self.rrf_k))
            self._rebuild_indexes_locked()

    def stats(self) -> dict:
        """Return retrieval index statistics."""
        with self._lock:
            active_docs = self._active_documents_locked()
            sources = Counter(doc.get("source", "") for doc in active_docs)
            return {
                "doc_count": len(active_docs),
                "deleted_doc_count": len(self._deleted_doc_ids),
                "embedding_dim": self._embedding_dim,
                "has_dense_index": self._dense_index is not None,
                "has_sparse_index": self._bm25 is not None,
                "sources": dict(sources),
            }
