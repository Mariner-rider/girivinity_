"""Hybrid RAG retrieval with dense FAISS, sparse BM25, RRF fusion, and reranking."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import pickle
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def _fallback_embeddings(texts: list[str], dim: int = 384) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


@dataclass
class RAGConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60
    embedding_dim: int = 384
    storage_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, config: Any) -> "RAGConfig":
        if isinstance(config, cls):
            return config
        if config is None:
            return cls()
        if isinstance(config, dict):
            known = {k: v for k, v in config.items() if k in cls.__dataclass_fields__}
            extra = {k: v for k, v in config.items() if k not in cls.__dataclass_fields__}
            return cls(**known, extra=extra)
        return cls(
            embedding_model=getattr(config, "embedding_model", cls.embedding_model),
            reranker_model=getattr(config, "reranker_model", cls.reranker_model),
            rrf_k=getattr(config, "rrf_k", cls.rrf_k),
            embedding_dim=getattr(config, "embedding_dim", cls.embedding_dim),
            storage_path=getattr(config, "storage_path", None),
        )


class _SimpleBM25:
    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus
        self.doc_freq: dict[str, int] = {}
        self.avgdl = sum(len(doc) for doc in corpus) / max(len(corpus), 1)
        for doc in corpus:
            for token in set(doc):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        n = max(len(self.corpus), 1)
        k1 = 1.5
        b = 0.75
        for doc in self.corpus:
            freqs: dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                if token not in freqs:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = freqs[token]
                denom = tf + k1 * (1 - b + b * len(doc) / max(self.avgdl, 1e-9))
                score += idf * (tf * (k1 + 1)) / denom
            scores.append(score)
        return scores


class RAGEngine:
    """
    Hybrid retrieval: FAISS (dense) + BM25 (sparse) combined with RRF.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = RAGConfig.from_any(config)
        self._lock = threading.RLock()
        self._docs: list[dict[str, Any]] = []
        self._embeddings: Any | None = None
        self._faiss_index: Any | None = None
        self._bm25: Any | None = None
        self._embedder: Any | None = None
        self._reranker: Any | None = None
        if self.config.storage_path:
            path = Path(self.config.storage_path)
            if path.exists():
                self.load(str(path))

    def _new_id(self, text: str, source: str, metadata: dict[str, Any]) -> str:
        raw = json.dumps({"text": text, "source": source, "metadata": metadata}, sort_keys=True)
        seed = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"doc_{len(self._docs):08d}_{seed}"

    def _get_embedder(self) -> Any | None:
        if self._embedder is None:
            spec = importlib.util.find_spec("sentence_transformers")
            if spec is None:
                return None
            module = importlib.import_module("sentence_transformers")
            self._embedder = module.SentenceTransformer(self.config.embedding_model)
        return self._embedder

    def _encode(self, texts: list[str]) -> Any:
        embedder = self._get_embedder()
        if embedder is None:
            return _fallback_embeddings(texts, self.config.embedding_dim)
        return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def _rebuild_indexes(self) -> None:
        texts = [doc["text"] for doc in self._docs]
        if not texts:
            self._embeddings = None
            self._faiss_index = None
            self._bm25 = None
            return

        embeddings = self._encode(texts)
        faiss_spec = importlib.util.find_spec("faiss")
        if faiss_spec is not None and not isinstance(embeddings, list):
            faiss = importlib.import_module("faiss")
            np = importlib.import_module("numpy")
            matrix = np.asarray(embeddings, dtype="float32")
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._embeddings = matrix
            self._faiss_index = index
        else:
            self._embeddings = embeddings
            self._faiss_index = None

        corpus = [_tokens(text) for text in texts]
        rank_bm25_spec = importlib.util.find_spec("rank_bm25")
        if rank_bm25_spec is not None:
            module = importlib.import_module("rank_bm25")
            self._bm25 = module.BM25Okapi(corpus)
        else:
            self._bm25 = _SimpleBM25(corpus)

    def add(self, text: str, source: str = "", metadata: dict | None = None) -> str:
        """Embed and store a document. Returns doc_id. Thread-safe."""
        clean_metadata = dict(metadata or {})
        with self._lock:
            doc_id = self._new_id(text, source, clean_metadata)
            self._docs.append(
                {"id": doc_id, "text": text, "source": source, "metadata": clean_metadata}
            )
            self._rebuild_indexes()
            return doc_id

    def add_batch(self, docs: list[dict]) -> list[str]:
        """Batch insert. Each dict: {text, source?, metadata?}. Returns doc_ids."""
        ids: list[str] = []
        with self._lock:
            for doc in docs:
                metadata = dict(doc.get("metadata") or {})
                doc_id = self._new_id(doc["text"], doc.get("source", ""), metadata)
                self._docs.append(
                    {
                        "id": doc_id,
                        "text": doc["text"],
                        "source": doc.get("source", ""),
                        "metadata": metadata,
                    }
                )
                ids.append(doc_id)
            self._rebuild_indexes()
        return ids

    def _metadata_matches(self, metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if key == "date_after":
                raw_date = metadata.get("date") or metadata.get("published") or metadata.get("created_at")
                if raw_date is None:
                    return False
                if self._parse_date(raw_date) <= self._parse_date(expected):
                    return False
            elif key == "date_before":
                raw_date = metadata.get("date") or metadata.get("published") or metadata.get("created_at")
                if raw_date is None:
                    return False
                if self._parse_date(raw_date) >= self._parse_date(expected):
                    return False
            elif metadata.get(key) != expected:
                return False
        return True

    @staticmethod
    def _parse_date(value: Any) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text[:10]).date()

    def _dense_ranking(self, query: str, candidate_count: int) -> list[tuple[int, float]]:
        if not self._docs:
            return []
        q_vec = self._encode([query])
        if self._faiss_index is not None:
            np = importlib.import_module("numpy")
            faiss = importlib.import_module("faiss")
            q = np.asarray(q_vec, dtype="float32")
            faiss.normalize_L2(q)
            scores, indices = self._faiss_index.search(q, min(candidate_count, len(self._docs)))
            return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0], strict=False)]

        query_vec = q_vec[0]
        scored = []
        for idx, vec in enumerate(self._embeddings or []):
            score = sum(float(a) * float(b) for a, b in zip(query_vec, vec, strict=False))
            scored.append((idx, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:candidate_count]

    def _sparse_ranking(self, query: str, candidate_count: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokens(query))
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:candidate_count]]

    def _get_reranker(self) -> Any | None:
        if self._reranker is None:
            spec = importlib.util.find_spec("sentence_transformers")
            if spec is None:
                return None
            module = importlib.import_module("sentence_transformers")
            self._reranker = module.CrossEncoder(self.config.reranker_model)
        return self._reranker

    def _rerank(self, query: str, doc_indices: list[int]) -> list[tuple[int, float]]:
        if not doc_indices:
            return []
        reranker = self._get_reranker()
        if reranker is None:
            return [(idx, 0.0) for idx in doc_indices]
        pairs = [(query, self._docs[idx]["text"]) for idx in doc_indices]
        scores = reranker.predict(pairs)
        ranked = [(idx, float(score)) for idx, score in zip(doc_indices, scores, strict=False)]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def query(
        self, query: str, top_k: int = 8, filter_metadata: dict | None = None
    ) -> list[dict]:
        """Run dense + sparse retrieval, RRF fusion, metadata filter, and reranking."""
        with self._lock:
            if not self._docs:
                return []
            candidate_count = min(len(self._docs), max(top_k * 3, 20))
            dense = self._dense_ranking(query, candidate_count)
            sparse = self._sparse_ranking(query, candidate_count)

            fused: dict[int, float] = {}
            for ranking in (dense, sparse):
                for rank, (idx, _) in enumerate(ranking, start=1):
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.config.rrf_k + rank)

            filtered = [
                (idx, score)
                for idx, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)
                if self._metadata_matches(self._docs[idx]["metadata"], filter_metadata)
            ]
            top_for_rerank = [idx for idx, _ in filtered[:20]]
            reranked = self._rerank(query, top_for_rerank)
            rerank_scores = dict(reranked)
            final_indices = [idx for idx, _ in reranked[:top_k]] or [idx for idx, _ in filtered[:top_k]]

            results = []
            for idx in final_indices:
                doc = self._docs[idx]
                results.append(
                    {
                        "id": doc["id"],
                        "text": doc["text"],
                        "source": doc["source"],
                        "score": rerank_scores.get(idx, fused.get(idx, 0.0)),
                        "metadata": dict(doc["metadata"]),
                    }
                )
            return results

    def delete(self, doc_ids: list[str]) -> int:
        with self._lock:
            wanted = set(doc_ids)
            before = len(self._docs)
            self._docs = [doc for doc in self._docs if doc["id"] not in wanted]
            deleted = before - len(self._docs)
            if deleted:
                self._rebuild_indexes()
            return deleted

    def save(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {"config": self.config, "docs": self._docs}
            with (target / "rag_engine.pkl").open("wb") as fh:
                pickle.dump(payload, fh)

    def load(self, path: str) -> None:
        source = Path(path)
        with (source / "rag_engine.pkl").open("rb") as fh:
            payload = pickle.load(fh)
        with self._lock:
            self.config = payload.get("config", self.config)
            self._docs = payload.get("docs", [])
            self._rebuild_indexes()

    def stats(self) -> dict:
        with self._lock:
            sources = {doc["source"] for doc in self._docs if doc.get("source")}
            source_types: dict[str, int] = {}
            for doc in self._docs:
                stype = doc.get("metadata", {}).get("source_type", "unknown")
                source_types[stype] = source_types.get(stype, 0) + 1
            return {
                "documents": len(self._docs),
                "sources": len(sources),
                "source_types": source_types,
                "dense_index": self._faiss_index is not None,
                "sparse_index": self._bm25 is not None,
                "embedding_model": self.config.embedding_model,
                "reranker_model": self.config.reranker_model,
            }
