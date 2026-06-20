"""Embedding-based context optimization for maximizing useful LLM prompt tokens."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ContextChunk:
    text: str
    source: str = ""


@dataclass(slots=True)
class OptimizedPrompt:
    prompt: str
    selected_chunks: list[str]
    dropped_chunks: int


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _fallback_embed(texts: list[str], dim: int = 384) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


class ContextOptimizationSystem:
    """Rank, rerank, clean, and compress retrieved context chunks."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self._embedder: Any | None = None
        self._reranker: Any | None = None

    def _get_embedder(self) -> Any | None:
        if self._embedder is None:
            spec = importlib.util.find_spec("sentence_transformers")
            if spec is None:
                return None
            module = importlib.import_module("sentence_transformers")
            self._embedder = module.SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _encode(self, texts: list[str]) -> Any:
        embedder = self._get_embedder()
        if embedder is None:
            return _fallback_embed(texts)
        return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def rank_context_relevance(
        self, query: str, chunks: list[ContextChunk]
    ) -> list[tuple[float, ContextChunk]]:
        """Rank chunks with dense embeddings and cosine/IP similarity."""
        if not chunks:
            return []

        embeddings = self._encode([query, *[chunk.text for chunk in chunks]])
        query_vec = embeddings[0]
        chunk_vecs = embeddings[1:]

        faiss_spec = importlib.util.find_spec("faiss")
        if faiss_spec is not None and not isinstance(chunk_vecs, list):
            faiss = importlib.import_module("faiss")
            np = importlib.import_module("numpy")
            matrix = np.asarray(chunk_vecs, dtype="float32")
            q = np.asarray([query_vec], dtype="float32")
            faiss.normalize_L2(matrix)
            faiss.normalize_L2(q)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            scores, indices = index.search(q, len(chunks))
            return [(float(scores[0][j]), chunks[int(i)]) for j, i in enumerate(indices[0])]

        scored = [(_cosine(list(query_vec), list(vec)), chunk) for vec, chunk in zip(chunk_vecs, chunks, strict=False)]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _get_reranker(self) -> Any | None:
        if self._reranker is None:
            spec = importlib.util.find_spec("sentence_transformers")
            if spec is None:
                return None
            module = importlib.import_module("sentence_transformers")
            self._reranker = module.CrossEncoder(self.reranker_model_name)
        return self._reranker

    def rerank(self, query: str, top_chunks: list[ContextChunk]) -> list[tuple[float, ContextChunk]]:
        """Rerank top candidates with a CrossEncoder, falling back to dense scores."""
        candidates = top_chunks[:20]
        if not candidates:
            return []
        reranker = self._get_reranker()
        if reranker is None:
            return self.rank_context_relevance(query, candidates)
        scores = reranker.predict([(query, chunk.text) for chunk in candidates])
        ranked = [(float(score), chunk) for score, chunk in zip(scores, candidates, strict=False)]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def compress_context(self, text: str, max_sentences: int = 2) -> str:
        """Compress by TF-IDF-like sentence scoring instead of taking the first N."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        doc_freq: dict[str, int] = {}
        tokenized = [_tokenize(sentence) for sentence in sentences]
        for tokens in tokenized:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        n_docs = len(sentences)
        scored: list[tuple[float, int, str]] = []
        for idx, (sentence, tokens) in enumerate(zip(sentences, tokenized, strict=False)):
            if not tokens:
                scored.append((0.0, idx, sentence))
                continue
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            score = sum((count / len(tokens)) * math.log((1 + n_docs) / (1 + doc_freq[token]) + 1.0) for token, count in tf.items())
            scored.append((score, idx, sentence))

        best = sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences]
        return " ".join(sentence for _, _, sentence in sorted(best, key=lambda item: item[1]))

    def remove_noise(self, text: str) -> str:
        """Extract readable content from HTML, then strip common boilerplate."""
        cleaned = text
        if "<" in text and ">" in text and importlib.util.find_spec("trafilatura") is not None:
            trafilatura = importlib.import_module("trafilatura")
            extracted = trafilatura.extract(text)
            if extracted:
                cleaned = extracted

        boilerplate = [
            r"\bcookie policy\b.*?(?=\.|$)",
            r"\baccept cookies\b",
            r"\bsubscribe now\b",
            r"\badvertisement\b",
            r"\bclick here\b",
            r"\blorem ipsum\b",
            r"\bsign up for (our )?newsletter\b",
            r"\ball rights reserved\b",
        ]
        for pattern in boilerplate:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def optimize(self, query: str, context_chunks: list[ContextChunk], max_chars: int = 1200) -> OptimizedPrompt:
        dense_ranked = self.rank_context_relevance(query, context_chunks)
        reranked = self.rerank(query, [chunk for _, chunk in dense_ranked[:20]])
        ranked_chunks = [chunk for _, chunk in reranked] + [chunk for _, chunk in dense_ranked[20:]]

        selected: list[str] = []
        total_chars = 0
        for chunk in ranked_chunks:
            denoised = self.remove_noise(chunk.text)
            compressed = self.compress_context(denoised)
            if not compressed:
                continue
            if total_chars + len(compressed) > max_chars:
                continue
            selected.append(compressed)
            total_chars += len(compressed)

        dropped = max(0, len(context_chunks) - len(selected))
        prompt = (
            f"User Query: {query}\n\n"
            "Relevant Context:\n"
            + "\n".join(f"- {chunk}" for chunk in selected)
            + "\n\nUse only the context above. If insufficient, state uncertainty."
        )
        return OptimizedPrompt(prompt=prompt, selected_chunks=selected, dropped_chunks=dropped)
