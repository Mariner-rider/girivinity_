"""Context optimization system to maximize useful tokens for LLM prompts."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ContextChunk:
    text: str
    source: str = ""


@dataclass(slots=True)
class OptimizedPrompt:
    prompt: str
    selected_chunks: list[str]
    dropped_chunks: int


class ContextOptimizationSystem:
    """Optimize retrieved context with dense ranking, reranking, and compression."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedder: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self._embedder = embedder
        self._reranker = reranker

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _get_reranker(self) -> Any:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker

    def _encode(self, texts: list[str]) -> Any:
        embeddings = self._get_embedder().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        import numpy as np

        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return embeddings / norms

    def _lexical_rank_fallback(self, query: str, chunks: list[ContextChunk]) -> list[tuple[float, ContextChunk]]:
        q_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        scored: list[tuple[float, ContextChunk]] = []
        for chunk in chunks:
            c_tokens = set(re.findall(r"[a-zA-Z0-9]+", chunk.text.lower()))
            overlap = len(q_tokens & c_tokens)
            density = overlap / max(len(c_tokens), 1)
            scored.append(((0.7 * overlap) + (0.3 * density), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def rank_context_relevance(self, query: str, chunks: list[ContextChunk]) -> list[tuple[float, ContextChunk]]:
        """Rank chunks by dense cosine similarity using FAISS inner product over L2-normalized embeddings."""
        if not query.strip() or not chunks:
            return []

        try:
            import numpy as np

            query_embedding = self._encode([query])
            chunk_embeddings = self._encode([chunk.text for chunk in chunks])

            try:
                import faiss

                index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
                index.add(chunk_embeddings)
                scores, indices = index.search(query_embedding, len(chunks))
                return [
                    (float(scores[0][i]), chunks[int(idx)])
                    for i, idx in enumerate(indices[0])
                    if int(idx) >= 0
                ]
            except Exception as exc:  # pragma: no cover - FAISS optional at runtime.
                logger.warning("FAISS context ranking unavailable; falling back to NumPy cosine similarity: %s", exc)
                scores = np.dot(chunk_embeddings, query_embedding[0])
                order = np.argsort(-scores)
                return [(float(scores[idx]), chunks[int(idx)]) for idx in order]
        except Exception as exc:  # pragma: no cover - keeps lightweight test/dev envs usable.
            logger.warning("Embedding context ranking unavailable; falling back to lexical ranking: %s", exc)
            return self._lexical_rank_fallback(query, chunks)

    def rerank(
        self,
        query: str,
        top_chunks: list[ContextChunk] | list[tuple[float, ContextChunk]],
        top_k: int | None = None,
    ) -> list[tuple[float, ContextChunk]]:
        """Rerank up to the top 20 candidates with a CrossEncoder."""
        if not top_chunks:
            return []

        normalized: list[tuple[float, ContextChunk]] = []
        for item in top_chunks:
            if isinstance(item, tuple):
                normalized.append((float(item[0]), item[1]))
            else:
                normalized.append((0.0, item))

        rerank_window = normalized[:20]
        pairs = [(query, chunk.text) for _, chunk in rerank_window]
        predictions = self._get_reranker().predict(pairs)
        scored = [(float(score), chunk) for score, (_, chunk) in zip(predictions, rerank_window, strict=False)]

        # Preserve candidates beyond the reranking window with their original dense score.
        scored.extend(normalized[20:])
        scored.sort(key=lambda item: item[0], reverse=True)
        if top_k is not None:
            return scored[:top_k]
        return scored

    def compress_context(self, text: str, max_sentences: int = 2) -> str:
        """Compress context by keeping the top-N TF-IDF scored sentences."""
        if max_sentences <= 0:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        tokenized = [re.findall(r"[a-zA-Z0-9]+", sentence.lower()) for sentence in sentences]
        document_frequency: dict[str, int] = {}
        for tokens in tokenized:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        sentence_count = len(sentences)
        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(tokenized):
            if not tokens:
                scored.append((0.0, index))
                continue
            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            score = 0.0
            for token, count in term_counts.items():
                tf = count / len(tokens)
                idf = math.log((sentence_count + 1) / (document_frequency[token] + 1)) + 1.0
                score += tf * idf
            scored.append((score, index))

        selected_indices = sorted(index for _, index in sorted(scored, reverse=True)[:max_sentences])
        return " ".join(sentences[index] for index in selected_indices)

    def remove_noise(self, text: str) -> str:
        """Extract readable text from HTML and strip common boilerplate from plain text."""
        candidate = text
        if re.search(r"</?[a-z][\s\S]*>", text, flags=re.IGNORECASE):
            try:
                import trafilatura

                extracted = trafilatura.extract(text)
                if extracted:
                    candidate = extracted
            except Exception as exc:  # pragma: no cover - depends on optional parser internals.
                logger.warning("trafilatura extraction failed; falling back to regex cleanup: %s", exc)

        boilerplate_patterns = [
            r"\b(lorem ipsum|click here|subscribe now|advertisement|cookie policy)\b",
            r"\b(sign up for our newsletter|all rights reserved|privacy policy|terms of service)\b",
            r"(?im)^\s*(share this|related articles|read more|back to top)\s*$",
        ]
        cleaned = candidate
        for pattern in boilerplate_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()

    def optimize(self, query: str, context_chunks: list[ContextChunk], max_chars: int = 1200) -> OptimizedPrompt:
        ranked = self.rank_context_relevance(query, context_chunks)
        try:
            ranked = self.rerank(query, ranked[:20]) + ranked[20:]
        except Exception as exc:  # pragma: no cover - protects deployments without reranker weights.
            logger.warning("CrossEncoder reranking unavailable; using dense ranking only: %s", exc)

        selected: list[str] = []
        total_chars = 0
        for _, chunk in ranked:
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
