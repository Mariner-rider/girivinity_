from __future__ import annotations

import hashlib
import importlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchResult:
    url: str
    title: str


@dataclass(slots=True)
class ScoredChunk:
    text: str
    url: str
    title: str
    score: float
    embedding: list[float]


class WebSearchPipeline:
    """Autonomous web intelligence pipeline for live search and training intake.

    The pipeline performs real DuckDuckGo searches, fetches returned pages, extracts
    readable article text, scores text chunks against the original query, and stores
    relevant raw chunks in ChromaDB for later training review.
    """

    def __init__(
        self,
        query: str,
        *,
        max_results: int = 5,
        chunk_tokens: int = 400,
        chunk_overlap: int = 50,
        relevance_threshold: float = 0.5,
        model_name: str = "all-MiniLM-L6-v2",
        chroma_collection: str = "pending_training",
    ) -> None:
        self.query = query.strip()
        self.max_results = max_results
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.relevance_threshold = relevance_threshold
        self.chroma_collection = chroma_collection

        sentence_transformers = importlib.import_module("sentence_transformers")
        self._embedder = sentence_transformers.SentenceTransformer(model_name)

    def run(self) -> dict[str, Any]:
        """Execute the full web intelligence pipeline and return structured results."""
        timestamp = datetime.now(timezone.utc).isoformat()
        if not self.query:
            return self._empty_response(timestamp)

        scored_chunks: list[ScoredChunk] = []
        for result in self._search():
            html = self._fetch_url(result.url)
            if not html:
                continue

            text = self._extract_text(html, result.url)
            if not text:
                continue

            chunks = self._chunk_text(text)
            if not chunks:
                continue

            scored_chunks.extend(self._score_chunks(chunks, result))

        relevant_chunks = [
            chunk for chunk in scored_chunks if chunk.score > self.relevance_threshold
        ]
        relevant_chunks.sort(key=lambda chunk: chunk.score, reverse=True)

        self._store_raw_chunks(relevant_chunks, timestamp)

        return {
            "answer_chunks": [chunk.text for chunk in relevant_chunks[:3]],
            "sources": self._build_sources(relevant_chunks),
            "raw_chunks": [self._serialize_chunk(chunk) for chunk in relevant_chunks],
            "query": self.query,
            "timestamp": timestamp,
        }

    def fetch(self) -> list[str]:
        """Compatibility helper returning only top answer chunk text."""
        return list(self.run()["answer_chunks"])

    def _empty_response(self, timestamp: str) -> dict[str, Any]:
        return {
            "answer_chunks": [],
            "sources": [],
            "raw_chunks": [],
            "query": self.query,
            "timestamp": timestamp,
        }

    def _search(self) -> list[SearchResult]:
        duckduckgo_search = importlib.import_module("duckduckgo_search")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        try:
            with duckduckgo_search.DDGS() as ddgs:
                search_rows = ddgs.text(self.query, max_results=self.max_results)
                for row in search_rows:
                    url = str(row.get("href") or row.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    title = self._clean_whitespace(str(row.get("title") or ""))
                    results.append(SearchResult(url=url, title=title))
                    seen_urls.add(url)
                    if len(results) >= self.max_results:
                        break
        except Exception as exc:  # DuckDuckGo/network failures should not crash pipeline.
            logger.warning("DuckDuckGo search failed for query %r: %s", self.query, exc)

        return results

    def _fetch_url(self, url: str) -> str | None:
        httpx = importlib.import_module("httpx")
        try:
            response = httpx.get(
                url,
                timeout=10,
                follow_redirects=True,
                headers={"User-Agent": "girivinity-web-intelligence/0.1"},
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:  # Per-URL failures should return partial results.
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def _extract_text(self, html: str, url: str) -> str:
        trafilatura = importlib.import_module("trafilatura")
        try:
            extracted = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=False,
                favor_recall=True,
            )
        except Exception as exc:
            logger.warning("Text extraction failed for %s: %s", url, exc)
            return ""
        return self._clean_whitespace(extracted or "")

    def _chunk_text(self, text: str) -> list[str]:
        tokens = text.split()
        if not tokens:
            return []

        size = max(1, self.chunk_tokens)
        overlap = min(max(0, self.chunk_overlap), size - 1)
        step = max(1, size - overlap)

        chunks: list[str] = []
        for start in range(0, len(tokens), step):
            chunk = " ".join(tokens[start : start + size]).strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= len(tokens):
                break
        return chunks

    def _score_chunks(self, chunks: list[str], result: SearchResult) -> list[ScoredChunk]:
        try:
            embeddings = self._embedder.encode(
                [self.query, *chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except TypeError:
            embeddings = self._embedder.encode([self.query, *chunks], convert_to_numpy=True)
            embeddings = self._normalize_rows(np.asarray(embeddings, dtype=np.float32))
        except Exception as exc:
            logger.warning("Embedding failed for %s: %s", result.url, exc)
            return []

        embedding_array = np.asarray(embeddings, dtype=np.float32)
        if embedding_array.ndim != 2 or embedding_array.shape[0] != len(chunks) + 1:
            logger.warning("Unexpected embedding shape for %s: %s", result.url, embedding_array.shape)
            return []

        query_embedding = embedding_array[0]
        scored: list[ScoredChunk] = []
        for text, chunk_embedding in zip(chunks, embedding_array[1:]):
            score = float(np.dot(query_embedding, chunk_embedding))
            scored.append(
                ScoredChunk(
                    text=text,
                    url=result.url,
                    title=result.title,
                    score=score,
                    embedding=chunk_embedding.astype(float).tolist(),
                )
            )
        return scored

    def _normalize_rows(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _store_raw_chunks(self, chunks: list[ScoredChunk], timestamp: str) -> None:
        if not chunks:
            return

        try:
            chromadb = importlib.import_module("chromadb")
            client = chromadb.Client()
            collection = client.get_or_create_collection(name=self.chroma_collection)
            collection.add(
                ids=[self._chunk_id(chunk, timestamp, index) for index, chunk in enumerate(chunks)],
                documents=[chunk.text for chunk in chunks],
                embeddings=[chunk.embedding for chunk in chunks],
                metadatas=[
                    {
                        "url": chunk.url,
                        "timestamp": timestamp,
                        "query": self.query,
                        "relevance_score": chunk.score,
                    }
                    for chunk in chunks
                ],
            )
        except Exception as exc:
            logger.warning("Failed to store raw chunks in ChromaDB: %s", exc)

    def _build_sources(self, chunks: list[ScoredChunk]) -> list[dict[str, Any]]:
        sources_by_url: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            existing = sources_by_url.get(chunk.url)
            if existing is None or chunk.score > existing["score"]:
                sources_by_url[chunk.url] = {
                    "url": chunk.url,
                    "title": chunk.title,
                    "score": chunk.score,
                }
        sources = sorted(sources_by_url.values(), key=lambda item: item["score"], reverse=True)
        return sources[: self.max_results]

    def _serialize_chunk(self, chunk: ScoredChunk) -> dict[str, Any]:
        return {
            "text": chunk.text,
            "url": chunk.url,
            "title": chunk.title,
            "score": chunk.score,
        }

    def _chunk_id(self, chunk: ScoredChunk, timestamp: str, index: int) -> str:
        payload = f"{timestamp}\0{self.query}\0{chunk.url}\0{index}\0{chunk.text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _clean_whitespace(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()
