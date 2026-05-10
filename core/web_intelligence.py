from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

LOG_PATH = Path("logs/crawler.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
if not any(
    isinstance(handler, logging.FileHandler) and handler.baseFilename == str(LOG_PATH.resolve())
    for handler in logger.handlers
):
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


@dataclass(slots=True)
class SearchResult:
    url: str
    title: str
    body: str = ""


@dataclass(slots=True)
class WebChunk:
    text: str
    url: str
    title: str
    chunk_index: int
    score: float = 0.0
    embedding: list[float] | None = None

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "url": self.url,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "score": self.score,
        }


class WebIntelligence:
    """Live web search, extraction, chunk scoring, and training-intake pipeline."""

    def __init__(
        self,
        query: str | None = None,
        *,
        max_results: int = 5,
        max_chars: int = 1600,
        overlap_chars: int = 200,
        relevance_threshold: float = 0.45,
        chroma_collection: str = "pending_training",
    ) -> None:
        self.query = (query or "").strip()
        self.max_results = max_results
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.relevance_threshold = relevance_threshold
        self.chroma_collection = chroma_collection

    def search(self, query: str | None = None) -> dict[str, Any]:
        query = (query or self.query).strip()
        timestamp = datetime.utcnow().isoformat()
        if not query:
            return self._no_results(query, timestamp)

        search_results = self._duckduckgo_search(query)
        scored_chunks: list[WebChunk] = []
        for result in search_results:
            html = self._fetch(result.url)
            if html is None:
                continue
            text = self._extract_text(html, result.url)
            if not text:
                continue
            scored_chunks.extend(self._score_chunks(query, self._chunk_text(text, result)))

        kept_chunks = [chunk for chunk in scored_chunks if chunk.score > self.relevance_threshold]
        kept_chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        if not kept_chunks:
            return self._no_results(query, timestamp)

        raw_chunks = [chunk.as_public_dict() for chunk in kept_chunks]
        self._store_raw_chunks(kept_chunks, query, timestamp)
        return {
            "answer_chunks": raw_chunks[:3],
            "raw_chunks": raw_chunks,
            "sources": self._sources(kept_chunks),
            "query": query,
            "timestamp": timestamp,
        }

    def run(self) -> dict[str, Any]:
        """Compatibility alias for older callers."""
        return self.search(self.query)

    def fetch(self) -> list[str]:
        """Compatibility helper returning answer chunk text only."""
        return [chunk["text"] for chunk in self.search(self.query).get("answer_chunks", [])]

    def _duckduckgo_search(self, query: str) -> list[SearchResult]:
        try:
            from duckduckgo_search import DDGS  # type: ignore

            with DDGS() as ddgs:
                rows = list(ddgs.text(query, max_results=self.max_results))
        except Exception as exc:
            logger.exception("DuckDuckGo search failed for query %r: %s", query, exc)
            return []

        results: list[SearchResult] = []
        seen: set[str] = set()
        for row in rows:
            url = str(row.get("href") or row.get("url") or "").strip()
            if not url or url in seen:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=str(row.get("title") or "").strip(),
                    body=str(row.get("body") or "").strip(),
                )
            )
            seen.add(url)
        return results[: self.max_results]

    def _fetch(self, url: str) -> str | None:
        try:
            import httpx

            response = httpx.get(url, timeout=8, follow_redirects=True)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            logger.exception("Failed to fetch %s: %s", url, exc)
            return None

    def _extract_text(self, html_content: str, url: str) -> str | None:
        try:
            import trafilatura

            text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
            )
        except Exception as exc:
            logger.exception("Trafilatura extraction failed for %s: %s", url, exc)
            return None
        if text is None:
            return None
        text = " ".join(text.split())
        return text or None

    def _chunk_text(self, text: str, result: SearchResult) -> list[WebChunk]:
        if not text:
            return []
        chunk_size = max(1, self.max_chars)
        overlap = min(max(0, self.overlap_chars), chunk_size - 1)
        step = max(1, chunk_size - overlap)
        chunks: list[WebChunk] = []
        for index, start in enumerate(range(0, len(text), step)):
            segment = text[start : start + chunk_size].strip()
            if segment:
                chunks.append(
                    WebChunk(
                        text=segment,
                        url=result.url,
                        title=result.title,
                        chunk_index=index,
                    )
                )
            if start + chunk_size >= len(text):
                break
        return chunks

    def _score_chunks(self, query: str, chunks: list[WebChunk]) -> list[WebChunk]:
        if not chunks:
            return []
        try:
            from core.query_router import QueryRouter

            embedder = QueryRouter._get_embedder()
            embeddings = embedder.encode(
                [query, *[chunk.text for chunk in chunks]],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except TypeError:
            embeddings = embedder.encode([query, *[chunk.text for chunk in chunks]], convert_to_numpy=True)
            embeddings = self._normalize_rows(np.asarray(embeddings, dtype=np.float32))
        except Exception as exc:
            logger.exception("Chunk relevance scoring failed: %s", exc)
            return []

        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(chunks) + 1:
            logger.error("Unexpected embedding shape during web scoring: %s", arr.shape)
            return []
        query_embedding = arr[0]
        for chunk, chunk_embedding in zip(chunks, arr[1:]):
            chunk.score = float(np.dot(query_embedding, chunk_embedding))
            chunk.embedding = chunk_embedding.astype(float).tolist()
        return chunks

    def _normalize_rows(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _store_raw_chunks(self, chunks: list[WebChunk], query: str, timestamp: str) -> None:
        if not chunks:
            return
        try:
            import chromadb  # type: ignore

            client = chromadb.Client()
            collection = client.get_or_create_collection(name=self.chroma_collection)
            collection.upsert(
                ids=[self._chunk_id(chunk) for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {
                        "url": chunk.url,
                        "query": query,
                        "score": chunk.score,
                        "timestamp": timestamp,
                    }
                    for chunk in chunks
                ],
                embeddings=[chunk.embedding for chunk in chunks] if chunks[0].embedding is not None else None,
            )
        except Exception as exc:
            logger.exception("Failed to upsert web chunks into ChromaDB: %s", exc)

    def _chunk_id(self, chunk: WebChunk) -> str:
        return hashlib.sha256(f"{chunk.url}{chunk.chunk_index}".encode("utf-8")).hexdigest()

    def _sources(self, chunks: list[WebChunk]) -> list[dict[str, Any]]:
        by_url: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            existing = by_url.get(chunk.url)
            if existing is None or chunk.score > existing["score"]:
                by_url[chunk.url] = {"url": chunk.url, "title": chunk.title, "score": chunk.score}
        return sorted(by_url.values(), key=lambda source: source["score"], reverse=True)

    def _no_results(self, query: str, timestamp: str) -> dict[str, Any]:
        return {
            "answer_chunks": [],
            "raw_chunks": [],
            "sources": [],
            "error": "no_results",
            "query": query,
            "timestamp": timestamp,
        }


class WebSearchPipeline(WebIntelligence):
    """Backward-compatible name for older callers."""
