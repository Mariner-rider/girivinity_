from __future__ import annotations

import datetime
import hashlib
import importlib
import logging
from pathlib import Path
from typing import Any

import yaml
from core.query_router import get_embedder

try:
    from duckduckgo_search import DDGS  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependency may be mocked in tests.
    DDGS = None  # type: ignore

try:
    import httpx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependency may be mocked in tests.
    httpx = None  # type: ignore

try:
    import trafilatura  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependency may be mocked in tests.
    trafilatura = None  # type: ignore

try:
    import chromadb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependency may be mocked in tests.
    chromadb = None  # type: ignore

try:
    from sentence_transformers import util  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependency may be mocked in tests.
    util = None  # type: ignore

try:
    import torch  # noqa: F401  # Imported for runtime tensor support in sentence-transformers.
except ModuleNotFoundError:  # pragma: no cover - torch may be absent in constrained installs.
    torch = None  # type: ignore  # noqa: F841

logger = logging.getLogger(__name__)


class WebIntelligence:
    def __init__(self):
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        chroma_path = cfg["rag"]["chroma_path"]
        chroma_module = self._module("chromadb", chromadb)
        self.client = chroma_module.PersistentClient(path=chroma_path)
        self.pending = self.client.get_or_create_collection("pending_training")

    def search(self, query: str) -> dict:
        urls_data = self._ddg_search(query)
        if not urls_data:
            return {
                "answer_chunks": [],
                "raw_chunks": [],
                "sources": [],
                "error": "no_ddg_results",
                "query": query,
            }

        all_chunks = []
        for item in urls_data:
            html = self._fetch_url(item["href"])
            if not html:
                continue
            trafilatura_module = self._module("trafilatura", trafilatura)
            text = trafilatura_module.extract(html, include_comments=False, include_tables=True)
            if not text:
                continue
            chunks = self._chunk_text(text, item["href"], item.get("title", ""))
            all_chunks.extend(chunks)

        if not all_chunks:
            return {
                "answer_chunks": [],
                "raw_chunks": [],
                "sources": [],
                "error": "extraction_failed",
                "query": query,
            }

        scored = self._score_chunks(query, all_chunks)
        above = [c for c in scored if c["score"] > 0.45]
        above.sort(key=lambda x: x["score"], reverse=True)

        self._store_pending(above, query)

        sources = []
        seen_urls = set()
        for c in above:
            if c["url"] not in seen_urls:
                sources.append({"url": c["url"], "title": c["title"], "score": c["score"]})
                seen_urls.add(c["url"])

        return {
            "answer_chunks": above[:3],
            "raw_chunks": above,
            "sources": sources,
            "query": query,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

    def _ddg_search(self, query: str) -> list:
        try:
            ddgs_cls = DDGS or self._module("duckduckgo_search", None).DDGS
            with ddgs_cls() as ddgs:
                return list(ddgs.text(query, max_results=5))
        except Exception as e:
            logger.warning(f"DDG search failed: {e}")
            return []

    def _fetch_url(self, url: str) -> str | None:
        try:
            httpx_module = self._module("httpx", httpx)
            r = httpx_module.get(
                url,
                timeout=8.0,
                follow_redirects=True,
                headers={"User-Agent": "GirivinityBot/1.0"},
            )
            return r.text if r.status_code == 200 else None
        except Exception as e:
            logger.warning(f"Fetch failed {url}: {e}")
            return None

    def _chunk_text(self, text: str, url: str, title: str) -> list:
        chunks = []
        size, overlap = 1600, 200
        i = 0
        idx = 0
        while i < len(text):
            chunk_text = text[i : i + size]
            chunks.append({"text": chunk_text, "url": url, "title": title, "chunk_index": idx})
            i += size - overlap
            idx += 1
        return chunks

    def _score_chunks(self, query: str, chunks: list) -> list:
        embedder = get_embedder()
        q_vec = embedder.encode(query, convert_to_tensor=True)
        texts = [c["text"] for c in chunks]
        c_vecs = embedder.encode(texts, convert_to_tensor=True)
        util_module = util or self._module("sentence_transformers", None).util
        scores = util_module.cos_sim(q_vec, c_vecs)[0].tolist()
        for i, c in enumerate(chunks):
            c["score"] = round(scores[i], 4)
        return chunks

    def _store_pending(self, chunks: list, query: str) -> None:
        if not chunks:
            return
        ids, docs, metas = [], [], []
        ts = datetime.datetime.utcnow().isoformat()
        for c in chunks:
            uid = hashlib.sha256(f"{c['url']}{c['chunk_index']}".encode()).hexdigest()
            ids.append(uid)
            docs.append(c["text"])
            metas.append({"url": c["url"], "query": query, "score": c["score"], "timestamp": ts})
        try:
            self.pending.upsert(ids=ids, documents=docs, metadatas=metas)
        except Exception as e:
            logger.warning(f"ChromaDB upsert failed: {e}")

    def _module(self, module_name: str, imported_module: Any):
        if imported_module is not None:
            return imported_module
        return importlib.import_module(module_name)


class WebSearchPipeline:
    """Compatibility wrapper for existing callers that use .run() or .fetch()."""

    def __init__(self, query: str):
        self.query = query
        self._web = WebIntelligence()

    def run(self) -> dict:
        return self._web.search(self.query)

    def fetch(self) -> list[str]:
        return [chunk.get("text", "") for chunk in self.run().get("answer_chunks", [])]
