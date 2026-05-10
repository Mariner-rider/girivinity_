from __future__ import annotations

import datetime
import hashlib
import importlib
import importlib.util
import logging
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

from app.core.query_router import get_embedder

logger = logging.getLogger(__name__)


def _optional_module(module_name: str) -> Any | None:
    if module_name in sys.modules:
        return sys.modules[module_name]
    if importlib.util.find_spec(module_name) is None:
        return None
    return importlib.import_module(module_name)


httpx = _optional_module("httpx")
trafilatura = _optional_module("trafilatura")
chromadb = _optional_module("chromadb")
_sentence_transformers = _optional_module("sentence_transformers")
util = getattr(_sentence_transformers, "util", None)
_duckduckgo_search = _optional_module("duckduckgo_search")
DDGS = getattr(_duckduckgo_search, "DDGS", None)


class WebIntelligence:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        chroma_path = cfg["rag"]["chroma_path"]
        chroma_module = self._require_module("chromadb", chromadb)
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

        all_chunks: list[dict] = []
        for item in urls_data:
            html = self._fetch(item.get("href", ""))
            if not html:
                continue
            trafilatura_module = self._require_module("trafilatura", trafilatura)
            text = trafilatura_module.extract(html, include_comments=False, include_tables=True)
            if not text:
                continue
            all_chunks.extend(self._chunk(text, item.get("href", ""), item.get("title", "")))

        if not all_chunks:
            return {
                "answer_chunks": [],
                "raw_chunks": [],
                "sources": [],
                "error": "extraction_failed",
                "query": query,
            }

        scored = self._score(query, all_chunks)
        above = sorted(
            [c for c in scored if c["score"] > 0.45],
            key=lambda x: x["score"],
            reverse=True,
        )

        threading.Thread(target=self._store_pending, args=(above, query), daemon=True).start()

        seen: set[str] = set()
        sources: list[dict] = []
        for c in above:
            if c["url"] not in seen:
                sources.append({"url": c["url"], "title": c["title"], "score": c["score"]})
                seen.add(c["url"])

        return {
            "answer_chunks": above[:3],
            "raw_chunks": above,
            "sources": sources,
            "query": query,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

    def _ddg_search(self, query: str) -> list[dict]:
        try:
            ddgs_cls = DDGS or getattr(self._require_module("duckduckgo_search", _duckduckgo_search), "DDGS")
            with ddgs_cls() as ddgs:
                return list(ddgs.text(query, max_results=5))
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []

    def _fetch(self, url: str) -> str | None:
        try:
            httpx_module = self._require_module("httpx", httpx)
            r = httpx_module.get(
                url,
                timeout=8.0,
                follow_redirects=True,
                headers={"User-Agent": "GirivinityBot/1.0"},
            )
            return r.text if r.status_code == 200 else None
        except Exception as exc:
            logger.warning("Fetch failed %s: %s", url, exc)
            return None

    def _chunk(self, text: str, url: str, title: str) -> list[dict]:
        chunks: list[dict] = []
        size, overlap, idx = 1600, 200, 0
        i = 0
        while i < len(text):
            chunks.append(
                {
                    "text": text[i : i + size],
                    "url": url,
                    "title": title,
                    "chunk_index": idx,
                }
            )
            i += size - overlap
            idx += 1
        return chunks

    def _score(self, query: str, chunks: list[dict]) -> list[dict]:
        embedder = get_embedder()
        q_vec = embedder.encode(query, convert_to_tensor=True)
        texts = [c["text"] for c in chunks]
        c_vecs = embedder.encode(texts, convert_to_tensor=True)
        util_module = self._require_module("sentence_transformers", _sentence_transformers).util if util is None else util
        scores = util_module.cos_sim(q_vec, c_vecs)[0].tolist()
        for i, c in enumerate(chunks):
            c["score"] = round(float(scores[i]), 4)
        return chunks

    def _store_pending(self, chunks: list[dict], query: str) -> None:
        if not chunks:
            return
        ts = datetime.datetime.utcnow().isoformat()
        ids, docs, metas = [], [], []
        for c in chunks:
            uid = hashlib.sha256(f"{c['url']}{c['chunk_index']}".encode()).hexdigest()
            ids.append(uid)
            docs.append(c["text"])
            metas.append(
                {
                    "url": c["url"],
                    "query": query,
                    "score": c["score"],
                    "timestamp": ts,
                }
            )
        try:
            self.pending.upsert(ids=ids, documents=docs, metadatas=metas)
        except Exception as exc:
            logger.warning("ChromaDB pending upsert failed: %s", exc)

    def _require_module(self, module_name: str, imported_module: Any | None) -> Any:
        if imported_module is not None:
            return imported_module
        module = _optional_module(module_name)
        if module is None:
            raise ModuleNotFoundError(f"{module_name} is required")
        return module
