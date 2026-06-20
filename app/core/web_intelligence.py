from __future__ import annotations

import datetime
import hashlib
import logging
import threading
from pathlib import Path

import chromadb
import httpx
import trafilatura
import yaml
from sentence_transformers import util

from app.core.query_router import get_embedder

logger = logging.getLogger(__name__)


class WebIntelligence:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        chroma_path: str = cfg["rag"]["chroma_path"]
        client = chromadb.PersistentClient(path=chroma_path)
        self.pending = client.get_or_create_collection("pending_training")

    def search(self, query: str) -> dict:
        hits = self._ddg(query)
        if not hits:
            return {"answer_chunks": [], "raw_chunks": [], "sources": [], "error": "no_ddg_results", "query": query}

        all_chunks: list[dict] = []
        for item in hits:
            html = self._fetch(item.get("href", ""))
            if not html:
                continue
            text = trafilatura.extract(html, include_comments=False, include_tables=True)
            if not text:
                continue
            all_chunks.extend(self._chunk(text, item.get("href", ""), item.get("title", "")))

        if not all_chunks:
            return {"answer_chunks": [], "raw_chunks": [], "sources": [], "error": "extraction_failed", "query": query}

        scored = self._score(query, all_chunks)
        above = sorted([c for c in scored if c["score"] > 0.45], key=lambda x: x["score"], reverse=True)

        if above:
            threading.Thread(
                target=self._store_pending,
                args=(above, query),
                daemon=True,
            ).start()

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

    def _ddg(self, query: str) -> list[dict]:
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=5))
        except Exception as exc:
            logger.warning("DDG search failed: %s", exc)
            return []

    def _fetch(self, url: str) -> str | None:
        try:
            r = httpx.get(
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
        size, overlap = 1600, 200
        chunks, i, idx = [], 0, 0
        while i < len(text):
            chunks.append({"text": text[i : i + size], "url": url, "title": title, "chunk_index": idx})
            i += size - overlap
            idx += 1
        return chunks

    def _score(self, query: str, chunks: list[dict]) -> list[dict]:
        embedder = get_embedder()
        q_vec = embedder.encode(query, convert_to_tensor=True)
        c_vecs = embedder.encode([c["text"] for c in chunks], convert_to_tensor=True)
        scores = util.cos_sim(q_vec, c_vecs)[0].tolist()
        for i, c in enumerate(chunks):
            c["score"] = round(float(scores[i]), 4)
        return chunks

    def _store_pending(self, chunks: list[dict], query: str) -> None:
        ts = datetime.datetime.utcnow().isoformat()
        ids, docs, metas = [], [], []
        for c in chunks:
            uid = hashlib.sha256(f"{c['url']}{c['chunk_index']}".encode()).hexdigest()
            ids.append(uid)
            docs.append(c["text"])
            metas.append({"url": c["url"], "query": query, "score": c["score"], "timestamp": ts})
        try:
            self.pending.upsert(ids=ids, documents=docs, metadatas=metas)
        except Exception as exc:
            logger.warning("ChromaDB upsert failed: %s", exc)
