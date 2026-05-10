from __future__ import annotations

import re
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import faiss  # type: ignore
import numpy as np
import requests
import yaml
try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    from html.parser import HTMLParser

    class _MiniSoupParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.text_parts: list[str] = []
            self.links: list[str] = []
            self.in_ignored = False

        def handle_starttag(self, tag, attrs):
            attrs_map = dict(attrs)
            if tag in {"script", "style", "nav", "footer", "aside"}:
                self.in_ignored = True
            href = attrs_map.get("href")
            if tag == "a" and href:
                self.links.append(href)

        def handle_endtag(self, tag):
            if tag in {"script", "style", "nav", "footer", "aside"}:
                self.in_ignored = False

        def handle_data(self, data):
            if not self.in_ignored:
                self.text_parts.append(data)

    class _MiniTag:
        def __init__(self, href: str):
            self._href = href

        def __getitem__(self, key: str) -> str:
            if key == "href":
                return self._href
            raise KeyError(key)

    class BeautifulSoup:  # type: ignore
        def __init__(self, html: str, parser: str = "html.parser") -> None:
            p = _MiniSoupParser()
            p.feed(html)
            self._text = " ".join(p.text_parts)
            self._links = p.links

        def __call__(self, tags):
            return []

        def find_all(self, name=None, href=False, class_=None):
            if name == "a" and href:
                return [_MiniTag(h) for h in self._links]
            return []

        def get_text(self, sep=" ", strip=False):
            txt = self._text
            return txt.strip() if strip else txt



@dataclass(slots=True)
class CrawledPage:
    url: str
    depth: int
    text: str


class KnowledgeIngestionPipeline:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        cfg = self._load_config(Path(config_path))
        self.seed_urls: list[str] = list(cfg["crawler"]["seed_urls"])
        self.max_depth: int = int(cfg["crawler"].get("max_depth", 2))
        self.index_path = Path(str(cfg["rag"]["index_path"]))
        self.chunk_db_path = Path(str(cfg["rag"]["chunk_db_path"]))

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "sentence-transformers is required. Install with `pip install sentence-transformers`."
            ) from exc
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_config(self, path: Path) -> dict[str, Any]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        crawler_cfg = raw.get("crawler", {})
        rag_cfg = raw.get("rag", {})
        if not crawler_cfg.get("seed_urls"):
            raise KeyError("Missing required config key: crawler.seed_urls")
        if "index_path" not in rag_cfg:
            raise KeyError("Missing required config key: rag.index_path")
        if "chunk_db_path" not in rag_cfg:
            raise KeyError("Missing required config key: rag.chunk_db_path")
        return {"crawler": crawler_cfg, "rag": rag_cfg}

    def _extract_clean_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        heuristic = re.compile(r"(nav|footer|menu|ads?|advert|banner|promo|sidebar)", re.I)
        for el in soup.find_all(class_=heuristic):
            el.decompose()
        text = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()

    def _chunk_text(self, text: str, size: int = 512, overlap: int = 64) -> list[str]:
        tokens = text.split()
        if not tokens:
            return []
        chunks: list[str] = []
        step = max(1, size - overlap)
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + size]
            if not chunk_tokens:
                continue
            chunks.append(" ".join(chunk_tokens))
            if i + size >= len(tokens):
                break
        return chunks

    def _crawl(self) -> list[CrawledPage]:
        queue: deque[tuple[str, int]] = deque((u, 0) for u in self.seed_urls)
        seen: set[str] = set()
        pages: list[CrawledPage] = []
        allowed_domains = {urlparse(u).netloc for u in self.seed_urls}

        while queue:
            url, depth = queue.popleft()
            url = urldefrag(url)[0]
            if url in seen or depth > self.max_depth:
                continue
            seen.add(url)

            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            clean_text = self._extract_clean_text(resp.text)
            if clean_text:
                pages.append(CrawledPage(url=url, depth=depth, text=clean_text))

            if depth == self.max_depth:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                nxt = urldefrag(urljoin(url, a["href"]))[0]
                parsed = urlparse(nxt)
                if parsed.scheme in {"http", "https"} and parsed.netloc in allowed_domains:
                    queue.append((nxt, depth + 1))
        return pages

    def _init_db(self) -> sqlite3.Connection:
        self.chunk_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.chunk_db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()
        return conn

    def _append_to_faiss(self, vectors: np.ndarray) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        dim = vectors.shape[1]
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            if index.d != dim:
                raise ValueError(f"FAISS index dimension mismatch: expected {index.d}, got {dim}")
        else:
            index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        faiss.write_index(index, str(self.index_path))

    def run(self) -> int:
        pages = self._crawl()
        all_chunks: list[tuple[str, str]] = []
        for page in pages:
            for chunk in self._chunk_text(page.text, size=512, overlap=64):
                all_chunks.append((page.url, chunk))
        if not all_chunks:
            return 0

        texts = [c[1] for c in all_chunks]
        vectors = self.embedder.encode(texts, convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype=np.float32)

        self._append_to_faiss(vectors)

        now = datetime.now(timezone.utc).isoformat()
        with self._init_db() as conn:
            conn.executemany(
                "INSERT INTO chunks(url, chunk_text, timestamp) VALUES (?, ?, ?)",
                [(url, chunk, now) for url, chunk in all_chunks],
            )
            conn.commit()
        return len(all_chunks)
