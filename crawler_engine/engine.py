from __future__ import annotations

import hashlib
import re
import time
import asyncio
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Protocol
from pathlib import Path
from urllib.parse import urlparse


class ContentQualityScorer(Protocol):
    def score(self, text: str, metadata: dict) -> float:
        ...


@dataclass(slots=True)
class PageData:
    url: str
    html: str
    fetched_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class CrawlResult:
    url: str
    title: str
    text: str
    metadata: dict
    language: str
    topics: list[str]
    trust_score: float
    quality_score: float


class _SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self.text_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data):
        cleaned = data.strip()
        if not cleaned:
            return
        if self._in_title and not self.title:
            self.title = cleaned
        self.text_parts.append(cleaned)


class HeuristicLLMQualityScorer:
    def score(self, text: str, metadata: dict) -> float:
        length_score = min(len(text) / 1500, 1.0)
        link_ratio = metadata.get("link_ratio", 0.0)
        spam_penalty = 0.4 if metadata.get("is_spam", False) else 0.0
        return max(0.0, min(1.0, (0.7 * length_score) + (0.3 * (1 - link_ratio)) - spam_penalty))


class CrawlerEngine:
    def __init__(
        self,
        seeds: list[str],
        quality_scorer: ContentQualityScorer | None = None,
        min_trust_score: float = 0.55,
        min_quality_score: float = 0.55,
        max_pages: int = 100,
    ) -> None:
        self.queue = deque(seeds)
        self.min_trust_score = min_trust_score
        self.min_quality_score = min_quality_score
        self.max_pages = max_pages
        self.quality_scorer = quality_scorer or HeuristicLLMQualityScorer()
        self.seen_urls: set[str] = set()
        self.content_hashes: set[str] = set()
        self.dedup_db = Path("data/crawler_seen.sqlite3")
        self.known_security_domains = {"nvd.nist.gov", "cisa.gov", "bleepingcomputer.com", "krebsonsecurity.com", "sans.org"}
        self.blocklist = {"malware.example", "phishing.example"}
        self.crawler_interval_seconds = 3600
        self._browser = None
        self._init_dedup_db()


    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "CrawlerEngine":
        import yaml
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
        crawler_cfg = (cfg or {}).get("crawler", {}) or {}
        return cls([], min_trust_score=float(crawler_cfg.get("trust_threshold", 0.6)), max_pages=int(crawler_cfg.get("max_depth", 100)))

    def _init_dedup_db(self) -> None:
        self.dedup_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.dedup_db) as db:
            db.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY, url TEXT, created_at REAL)")
            db.commit()

    async def _ensure_browser(self):
        if self._browser is not None:
            return self._browser
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch()
        return self._browser

    async def _crawl_url_async(self, url: str) -> str:
        try:
            import trafilatura
            fetched = trafilatura.fetch_url(url)
            if fetched and fetched.strip():
                return fetched
        except Exception:
            pass
        browser = await self._ensure_browser()
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=20000)
            return await page.content()
        finally:
            await page.close()

    def crawl_url(self, url: str) -> dict | None:
        html = asyncio.run(self._crawl_url_async(url))
        if not html or self._deduplicate(url, html):
            return None
        result = self.process_page(PageData(url=url, html=html))
        return None if result is None else result.__dict__

    def trust_score(self, url: str, content: str) -> float:
        parsed = urlparse(url); domain = parsed.netloc.lower()
        score = 0.0
        if parsed.scheme == "https": score += 0.2
        if any(domain.endswith(d) for d in self.known_security_domains): score += 0.3
        if len(content or "") > 500: score += 0.1
        if not re.search(r"ignore\s+previous|<script|javascript:|prompt\s*injection", content or "", re.I): score += 0.2
        if domain not in self.blocklist: score += 0.2
        return round(min(1.0, score), 3)

    def cybersecurity_crawl(self) -> list[dict]:
        urls = [
            "https://nvd.nist.gov/vuln",
            "https://www.cisa.gov/news-events/cybersecurity-advisories",
            "https://www.bleepingcomputer.com/",
            "https://krebsonsecurity.com/",
            "https://www.sans.org/blog/",
        ]
        results = []
        for url in urls:
            try:
                item = self.crawl_url(url)
                if item: results.append(item)
            except Exception:
                continue
        return results

    def crawl_batch(self) -> list[dict]:
        results = []
        if self.queue:
            for _ in range(min(len(self.queue), self.max_pages)):
                url = self.queue.popleft()
                item = self.crawl_url(url)
                if item: results.append(item)
        else:
            results.extend(self.cybersecurity_crawl())
        return results

    def _deduplicate(self, url: str, content: str) -> bool:
        digest = hashlib.sha256((url + (content or "")[:500]).encode("utf-8")).hexdigest()
        with sqlite3.connect(self.dedup_db) as db:
            exists = db.execute("SELECT 1 FROM seen WHERE hash=?", (digest,)).fetchone()
            if exists: return True
            db.execute("INSERT INTO seen(hash,url,created_at) VALUES(?,?,?)", (digest, url, time.time()))
            db.commit()
        return False

    def domain_trust_score(self, url: str) -> float:
        domain = urlparse(url).netloc.lower()
        if not domain:
            return 0.0
        if domain.endswith(".gov") or domain.endswith(".edu"):
            return 0.95
        if domain.endswith(".org"):
            return 0.8
        return 0.65 if "." in domain else 0.2

    def detect_language(self, text: str) -> str:
        lower = text.lower()
        if any(word in lower for word in [" the ", " and ", " is "]):
            return "en"
        if any(word in lower for word in [" el ", " la ", " de "]):
            return "es"
        return "unknown"

    def topic_tagging(self, text: str) -> list[str]:
        topic_rules = {
            "ai": ["model", "llm", "neural", "machine learning"],
            "finance": ["stock", "market", "revenue", "investment"],
            "health": ["patient", "clinical", "disease", "treatment"],
            "security": ["vulnerability", "attack", "encryption", "security"],
        }
        lower = text.lower()
        topics = [topic for topic, keywords in topic_rules.items() if any(k in lower for k in keywords)]
        return topics or ["general"]

    def _extract(self, page: PageData) -> tuple[str, str, dict]:
        parser = _SimpleHTMLTextExtractor()
        parser.feed(page.html)
        text = " ".join(parser.text_parts)
        out_links = re.findall(r"href=['\"](.*?)['\"]", page.html, flags=re.IGNORECASE)
        is_spam = bool(re.search(r"\b(buy now|free money|click here)\b", text.lower()))
        metadata = {
            "fetched_at": page.fetched_at,
            "domain": urlparse(page.url).netloc,
            "word_count": len(text.split()),
            "outbound_links": len(out_links),
            "link_ratio": len(out_links) / max(len(text.split()), 1),
            "is_spam": is_spam,
        }
        return parser.title, text, metadata

    def _is_duplicate(self, url: str, text: str) -> bool:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if url in self.seen_urls or content_hash in self.content_hashes:
            return True
        self.seen_urls.add(url)
        self.content_hashes.add(content_hash)
        return False

    def process_page(self, page: PageData) -> CrawlResult | None:
        trust_score = max(self.domain_trust_score(page.url), self.trust_score(page.url, page.html))
        if trust_score < self.min_trust_score:
            return None

        title, text, metadata = self._extract(page)
        if self._is_duplicate(page.url, text):
            return None

        quality = self.quality_scorer.score(text, metadata)
        if metadata["is_spam"] or quality < self.min_quality_score:
            return None

        language = self.detect_language(text)
        topics = self.topic_tagging(text)
        return CrawlResult(
            url=page.url,
            title=title,
            text=text,
            metadata=metadata,
            language=language,
            topics=topics,
            trust_score=trust_score,
            quality_score=quality,
        )

    def run(self, pages: list[PageData]) -> list[CrawlResult]:
        accepted: list[CrawlResult] = []
        for page in pages[: self.max_pages]:
            result = self.process_page(page)
            if result:
                accepted.append(result)
        return accepted
