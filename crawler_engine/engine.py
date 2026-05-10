from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Protocol
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
        trust_score = self.domain_trust_score(page.url)
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
