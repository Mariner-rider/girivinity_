from __future__ import annotations

import scrapy

from app.crawler.items import CrawledDocument
from app.crawler.queue import URLQueue
from app.security.policy import SecurityGuard, secure_operation


class ScalableSpider(scrapy.Spider):
    name = "scalable_spider"

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DUPEFILTER_CLASS": "scrapy.dupefilters.RFPDupeFilter",
        "ITEM_PIPELINES": {
            "app.crawler.pipelines.VectorDBPipeline": 300,
        },
    }

    def __init__(self, start_urls: list[str] | None = None, max_depth: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = int(max_depth)
        self.url_queue = URLQueue()
        self.security_guard = SecurityGuard()
        for url in (start_urls or []):
            trust = self.security_guard.score_url_trust(url)
            if trust.trusted:
                self.url_queue.push(url, depth=0)

    @secure_operation("crawler.start_requests")
    def start_requests(self):
        while len(self.url_queue) > 0:
            task = self.url_queue.pop()
            trust = self.security_guard.require_trusted_url(task.url)
            yield scrapy.Request(
                task.url,
                callback=self.parse,
                meta={
                    "depth": task.depth,
                    "trust_score": trust.score,
                    "trust_reasons": trust.reasons,
                },
            )

    @secure_operation("crawler.parse")
    def parse(self, response):
        depth = int(response.meta.get("depth", 0))
        title = response.css("title::text").get(default="").strip()

        paragraphs = response.css("p::text").getall()
        body_text = "\n".join(part.strip() for part in paragraphs if part.strip())

        metadata = {
            "status": response.status,
            "content_type": response.headers.get("Content-Type", b"").decode(
                "utf-8", errors="ignore"
            ),
            "depth": depth,
            "trust_score": response.meta.get("trust_score", 0.0),
            "trust_reasons": tuple(response.meta.get("trust_reasons", ())),
        }

        yield CrawledDocument(
            url=response.url,
            title=title,
            text=body_text,
            metadata=metadata,
        )

        if depth >= self.max_depth:
            return

        for next_url in response.css("a::attr(href)").getall():
            absolute = response.urljoin(next_url)
            trust = self.security_guard.score_url_trust(absolute)
            if trust.trusted and self.url_queue.push(absolute, depth=depth + 1):
                yield scrapy.Request(
                    absolute,
                    callback=self.parse,
                    meta={
                        "depth": depth + 1,
                        "trust_score": trust.score,
                        "trust_reasons": trust.reasons,
                    },
                )
