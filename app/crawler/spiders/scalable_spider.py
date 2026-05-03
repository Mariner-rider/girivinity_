from __future__ import annotations

import scrapy

from app.crawler.items import CrawledDocument
from app.crawler.queue import URLQueue


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
        for url in (start_urls or []):
            self.url_queue.push(url, depth=0)

    def start_requests(self):
        while len(self.url_queue) > 0:
            task = self.url_queue.pop()
            yield scrapy.Request(task.url, callback=self.parse, meta={"depth": task.depth})

    def parse(self, response):
        depth = int(response.meta.get("depth", 0))
        title = response.css("title::text").get(default="").strip()

        paragraphs = response.css("p::text").getall()
        body_text = "\n".join(part.strip() for part in paragraphs if part.strip())

        metadata = {
            "status": response.status,
            "content_type": response.headers.get("Content-Type", b"").decode("utf-8", errors="ignore"),
            "depth": depth,
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
            if self.url_queue.push(absolute, depth=depth + 1):
                yield scrapy.Request(absolute, callback=self.parse, meta={"depth": depth + 1})
