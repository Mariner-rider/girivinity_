from __future__ import annotations

from scrapy.crawler import CrawlerProcess

from app.crawler.spiders.scalable_spider import ScalableSpider
from app.crawler.vector_db import InMemoryVectorDBClient


def run_crawler(start_urls: list[str], max_depth: int = 2) -> InMemoryVectorDBClient:
    vector_client = InMemoryVectorDBClient()

    process = CrawlerProcess(
        settings={
            "ROBOTSTXT_OBEY": True,
            "VECTOR_DB_CLIENT": vector_client,
        }
    )
    process.crawl(ScalableSpider, start_urls=start_urls, max_depth=max_depth)
    process.start()
    return vector_client
