from __future__ import annotations

from app.crawler.items import CrawledDocument
from app.crawler.vector_db import VectorDBClient, build_document_id
from app.security.policy import secure_operation


class VectorDBPipeline:
    def __init__(self, vector_db_client: VectorDBClient) -> None:
        self.vector_db_client = vector_db_client

    @classmethod
    def from_crawler(cls, crawler):
        client = crawler.settings.get("VECTOR_DB_CLIENT")
        if client is None:
            raise ValueError("VECTOR_DB_CLIENT setting must be configured")
        return cls(client)

    @secure_operation("crawler.vector_pipeline")
    def process_item(self, item, spider):
        if isinstance(item, dict):
            document = CrawledDocument(**item)
        else:
            document = item

        doc_id = build_document_id(document)
        payload_metadata = {"title": document.title, **document.metadata}
        self.vector_db_client.upsert_document(doc_id, document.text, payload_metadata)
        return item
