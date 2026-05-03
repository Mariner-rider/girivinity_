from __future__ import annotations

from typing import Protocol

from app.crawler.items import CrawledDocument


class VectorDBClient(Protocol):
    def upsert_document(self, document_id: str, text: str, metadata: dict) -> None:
        ...


class InMemoryVectorDBClient:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[str, dict]] = {}

    def upsert_document(self, document_id: str, text: str, metadata: dict) -> None:
        self.documents[document_id] = (text, metadata)


def build_document_id(document: CrawledDocument) -> str:
    return document.url
