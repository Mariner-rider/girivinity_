from app.crawler.items import CrawledDocument
from app.crawler.pipelines import VectorDBPipeline
from app.crawler.vector_db import InMemoryVectorDBClient


def test_pipeline_pushes_to_vector_db():
    client = InMemoryVectorDBClient()
    pipeline = VectorDBPipeline(client)

    doc = CrawledDocument(url="https://example.com", title="Example", text="hello", metadata={"lang": "en"})
    pipeline.process_item(doc, spider=None)

    assert "https://example.com" in client.documents
