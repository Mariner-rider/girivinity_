from __future__ import annotations

import sys
import types

import numpy as np

from core.web_intelligence import WebSearchPipeline


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        assert model_name == "all-MiniLM-L6-v2"

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        assert convert_to_numpy is True
        embeddings = []
        for index, _text in enumerate(texts):
            embeddings.append([1.0, 0.0] if index < 3 else [0.0, 1.0])
        return np.asarray(embeddings, dtype=np.float32)


class FakeDDGS:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query: str, max_results: int):
        assert query == "test query"
        assert max_results == 5
        return [
            {"href": "https://example.com/one", "title": "One"},
            {"href": "https://example.com/fail", "title": "Fail"},
        ]


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class FakeCollection:
    def __init__(self) -> None:
        self.added = None

    def add(self, **kwargs) -> None:
        self.added = kwargs


class FakeChromaClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection

    def get_or_create_collection(self, name: str):
        assert name == "pending_training"
        return self.collection


def test_web_search_pipeline_returns_scored_chunks_and_stores_training(monkeypatch):
    collection = FakeCollection()

    def fake_http_get(url: str, **kwargs):
        if url.endswith("fail"):
            raise RuntimeError("network down")
        assert kwargs["timeout"] == 10
        return FakeResponse("<html>content</html>")

    def fake_extract(html: str, **kwargs):
        assert html == "<html>content</html>"
        return " ".join(f"token{i}" for i in range(700))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(sys.modules, "duckduckgo_search", types.SimpleNamespace(DDGS=FakeDDGS))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(get=fake_http_get))
    monkeypatch.setitem(sys.modules, "trafilatura", types.SimpleNamespace(extract=fake_extract))
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        types.SimpleNamespace(Client=lambda: FakeChromaClient(collection)),
    )

    result = WebSearchPipeline("test query").run()

    assert result["query"] == "test query"
    assert len(result["answer_chunks"]) == 2
    assert len(result["raw_chunks"]) == 2
    assert result["sources"] == [{"url": "https://example.com/one", "title": "One", "score": 1.0}]
    assert collection.added is not None
    assert collection.added["documents"] == result["answer_chunks"]
    assert collection.added["metadatas"][0]["url"] == "https://example.com/one"
    assert collection.added["metadatas"][0]["query"] == "test query"
    assert collection.added["metadatas"][0]["relevance_score"] == 1.0
