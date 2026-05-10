from __future__ import annotations

import time
import types

import numpy as np

from app.core import query_router


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        assert model_name == "all-MiniLM-L6-v2"

    def encode(self, query: str):
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)


class FakeCollection:
    def __init__(self, result):
        self.result = result

    def query(self, **kwargs):
        assert kwargs["n_results"] == 5
        assert kwargs["include"] == ["documents", "metadatas", "distances"]
        return self.result


class FakePersistentClient:
    def __init__(self, *, path: str, result) -> None:
        assert path == "data/chroma"
        self.result = result

    def get_or_create_collection(self, name: str):
        assert name == "girivinity_knowledge"
        return FakeCollection(self.result)


def _mock_kb(monkeypatch, result) -> None:
    query_router._EMBEDDER = None
    monkeypatch.setattr(query_router, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(
        query_router,
        "chromadb",
        types.SimpleNamespace(PersistentClient=lambda path: FakePersistentClient(path=path, result=result)),
    )


def test_kb_hit(monkeypatch):
    _mock_kb(
        monkeypatch,
        {
            "documents": [["kb chunk"]],
            "metadatas": [[{"url": "https://kb.example/source"}]],
            "distances": [[0.1]],
        },
    )

    result = query_router.QueryRouter().route("What is Girivinity?")

    assert result["source"] == "knowledge_base"
    assert result["trigger_web"] is False


def test_kb_miss(monkeypatch):
    _mock_kb(
        monkeypatch,
        {
            "documents": [["weak chunk"]],
            "metadatas": [[{}]],
            "distances": [[1.9]],
        },
    )

    class FakeWebIntelligence:
        def search(self, query: str):
            return {
                "answer_chunks": [{"text": "test", "score": 0.6}],
                "raw_chunks": [],
                "sources": [],
            }

    import app.core.web_intelligence as web_intelligence

    monkeypatch.setattr(web_intelligence, "WebIntelligence", FakeWebIntelligence)

    result = query_router.QueryRouter().route("Needs web")

    assert result["source"] == "web"


def test_training_is_async(monkeypatch):
    _mock_kb(
        monkeypatch,
        {
            "documents": [["weak chunk"]],
            "metadatas": [[{}]],
            "distances": [[1.9]],
        },
    )

    class FakeWebIntelligence:
        def search(self, query: str):
            return {
                "answer_chunks": [{"text": "test", "score": 0.6}],
                "raw_chunks": [{"text": "raw", "score": 0.6}],
                "sources": [],
            }

    import app.core.web_intelligence as web_intelligence

    monkeypatch.setattr(web_intelligence, "WebIntelligence", FakeWebIntelligence)
    monkeypatch.setattr(query_router.QueryRouter, "_queue_training", lambda self, query, chunks: time.sleep(1.0))

    started = time.time()
    result = query_router.QueryRouter().route("Needs background queue")
    elapsed = time.time() - started

    assert result["source"] == "web"
    assert elapsed < 0.5
