from __future__ import annotations

import time
import types

import numpy as np

from core import query_router


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


def _mock_kb(monkeypatch, result):
    query_router.QueryRouter.reset_model_cache()
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
    assert result["confidence"] == 0.95
    assert result["chunks"][0]["text"] == "kb chunk"
    assert result["context_string"] == "Context:\n[1] kb chunk"


def test_kb_miss(monkeypatch):
    _mock_kb(
        monkeypatch,
        {
            "documents": [["weak chunk"]],
            "metadatas": [[{}]],
            "distances": [[1.8]],
        },
    )

    class FakeWebIntelligence:
        def search(self, query: str):
            return {
                "answer_chunks": [{"text": "test", "score": 0.6}],
                "raw_chunks": [],
                "sources": [],
            }

    import core.web_intelligence as web_intelligence

    monkeypatch.setattr(web_intelligence, "WebIntelligence", FakeWebIntelligence)

    result = query_router.QueryRouter().route("Needs web")

    assert result["source"] == "web"
    assert result["trigger_web"] is True
    assert result["chunks"] == [{"text": "test", "score": 0.6}]


def test_background_thread_does_not_block(monkeypatch):
    _mock_kb(
        monkeypatch,
        {
            "documents": [["weak chunk"]],
            "metadatas": [[{}]],
            "distances": [[1.8]],
        },
    )

    class FakeWebIntelligence:
        def search(self, query: str):
            return {
                "answer_chunks": [{"text": "test", "score": 0.6}],
                "raw_chunks": [{"text": "raw", "score": 0.6}],
                "sources": [],
            }

    class SlowSelfTrainer:
        def queue(self, query: str, chunks: list) -> None:
            time.sleep(1.0)

    import core.self_trainer as self_trainer
    import core.web_intelligence as web_intelligence

    monkeypatch.setattr(web_intelligence, "WebIntelligence", FakeWebIntelligence)
    monkeypatch.setattr(self_trainer, "SelfTrainer", SlowSelfTrainer)

    started = time.time()
    result = query_router.QueryRouter().route("Needs background queue")
    elapsed = time.time() - started

    assert result["source"] == "web"
    assert elapsed < 0.5
