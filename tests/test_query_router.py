from __future__ import annotations

import sys
import time
import types


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        assert model_name == "all-MiniLM-L6-v2"

    def encode(self, query: str):
        return types.SimpleNamespace(tolist=lambda: [0.1, 0.2, 0.3])


class FakeCollection:
    def __init__(self, result: dict) -> None:
        self.result = result

    def query(self, **kwargs):
        assert kwargs["n_results"] == 5
        return self.result


class FakePersistentClient:
    def __init__(self, *, path: str, result: dict) -> None:
        assert path == "data/chroma"
        self.result = result

    def get_or_create_collection(self, name: str):
        assert name == "girivinity_knowledge"
        return FakeCollection(self.result)


def _load_router(monkeypatch, result: dict):
    fake_chromadb = types.SimpleNamespace(
        PersistentClient=lambda path: FakePersistentClient(path=path, result=result)
    )
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    import importlib

    mod = importlib.import_module("app.core.query_router")
    mod = importlib.reload(mod)
    return mod


def test_kb_hit(monkeypatch):
    query_router = _load_router(
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
    query_router = _load_router(
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
                "answer_chunks": [{"text": "web result", "score": 0.6}],
                "raw_chunks": [],
                "sources": [],
            }

    fake_web_module = types.SimpleNamespace(WebIntelligence=FakeWebIntelligence)
    monkeypatch.setitem(sys.modules, "app.core.web_intelligence", fake_web_module)

    result = query_router.QueryRouter().route("Needs web")
    assert result["source"] == "web"


def test_no_block(monkeypatch):
    query_router = _load_router(
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
                "answer_chunks": [{"text": "web result", "score": 0.6}],
                "raw_chunks": [{"text": "raw"}],
                "sources": [],
            }

    class SlowSelfTrainer:
        def queue(self, query: str, chunks: list[dict]) -> None:
            time.sleep(1.2)

    monkeypatch.setitem(
        sys.modules,
        "app.core.web_intelligence",
        types.SimpleNamespace(WebIntelligence=FakeWebIntelligence),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.core.self_trainer",
        types.SimpleNamespace(SelfTrainer=SlowSelfTrainer),
    )

    started = time.time()
    query_router.QueryRouter().route("Needs background queue")
    elapsed = time.time() - started
    assert elapsed < 1.0
