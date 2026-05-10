from __future__ import annotations

import sys
import types

from core import query_router


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        assert model_name == "all-MiniLM-L6-v2"

    def encode(self, text: str, convert_to_numpy: bool = True):
        return [0.1, 0.2, 0.3]


class FakeCollection:
    def __init__(self, result):
        self._result = result

    def query(self, **kwargs):
        return self._result


class FakeClient:
    def __init__(self, result):
        self._result = result

    def get_or_create_collection(self, name: str):
        assert name == "knowledge_base"
        return FakeCollection(self._result)


def _mock_modules(monkeypatch, chroma_result):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        types.SimpleNamespace(Client=lambda: FakeClient(chroma_result)),
    )


def test_route_returns_knowledge_base_when_threshold_hit(monkeypatch):
    _mock_modules(
        monkeypatch,
        chroma_result={"documents": [["chunk-a", "chunk-b"]], "distances": [[0.1, 0.2]]},
    )

    router = query_router.QueryRouter(threshold=0.75)
    result = router.route("What is AI?")

    assert result["source"] == "knowledge_base"
    assert result["chunks"] == ["chunk-a", "chunk-b"]
    assert "trigger_training" not in result


def test_route_returns_web_and_triggers_background_training(monkeypatch):
    _mock_modules(monkeypatch, chroma_result={"documents": [[]], "distances": [[]]})

    queued = {"called": False, "chunks": None}

    class FakeWebSearchPipeline:
        def __init__(self, query: str) -> None:
            self.query = query

        def fetch(self):
            return ["web-chunk-1", "web-chunk-2"]

    class FakeSelfTrainer:
        def __init__(self, chunks):
            self.chunks = chunks

        def queue_for_training(self):
            queued["called"] = True
            queued["chunks"] = self.chunks

    monkeypatch.setattr(query_router, "WebSearchPipeline", FakeWebSearchPipeline)
    monkeypatch.setattr(query_router, "SelfTrainer", FakeSelfTrainer)

    class ImmediateThread:
        def __init__(self, target, daemon, name):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(query_router.threading, "Thread", ImmediateThread)

    router = query_router.QueryRouter(threshold=0.75)
    result = router.route("Latest tech news")

    assert result["source"] == "web"
    assert result["chunks"] == ["web-chunk-1", "web-chunk-2"]
    assert result["trigger_training"] is True
    assert queued["called"] is True
    assert queued["chunks"] == ["web-chunk-1", "web-chunk-2"]
