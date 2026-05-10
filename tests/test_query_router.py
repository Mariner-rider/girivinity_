from __future__ import annotations

import sys
import time
import types

from core import query_router


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        assert model_name == "all-MiniLM-L6-v2"

    def encode(self, text: str, convert_to_numpy: bool = True, normalize_embeddings: bool = True):
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        return [0.1, 0.2, 0.3]


class FakeCollection:
    def __init__(self, result):
        self._result = result

    def query(self, **kwargs):
        assert kwargs["n_results"] == 5
        return self._result


class FakeClient:
    def __init__(self, result):
        self._result = result

    def get_or_create_collection(self, name: str):
        assert name == "girivinity_knowledge"
        return FakeCollection(self._result)


def _config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text('rag:\n  chroma_path: null\n')
    return config_path


def _mock_modules(monkeypatch, chroma_result):
    query_router.QueryRouter.reset_model_cache()
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


def test_kb_hit(monkeypatch, tmp_path):
    _mock_modules(
        monkeypatch,
        {
            "documents": [["chunk-a", "chunk-b"]],
            "metadatas": [[{"url": "https://kb.example/a"}, {"url": "https://kb.example/b"}]],
            "distances": [[0.2, 0.25]],
        },
    )

    result = query_router.QueryRouter(config_path=_config(tmp_path)).route("What is AI?")

    assert result["source"] == "kb"
    assert result["chunks"] == ["chunk-a", "chunk-b"]
    assert result["trigger_web"] is False
    assert result["confidence"] == 0.8
    assert result["urls"] == ["https://kb.example/a", "https://kb.example/b"]
    assert "Context:\n[1] chunk-a\n[2] chunk-b" in result["context_string"]


def test_kb_miss(monkeypatch, tmp_path):
    _mock_modules(
        monkeypatch,
        {
            "documents": [["weak-kb-chunk"]],
            "metadatas": [[{"url": "https://kb.example/weak"}]],
            "distances": [[0.7]],
        },
    )

    class FakeWebIntelligence:
        def __init__(self, query: str) -> None:
            self.query = query

        def search(self):
            return {
                "answer_chunks": ["web-chunk-1", "web-chunk-2"],
                "raw_chunks": [{"text": "raw-web-chunk", "url": "https://web.example/a", "score": 0.91}],
                "sources": [{"url": "https://web.example/a", "score": 0.91}],
            }

    class FakeSelfTrainer:
        def queue(self, query: str, chunks: list):
            return None

    import core.web_intelligence as web_intelligence
    import core.self_trainer as self_trainer

    monkeypatch.setattr(web_intelligence, "WebIntelligence", FakeWebIntelligence)
    monkeypatch.setattr(self_trainer, "SelfTrainer", FakeSelfTrainer)

    result = query_router.QueryRouter(config_path=_config(tmp_path)).route("Latest AI news")

    assert result["source"] == "web"
    assert result["chunks"] == ["web-chunk-1", "web-chunk-2"]
    assert result["trigger_web"] is True
    assert result["raw_for_training"] == [
        {"text": "raw-web-chunk", "url": "https://web.example/a", "score": 0.91}
    ]
    assert result["urls"] == ["https://web.example/a"]


def test_thread_does_not_block(monkeypatch, tmp_path):
    _mock_modules(
        monkeypatch,
        {"documents": [["weak"]], "metadatas": [[{}]], "distances": [[0.7]]},
    )

    class FakeWebIntelligence:
        def __init__(self, query: str) -> None:
            self.query = query

        def search(self):
            return {
                "answer_chunks": ["web-chunk"],
                "raw_chunks": ["raw-web-chunk"],
                "sources": [{"url": "https://web.example/a", "score": 0.8}],
            }

    class SlowSelfTrainer:
        def queue(self, query: str, chunks: list):
            time.sleep(0.25)

    import core.web_intelligence as web_intelligence
    import core.self_trainer as self_trainer

    monkeypatch.setattr(web_intelligence, "WebIntelligence", FakeWebIntelligence)
    monkeypatch.setattr(self_trainer, "SelfTrainer", SlowSelfTrainer)

    started = time.perf_counter()
    result = query_router.QueryRouter(config_path=_config(tmp_path)).route("Needs web")
    elapsed = time.perf_counter() - started

    assert result["source"] == "web"
    assert elapsed < 0.1
