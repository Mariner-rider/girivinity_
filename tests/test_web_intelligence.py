from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from app.core import query_router, web_intelligence
from app.core.query_router import QueryRouter
from app.core.web_intelligence import WebIntelligence


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        assert model_name == "all-MiniLM-L6-v2"

    def encode(self, texts, convert_to_tensor=False, convert_to_numpy=False, normalize_embeddings=False):
        if isinstance(texts, str):
            return np.asarray([1.0, 0.0], dtype=np.float32)
        embeddings = []
        for index, _text in enumerate(texts):
            embeddings.append([1.0, 0.0] if index < 3 else [0.0, 1.0])
        return np.asarray(embeddings, dtype=np.float32)


class FakeUtil:
    @staticmethod
    def cos_sim(q_vec, c_vecs):
        q = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
        c = np.asarray(c_vecs, dtype=np.float32)
        return np.matmul(q, c.T)


class FakeDDGS:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query: str, max_results: int):
        assert query == "test query"
        assert max_results == 5
        return [
            {"href": "https://example.com/one", "title": "One", "body": "snippet"},
            {"href": "https://example.com/fail", "title": "Fail", "body": "bad"},
        ]


class FakeResponse:
    status_code = 200

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class FakeCollection:
    def __init__(self) -> None:
        self.upserted = None

    def upsert(self, **kwargs) -> None:
        self.upserted = kwargs


class FakeChromaClient:
    def __init__(self, path: str, collection: FakeCollection) -> None:
        assert path == "data/chroma"
        self.collection = collection

    def get_or_create_collection(self, name: str):
        assert name == "pending_training"
        return self.collection


def test_web_intelligence_returns_scored_chunks_and_upserts_training(monkeypatch):
    QueryRouter.reset_model_cache()
    monkeypatch.setattr(query_router, "SentenceTransformer", FakeSentenceTransformer)
    collection = FakeCollection()

    def fake_http_get(url: str, **kwargs):
        if url.endswith("fail"):
            raise RuntimeError("network down")
        assert kwargs["timeout"] == 8
        assert kwargs["follow_redirects"] is True
        return FakeResponse("<html>content</html>")

    def fake_extract(html: str, **kwargs):
        assert html == "<html>content</html>"
        assert kwargs["include_comments"] is False
        assert kwargs["include_tables"] is True
        return " ".join(f"token{i}" for i in range(700))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer, util=FakeUtil),
    )
    monkeypatch.setitem(sys.modules, "duckduckgo_search", types.SimpleNamespace(DDGS=FakeDDGS))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(get=fake_http_get))
    monkeypatch.setitem(sys.modules, "trafilatura", types.SimpleNamespace(extract=fake_extract))
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        types.SimpleNamespace(PersistentClient=lambda path: FakeChromaClient(path, collection)),
    )
    monkeypatch.setattr(web_intelligence, "DDGS", FakeDDGS)
    monkeypatch.setattr(web_intelligence, "httpx", types.SimpleNamespace(get=fake_http_get))
    monkeypatch.setattr(web_intelligence, "trafilatura", types.SimpleNamespace(extract=fake_extract))
    monkeypatch.setattr(
        web_intelligence,
        "chromadb",
        types.SimpleNamespace(PersistentClient=lambda path: FakeChromaClient(path, collection)),
    )
    monkeypatch.setattr(web_intelligence, "util", FakeUtil)

    result = WebIntelligence().search("test query")

    assert result["query"] == "test query"
    assert len(result["answer_chunks"]) == 3
    assert len(result["raw_chunks"]) == 3
    assert result["sources"] == [{"url": "https://example.com/one", "title": "One", "score": 1.0}]
    for _ in range(20):
        if collection.upserted is not None:
            break
        import time

        time.sleep(0.01)

    assert collection.upserted is not None
    assert collection.upserted["documents"] == [chunk["text"] for chunk in result["raw_chunks"]]
    assert collection.upserted["metadatas"][0]["url"] == "https://example.com/one"
    assert collection.upserted["metadatas"][0]["query"] == "test query"
    assert collection.upserted["metadatas"][0]["score"] == 1.0
    assert len(collection.upserted["ids"][0]) == 64


def test_web_intelligence_no_results_when_all_urls_fail(monkeypatch):
    QueryRouter.reset_model_cache()
    monkeypatch.setattr(query_router, "SentenceTransformer", FakeSentenceTransformer)

    class AllFailDDGS(FakeDDGS):
        def text(self, query: str, max_results: int):
            return [{"href": "https://example.com/fail", "title": "Fail", "body": "bad"}]

    monkeypatch.setitem(sys.modules, "duckduckgo_search", types.SimpleNamespace(DDGS=AllFailDDGS))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail"))))
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        types.SimpleNamespace(PersistentClient=lambda path: FakeChromaClient(path, FakeCollection())),
    )
    monkeypatch.setattr(web_intelligence, "DDGS", AllFailDDGS)
    monkeypatch.setattr(
        web_intelligence,
        "httpx",
        types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail"))),
    )
    monkeypatch.setattr(
        web_intelligence,
        "chromadb",
        types.SimpleNamespace(PersistentClient=lambda path: FakeChromaClient(path, FakeCollection())),
    )

    result = WebIntelligence().search("test query")

    assert result["answer_chunks"] == []
    assert result["raw_chunks"] == []
    assert result["sources"] == []
    assert result["error"] == "extraction_failed"
    assert result["query"] == "test query"


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("RUN_WEB_INTEGRATION"), reason="set RUN_WEB_INTEGRATION=1 to run live web integration test")
def test_live_search_india_history_maurya_empire():
    pytest.importorskip("duckduckgo_search")
    pytest.importorskip("httpx")
    pytest.importorskip("trafilatura")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("chromadb")

    result = WebIntelligence().search("India history Maurya Empire")

    assert result["answer_chunks"]
    assert all(source["url"].startswith("http") for source in result["sources"])
