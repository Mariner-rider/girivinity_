from __future__ import annotations

import importlib
import sys
import time
import types


class FakeCollection:
    def upsert(self, **kwargs) -> None:
        return None


class SlowCollection:
    def upsert(self, **kwargs) -> None:
        time.sleep(1.2)


class FakeChromaClient:
    def __init__(self, path: str, collection) -> None:
        assert path == "data/chroma"
        self.collection = collection

    def get_or_create_collection(self, name: str):
        assert name == "pending_training"
        return self.collection


class FakeEmbedder:
    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            return [1.0, 0.0]
        return [[1.0, 0.0] for _ in texts]


class FakeCos:
    def __init__(self, n: int = 1) -> None:
        self.n = n

    def __getitem__(self, idx):
        return self

    def tolist(self):
        return [0.9] * self.n


def _load_module(monkeypatch, collection):
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        types.SimpleNamespace(PersistentClient=lambda path: FakeChromaClient(path, collection)),
    )
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=lambda *_: {"rag": {"chroma_path": "data/chroma"}}))
    monkeypatch.setitem(sys.modules, "trafilatura", types.SimpleNamespace(extract=lambda *_args, **_kwargs: "x" * 4000))
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(
            SentenceTransformer=lambda *a, **k: types.SimpleNamespace(
                encode=lambda *a, **k: [0.1] * 384
            ),
            util=types.SimpleNamespace(cos_sim=lambda q, c: FakeCos(len(c))),
        ),
    )

    import app.core.query_router as query_router

    monkeypatch.setattr(query_router, "get_embedder", lambda: FakeEmbedder())

    module = importlib.import_module("app.core.web_intelligence")
    return importlib.reload(module)


def test_search_returns_chunks(monkeypatch):
    module = _load_module(monkeypatch, FakeCollection())

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query: str, max_results: int):
            return [
                {"href": "https://example.com/1", "title": "One"},
                {"href": "https://example.com/2", "title": "Two"},
            ]

    monkeypatch.setitem(sys.modules, "duckduckgo_search", types.SimpleNamespace(DDGS=FakeDDGS))
    monkeypatch.setattr(module.httpx, "get", lambda *_args, **_kwargs: types.SimpleNamespace(status_code=200, text="<html/>"))
    monkeypatch.setattr(module.trafilatura, "extract", lambda *_args, **_kwargs: "x" * 4000)

    result = module.WebIntelligence().search("query")
    assert isinstance(result["answer_chunks"], list)


def test_all_fetches_fail(monkeypatch):
    module = _load_module(monkeypatch, FakeCollection())

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query: str, max_results: int):
            return [{"href": "https://example.com/1", "title": "One"}]

    monkeypatch.setitem(sys.modules, "duckduckgo_search", types.SimpleNamespace(DDGS=FakeDDGS))

    def _raise(*_args, **_kwargs):
        raise TimeoutError("timeout")

    monkeypatch.setattr(module.httpx, "get", _raise)

    result = module.WebIntelligence().search("query")
    assert "error" in result


def test_store_is_async(monkeypatch):
    module = _load_module(monkeypatch, SlowCollection())

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query: str, max_results: int):
            return [{"href": "https://example.com/1", "title": "One"}]

    monkeypatch.setitem(sys.modules, "duckduckgo_search", types.SimpleNamespace(DDGS=FakeDDGS))
    monkeypatch.setattr(module.httpx, "get", lambda *_args, **_kwargs: types.SimpleNamespace(status_code=200, text="<html/>"))
    monkeypatch.setattr(module.trafilatura, "extract", lambda *_args, **_kwargs: "x" * 4000)

    start = time.time()
    module.WebIntelligence().search("query")
    assert time.time() - start < 1.0
