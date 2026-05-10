from __future__ import annotations

import sys
import types

from core import truth_engine
from core.truth_engine import TruthEngine


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        assert model_name == "all-MiniLM-L6-v2"

    def encode(self, text: str, convert_to_numpy: bool = True, normalize_embeddings: bool = True):
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        return [0.1, 0.2, 0.3]


class FakeCollection:
    def __init__(self, result):
        self.result = result

    def query(self, **kwargs):
        assert kwargs["n_results"] == 3
        return self.result


class FakeClient:
    def __init__(self, result):
        self.result = result

    def get_or_create_collection(self, name: str):
        assert name == "knowledge_base"
        return FakeCollection(self.result)


def _mock_chroma(monkeypatch, result):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(sys.modules, "chromadb", types.SimpleNamespace(Client=lambda: FakeClient(result)))


def test_truth_engine_cites_only_real_kb_urls(monkeypatch):
    _mock_chroma(
        monkeypatch,
        {
            "documents": [["Paris is documented as France's capital."]],
            "metadatas": [[{"url": "https://example.org/paris", "timestamp": "2026-05-01T00:00:00+00:00"}]],
            "distances": [[0.1]],
        },
    )

    result = TruthEngine().verify_response("Paris is the capital of France.")

    assert "Paris is the capital of France. [1]" in result.response
    assert "Sources:\n[1] https://example.org/paris" in result.response
    assert result.sources == [
        {
            "citation_number": 1,
            "url": "https://example.org/paris",
            "title": "",
            "score": 0.9,
            "timestamp": "2026-05-01T00:00:00+00:00",
        }
    ]
    assert result.unverified_claims == []
    assert result.confidence > 0.8


def test_truth_engine_marks_unverified_and_triggers_web_search(monkeypatch):
    _mock_chroma(
        monkeypatch,
        {"documents": [[]], "metadatas": [[]], "distances": [[]]},
    )

    class FakeWebSearchPipeline:
        def __init__(self, query: str) -> None:
            self.query = query

        def run(self):
            return {
                "sources": [
                    {"url": "https://retrieved.example/source", "title": "Retrieved", "score": 0.83}
                ]
            }

    monkeypatch.setattr(truth_engine, "WebSearchPipeline", FakeWebSearchPipeline)

    result = TruthEngine().verify_response("The moon is made of cheese. Grass is blue.")

    assert result.triggered_web_search is True
    assert result.response.startswith("I found limited verified information on this.")
    assert "[UNVERIFIED] The moon is made of cheese." in result.response
    assert "[UNVERIFIED] Grass is blue." in result.response
    assert "https://retrieved.example/source" not in result.response
    assert result.sources == []
    assert result.confidence < 0.5


def test_truth_engine_formats_current_session_web_citations_without_fabrication(monkeypatch):
    _mock_chroma(
        monkeypatch,
        {"documents": [[]], "metadatas": [[]], "distances": [[]]},
    )
    session_sources = [{"url": "https://news.example/a", "title": "News A", "score": 0.7}]

    result = TruthEngine().verify_response(
        "A current event happened today.",
        source="web",
        session_sources=session_sources,
    )

    assert "A current event happened today. [1]" in result.response
    assert "Sources:\n[1] https://news.example/a" in result.response
    assert "[WEB_SOURCED]" not in result.response
    assert result.sources[0]["url"] == "https://news.example/a"
    assert result.unverified_claims == []


def test_truth_engine_refuses_fake_web_citation_when_no_session_url(monkeypatch):
    _mock_chroma(
        monkeypatch,
        {"documents": [[]], "metadatas": [[]], "distances": [[]]},
    )
    monkeypatch.setattr(
        truth_engine,
        "WebSearchPipeline",
        lambda query: types.SimpleNamespace(run=lambda: {"sources": []}),
    )

    result = TruthEngine().verify_response("A current event happened today.", source="web")

    assert "[UNVERIFIED] A current event happened today." in result.response
    assert "Sources:" not in result.response
    assert result.unverified_claims == ["A current event happened today."]
