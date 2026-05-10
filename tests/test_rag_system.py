import numpy as np

from app.rag.system import RAGSystem, RetrievedChunk


class FakeEmbedder:
    def encode(self, text: str) -> np.ndarray:
        return np.array([len(text), 1.0], dtype="float32")


class FakeSearcher:
    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        _ = query_vector
        return [
            RetrievedChunk("doc-1", "RAG retrieves relevant data.", 0.91, {"url": "https://a"}),
            RetrievedChunk("doc-2", "Scores indicate similarity confidence.", 0.82, {"url": "https://b"}),
        ][:top_k]


class EmptySearcher:
    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        _ = (query_vector, top_k)
        return []


class FakeGenerator:
    def __init__(self):
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        assert "Context:" in prompt
        return "- RAG combines retrieval with generation [source:doc-1]."


def test_rag_generate_includes_sources_citations_and_confidence():
    generator = FakeGenerator()
    rag = RAGSystem(embedder=FakeEmbedder(), searcher=FakeSearcher(), generator=generator)
    response = rag.generate("What is RAG?", top_k=2)

    assert "source:doc-1" in response.answer
    assert len(response.sources) == 2
    assert response.citations == ["source:doc-1", "source:doc-2"]
    assert response.confidence == 0.865
    assert "[source:doc-1]" in response.context


def test_rag_prompt_includes_user_level():
    generator = FakeGenerator()
    rag = RAGSystem(embedder=FakeEmbedder(), searcher=FakeSearcher(), generator=generator)
    rag.generate("Explain faiss quantization and retrieval", top_k=1)
    assert "User level:" in generator.last_prompt


def test_rag_returns_insufficient_information_when_no_data():
    generator = FakeGenerator()
    rag = RAGSystem(embedder=FakeEmbedder(), searcher=EmptySearcher(), generator=generator)
    response = rag.generate("Unknown query", top_k=3)

    assert response.answer == "insufficient information"
    assert response.sources == []
    assert response.citations == []
    assert response.confidence == 0.0
    assert response.context == ""
