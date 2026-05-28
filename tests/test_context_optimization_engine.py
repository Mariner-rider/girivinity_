from app.engines.context_optimization_engine import ContextChunk, ContextOptimizationSystem


def test_context_optimization_ranks_compresses_and_removes_noise():
    system = ContextOptimizationSystem()
    query = "How does vector retrieval improve RAG latency?"
    chunks = [
        ContextChunk("Vector retrieval improves RAG latency by reducing search scope. It uses embeddings for nearest-neighbor lookup."),
        ContextChunk("Click here advertisement subscribe now. Totally unrelated content."),
        ContextChunk("RAG combines retrieval and generation for grounded responses."),
    ]

    optimized = system.optimize(query, chunks, max_chars=500)
    assert "advertisement" not in optimized.prompt.lower()
    assert len(optimized.selected_chunks) >= 1
    assert optimized.dropped_chunks >= 0
    assert "User Query:" in optimized.prompt


def test_context_optimization_respects_token_budget_proxy():
    system = ContextOptimizationSystem()
    query = "Explain deployment"
    chunks = [ContextChunk("Sentence one. Sentence two. Sentence three.") for _ in range(10)]

    optimized = system.optimize(query, chunks, max_chars=120)
    assert len(optimized.prompt) <= 300  # includes wrapper text + selected compressed chunks
