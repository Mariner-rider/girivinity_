import numpy as np

from app.memory.system import InMemoryNeo4j, InMemoryRedis, MemorySystem


class FakeEmbedder:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 997
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=(self.dim,)).astype("float32")
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return np.stack(vectors).astype("float32")


def test_store_and_retrieve_memory():
    memory = MemorySystem(embedding_dim=4, embedder=FakeEmbedder())
    memory_id = memory.store_memory("hello world", {"source": "unit"})

    stored = memory.retrieve_memory(memory_id)
    assert stored.text == "hello world"
    assert stored.metadata["source"] == "unit"


def test_retrieve_context_returns_results():
    memory = MemorySystem(embedding_dim=4, embedder=FakeEmbedder())
    memory.store_memory("apple pie")
    memory.store_memory("banana split")

    results = memory.retrieve_context("apple", top_k=1)
    assert len(results) == 1
    assert isinstance(results[0].text, str)


def test_graph_relationship_builder_persists_edges():
    graph = InMemoryNeo4j()
    memory = MemorySystem(embedding_dim=4, embedder=FakeEmbedder(), neo4j_client=graph)
    m1 = memory.store_memory("alice")
    m2 = memory.store_memory("bob")

    memory.graph_relationship_builder(m1, m2, "KNOWS")
    neighbors = graph.neighbors(m1)
    assert neighbors == [{"target": m2, "relation": "KNOWS"}]


def test_store_memory_updates_short_term_redis():
    redis = InMemoryRedis()
    memory = MemorySystem(embedding_dim=4, embedder=FakeEmbedder(), redis_client=redis)
    memory.store_memory("first")
    memory.store_memory("second")

    assert redis.lrange("memory:short_term", 0, -1) == ["1", "0"]
