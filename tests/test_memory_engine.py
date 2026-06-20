import sys
from unittest.mock import MagicMock, patch

# Mock chromadb at sys.modules level BEFORE any import
# so module-level "import chromadb" in memory_engine.py succeeds
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.PersistentClient", MagicMock())

from app.core.memory_engine import MemoryEngine, MemoryNode


def _make_engine() -> MemoryEngine:
    engine = MemoryEngine.__new__(MemoryEngine)
    engine.max_memories_per_user = 1000
    engine.recall_top_k = 5
    engine.importance_threshold = 0.3
    engine.collection = MagicMock()
    return engine


def test_make_id_is_deterministic():
    engine = _make_engine()
    id1 = engine._make_id("user1", "test content")
    id2 = engine._make_id("user1", "test content")
    assert id1 == id2
    assert len(id1) == 24


def test_make_id_differs_for_different_users():
    engine = _make_engine()
    id1 = engine._make_id("user1", "test content")
    id2 = engine._make_id("user2", "test content")
    assert id1 != id2


def test_extract_facts_returns_sentences():
    engine = _make_engine()
    facts = engine._extract_facts(
        "Machine learning is powerful. "
        "It uses data to train models. "
        "Neural networks are a subset of deep learning."
    )
    assert len(facts) >= 1
    assert all(isinstance(f, str) for f in facts)


def test_extract_facts_filters_short():
    engine = _make_engine()
    facts = engine._extract_facts("Short. Also short. " + "x" * 200)
    for f in facts:
        assert len(f.strip()) > 20


def test_score_importance_range():
    engine = _make_engine()
    score = engine._score_importance(
        "machine learning uses data", "machine learning"
    )
    assert 0.0 <= score <= 1.0


def test_score_importance_higher_on_overlap():
    engine = _make_engine()
    high = engine._score_importance(
        "machine learning algorithm", "machine learning"
    )
    low = engine._score_importance(
        "cooking recipes for dinner", "machine learning"
    )
    assert high > low


def test_extract_main_topic_returns_string():
    engine = _make_engine()
    topic = engine._extract_main_topic("explain machine learning clearly")
    assert topic is None or isinstance(topic, str)


def test_detect_preferences_on_explicit_preference():
    engine = _make_engine()
    prefs = engine._detect_preferences(
        "I always prefer detailed explanations with examples"
    )
    assert isinstance(prefs, list)


def test_build_memory_context_empty():
    engine = _make_engine()
    result = engine.build_memory_context([])
    assert result == ""


def test_build_memory_context_with_nodes():
    from datetime import datetime, timezone
    engine = _make_engine()
    now = datetime.now(timezone.utc).isoformat()
    nodes = [
        MemoryNode(
            node_id="abc",
            user_id="user1",
            content="User is a developer",
            node_type="fact",
            importance=0.8,
            created_at=now,
            last_accessed=now,
        )
    ]
    result = engine.build_memory_context(nodes)
    assert "User is a developer" in result
    assert "[Relevant memories" in result
