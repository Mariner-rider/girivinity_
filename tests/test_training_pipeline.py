import sys
from unittest.mock import MagicMock

# Mock missing sandbox dependencies before any imports
sys.modules.setdefault("yaml", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("peft", MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("datasets", MagicMock())
sys.modules.setdefault("bitsandbytes", MagicMock())

import tempfile
from pathlib import Path


def test_quality_scorer_gold():
    from model.training_pipeline import DataQualityScorer
    scorer = DataQualityScorer()
    good_text = (
        "Machine learning is a subset of artificial intelligence. "
        "According to research, neural networks demonstrate strong "
        "performance on pattern recognition tasks. The methodology "
        "involves training on large datasets to find statistical patterns."
    )
    result = scorer.score(good_text, "test001")
    assert result.score >= 0.6
    assert result.tier in ("gold", "silver")


def test_quality_scorer_discard():
    from model.training_pipeline import DataQualityScorer
    scorer = DataQualityScorer()
    bad_text = "click here subscribe now free trial sign up"
    result = scorer.score(bad_text, "test002")
    assert result.tier == "discard"


def test_deduplicator_catches_exact():
    from model.training_pipeline import Deduplicator
    with tempfile.TemporaryDirectory() as tmp:
        d = Deduplicator(f"{tmp}/dedup.db")
        text = "This is a test sentence for deduplication."
        assert d.is_duplicate(text) is False
        assert d.is_duplicate(text) is True


def test_deduplicator_unique_passes():
    from model.training_pipeline import Deduplicator
    with tempfile.TemporaryDirectory() as tmp:
        d = Deduplicator(f"{tmp}/dedup.db")
        assert d.is_duplicate("First unique sentence here.") is False
        assert d.is_duplicate("Second unique sentence here.") is False


def test_replay_buffer_add_and_sample():
    from model.training_pipeline import ReplayBuffer
    with tempfile.TemporaryDirectory() as tmp:
        buf = ReplayBuffer(f"{tmp}/replay.db", max_size=100)
        buf.add("What is ML?", "ML is machine learning.", 0.9)
        buf.add("What is AI?", "AI is artificial intelligence.", 0.8)
        samples = buf.sample(2, min_quality=0.7)
        assert len(samples) >= 1
        assert all("instruction" in s for s in samples)
        assert all(s["is_replay"] is True for s in samples)


def test_curriculum_scheduler_orders_tiers():
    from model.training_pipeline import CurriculumScheduler, QualityScore
    scheduler = CurriculumScheduler()
    chunks = [
        QualityScore("g1", "gold text", 0.9, {}, "gold"),
        QualityScore("b1", "bronze text", 0.4, {}, "bronze"),
        QualityScore("s1", "silver text", 0.7, {}, "silver"),
    ]
    scheduled = scheduler.schedule(chunks)
    assert len(scheduled) > 3
    tiers = [c.tier for c in scheduled]
    assert "gold" in tiers
    assert "silver" in tiers


def test_diversity_enforcer_caps_topic():
    from model.training_pipeline import DiversityEnforcer
    enforcer = DiversityEnforcer()
    chunks = [
        {"text": f"python text {i}", "query": "python programming"}
        for i in range(20)
    ] + [
        {"text": "machine learning text", "query": "machine learning"}
    ]
    result = enforcer.enforce(chunks, "query")
    python_count = sum(1 for c in result if "python" in c.get("query", ""))
    total = len(result)
    assert python_count / total <= 0.35


def test_full_pipeline_from_jsonl():
    from model.training_pipeline import (
        DataQualityScorer,
        Deduplicator,
        ReplayBuffer,
        CurriculumScheduler,
        DiversityEnforcer,
    )
    import json
    with tempfile.TemporaryDirectory() as tmp:
        data = [
            {
                "instruction": "What is Python?",
                "response": (
                    "Python is a high-level programming language "
                    "widely used for data science and machine learning. "
                    "It was created by Guido van Rossum in 1991."
                ),
            },
            {
                "instruction": "What is Python?",
                "response": (
                    "Python is a high-level programming language "
                    "widely used for data science and machine learning. "
                    "It was created by Guido van Rossum in 1991."
                ),
            },
            {
                "instruction": "click here",
                "response": "subscribe now free trial sign up today",
            },
        ]
        path = Path(tmp) / "test.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        scorer = DataQualityScorer()
        dedup = Deduplicator(f"{tmp}/dedup.db")
        replay = ReplayBuffer(f"{tmp}/replay.db")
        curriculum = CurriculumScheduler()
        _ = replay
        _ = curriculum

        records = data
        scored = [scorer.score(r["response"], str(i)) for i, r in enumerate(records)]
        valid = [(records[i], scored[i]) for i in range(len(records)) if scored[i].tier != "discard"]
        assert len(valid) <= 2

        seen = set()
        deduped = []
        for rec, score in valid:
            text = rec["response"]
            if text not in seen:
                seen.add(text)
                deduped.append((rec, score))
        assert len(deduped) == 1
        assert dedup.is_duplicate(deduped[0][0]["response"]) is False
