from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

from core.successor_engine import (
    SuccessorConfig,
    SuccessorEngine,
    TrainingSummary,
    approve_successor,
    list_model_versions,
    read_notifications,
    reject_successor,
)


class FakeCollection:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents

    def count(self) -> int:
        return len(self.documents)

    def get(self, **kwargs):
        return {
            "ids": [f"doc-{index}" for index, _doc in enumerate(self.documents)],
            "documents": self.documents,
            "metadatas": [{"source": "unit-test"} for _doc in self.documents],
        }


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection

    def get_or_create_collection(self, name: str):
        assert name == "trained"
        return self.collection


class FakeTrainer:
    def __init__(self) -> None:
        self.calls = []

    def train_and_evaluate(
        self,
        corpus_path: Path,
        version: str,
        previous_version: str | None,
        trained_on_chunks: int,
    ) -> TrainingSummary:
        self.calls.append((corpus_path, version, previous_version, trained_on_chunks))
        return TrainingSummary(
            version=version,
            model_path="models/versions/" + version,
            perplexity=8.0,
            previous_version=previous_version,
            previous_perplexity=10.0,
            trained_on_chunks=trained_on_chunks,
        )


def _config(tmp_path, **overrides) -> SuccessorConfig:
    values = {
        "check_interval_seconds": 1,
        "chunk_threshold": 2,
        "quality_threshold": 3.5,
        "feedback_db_path": tmp_path / "feedback.sqlite3",
        "training_root": tmp_path / "successor_training",
        "versions_dir": tmp_path / "versions",
        "active_model_symlink": tmp_path / "active",
        "notifications_path": tmp_path / "admin_notifications.jsonl",
        "state_path": tmp_path / "successor_state.json",
        "train_seq_len": 8,
    }
    values.update(overrides)
    return SuccessorConfig(**values)


def test_successor_engine_exports_corpus_trains_and_notifies(tmp_path, monkeypatch):
    collection = FakeCollection(["alpha", "beta", "gamma"])
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        types.SimpleNamespace(Client=lambda: FakeClient(collection)),
    )
    trainer = FakeTrainer()
    config = _config(tmp_path)
    engine = SuccessorEngine(config=config, trainer=trainer)

    summary = engine.check_once()

    assert summary is not None
    assert trainer.calls[0][3] == 3
    corpus_path = trainer.calls[0][0]
    records = [json.loads(line) for line in corpus_path.read_text().splitlines()]
    assert [record["text"] for record in records] == ["alpha", "beta", "gamma"]
    notifications = read_notifications(config)
    assert notifications[0]["type"] == "successor_ready"
    assert notifications[0]["status"] == "awaiting_admin_approval"
    assert notifications[0]["trained_on_chunks"] == 3
    state = json.loads(config.state_path.read_text())
    assert state["last_candidate_chunk_count"] == 3
    assert "last_model_chunk_count" not in state


def test_successor_engine_triggers_on_low_feedback_even_below_chunk_threshold(tmp_path, monkeypatch):
    collection = FakeCollection(["alpha"])
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        types.SimpleNamespace(Client=lambda: FakeClient(collection)),
    )
    config = _config(tmp_path, chunk_threshold=99)
    with sqlite3.connect(config.feedback_db_path) as conn:
        conn.execute("CREATE TABLE user_feedback (score REAL, created_at TEXT)")
        conn.executemany(
            "INSERT INTO user_feedback (score, created_at) VALUES (?, ?)",
            [(2.0, "2026-01-02"), (3.0, "2026-01-01")],
        )
    engine = SuccessorEngine(config=config, trainer=FakeTrainer())

    assert engine.check_once() is not None


def test_admin_approve_reject_and_list_versions(tmp_path):
    config = _config(tmp_path)
    version_dir = config.versions_dir / "successor-v1"
    version_dir.mkdir(parents=True)
    (version_dir / "metrics.json").write_text('{"perplexity": 7.5}')
    config.notifications_path.write_text(
        json.dumps(
            {
                "version": "successor-v1",
                "status": "awaiting_admin_approval",
                "trained_on_chunks": 42,
            }
        )
        + "\n"
    )

    approved = approve_successor("successor-v1", config)

    assert approved["status"] == "approved"
    assert config.active_model_symlink.resolve() == version_dir.resolve()
    assert read_notifications(config)[0]["status"] == "approved"
    assert json.loads(config.state_path.read_text())["last_model_chunk_count"] == 42
    versions = list_model_versions(config)
    assert versions == [
        {
            "version": "successor-v1",
            "path": str(version_dir),
            "active": True,
            "metrics": {"perplexity": 7.5},
        }
    ]

    rejected = reject_successor("successor-v1", config)

    assert rejected == {"version": "successor-v1", "status": "rejected"}
    assert read_notifications(config)[0]["status"] == "rejected"
