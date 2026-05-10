from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

from app.core.self_trainer import SelfTrainer


def _write_config(tmp_path: Path, *, threshold: int = 50) -> None:
    config = {
        "training": {"queue_db": str(tmp_path / "training_queue.sqlite3")},
        "modules": {
            "self_training": {
                "interval_seconds": 1800,
                "chunk_threshold": threshold,
                "base_model_path": str(tmp_path / "models" / "base"),
                "adapters_dir": str(tmp_path / "models" / "adapters"),
                "training_queue_dir": str(tmp_path / "training_queue"),
                "event_log_path": str(tmp_path / "logs" / "self_training.jsonl"),
                "alerts_log_path": str(tmp_path / "logs" / "alerts.jsonl"),
                "epochs": 3,
                "learning_rate": 0.0002,
                "loss_abort_threshold": 2.0,
            }
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_queue_saves_to_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    trainer = SelfTrainer()

    trainer.queue(
        query="Maurya Empire",
        chunks=[
            {
                "text": "Chandragupta Maurya founded the Maurya Empire.",
                "url": "https://example.org/maurya",
                "score": 0.92,
            }
        ],
    )

    with sqlite3.connect(trainer.db_path) as conn:
        row = conn.execute(
            "SELECT query, chunk_text, url, score, status FROM training_queue"
        ).fetchone()

    assert row == (
        "Maurya Empire",
        "Chandragupta Maurya founded the Maurya Empire.",
        "https://example.org/maurya",
        0.92,
        "pending",
    )


def test_check_and_train_writes_dataset_and_marks_trained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, threshold=2)
    trainer = SelfTrainer()
    trainer.queue(
        query="India history",
        chunks=[
            {
                "text": "The Maurya Empire was a major ancient Indian empire.",
                "url": "https://example.org/1",
                "score": 0.8,
            },
            {
                "text": "Ashoka promoted dhamma after the Kalinga war.",
                "url": "https://example.org/2",
                "score": 0.7,
            },
        ],
    )

    calls: dict[str, Path | str] = {}

    def fake_lora(dataset_path: Path, version: str) -> bool:
        calls["dataset_path"] = dataset_path
        calls["version"] = version
        return True

    monkeypatch.setattr(trainer, "_run_lora_update", fake_lora)

    trainer._check_and_train()

    dataset_path = calls["dataset_path"]
    assert isinstance(dataset_path, Path)
    records = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    assert records == [
        {
            "instruction": "What do you know about: India history",
            "response": "The Maurya Empire was a major ancient Indian empire.",
            "source_url": "https://example.org/1",
        },
        {
            "instruction": "What do you know about: India history",
            "response": "Ashoka promoted dhamma after the Kalinga war.",
            "source_url": "https://example.org/2",
        },
    ]

    with sqlite3.connect(trainer.db_path) as conn:
        statuses = [row[0] for row in conn.execute("SELECT status FROM training_queue ORDER BY id")]
    assert statuses == ["trained", "trained"]

    event = json.loads(trainer.event_log.read_text().splitlines()[0])
    assert event["chunks_trained"] == 2
    assert event["status"] == "success"


def test_check_and_train_skips_if_below_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, threshold=50)
    trainer = SelfTrainer()
    for index in range(10):
        trainer.queue(query=f"query-{index}", chunks=[{"text": f"chunk-{index}", "score": 0.5}])

    called = {"lora": False}

    def fake_lora(dataset_path: Path, version: str) -> bool:
        called["lora"] = True
        return True

    monkeypatch.setattr(trainer, "_run_lora_update", fake_lora)

    trainer._check_and_train()

    assert called["lora"] is False
    assert not trainer.training_dir.exists()
