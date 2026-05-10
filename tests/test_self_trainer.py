from __future__ import annotations

import json
import sqlite3

from core.self_trainer import SelfTrainer, SelfTrainerConfig


def _config(tmp_path, *, min_batch_size: int = 50) -> SelfTrainerConfig:
    return SelfTrainerConfig(
        queue_db=tmp_path / "training_queue.sqlite3",
        training_queue_dir=tmp_path / "training_queue",
        event_log_path=tmp_path / "logs" / "self_training.jsonl",
        check_interval_minutes=30,
        min_batch_size=min_batch_size,
        epochs=3,
        base_model_path=tmp_path / "models" / "base",
        adapters_dir=tmp_path / "models" / "adapters",
        latest_adapter_path=tmp_path / "models" / "adapters" / "latest",
        pid_path=tmp_path / ".self_trainer.pid",
    )


def test_queue_saves_to_db(tmp_path):
    config = _config(tmp_path)
    trainer = SelfTrainer(config=config)

    trainer.queue(
        query="Maurya Empire",
        chunks=[{"text": "Chandragupta Maurya founded the Maurya Empire.", "url": "https://example.org/maurya", "score": 0.92}],
    )

    with sqlite3.connect(config.queue_db) as conn:
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


def test_format_instruction_pairs(tmp_path):
    config = _config(tmp_path)
    trainer = SelfTrainer(config=config)
    trainer.queue(
        query="India history",
        chunks=[
            {"text": "The Maurya Empire was a major ancient Indian empire.", "url": "https://example.org/1", "score": 0.8},
            {"text": "Ashoka promoted dhamma after the Kalinga war.", "url": "https://example.org/2", "score": 0.7},
        ],
    )

    rows = trainer._pending_rows()
    dataset_path = trainer._write_instruction_pairs(rows, "20260510_120000")

    records = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    assert records == [
        {
            "instruction": "What do you know about: India history",
            "response": "The Maurya Empire was a major ancient Indian empire.",
        },
        {
            "instruction": "What do you know about: India history",
            "response": "Ashoka promoted dhamma after the Kalinga war.",
        },
    ]


def test_run_daemon_skips_if_below_threshold(tmp_path, monkeypatch):
    config = _config(tmp_path, min_batch_size=50)
    trainer = SelfTrainer(config=config)
    for index in range(10):
        trainer.queue(query=f"query-{index}", chunks=[{"text": f"chunk-{index}", "score": 0.5}])

    called = {"lora": False}

    def fake_lora(dataset_path: str) -> bool:
        called["lora"] = True
        return True

    monkeypatch.setattr(trainer, "_run_lora_update", fake_lora)

    assert trainer._run_once() is False
    assert called["lora"] is False
