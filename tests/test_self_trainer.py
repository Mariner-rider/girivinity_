from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.core.self_trainer import SelfTrainer


def _write_config(tmp_path: Path, threshold: int = 5) -> None:
    (tmp_path / "config.yaml").write_text(
        f"""
modules:
  self_training:
    interval_seconds: 60
    chunk_threshold: {threshold}
    base_model_path: "{tmp_path / 'models' / 'base'}"
    adapters_dir: "{tmp_path / 'models' / 'adapters'}"
    training_queue_dir: "{tmp_path / 'training_queue'}"
    event_log_path: "{tmp_path / 'logs' / 'events.jsonl'}"
    alerts_log_path: "{tmp_path / 'logs' / 'alerts.jsonl'}"
    epochs: 1
    learning_rate: 0.0001
    loss_abort_threshold: 2.0
training:
  queue_db: "{tmp_path / 'db' / 'queue.sqlite3'}"
"""
    )


def test_queue_writes_to_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    trainer = SelfTrainer()
    trainer.queue(query="test", chunks=[{"text": "a", "url": "b", "score": 0.5}])

    with sqlite3.connect(trainer.db_path) as conn:
        row = conn.execute("SELECT query, chunk_text, url, score, status FROM training_queue").fetchone()

    assert row == ("test", "a", "b", 0.5, "pending")


def test_below_threshold_skips_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, threshold=3)
    trainer = SelfTrainer()
    trainer.queue(query="test", chunks=[{"text": "a", "url": "b", "score": 0.5}])

    called = {"lora": False}

    def fake_run_lora_update(dataset_path, version):
        called["lora"] = True
        return True

    monkeypatch.setattr(trainer, "_run_lora_update", fake_run_lora_update)
    trainer._check_and_train()
    assert called["lora"] is False


def test_event_logged_on_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, threshold=5)
    trainer = SelfTrainer()

    for i in range(5):
        trainer.queue(query=f"q{i}", chunks=[{"text": f"chunk-{i}", "url": "u", "score": 0.5}])

    monkeypatch.setattr(trainer, "_run_lora_update", lambda dataset_path, version: True)

    trainer._check_and_train()

    assert trainer.event_log.exists()
    lines = trainer.event_log.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["status"] == "success"
    assert payload["chunks_trained"] == 5
