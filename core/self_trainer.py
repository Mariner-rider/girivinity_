from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SelfTrainerConfig:
    queue_db: Path = Path("data/training_queue.sqlite3")
    training_queue_dir: Path = Path("data/training_queue")
    event_log_path: Path = Path("logs/self_training.jsonl")
    check_interval_minutes: float = 30.0
    min_batch_size: int = 50
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    max_length: int = 1024
    base_model_path: Path = Path("models/base")
    adapters_dir: Path = Path("models/adapters")
    latest_adapter_path: Path = Path("models/adapters/latest")
    pid_path: Path = Path(".self_trainer.pid")

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config.yaml") -> SelfTrainerConfig:
        path = Path(config_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        raw = raw or {}
        training = raw.get("training") or {}
        self_training = raw.get("self_training") or {}
        model = raw.get("model") or {}
        return cls(
            queue_db=Path(training.get("queue_db", self_training.get("queue_db", "data/training_queue.sqlite3"))),
            training_queue_dir=Path(
                training.get("training_queue_dir", self_training.get("training_queue_dir", "data/training_queue"))
            ),
            event_log_path=Path(
                training.get("event_log_path", self_training.get("event_log_path", "logs/self_training.jsonl"))
            ),
            check_interval_minutes=float(
                training.get(
                    "check_interval_minutes",
                    self_training.get("check_interval_minutes", self_training.get("interval_seconds", 1800) / 60),
                )
            ),
            min_batch_size=int(training.get("min_batch_size", self_training.get("chunk_threshold", 50))),
            epochs=int(training.get("epochs", self_training.get("epochs", 3))),
            learning_rate=float(training.get("learning_rate", self_training.get("learning_rate", 2e-4))),
            batch_size=int(training.get("batch_size", self_training.get("batch_size", 2))),
            max_length=int(training.get("max_length", self_training.get("max_length", 1024))),
            base_model_path=Path(model.get("base_model_path", self_training.get("base_model_path", "models/base"))),
            adapters_dir=Path(training.get("adapters_dir", self_training.get("adapters_dir", "models/adapters"))),
            latest_adapter_path=Path(
                training.get("latest_adapter_path", self_training.get("latest_adapter_path", "models/adapters/latest"))
            ),
            pid_path=Path(training.get("pid_path", self_training.get("pid_path", ".self_trainer.pid"))),
        )


# Backwards-compatible name used by older tests/callers.
SelfTrainingConfig = SelfTrainerConfig


class SelfTrainer:
    """Continuous self-learning queue and LoRA update daemon."""

    def __init__(
        self,
        *,
        config_path: str | Path = "config.yaml",
        config: SelfTrainerConfig | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = config or SelfTrainerConfig.from_yaml(config_path)
        self._ensure_schema()

    def queue(self, query: str, chunks: list[dict[str, Any]] | list[Any]) -> None:
        """Persist retrieved chunks to the SQLite training queue as pending rows."""
        if not chunks:
            return
        timestamp = datetime.utcnow().isoformat()
        rows: list[tuple[str, str, str, float, str, str]] = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk_text = str(chunk.get("text") or chunk.get("chunk") or "").strip()
                url = str(chunk.get("url") or "")
                score = float(chunk.get("score") or chunk.get("relevance_score") or 0.0)
            else:
                chunk_text = str(chunk).strip()
                url = ""
                score = 0.0
            if chunk_text:
                rows.append((query, chunk_text, url, score, timestamp, "pending"))
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO training_queue (query, chunk_text, url, score, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def run_daemon(self) -> None:
        """Run forever, periodically training on pending queued chunks."""
        while True:
            self._run_once()
            time.sleep(max(0.0, self.config.check_interval_minutes) * 60)

    @classmethod
    def start(cls, *, config_path: str | Path = "config.yaml") -> mp.Process:
        """Start the self-trainer daemon in a separate process and write its PID."""
        process = mp.Process(
            target=_run_daemon_entrypoint,
            kwargs={"config_path": str(config_path)},
            daemon=True,
            name="girivinity-self-trainer",
        )
        process.start()
        config = SelfTrainerConfig.from_yaml(config_path)
        if config.pid_path.parent != Path(""):
            config.pid_path.parent.mkdir(parents=True, exist_ok=True)
        config.pid_path.write_text(str(process.pid), encoding="utf-8")
        return process

    def _run_once(self) -> bool:
        pending_count = self._pending_count()
        if pending_count < self.config.min_batch_size:
            return False

        rows = self._pending_rows()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dataset_path = self._write_instruction_pairs(rows, timestamp)
        row_ids = [int(row["id"]) for row in rows]
        error: str | None = None
        success = False
        try:
            success = self._run_lora_update(str(dataset_path))
            if success:
                self._mark_trained(row_ids)
        except Exception as exc:
            error = str(exc)
            logger.exception("Self-training LoRA update failed: %s", exc)
            success = False

        self._log_event(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "chunks_trained": len(rows) if success else 0,
                "status": "trained" if success else "failed",
                "error_if_any": error,
            }
        )
        return success

    def _run_lora_update(self, dataset_path: str) -> bool:
        """Run a real LoRA update against the configured base model."""
        if not self.config.base_model_path.exists():
            logger.warning("base model not yet built")
            return False

        try:
            import torch
            from datasets import load_dataset
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        except Exception as exc:
            logger.exception("Self-training dependencies are unavailable: %s", exc)
            return False

        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = self.config.adapters_dir / version
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.config.base_model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                str(self.config.base_model_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            if self.config.latest_adapter_path.exists() or self.config.latest_adapter_path.is_symlink():
                model = PeftModel.from_pretrained(model, str(self.config.latest_adapter_path), is_trainable=True)
            else:
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_config)

            dataset = load_dataset("json", data_files=dataset_path, split="train")

            def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
                prompts = [
                    f"Instruction: {instruction}\nAnswer: {response}"
                    for instruction, response in zip(batch["instruction"], batch["response"])
                ]
                tokens = tokenizer(prompts, truncation=True, max_length=self.config.max_length)
                tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
                return tokens

            tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                num_train_epochs=self.config.epochs,
                logging_steps=10,
                save_strategy="epoch",
                fp16=torch.cuda.is_available(),
                bf16=False,
                report_to=[],
            )
            trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
            trainer.train()
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            self._promote_latest_adapter(output_dir)
            return True
        except Exception as exc:
            logger.exception("Self-training LoRA update failed: %s", exc)
            return False

    def _write_instruction_pairs(self, rows: list[sqlite3.Row], timestamp: str) -> Path:
        output_path = self.config.training_queue_dir / f"{timestamp}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            "instruction": f"What do you know about: {row['query']}",
                            "response": row["chunk_text"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return output_path

    def _ensure_schema(self) -> None:
        self.config.queue_db.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    url TEXT,
                    score REAL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.queue_db)
        conn.row_factory = sqlite3.Row
        return conn

    def _pending_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM training_queue WHERE status='pending'").fetchone()[0])

    def _pending_rows(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(conn.execute("SELECT * FROM training_queue WHERE status='pending' ORDER BY id"))

    def _mark_trained(self, row_ids: list[int]) -> None:
        if not row_ids:
            return
        placeholders = ",".join("?" for _ in row_ids)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE training_queue SET status='trained' WHERE id IN ({placeholders})",
                row_ids,
            )

    def _log_event(self, event: dict[str, Any]) -> None:
        self.config.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _promote_latest_adapter(self, output_dir: Path) -> None:
        latest_path = self.config.latest_adapter_path
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_link = latest_path.parent / f".{latest_path.name}.tmp"
        if temporary_link.exists() or temporary_link.is_symlink():
            temporary_link.unlink()
        os.symlink(output_dir.resolve(), temporary_link)
        if latest_path.exists() or latest_path.is_symlink():
            if latest_path.is_dir() and not latest_path.is_symlink():
                import shutil

                shutil.rmtree(latest_path)
            else:
                latest_path.unlink()
        os.replace(temporary_link, latest_path)


def _run_daemon_entrypoint(config_path: str) -> None:
    SelfTrainer(config_path=config_path).run_daemon()
