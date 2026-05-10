from __future__ import annotations

import json
import logging
import multiprocessing
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class SelfTrainer:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        st = cfg["modules"]["self_training"]
        tr = cfg["training"]
        self.db_path = Path(tr["queue_db"])
        self.interval_s = int(st["interval_seconds"])
        self.threshold = int(st["chunk_threshold"])
        self.base_model = Path(st["base_model_path"])
        self.adapters_dir = Path(st["adapters_dir"])
        self.training_dir = Path(st["training_queue_dir"])
        self.event_log = Path(st["event_log_path"])
        self.alerts_log = Path(st["alerts_log_path"])
        self.epochs = int(st["epochs"])
        self.lr = float(st["learning_rate"])
        self.loss_abort = float(st["loss_abort_threshold"])
        self._init_db()

    # ── Public API ──────────────────────────────────────────────────────────

    def queue(self, query: str, chunks: list[dict]) -> None:
        """Fast write to SQLite — called from background thread, must not block."""
        ts = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO training_queue "
                "(query,chunk_text,url,score,timestamp,status) VALUES (?,?,?,?,?,?)",
                [
                    (
                        query,
                        c.get("text", ""),
                        c.get("url", ""),
                        c.get("score", 0.0),
                        ts,
                        "pending",
                    )
                    for c in chunks
                ],
            )

    @classmethod
    def start(cls) -> multiprocessing.Process:
        """Launch daemon process. Call once at FastAPI startup."""
        instance = cls()
        p = multiprocessing.Process(target=instance._run_daemon, daemon=True)
        p.start()
        Path(".self_trainer.pid").write_text(str(p.pid))
        logger.info("SelfTrainer daemon started, PID=%s", p.pid)
        return p

    # ── Daemon loop ──────────────────────────────────────────────────────────

    def _run_daemon(self) -> None:
        logger.info("SelfTrainer daemon running, interval=%ds", self.interval_s)
        while True:
            try:
                self._check_and_train()
            except Exception as exc:
                logger.error("SelfTrainer daemon error: %s", exc)
            time.sleep(self.interval_s)

    def _check_and_train(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM training_queue WHERE status='pending'"
            ).fetchone()[0]

        if count < self.threshold:
            logger.info(
                "SelfTrainer: %d pending < threshold %d, skipping",
                count,
                self.threshold,
            )
            return

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id,query,chunk_text,url FROM training_queue WHERE status='pending'"
            ).fetchall()

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self.training_dir / f"{ts}.jsonl"

        with open(dataset_path, "w", encoding="utf-8") as f:
            for _, query, chunk_text, url in rows:
                f.write(
                    json.dumps(
                        {
                            "instruction": f"What do you know about: {query}",
                            "response": chunk_text,
                            "source_url": url,
                        }
                    )
                    + "\n"
                )

        success = self._run_lora_update(dataset_path, ts)

        if success:
            ids = [r[0] for r in rows]
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"UPDATE training_queue SET status='trained' "
                    f"WHERE id IN ({','.join('?' * len(ids))})",
                    ids,
                )
            self._log_event(ts, len(rows), "success")
            logger.info("SelfTrainer: trained %d chunks, adapter=%s", len(rows), ts)

    # ── LoRA update ──────────────────────────────────────────────────────────

    def _run_lora_update(self, dataset_path: Path, version: str) -> bool:
        if not self.base_model.exists():
            logger.warning(
                "Base model not found at %s — skipping LoRA update. "
                "Build the base model first (python model/architecture.py).",
                self.base_model,
            )
            return False

        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )

            tokenizer = AutoTokenizer.from_pretrained(str(self.base_model))
            model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model), torch_dtype=torch.float32
            )

            latest = self.adapters_dir / "latest"
            if latest.exists() and latest.is_symlink():
                model = PeftModel.from_pretrained(model, str(latest))
            else:
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=0.05,
                        task_type=TaskType.CAUSAL_LM,
                    ),
                )

            records = []
            with open(dataset_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    records.append(
                        {
                            "text": f"### Instruction:\n{r['instruction']}\n\n"
                            f"### Response:\n{r['response']}"
                        }
                    )

            def tokenize(ex):
                return tokenizer(ex["text"], truncation=True, max_length=1024, padding="max_length")

            tokenized = Dataset.from_list(records).map(
                tokenize, batched=True, remove_columns=["text"]
            )

            adapter_out = self.adapters_dir / version
            adapter_out.mkdir(parents=True, exist_ok=True)

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(adapter_out),
                    num_train_epochs=self.epochs,
                    per_device_train_batch_size=2,
                    learning_rate=self.lr,
                    logging_steps=10,
                    save_strategy="no",
                    report_to="none",
                ),
                train_dataset=tokenized,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )

            result = trainer.train()
            loss = result.training_loss

            if loss > self.loss_abort:
                self._log_alert(version, loss)
                logger.error(
                    "LoRA aborted — loss %.4f exceeds threshold %.4f",
                    loss,
                    self.loss_abort,
                )
                return False

            model.save_pretrained(str(adapter_out))

            latest_link = self.adapters_dir / "latest"
            if latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(adapter_out.resolve())

            return True

        except Exception as exc:
            logger.error("LoRA update failed: %s", exc)
            return False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_queue (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    query     TEXT    NOT NULL,
                    chunk_text TEXT   NOT NULL,
                    url       TEXT    DEFAULT '',
                    score     REAL    DEFAULT 0.0,
                    timestamp TEXT    NOT NULL,
                    status    TEXT    DEFAULT 'pending'
                )
            """
            )

    def _log_event(self, ts: str, count: int, status: str) -> None:
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.event_log, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": ts,
                        "chunks_trained": count,
                        "adapter_version": ts,
                        "status": status,
                    }
                )
                + "\n"
            )

    def _log_alert(self, ts: str, loss: float) -> None:
        self.alerts_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.alerts_log, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": ts,
                        "alert": "loss_abort",
                        "loss": round(loss, 4),
                    }
                )
                + "\n"
            )
