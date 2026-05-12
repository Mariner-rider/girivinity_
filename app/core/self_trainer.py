from __future__ import annotations

import json
import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path

import yaml
from app.core import db

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

    def queue(self, query: str, chunks: list[dict]) -> None:
        db.executemany(
            """
            INSERT INTO training_queue
                (query, chunk_text, url, score, status)
            VALUES (%s, %s, %s, %s, 'pending')
            """,
            [
                (query, c.get("text", ""), c.get("url", ""), c.get("score", 0.0))
                for c in chunks
            ],
        )

    @classmethod
    def start(cls) -> multiprocessing.Process:
        """Launch the daemon process. Call once at FastAPI startup."""
        instance = cls()
        p = multiprocessing.Process(target=instance._run_daemon, daemon=True)
        p.start()
        Path(".self_trainer.pid").write_text(str(p.pid))
        logger.info("SelfTrainer daemon started PID=%s", p.pid)
        return p

    def _run_daemon(self) -> None:
        logger.info("SelfTrainer running, interval=%ds", self.interval_s)
        while True:
            try:
                self._check_and_train()
            except Exception as exc:
                logger.error("SelfTrainer error: %s", exc)
            time.sleep(self.interval_s)

    def _check_and_train(self) -> None:
        row = db.fetchone("SELECT COUNT(*) FROM training_queue WHERE status='pending'")
        count = row[0] if row else 0

        if count < self.threshold:
            logger.info(
                "SelfTrainer: %d pending < threshold %d, skipping",
                count, self.threshold,
            )
            return

        rows = db.fetchall("SELECT id, query, chunk_text, url FROM training_queue WHERE status='pending'")

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self.training_dir / f"{ts}.jsonl"

        with open(dataset_path, "w", encoding="utf-8") as f:
            for row in rows:
                _, query, chunk_text, url = row
                f.write(json.dumps({
                    "instruction": f"What do you know about: {query}",
                    "response": chunk_text,
                    "source_url": url,
                }) + "\n")

        success = self._run_lora_update(dataset_path, ts)

        if success:
            ids = [r[0] for r in rows]
            placeholders = ",".join(["%s"] * len(ids))
            db.execute(f"UPDATE training_queue SET status='trained' WHERE id IN ({placeholders})", tuple(ids))
            self._log_event(ts, len(rows), "success")

    def _run_lora_update(self, dataset_path: Path, version: str) -> bool:
        if not self.base_model.exists():
            logger.warning(
                "Base model not found at %s — skipping. "
                "Build it first with: python model/architecture.py",
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
            model = AutoModelForCausalLM.from_pretrained(str(self.base_model), torch_dtype=torch.float32)

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
                    records.append({"text": f"### Instruction:\n{r['instruction']}\n\n" f"### Response:\n{r['response']}"})

            def tokenize(ex: dict) -> dict:
                return tokenizer(ex["text"], truncation=True, max_length=1024, padding="max_length")

            tokenized = Dataset.from_list(records).map(tokenize, batched=True, remove_columns=["text"])

            adapter_out = self.adapters_dir / version
            adapter_out.mkdir(parents=True, exist_ok=True)

            result = Trainer(
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
            ).train()

            if result.training_loss > self.loss_abort:
                self._log_alert(version, result.training_loss)
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

    def _init_db(self) -> None:
        pass  # Tables created by migrations.py at startup

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
            f.write(json.dumps({"timestamp": ts, "alert": "loss_abort", "loss": round(loss, 4)}) + "\n")
