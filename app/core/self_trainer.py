from __future__ import annotations

import json
import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path

import yaml

from app.core.db import get_conn

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
        """Write chunks to PostgreSQL queue. Fast — never blocks caller."""
        ts = datetime.utcnow().isoformat()
        with get_conn() as conn:
            with conn.cursor() as db:
                db.executemany(
                    "INSERT INTO training_queue "
                    "(query, chunk_text, url, score, timestamp, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
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
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute("SELECT COUNT(*) FROM training_queue WHERE status='pending'")
                count = db.fetchone()[0]

        if count < self.threshold:
            logger.info("SelfTrainer: %d pending < threshold %d, skipping", count, self.threshold)
            return

        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute("SELECT id, query, chunk_text, url FROM training_queue WHERE status='pending'")
                rows = db.fetchall()

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
            placeholders = ",".join(["%s"] * len(ids))
            with get_conn() as conn:
                with conn.cursor() as db:
                    db.execute(
                        f"UPDATE training_queue SET status='trained' WHERE id IN ({placeholders})",
                        ids,
                    )
            self._log_event(ts, len(rows), "success")

    def _run_lora_update(self, dataset_path: Path, version: str) -> bool:
        if not self.base_model.exists():
            logger.warning(
                "Base model not found at %s — skipping. "
                "Build it first with: python -m app.training.pretrain --config config.yaml",
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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            on_gpu = device.type == "cuda"
            if on_gpu:
                free_vram, _total_vram = torch.cuda.mem_get_info()
                if free_vram < 4 * 1024**3:
                    logger.warning(
                        "GPU has %.1fGB free VRAM (<4GB); falling back to CPU for LoRA training",
                        free_vram / 1e9,
                    )
                    device = torch.device("cpu")
                    on_gpu = False
            if on_gpu:
                logger.info("Training on GPU: %s", torch.cuda.get_device_name(0))
            else:
                logger.info("Training on CPU — this will be slow, consider a GPU instance")

            scaler = torch.cuda.amp.GradScaler(enabled=on_gpu)
            _ = scaler

            tokenizer = AutoTokenizer.from_pretrained(str(self.base_model))
            model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model),
                torch_dtype=torch.float16 if on_gpu else torch.float32,
                device_map=None,
            ).to(device)

            latest = self.adapters_dir / "latest"
            if latest.exists() and latest.is_symlink():
                model = PeftModel.from_pretrained(model, str(latest)).to(device)
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
                ).to(device)

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
                    fp16=on_gpu,
                    dataloader_num_workers=4 if on_gpu else 0,
                    no_cuda=not on_gpu,
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
        finally:
            if "on_gpu" in locals() and on_gpu:
                torch.cuda.empty_cache()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS training_queue (
                        id         SERIAL PRIMARY KEY,
                        query      TEXT    NOT NULL,
                        chunk_text TEXT    NOT NULL,
                        url        TEXT    DEFAULT '',
                        score      REAL    DEFAULT 0.0,
                        timestamp  TEXT    NOT NULL,
                        status     TEXT    DEFAULT 'pending'
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
            f.write(json.dumps({"timestamp": ts, "alert": "loss_abort", "loss": round(loss, 4)}) + "\n")
