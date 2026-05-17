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

    def _run_lora_update(
        self, dataset_path: Path, version: str
    ) -> bool:
        """
        Use the improved training pipeline instead of
        the basic LoRATrainer.
        """
        try:
            from model.training_pipeline import ImprovedLoRATrainer
            trainer = ImprovedLoRATrainer()
            return trainer.train_from_jsonl(str(dataset_path), version)
        except Exception as exc:
            logger.error("ImprovedLoRATrainer failed: %s", exc)
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
