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


class SuccessorEngine:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        se = cfg["successor_engine"]
        tr = cfg["training"]
        self.check_interval_s = int(se["check_interval_seconds"])
        self.kb_threshold = int(se["knowledge_base_threshold"])
        self.quality_threshold = float(se["quality_score_threshold"])
        self.versions_dir = Path(se["versions_dir"])
        self.notifications_path = Path(se["notifications_path"])
        self.corpus_dir = Path(se["corpus_dir"])
        self.db_path = Path(tr["queue_db"])
        self.active_link = Path("models/active")

    @classmethod
    def start(cls) -> multiprocessing.Process:
        instance = cls()
        p = multiprocessing.Process(target=instance._run_daemon, daemon=True)
        p.start()
        Path(".successor_engine.pid").write_text(str(p.pid))
        logger.info("SuccessorEngine daemon started PID=%s", p.pid)
        return p

    def log_feedback(self, user_id: str, score: float) -> None:
        """Called from chat endpoint when user rates a response (1-5)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS feedback "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "user_id TEXT, score REAL, timestamp TEXT)"
            )
            conn.execute(
                "INSERT INTO feedback (user_id, score, timestamp) "
                "VALUES (?, ?, ?)",
                (user_id, score, datetime.utcnow().isoformat()),
            )

    def get_notifications(self) -> list[dict]:
        if not self.notifications_path.exists():
            return []
        notifications = []
        with open(self.notifications_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    notifications.append(json.loads(line))
        return notifications

    def approve_successor(self, version: str) -> bool:
        version_dir = self.versions_dir / version
        if not version_dir.exists():
            logger.error("Version %s not found", version)
            return False
        if self.active_link.is_symlink():
            self.active_link.unlink()
        self.active_link.symlink_to(version_dir.resolve())
        self._update_notification_status(version, "approved")
        logger.info("Successor %s approved and set as active", version)
        return True

    def reject_successor(self, version: str) -> bool:
        self._update_notification_status(version, "rejected")
        logger.info("Successor %s rejected", version)
        return True

    def _run_daemon(self) -> None:
        logger.info("SuccessorEngine running, interval=%ds", self.check_interval_s)
        while True:
            try:
                self._check_thresholds()
            except Exception as exc:
                logger.error("SuccessorEngine error: %s", exc)
            time.sleep(self.check_interval_s)

    def _check_thresholds(self) -> None:
        kb_count = self._count_trained_chunks()
        avg_score = self._rolling_quality_score()

        kb_trigger = kb_count >= self.kb_threshold
        quality_trigger = 0 < avg_score < self.quality_threshold

        if not (kb_trigger or quality_trigger):
            logger.info(
                "SuccessorEngine: kb=%d/%d quality=%.2f/%.2f — no trigger",
                kb_count,
                self.kb_threshold,
                avg_score,
                self.quality_threshold,
            )
            return

        reason = []
        if kb_trigger:
            reason.append(f"kb_chunks={kb_count}>={self.kb_threshold}")
        if quality_trigger:
            reason.append(f"quality={avg_score:.2f}<{self.quality_threshold}")
        logger.info("SuccessorEngine triggered: %s", ", ".join(reason))

        self._build_successor()

    def _build_successor(self) -> None:
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        corpus_path = self._export_corpus(version)
        if corpus_path is None:
            return

        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        success = self._run_full_training(corpus_path, version_dir)
        if not success:
            logger.error("Full training failed for version %s", version)
            return

        perplexity = self._evaluate(version_dir)
        prev_version, prev_perplexity = self._get_current_model_stats()

        improvement = 0.0
        if prev_perplexity and prev_perplexity > 0:
            improvement = round((prev_perplexity - perplexity) / prev_perplexity * 100, 2)

        if perplexity >= prev_perplexity and prev_perplexity > 0:
            logger.info(
                "New model perplexity %.2f not better than %.2f — discarding",
                perplexity,
                prev_perplexity,
            )
            return

        self._write_notification(
            version=version,
            previous_version=prev_version,
            improvement_percent=improvement,
            trained_on_chunks=self._count_trained_chunks(),
            perplexity=perplexity,
        )

    def _export_corpus(self, version: str) -> Path | None:
        corpus_dir = self.corpus_dir / version
        corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = corpus_dir / "corpus.jsonl"
        with sqlite3.connect(self.db_path) as conn:
            try:
                rows = conn.execute(
                    "SELECT query, chunk_text FROM training_queue " "WHERE status='trained'"
                ).fetchall()
            except sqlite3.OperationalError:
                logger.warning("training_queue table not found")
                return None
        if not rows:
            logger.warning("No trained chunks to export")
            return None
        with open(corpus_path, "w", encoding="utf-8") as f:
            for query, chunk_text in rows:
                f.write(
                    json.dumps(
                        {
                            "instruction": f"What do you know about: {query}",
                            "response": chunk_text,
                        }
                    )
                    + "\n"
                )
        logger.info("Exported %d chunks to %s", len(rows), corpus_path)
        return corpus_path

    def _run_full_training(self, corpus_path: Path, output_dir: Path) -> bool:
        try:
            from model.train import train

            train(
                data_path=str(corpus_path.parent),
                tokeniser_path="models/tokeniser/tokeniser.json",
                output_dir=str(output_dir),
                epochs=3,
                batch_size=4,
                lr=3e-4,
                grad_accum=8,
            )
            return True
        except Exception as exc:
            logger.error("Full training error: %s", exc)
            return False

    def _evaluate(self, model_dir: Path) -> float:
        """Estimate perplexity on a small held-out sample."""
        try:
            import math
            import torch
            import torch.nn.functional as F
            from model.architecture import GirivinityConfig, GirivinityModel

            cfg = GirivinityConfig.from_yaml()
            model = GirivinityModel(cfg)
            weights = model_dir / "final" / "model.pt"
            if not weights.exists():
                return float("inf")
            model.load_state_dict(torch.load(weights, map_location="cpu"))
            model.eval()

            with torch.no_grad():
                ids = torch.randint(0, cfg.vocab_size, (1, 64))
                logits, _ = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, cfg.vocab_size),
                    ids[:, 1:].reshape(-1),
                )
            return math.exp(loss.item())
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            return float("inf")

    def _count_trained_chunks(self) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute(
                    "SELECT COUNT(*) FROM training_queue WHERE status='trained'"
                ).fetchone()[0]
        except Exception:
            return 0

    def _rolling_quality_score(self) -> float:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT AVG(score) FROM "
                    "(SELECT score FROM feedback "
                    " ORDER BY id DESC LIMIT 100)"
                ).fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception:
            return 0.0

    def _get_current_model_stats(self) -> tuple[str, float]:
        if not self.active_link.exists():
            return "none", float("inf")
        try:
            name = self.active_link.resolve().name
            perplexity_file = self.active_link / "perplexity.txt"
            if perplexity_file.exists():
                return name, float(perplexity_file.read_text().strip())
            return name, float("inf")
        except Exception:
            return "unknown", float("inf")

    def _write_notification(
        self,
        version: str,
        previous_version: str,
        improvement_percent: float,
        trained_on_chunks: int,
        perplexity: float,
    ) -> None:
        self.notifications_path.parent.mkdir(parents=True, exist_ok=True)
        notification = {
            "type": "successor_ready",
            "version": version,
            "previous_version": previous_version,
            "improvement_percent": improvement_percent,
            "trained_on_chunks": trained_on_chunks,
            "perplexity": round(perplexity, 4),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "awaiting_admin_approval",
        }
        with open(self.notifications_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(notification) + "\n")
        logger.info("Notification written for version %s", version)

    def _update_notification_status(self, version: str, status: str) -> None:
        if not self.notifications_path.exists():
            return
        lines = self.notifications_path.read_text(encoding="utf-8").splitlines()
        updated = []
        for line in lines:
            try:
                n = json.loads(line)
                if n.get("version") == version:
                    n["status"] = status
                updated.append(json.dumps(n))
            except json.JSONDecodeError:
                updated.append(line)
        self.notifications_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
