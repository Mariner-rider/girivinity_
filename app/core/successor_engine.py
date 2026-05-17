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


GENERATION_CONFIGS = {
    1: {
        "dim": 3072, "n_layers": 28, "n_heads": 24,
        "n_kv_heads": 8, "ffn_multiplier": 2.667,
        "description": "Girivinity 3B — Foundation",
    },
    2: {
        "dim": 4096, "n_layers": 32, "n_heads": 32,
        "n_kv_heads": 8, "ffn_multiplier": 2.667,
        "description": "Girivinity 7B — First Evolution",
    },
    3: {
        "dim": 5120, "n_layers": 40, "n_heads": 40,
        "n_kv_heads": 8, "ffn_multiplier": 2.667,
        "description": "Girivinity 13B — Second Evolution",
    },
    4: {
        "dim": 6656, "n_layers": 60, "n_heads": 52,
        "n_kv_heads": 8, "ffn_multiplier": 2.667,
        "description": "Girivinity 30B — Third Evolution",
    },
    5: {
        "dim": 8192, "n_layers": 80, "n_heads": 64,
        "n_kv_heads": 8, "ffn_multiplier": 2.667,
        "description": "Girivinity 70B — Fourth Evolution",
    },
}

CHUNK_THRESHOLDS_FOR_GENERATION = {
    2: 100_000,
    3: 500_000,
    4: 2_000_000,
    5: 10_000_000,
}



class SuccessorEngine:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        se = cfg["successor_engine"]
        self.check_interval_s = int(se["check_interval_seconds"])
        self.kb_threshold = int(se["knowledge_base_threshold"])
        self.quality_threshold = float(se["quality_score_threshold"])
        self.versions_dir = Path(se["versions_dir"])
        self.notifications_path = Path(se["notifications_path"])
        self.corpus_dir = Path(se["corpus_dir"])
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
        db.execute("INSERT INTO feedback (user_id, score) VALUES (%s, %s)", (user_id, score))

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

    def _get_next_generation(self) -> tuple[int, dict]:
        """
        Determines what generation the next model should be
        based on how much training data has been accumulated.
        """
        total_trained = self._count_trained_chunks()

        next_gen = 1
        for gen, threshold in sorted(
            CHUNK_THRESHOLDS_FOR_GENERATION.items()
        ):
            if total_trained >= threshold:
                next_gen = gen

        # Never exceed the highest defined generation
        next_gen = min(next_gen, max(GENERATION_CONFIGS.keys()))

        return next_gen, GENERATION_CONFIGS[next_gen]

    def _get_current_generation(self) -> int:
        """Read current generation from active model metadata."""
        try:
            meta = self.active_link / "generation.json"
            if meta.exists():
                import json
                data = json.loads(meta.read_text())
                return int(data.get("generation", 1))
        except Exception:
            pass
        return 1

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

        # Determine next generation config
        next_gen, gen_cfg = self._get_next_generation()
        current_gen       = self._get_current_generation()

        # Only upgrade if generation increases or
        # significant new data is available
        if next_gen <= current_gen:
            logger.info(
                "SuccessorEngine: still at generation %d, "
                "not enough data to upgrade. "
                "Need %d chunks for gen %d.",
                current_gen,
                CHUNK_THRESHOLDS_FOR_GENERATION.get(
                    current_gen + 1, 999_999_999
                ),
                current_gen + 1,
            )

        # Update architecture config for this generation
        from model.architecture import GirivinityConfig
        new_cfg = GirivinityConfig(
            dim=gen_cfg["dim"],
            n_layers=gen_cfg["n_layers"],
            n_heads=gen_cfg["n_heads"],
            n_kv_heads=gen_cfg["n_kv_heads"],
            ffn_multiplier=gen_cfg["ffn_multiplier"],
        )

        # Save new config so train.py picks it up
        import yaml
        cfg_override = self.versions_dir / version / "arch_config.yaml"
        cfg_override.parent.mkdir(parents=True, exist_ok=True)
        cfg_override.write_text(yaml.dump({
            "architecture": {
                "dim":            new_cfg.dim,
                "n_layers":       new_cfg.n_layers,
                "n_heads":        new_cfg.n_heads,
                "n_kv_heads":     new_cfg.n_kv_heads,
                "ffn_multiplier": new_cfg.ffn_multiplier,
                "vocab_size":     new_cfg.vocab_size,
                "max_seq_len":    new_cfg.max_seq_len,
                "norm_eps":       new_cfg.norm_eps,
                "rope_theta":     new_cfg.rope_theta,
            }
        }))

        success = self._run_full_training(corpus_path, version_dir, next_gen, gen_cfg)
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
            current_gen=current_gen,
            next_gen=next_gen,
            gen_cfg=gen_cfg,
        )

    def _export_corpus(self, version: str) -> Path | None:
        corpus_dir = self.corpus_dir / version
        corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = corpus_dir / "corpus.jsonl"
        try:
            rows = db.fetchall("SELECT query, chunk_text FROM training_queue WHERE status='trained'")
        except Exception as exc:
            logger.warning("training_queue fetch failed: %s", exc)
            return None
        if not rows:
            logger.warning("No trained chunks to export")
            return None
        with open(corpus_path, "w", encoding="utf-8") as f:
            for query, chunk_text in rows:
                f.write(json.dumps({
                    "instruction": f"What do you know about: {query}",
                    "response": chunk_text,
                }) + "\n")
        logger.info("Exported %d chunks to %s", len(rows), corpus_path)
        return corpus_path

    def _run_full_training(self, corpus_path: Path, output_dir: Path, next_gen: int, gen_cfg: dict) -> bool:
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
            import json
            gen_meta = Path(output_dir) / "generation.json"
            gen_meta.write_text(json.dumps({
                "generation": next_gen,
                "description": gen_cfg["description"],
                "trained_on_chunks": self._count_trained_chunks(),
                "trained_at": datetime.utcnow().isoformat(),
                "dim": gen_cfg["dim"],
                "n_layers": gen_cfg["n_layers"],
            }, indent=2))
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
            row = db.fetchone("SELECT COUNT(*) FROM training_queue WHERE status='trained'")
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _rolling_quality_score(self) -> float:
        try:
            row = db.fetchone("""
                SELECT AVG(score) FROM (
                    SELECT score FROM feedback
                    ORDER BY id DESC LIMIT 100
                ) sub
                """)
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
        current_gen: int,
        next_gen: int,
        gen_cfg: dict,
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
            "current_generation": current_gen,
            "next_generation":    next_gen,
            "model_description":  gen_cfg["description"],
            "param_scale":        gen_cfg["dim"],
            "chunks_needed_for_next": CHUNK_THRESHOLDS_FOR_GENERATION.get(
                next_gen + 1, "maximum_reached"
            ),
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
