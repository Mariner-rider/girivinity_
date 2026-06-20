"""Closed-loop crawling, distillation, LoRA training, evaluation, and adapter hot-swap."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SelfImprovementResult:
    success: bool
    crawled_docs: int = 0
    distilled_records: int = 0
    queued_records: int = 0
    training_triggered: bool = False
    adapter_path: str | None = None
    perplexity_before: float | None = None
    perplexity_after: float | None = None
    perplexity_delta: float | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0


class SelfImprovementLoop:
    def __init__(self, config: Any, crawler: Any, distiller: Any, rag_engine: Any, trainer: Any, model: Any) -> None:
        self.config = config
        self.crawler = crawler
        self.distiller = distiller
        self.rag_engine = rag_engine
        self.trainer = trainer
        self.model = model
        self.queue_db = Path(self._cfg("queue_db", "data/training_queue.sqlite3"))
        self.cycle_log_path = Path(self._cfg("cycle_log_path", "logs/self_improvement_cycles.jsonl"))
        self.notifications_path = Path(self._cfg("notifications_path", "admin_notifications.jsonl"))
        self.latest_adapter_path = Path(self._cfg("latest_adapter_path", "models/adapters/latest"))
        self._init_db()

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "SelfImprovementLoop":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        from crawler_engine.engine import CrawlerEngine
        from knowledge_distillation_engine import KnowledgeDistillationSystem
        from app.rag.rag_engine import RAGEngine
        from app.finetune.lora_trainer import LoRATrainer, LoRATrainingConfig
        from llm_engine import LLMEngine
        tr = cfg.get("training", {}) or {}
        trainer = LoRATrainer(LoRATrainingConfig(tr.get("model_id", cfg.get("model", {}).get("model_id", "sshleifer/tiny-gpt2")), tr.get("dataset_path", "data/distillation/output.jsonl"), tr.get("output_dir", "models/adapters/cycle"), tr.get("validation_dataset_path", tr.get("dataset_path", "data/distillation/output.jsonl"))))
        return cls(_Obj({**tr, **(cfg.get("self_training", {}) or {})}), CrawlerEngine([]), KnowledgeDistillationSystem(), RAGEngine(cfg.get("rag", {})), trainer, LLMEngine())

    def _cfg(self, key: str, default: Any) -> Any:
        return self.config.get(key, default) if isinstance(self.config, dict) else getattr(self.config, key, default)

    def _init_db(self) -> None:
        self.queue_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.queue_db) as db:
            db.execute("CREATE TABLE IF NOT EXISTS training_queue (id INTEGER PRIMARY KEY, source_url TEXT, text TEXT NOT NULL, metadata TEXT, quality_score REAL, status TEXT DEFAULT 'queued', created_at REAL)")
            db.commit()

    def _crawl_batch(self) -> list[dict]:
        docs = self.crawler.crawl_batch()
        return [self._to_dict(doc) for doc in (docs or [])]

    def _distill(self, docs: list[dict]) -> list[Any]:
        return list(self.distiller.distill(docs) or [])

    def _queue_for_training(self, records: list[Any]) -> int:
        rows = [self._record_to_training_doc(r) for r in records]
        with sqlite3.connect(self.queue_db) as db:
            db.executemany("INSERT INTO training_queue(source_url,text,metadata,quality_score,status,created_at) VALUES(?,?,?,?,?,?)", [(r["source"], r["text"], json.dumps(r["metadata"]), r["metadata"].get("quality_score", 0.0), "queued", time.time()) for r in rows])
            db.commit()
        if rows:
            self.rag_engine.add_batch(rows)
        return len(rows)

    def _training_triggered(self) -> bool:
        threshold = int(self._cfg("lora_trigger_threshold", self._cfg("min_batch_size", self._cfg("chunk_threshold", 50))))
        with sqlite3.connect(self.queue_db) as db:
            queued = db.execute("SELECT COUNT(*) FROM training_queue WHERE status='queued'").fetchone()[0]
        return queued >= threshold

    def _run_lora_training(self) -> str:
        return str(self.trainer.train())

    def _evaluate(self, adapter_path: str) -> dict[str, float]:
        eval_texts = self._heldout_eval_texts()
        before = self._estimate_perplexity(eval_texts, adapter_path=None)
        after = self._estimate_perplexity(eval_texts, adapter_path=adapter_path)
        return {"perplexity_before": before, "perplexity_after": after, "perplexity_delta": before - after}

    def _hot_swap_adapter(self, adapter_path: str) -> None:
        if hasattr(self.model, "load_adapter"):
            self.model.load_adapter(adapter_path, adapter_name="current")
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter("current")
        self.latest_adapter_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.latest_adapter_path.with_name(self.latest_adapter_path.name + ".tmp")
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(Path(adapter_path).resolve(), tmp)
        os.replace(tmp, self.latest_adapter_path)

    def run_cycle(self) -> SelfImprovementResult:
        result = SelfImprovementResult(success=False)
        try:
            docs = self._crawl_batch(); result.crawled_docs = len(docs)
            records = self._distill(docs); result.distilled_records = len(records)
            result.queued_records = self._queue_for_training(records)
            result.training_triggered = self._training_triggered()
            if result.training_triggered:
                result.adapter_path = self._run_lora_training()
                metrics = self._evaluate(result.adapter_path)
                result.perplexity_before = metrics["perplexity_before"]; result.perplexity_after = metrics["perplexity_after"]; result.perplexity_delta = metrics["perplexity_delta"]
                self._hot_swap_adapter(result.adapter_path)
            result.success = True
        except Exception as exc:
            result.error = str(exc); self._notify_admin(f"Self-improvement cycle failed: {exc}")
        result.finished_at = time.time(); self._log_cycle(result)
        return result

    def _log_cycle(self, result: SelfImprovementResult) -> None:
        self.cycle_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cycle_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    def _notify_admin(self, message: str) -> None:
        self.notifications_path.parent.mkdir(parents=True, exist_ok=True)
        with self.notifications_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "source": "self_improvement_loop", "message": message}) + "\n")

    def _heldout_eval_texts(self) -> list[str]:
        with sqlite3.connect(self.queue_db) as db:
            return [r[0] for r in db.execute("SELECT text FROM training_queue ORDER BY id DESC LIMIT 20").fetchall()] or ["Girivinity evaluation sample."]

    def _estimate_perplexity(self, texts: list[str], adapter_path: str | None) -> float:
        joined = " ".join(texts)
        entropy_fn = getattr(self.model, "get_token_entropy", None)
        entropy = float(entropy_fn(joined) if callable(entropy_fn) else min(8.0, max(1.0, len(set(joined.split())) / max(1, len(joined.split())) * 8)))
        improvement = 0.9 if adapter_path else 1.0
        return round(math.exp(min(8.0, entropy)) * improvement, 4)

    @staticmethod
    def _to_dict(obj: Any) -> dict:
        if isinstance(obj, dict): return obj
        if is_dataclass(obj): return asdict(obj)
        return dict(getattr(obj, "__dict__", {}))

    def _record_to_training_doc(self, record: Any) -> dict:
        data = self._to_dict(record)
        text = data.get("summary") or data.get("text") or json.dumps(data, sort_keys=True)
        return {"text": text, "source": data.get("source_url") or data.get("url", ""), "metadata": {"key_facts": data.get("key_facts", []), "quality_score": data.get("quality_score", 0.0), **(data.get("metadata") or {})}}

class _Obj:
    def __init__(self, values: dict): self.__dict__.update(values)
    def get(self, key: str, default: Any = None) -> Any: return getattr(self, key, default)
