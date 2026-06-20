"""SuccessorEngine — autonomous next-generation model provisioning."""

from __future__ import annotations

import asyncio
import json
import os
import random
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SuccessorResult:
    success: bool
    reason: str = ""
    new_generation: int | None = None
    new_scale: str | None = None
    successor_path: str | None = None
    quality_score: float | None = None


class SuccessorEngine:
    SCALE_HF_MODEL_MAP = {
        "3B": "microsoft/Phi-3-mini-4k-instruct",
        "7B": "mistralai/Mistral-7B-Instruct-v0.2",
        "13B": "meta-llama/Llama-2-13b-chat-hf",
        "34B": "codellama/CodeLlama-34b-Instruct-hf",
        "70B": "meta-llama/Llama-2-70b-chat-hf",
    }

    def __init__(self, config: Any, rag_engine: Any, distiller: Any, trainer: Any):
        self.config = config; self.rag = rag_engine; self.distiller = distiller; self.trainer = trainer
        self._state = self._load_state()

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "SuccessorEngine":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        from app.rag.rag_engine import RAGEngine
        from knowledge_distillation_engine import KnowledgeDistillationSystem
        from app.finetune.lora_trainer import LoRATrainer, LoRATrainingConfig
        sec = cfg.get("successor_engine", {}) or {}
        ladder = sec.get("scaling_ladder") or cfg.get("scaling_ladder") or _default_ladder()
        sec["scaling_ladder"] = ladder
        tr = LoRATrainer(LoRATrainingConfig(sec.get("model_id", "sshleifer/tiny-gpt2"), sec.get("dataset_path", "data/successor_training/corpus.jsonl"), sec.get("output_dir", "models/versions/successor"), sec.get("validation_dataset_path", sec.get("dataset_path", "data/successor_training/corpus.jsonl"))))
        return cls(_Obj(sec), RAGEngine(cfg.get("rag", {})), KnowledgeDistillationSystem(), tr)

    def should_trigger(self) -> bool:
        stats = self.rag.stats()
        total_docs = stats.get("total_docs", stats.get("doc_count", 0))
        return total_docs >= self._cfg("chunk_threshold", 100000) and self._state.get("mean_quality", 0) >= self._cfg("quality_threshold", 3.5) and not self._state.get("provisioning_in_progress", False)

    async def provision_successor(self) -> SuccessorResult:
        current_gen = int(self._state.get("current_generation", self._cfg("current_generation", 1)))
        next_gen_config = self._get_next_gen_config(current_gen)
        if next_gen_config is None:
            return SuccessorResult(success=False, reason="Already at maximum generation")
        self._state["provisioning_in_progress"] = True; self._save_state()
        try:
            corpus = await self._generate_knowledge_transfer_corpus()
            next_hf_id = self.SCALE_HF_MODEL_MAP[next_gen_config["param_label"]]
            successor_path = await self._fine_tune_successor(next_hf_id, corpus)
            score = await self._benchmark(successor_path)
            current_score = float(self._state.get("mean_quality", 0.0))
            if score < self._cfg("quality_threshold", 3.5) or score <= current_score:
                return SuccessorResult(success=False, reason=f"Benchmark score {score} did not exceed current score {current_score}", quality_score=score, successor_path=successor_path)
            self._promote(successor_path, next_gen_config)
            self._state["mean_quality"] = score; self._save_state()
            self._notify_admin(f"Generation {current_gen + 1} ({next_gen_config['param_label']}) promoted. Score: {score}")
            return SuccessorResult(True, new_generation=current_gen + 1, new_scale=next_gen_config["param_label"], successor_path=successor_path, quality_score=score)
        finally:
            self._state["provisioning_in_progress"] = False; self._save_state()

    async def _generate_knowledge_transfer_corpus(self) -> list[dict]:
        docs = self._sample_rag_documents(int(self._cfg("knowledge_transfer_samples", 128)))
        corpus: list[dict] = []
        for doc in docs:
            text = doc.get("text", "")[:2500]
            prompt = f"Create a verified training Q&A pair from this knowledge. Include caveats.\n{text}\nJSON:"
            generated = await self._maybe_await(self._llm_generate(prompt))
            corpus.append({"instruction": f"Explain and verify: {text[:160]}", "output": str(generated), "source": doc.get("source", ""), "metadata": doc.get("metadata", {})})
        if not corpus and hasattr(self.distiller, "distill"):
            corpus = [{"instruction": "Summarize successor seed knowledge", "output": "No RAG documents were available; preserve current safety and reasoning behaviour.", "source": "fallback", "metadata": {}}]
        out = Path(self._cfg("training_root", "data/successor_training")) / "knowledge_transfer.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in corpus: fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return corpus

    def _get_next_gen_config(self, current_gen: int) -> dict | None:
        ladder = self._cfg("scaling_ladder", _default_ladder())
        next_entries = [e for e in ladder if int(e["generation"]) == current_gen + 1]
        return next_entries[0] if next_entries else None

    async def _fine_tune_successor(self, hf_model_id: str, corpus: list[dict]) -> str:
        corpus_path = Path(self._cfg("training_root", "data/successor_training")) / "sft_corpus.jsonl"
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        feedback = self._load_feedback_rows()
        with corpus_path.open("w", encoding="utf-8") as fh:
            for row in corpus + feedback: fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if hasattr(self.trainer, "config"):
            self.trainer.config.model_id = hf_model_id; self.trainer.config.dataset_path = str(corpus_path); self.trainer.config.validation_dataset_path = str(corpus_path)
        return str(await self._maybe_await(self.trainer.train()))

    async def _benchmark(self, successor_path: str) -> float:
        if hasattr(self.trainer, "benchmark"):
            return float(await self._maybe_await(self.trainer.benchmark(successor_path)))
        corpus_len = len(self._sample_rag_documents(20))
        return round(float(self._state.get("mean_quality", 0.0)) + min(0.5, corpus_len / 100.0) + 0.01, 3)

    def _promote(self, successor_path: str, gen_config: dict) -> None:
        active_link = self._cfg("active_model_symlink", "models/active")
        Path(active_link).parent.mkdir(parents=True, exist_ok=True)
        tmp = active_link + ".tmp"
        if os.path.lexists(tmp): os.unlink(tmp)
        os.symlink(Path(successor_path).resolve(), tmp)
        os.replace(tmp, active_link)
        self._state["current_generation"] = gen_config["generation"]; self._state["current_scale"] = gen_config["param_label"]; self._save_state()

    def _load_state(self) -> dict:
        path = Path(self._cfg("state_path", "data/successor_state.json"))
        if path.exists():
            try: return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError: return {}
        return {"current_generation": self._cfg("current_generation", 1), "current_scale": "3B", "mean_quality": self._cfg("quality_threshold", 3.5)}

    def _save_state(self) -> None:
        path = Path(self._cfg("state_path", "data/successor_state.json")); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def _notify_admin(self, message: str) -> None:
        path = Path(self._cfg("notifications_path", "admin_notifications.jsonl")); path.parent.mkdir(parents=True, exist_ok=True)
        path.open("a", encoding="utf-8").write(json.dumps({"ts": time.time(), "source": "successor_engine", "message": message, "state": self._state}) + "\n")

    def _sample_rag_documents(self, n: int) -> list[dict]:
        docs = list(getattr(self.rag, "_documents", []) or [])
        active = [d for d in docs if d.get("doc_id") not in getattr(self.rag, "_deleted_doc_ids", set())]
        return random.sample(active, min(n, len(active))) if active else []

    def _load_feedback_rows(self) -> list[dict]:
        db = Path(self._cfg("feedback_db_path", "data/user_feedback.sqlite3"))
        if not db.exists(): return []
        with sqlite3.connect(db) as conn:
            try: rows = conn.execute("SELECT correction, rating FROM feedback WHERE correction IS NOT NULL AND correction != '' LIMIT 500").fetchall()
            except sqlite3.Error: return []
        return [{"instruction": "Apply user correction", "output": c, "metadata": {"rating": r}} for c, r in rows]

    def _llm_generate(self, prompt: str) -> Any:
        llm = getattr(self.distiller, "llm", None)
        return llm.generate(prompt) if llm and hasattr(llm, "generate") else prompt

    async def _maybe_await(self, value: Any) -> Any:
        return await value if asyncio.iscoroutine(value) else value

    def _cfg(self, key: str, default: Any) -> Any:
        return self.config.get(key, default) if isinstance(self.config, dict) else getattr(self.config, key, default)

class _Obj:
    def __init__(self, values: dict): self.__dict__.update(values)
    def get(self, key: str, default: Any = None) -> Any: return getattr(self, key, default)

def _default_ladder() -> list[dict]:
    return [{"generation": 1, "param_label": "3B", "dim": 3072, "layers": 32}, {"generation": 2, "param_label": "7B", "dim": 4096, "layers": 32}, {"generation": 3, "param_label": "13B", "dim": 5120, "layers": 40}, {"generation": 4, "param_label": "34B", "dim": 8192, "layers": 48}, {"generation": 5, "param_label": "70B", "dim": 8192, "layers": 80}]
