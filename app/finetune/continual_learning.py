from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.finetune.dataset_builder import build_dataset_from_logs
from app.finetune.update_gate import approve_model_update
from app.security.policy import SecurityGuard, SecurityPolicyError, secure_operation


@dataclass(slots=True)
class ContinualLearningConfig:
    logs_path: str
    dataset_output_path: str
    adapter_output_dir: str
    production_adapter_dir: str
    benchmark_name: str = "validation_exact_match"
    min_quality_score: float = 0.6
    min_delta: float = 0.0


@dataclass(slots=True)
class ContinualLearningResult:
    collected_samples: int
    kept_samples: int
    candidate_score: float
    baseline_score: float
    promoted: bool
    rollback_triggered: bool


class ContinualLearningSystem:
    """Continual learning pipeline with LoRA-only updates (no full retraining)."""

    def __init__(self, config: ContinualLearningConfig, security_guard: SecurityGuard | None = None) -> None:
        self.config = config
        self.security_guard = security_guard or SecurityGuard()

    def collect_logs(self) -> list[dict]:
        rows: list[dict] = []
        with Path(self.config.logs_path).open("r", encoding="utf-8") as src:
            for raw in src:
                raw = raw.strip()
                if not raw:
                    continue
                rows.append(json.loads(raw))
        return rows

    def filter_high_quality_samples(self, rows: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        for row in rows:
            quality = float(row.get("quality_score", 0.0))
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if quality >= self.config.min_quality_score and prompt and response:
                filtered.append({"prompt": prompt, "response": response})
        return filtered

    def build_dataset(self, filtered_rows: list[dict]) -> int:
        logs_for_builder = Path(self.config.dataset_output_path).with_suffix(".filtered_logs.jsonl")
        logs_for_builder.parent.mkdir(parents=True, exist_ok=True)
        with logs_for_builder.open("w", encoding="utf-8") as out:
            for row in filtered_rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        return build_dataset_from_logs(logs_for_builder, self.config.dataset_output_path)

    def run_lora_finetune(self, train_fn) -> str:
        # Explicitly LoRA-only path (no full retraining).
        return train_fn(self.config.dataset_output_path, self.config.adapter_output_dir)

    def evaluate_benchmarks(self, eval_fn) -> tuple[float, float]:
        baseline_score, candidate_score = eval_fn(self.config.production_adapter_dir, self.config.adapter_output_dir)
        return baseline_score, candidate_score

    def rollback(self) -> None:
        adapter_dir = Path(self.config.adapter_output_dir)
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)

    @secure_operation("finetune.continual_learning")
    def run(self, train_fn, eval_fn) -> ContinualLearningResult:
        collected = self.collect_logs()
        filtered = self.filter_high_quality_samples(collected)
        kept_samples = self.build_dataset(filtered)

        self.run_lora_finetune(train_fn)
        baseline_score, candidate_score = self.evaluate_benchmarks(eval_fn)

        promoted = False
        rollback_triggered = False
        try:
            promoted = approve_model_update(
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                benchmark_name=self.config.benchmark_name,
                min_delta=self.config.min_delta,
                security_guard=self.security_guard,
            )
        except SecurityPolicyError:
            rollback_triggered = True
            self.rollback()

        return ContinualLearningResult(
            collected_samples=len(collected),
            kept_samples=kept_samples,
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            promoted=promoted,
            rollback_triggered=rollback_triggered,
        )
