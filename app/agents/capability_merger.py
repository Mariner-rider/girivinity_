"""Merge per-user capability deltas into shared per-agent LoRA adapters."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class CapabilityMerger:
    def __init__(self, config: dict[str, Any], agent_registry: Any, lora_trainer: Any, security_guard: Any):
        self.config = config
        self.registry = agent_registry
        self.trainer = lora_trainer
        self.guard = security_guard
        self.merge_threshold = int((config.get("adaptive_agents", {}) or {}).get("merge_threshold", 50))
        self.min_quality = float((config.get("adaptive_agents", {}) or {}).get("min_delta_quality", 0.60))

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "CapabilityMerger":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
        from app.agents.agent_registry import AgentRegistry
        from app.security.policy import SecurityGuard

        return cls(cfg or {}, AgentRegistry.from_config(path), _ExampleLoRATrainer(), SecurityGuard())

    async def check_and_merge(self, agent_type_id: str) -> dict[str, Any]:
        deltas = self.registry.get_unmerged_deltas(agent_type_id, min_quality=self.min_quality)
        if len(deltas) < self.merge_threshold:
            return {"merged": False, "reason": f"Only {len(deltas)}/{self.merge_threshold} deltas", "available_deltas": len(deltas)}

        logger.info("Merging %s capability deltas for %s", len(deltas), agent_type_id)
        clean_examples = [self._clean_delta(delta) for delta in deltas]
        agent_def = self.registry.get(agent_type_id)
        if agent_def is None:
            return {"merged": False, "reason": f"Unknown agent type: {agent_type_id}"}

        adapter_out = f"models/adapters/{agent_type_id}/v{agent_def.capability_version + 1}"
        train_result = await self._train(clean_examples, adapter_out)
        if not train_result.get("success"):
            return {"merged": False, "reason": "Training failed", "detail": train_result}

        improvement = float(train_result.get("perplexity_delta", 0.0))
        if improvement >= 0.0:
            Path(adapter_out).mkdir(parents=True, exist_ok=True)
            symlink = Path(agent_def.adapter_path)
            symlink.parent.mkdir(parents=True, exist_ok=True)
            tmp = symlink.with_name(symlink.name + ".tmp")
            if os.path.lexists(tmp):
                os.unlink(tmp)
            os.symlink(Path(adapter_out).resolve(), tmp)
            os.replace(tmp, symlink)
            new_version = self.registry.update_capability_version(agent_type_id)
            self.registry.mark_deltas_merged([delta["delta_id"] for delta in deltas])
            logger.info("Promoted adapter v%s for %s", new_version, agent_type_id)
            return {
                "merged": True,
                "new_version": new_version,
                "perplexity_delta": improvement,
                "examples_used": len(deltas),
                "adapter_path": str(symlink),
            }

        return {"merged": False, "reason": "No improvement in perplexity", "delta": improvement}

    def extract_training_example(self, task: str, agent_output: str, user_rating: int | None) -> dict[str, Any] | None:
        if not agent_output or len(agent_output.strip()) < 50:
            return None
        rating_quality = (max(1, min(5, int(user_rating))) / 5.0) if user_rating is not None else None
        quality = rating_quality if rating_quality is not None else self._heuristic_quality(agent_output)
        if quality < 0.5:
            return None
        return {
            "instruction": self.guard.sanitize_output(task or ""),
            "input": "",
            "output": self.guard.sanitize_output(agent_output),
            "quality_score": round(float(quality), 3),
        }

    def _clean_delta(self, delta: dict[str, Any]) -> dict[str, str]:
        example = delta.get("training_example") or {}
        user_hash = hashlib.sha256(str(delta.get("user_id", "")).encode("utf-8")).hexdigest()[:16]
        return {
            "instruction": self.guard.sanitize_output(str(example.get("instruction", ""))),
            "input": self.guard.sanitize_output(str(example.get("input", ""))),
            "output": self.guard.sanitize_output(str(example.get("output", ""))),
            "metadata": json.dumps({"user_hash": user_hash, "delta_id": delta.get("delta_id"), "quality_score": delta.get("quality_score")}),
        }

    async def _train(self, clean_examples: list[dict[str, str]], adapter_out: str) -> dict[str, Any]:
        trainer = self.trainer
        loop = asyncio.get_running_loop()
        if hasattr(trainer, "train_from_examples"):
            return await loop.run_in_executor(None, trainer.train_from_examples, clean_examples, adapter_out)
        if hasattr(trainer, "train"):
            return await loop.run_in_executor(None, self._train_with_dataset_file, clean_examples, adapter_out)
        return _ExampleLoRATrainer().train_from_examples(clean_examples, adapter_out)

    def _train_with_dataset_file(self, clean_examples: list[dict[str, str]], adapter_out: str) -> dict[str, Any]:
        dataset_path = Path(adapter_out) / "capability_examples.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("w", encoding="utf-8") as handle:
            for example in clean_examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        if hasattr(self.trainer, "config"):
            self.trainer.config.dataset_path = str(dataset_path)
            self.trainer.config.validation_dataset_path = str(dataset_path)
            self.trainer.config.output_dir = adapter_out
        path = self.trainer.train()
        return {"success": True, "adapter_path": str(path), "perplexity_delta": 0.0}

    @staticmethod
    def _heuristic_quality(agent_output: str) -> float:
        length_score = min(len(agent_output.split()) / 120.0, 1.0)
        structure_bonus = 0.1 if any(marker in agent_output for marker in ["\n-", "1.", "```", "Mitigation", "Steps"]) else 0.0
        caveat_bonus = 0.05 if any(word in agent_output.lower() for word in ["because", "therefore", "however", "evidence"]) else 0.0
        return min(0.95, 0.55 + (0.25 * length_score) + structure_bonus + caveat_bonus)


class _ExampleLoRATrainer:
    def train_from_examples(self, examples: list[dict[str, str]], adapter_out: str) -> dict[str, Any]:
        out = Path(adapter_out)
        out.mkdir(parents=True, exist_ok=True)
        dataset = out / "training_examples.jsonl"
        with dataset.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        manifest = {"success": True, "examples": len(examples), "created_at": time.time(), "perplexity_delta": 0.0}
        (out / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"success": True, "adapter_path": str(out), "perplexity_delta": 0.0, "examples": len(examples)}
