from app.finetune.dataset_builder import build_dataset_from_logs
from app.finetune.update_gate import approve_model_update

__all__ = [
    "build_dataset_from_logs",
    "LoRATrainer",
    "LoRATrainingConfig",
    "evaluate_adapter",
    "approve_model_update",
]


def __getattr__(name: str):
    if name in {"LoRATrainer", "LoRATrainingConfig"}:
        from app.finetune.lora_trainer import LoRATrainer, LoRATrainingConfig

        return {"LoRATrainer": LoRATrainer, "LoRATrainingConfig": LoRATrainingConfig}[name]
    if name == "evaluate_adapter":
        from app.finetune.evaluate import evaluate_adapter

        return evaluate_adapter
    raise AttributeError(name)
