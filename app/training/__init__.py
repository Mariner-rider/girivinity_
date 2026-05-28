from app.finetune import approve_model_update, build_dataset_from_logs

__all__ = [
    "approve_model_update",
    "build_dataset_from_logs",
    "LoRATrainer",
    "LoRATrainingConfig",
    "evaluate_adapter",
    "PretrainConfig",
    "GirivinityPretrainer",
]


def __getattr__(name: str):
    if name in {"LoRATrainer", "LoRATrainingConfig", "evaluate_adapter"}:
        import app.finetune as finetune

        return getattr(finetune, name)
    if name in {"PretrainConfig", "GirivinityPretrainer"}:
        import app.training.pretrain as pretrain

        return getattr(pretrain, name)
    raise AttributeError(name)
