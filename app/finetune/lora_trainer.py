from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from app.security.policy import SecurityGuard, secure_operation

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoRATrainingConfig:
    model_id: str
    dataset_path: str
    output_dir: str
    validation_dataset_path: str
    learning_rate: float = 2e-4
    epochs: int = 1
    batch_size: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    min_benchmark_delta: float = 0.0


class LoRATrainer:
    """Parameter-efficient (LoRA-only) trainer for plug-and-play model updates."""

    def __init__(
        self,
        config: LoRATrainingConfig,
        security_guard: SecurityGuard | None = None,
    ) -> None:
        self.config = config
        self.security_guard = security_guard or SecurityGuard()

    def _tokenize(self, tokenizer):
        def fn(batch):
            merged = [
                f"Instruction: {instruction}\nAnswer: {output}"
                for instruction, output in zip(batch["instruction"], batch["output"])
            ]
            tokens = tokenizer(merged, truncation=True, max_length=1024)
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        return fn

    @secure_operation("finetune.lora_train")
    def train(self) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        on_gpu = device.type == "cuda"
        if on_gpu:
            logger.info("Training on GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.info("Training on CPU — this will be slow, consider a GPU instance")

        self.security_guard.require_validation_dataset(self.config.validation_dataset_path)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if on_gpu else torch.float32,
            device_map=None,
        ).to(device)

        scaler = torch.cuda.amp.GradScaler(enabled=on_gpu)
        _ = scaler

        peft_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, peft_cfg).to(device)

        ds = load_dataset("json", data_files=self.config.dataset_path, split="train")
        val_ds = load_dataset("json", data_files=self.config.validation_dataset_path, split="train")
        tokenized = ds.map(self._tokenize(tokenizer), batched=True, remove_columns=ds.column_names)
        tokenized_val = val_ds.map(
            self._tokenize(tokenizer),
            batched=True,
            remove_columns=val_ds.column_names,
        )

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.epochs,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            fp16=on_gpu,
            bf16=False,
            dataloader_num_workers=4 if on_gpu else 0,
            no_cuda=not on_gpu,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            eval_dataset=tokenized_val,
        )
        try:
            trainer.train()
            model.save_pretrained(self.config.output_dir)
            tokenizer.save_pretrained(self.config.output_dir)
            return self.config.output_dir
        finally:
            if on_gpu:
                torch.cuda.empty_cache()
