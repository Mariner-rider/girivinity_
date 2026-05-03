from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


@dataclass(slots=True)
class LoRATrainingConfig:
    model_id: str
    dataset_path: str
    output_dir: str
    learning_rate: float = 2e-4
    epochs: int = 1
    batch_size: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class LoRATrainer:
    """Parameter-efficient (LoRA-only) trainer for plug-and-play model updates."""

    def __init__(self, config: LoRATrainingConfig) -> None:
        self.config = config

    def _tokenize(self, tokenizer):
        def fn(batch):
            merged = [f"Instruction: {i}\nAnswer: {o}" for i, o in zip(batch["instruction"], batch["output"])]
            tokens = tokenizer(merged, truncation=True, max_length=1024)
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        return fn

    def train(self) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        peft_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, peft_cfg)

        ds = load_dataset("json", data_files=self.config.dataset_path, split="train")
        tokenized = ds.map(self._tokenize(tokenizer), batched=True, remove_columns=ds.column_names)

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.epochs,
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            bf16=False,
            report_to=[],
        )

        trainer = Trainer(model=model, args=args, train_dataset=tokenized)
        trainer.train()
        model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        return self.config.output_dir
