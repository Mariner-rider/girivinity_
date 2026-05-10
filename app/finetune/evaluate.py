from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.security.policy import SecurityGuard, secure_operation


@dataclass(slots=True)
class EvalResult:
    samples: int
    exact_match: float


@secure_operation("finetune.evaluate_adapter")
def evaluate_adapter(
    model_id: str,
    adapter_path: str,
    eval_dataset_path: str,
    max_new_tokens: int = 64,
    security_guard: SecurityGuard | None = None,
) -> EvalResult:
    guard = security_guard or SecurityGuard()
    guard.require_validation_dataset(eval_dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    dataset = load_dataset("json", data_files=eval_dataset_path, split="train")

    matches = 0
    for row in dataset:
        prompt = f"Instruction: {row['instruction']}\nAnswer:"
        gold = row["output"].strip()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prediction = generated.split("Answer:")[-1].strip()
        if prediction == gold:
            matches += 1

    total = len(dataset)
    score = matches / total if total else 0.0
    return EvalResult(samples=total, exact_match=round(score, 4))
