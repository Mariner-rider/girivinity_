"""Config-driven modular LLM loader with 4-bit quantization support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase


@dataclass(slots=True)
class ModelConfig:
    model_id: str
    device_map: str = "auto"
    trust_remote_code: bool = False
    attn_implementation: str = "sdpa"
    torch_dtype: str = "float16"
    kv_cache: bool = True

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"


@dataclass(slots=True)
class LoadedModel:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    config: ModelConfig


class LLMConfigError(ValueError):
    """Raised when configuration content is invalid."""


def _to_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype_name.lower()]
    except KeyError as exc:
        raise LLMConfigError(f"Unsupported dtype '{dtype_name}'.") from exc


def load_config(path: str | Path = "config.yaml") -> ModelConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    llm_cfg = raw.get("llm")
    if not isinstance(llm_cfg, dict):
        raise LLMConfigError("Expected top-level 'llm' mapping in config.yaml")

    return ModelConfig(**llm_cfg)


class LLMFactory:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def _quantization(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=_to_dtype(self.config.bnb_4bit_compute_dtype),
        )

    def load(self) -> LoadedModel:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            quantization_config=self._quantization(),
            device_map=self.config.device_map,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=_to_dtype(self.config.torch_dtype),
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
        )

        model.config.use_cache = self.config.kv_cache
        return LoadedModel(tokenizer=tokenizer, model=model, config=self.config)


def load_from_yaml(path: str | Path = "config.yaml") -> LoadedModel:
    cfg = load_config(path)
    return LLMFactory(cfg).load()
