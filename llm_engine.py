"""HuggingFace Transformers inference engine with streaming and safe CPU fallback."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from app.core.config_loader import ConfigLoader
from app.security.policy import secure_operation

logger = logging.getLogger(__name__)


_DTYPE_NAMES = {"float16", "bfloat16", "float32"}


@dataclass(frozen=True, slots=True)
class LLMEngineConfig:
    model_id: str
    device_map: str = "auto"
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    kv_cache: bool = True
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "LLMEngineConfig":
        config = ConfigLoader(path).load()
        raw_model = dict(config.raw.get("model", {}))
        modules_llm = config.raw.get("modules", {}).get("llm", {})
        if isinstance(modules_llm, dict):
            raw_model = {**modules_llm, **raw_model}
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in raw_model.items() if key in allowed})


@dataclass(slots=True)
class GenerationMetrics:
    latency_ms: float
    tokens_generated: int
    used_gpu: bool


@dataclass(slots=True)
class GenerationResult:
    text: str
    metrics: GenerationMetrics


class LLMEngine:
    """Lazy-loading inference engine optimized for low-memory deployments."""

    def __init__(self, config: LLMEngineConfig) -> None:
        self.config = config
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.used_gpu = False

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "LLMEngine":
        return cls(LLMEngineConfig.from_yaml(path))

    @secure_operation("llm_engine.load")
    def load(self) -> "LLMEngine":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.used_gpu = bool(torch.cuda.is_available())
        dtype = self._resolve_torch_dtype(torch)
        quantization_config = None
        effective_device_map = self.config.device_map

        if self.used_gpu and self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        elif not self.used_gpu:
            effective_device_map = "cpu"
            dtype = torch.float32
            logger.warning(
                "GPU unavailable; falling back to CPU inference without 4-bit quantization"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map=effective_device_map,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.config.use_cache = self.config.kv_cache
        self.model.eval()
        return self

    @secure_operation("llm_engine.generate")
    def generate(self, prompt: str, **overrides: Any) -> GenerationResult:
        self._ensure_loaded()
        import torch

        start = time.perf_counter()
        inputs = self._tokenize(prompt)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **self._generation_kwargs(overrides),
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        input_tokens = int(inputs["input_ids"].shape[-1])
        generated_tokens = max(int(output_ids.shape[-1]) - input_tokens, 0)
        return GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                latency_ms=round((time.perf_counter() - start) * 1000, 3),
                tokens_generated=generated_tokens,
                used_gpu=self.used_gpu,
            ),
        )

    @secure_operation("llm_engine.stream")
    def stream(self, prompt: str, **overrides: Any) -> Iterator[str]:
        self._ensure_loaded()
        from transformers import TextIteratorStreamer

        inputs = self._tokenize(prompt)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs = {
            **inputs,
            **self._generation_kwargs(overrides),
            "streamer": streamer,
        }
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        yield from streamer
        thread.join()

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            self.load()

    def _tokenize(self, prompt: str) -> dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.used_gpu:
            return inputs.to(self.model.device)
        return inputs

    def _generation_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "use_cache": self.config.kv_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        return kwargs

    def _resolve_torch_dtype(self, torch_module: Any) -> Any:
        dtype_name = self.config.torch_dtype.lower()
        if dtype_name not in _DTYPE_NAMES:
            raise ValueError(f"Unsupported torch_dtype: {self.config.torch_dtype}")
        return getattr(torch_module, dtype_name)
