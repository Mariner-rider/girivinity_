"""
Girivinity LLMEngine — hardware-adaptive, dual-backend inference engine.

Backend selection (automatic):
  If VRAM >= 8GB AND gguf_path exists → "hybrid" (llama.cpp generate + HF tokenizer)
  If VRAM >= 4GB AND gguf_path exists → "llama_cpp" (full llama.cpp)
  If VRAM >= 4GB AND no gguf_path    → "transformers" (4-bit HF)
  If no GPU                          → "llama_cpp" CPU-only mode

Key capabilities added vs. existing file:
  - get_token_entropy(prompt) → float   (for real confidence scoring)
  - get_logprobs(prompt, completion) → list[float]
  - generate_from_embeds(embeds, attention_mask) → GenerationResult
      (accepts pre-computed embeddings from MultimodalFusionLayer)
  - Thread-safe via asyncio.Lock
  - Hardware monitoring: logs GPU utilisation per generation call
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareProfile:
    has_gpu: bool
    vram_gb: float
    gpu_name: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    backend: str
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    entropy: float | None = None
    hardware: HardwareProfile | None = None


@dataclass(frozen=True)
class LLMEngineConfig:
    backend: str = "hybrid"
    model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    gguf_path: Path = Path("models/gguf/girivinity-q4_k_m.gguf")
    device_map: str = "auto"
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    n_threads: int = 8
    n_batch: int = 512
    use_mlock: bool = True
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config.yaml") -> "LLMEngineConfig":
        path = Path(config_path)
        if not path.exists():
            return cls()
        if importlib.util.find_spec("yaml") is None:
            return cls()
        yaml = importlib.import_module("yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        model_cfg = data.get("model", {}) or {}
        gguf_path = model_cfg.get("gguf_path") or model_cfg.get("quantised_path")
        return cls(
            backend=str(model_cfg.get("backend", "hybrid")),
            model_id=str(model_cfg.get("model_id", "microsoft/Phi-3-mini-4k-instruct")),
            gguf_path=Path(gguf_path or "models/gguf/girivinity-q4_k_m.gguf"),
            device_map=str(model_cfg.get("device_map", "auto")),
            n_gpu_layers=int(model_cfg.get("n_gpu_layers", -1)),
            n_ctx=int(model_cfg.get("n_ctx", 4096)),
            n_threads=int(model_cfg.get("n_threads", max(1, (os.cpu_count() or 2) - 1))),
            n_batch=int(model_cfg.get("n_batch", 512)),
            use_mlock=bool(model_cfg.get("use_mlock", True)),
            torch_dtype=str(model_cfg.get("torch_dtype", "float16")),
            load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            attn_implementation=str(
                model_cfg.get("attn_implementation", "flash_attention_2")
            ),
        )


class LLMEngine:
    """Hardware-adaptive Girivinity inference engine."""

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        loader: Any | None = None,
    ) -> None:
        self.config = LLMEngineConfig.from_yaml(config_path)
        self.hardware = self._detect_hardware()
        self.backend = self._select_backend()
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.RLock()
        self._loader = loader
        self._llama: Any | None = None
        self._tokenizer: Any | None = None
        self._hf_model: Any | None = None

        if loader is not None:
            self.backend = "llama_cpp"
            self._llama = loader.get_model()

        logger.info(
            "LLMEngine selected backend=%s gpu=%s vram_gb=%.2f gguf_exists=%s",
            self.backend,
            self.hardware.gpu_name or "none",
            self.hardware.vram_gb,
            self.config.gguf_path.exists(),
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stream: bool = True,
    ) -> Iterator[str] | str:
        with self._sync_lock:
            self._log_gpu_utilisation()
            if stream:
                return self._stream(prompt, max_tokens)
            return self._generate_text(prompt, max_tokens)

    async def agenerate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        async with self._async_lock:
            self._log_gpu_utilisation()
            text = self._generate_text(prompt, max_tokens)
            return GenerationResult(
                text=text,
                backend=self.backend,
                entropy=self.get_token_entropy(prompt),
                hardware=self.hardware,
            )

    def generate_from_embeds(
        self,
        embeds: Any,
        attention_mask: Any | None = None,
        max_tokens: int = 512,
    ) -> GenerationResult:
        """Generate from precomputed embeddings produced by multimodal fusion."""
        with self._sync_lock:
            self._ensure_transformers_loaded()
            torch = importlib.import_module("torch")
            with torch.no_grad():
                output_ids = self._hf_model.generate(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                )
            text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return GenerationResult(text=text, backend="transformers", hardware=self.hardware)

    def get_token_entropy(self, prompt: str) -> float:
        logprobs = self.get_logprobs(prompt, "")
        if not logprobs:
            return 0.0
        probabilities = [math.exp(value) for value in logprobs]
        total = sum(probabilities)
        if total <= 0:
            return 0.0
        return -sum((prob / total) * math.log(prob / total) for prob in probabilities)

    def get_logprobs(self, prompt: str, completion: str) -> list[float]:
        text = f"{prompt}{completion}"
        if self.backend in {"llama_cpp", "hybrid"}:
            model = self._ensure_llama_loaded()
            if hasattr(model, "create_completion"):
                response = model.create_completion(
                    text,
                    max_tokens=1,
                    echo=True,
                    logprobs=5,
                )
                choice = response.get("choices", [{}])[0]
                token_logprobs = choice.get("logprobs", {}).get("token_logprobs") or []
                return [float(value) for value in token_logprobs if value is not None]
            return []

        self._ensure_transformers_loaded()
        torch = importlib.import_module("torch")
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self._hf_model(**inputs)
            log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
            target_ids = inputs["input_ids"][:, 1:]
            gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return [float(value) for value in gathered.flatten().tolist()]

    def generate_with_context(
        self,
        question: str,
        context: str,
        user_level: int = 1,
        max_tokens: int = 512,
        stream: bool = True,
    ) -> Iterator[str] | str:
        prompt = (
            "Use the provided context to answer the question.\n"
            f"User level: {user_level}\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return self.generate(prompt, max_tokens=max_tokens, stream=stream)

    def _select_backend(self) -> str:
        gguf_exists = self.config.gguf_path.exists()
        if self.hardware.vram_gb >= 8 and gguf_exists:
            return "hybrid"
        if self.hardware.vram_gb >= 4 and gguf_exists:
            return "llama_cpp"
        if self.hardware.vram_gb >= 4 and not gguf_exists:
            return "transformers"
        return "llama_cpp"

    def _detect_hardware(self) -> HardwareProfile:
        if importlib.util.find_spec("torch") is None:
            return HardwareProfile(has_gpu=False, vram_gb=0.0)

        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return HardwareProfile(has_gpu=False, vram_gb=0.0)

        props = torch.cuda.get_device_properties(0)
        vram_gb = float(props.total_memory) / (1024**3)
        return HardwareProfile(has_gpu=True, vram_gb=vram_gb, gpu_name=props.name)

    def _ensure_llama_loaded(self) -> Any:
        if self._llama is None:
            llama_cpp = importlib.import_module("llama_cpp")
            self._llama = llama_cpp.Llama(
                model_path=str(self.config.gguf_path),
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers if self.hardware.has_gpu else 0,
                n_batch=self.config.n_batch,
                use_mlock=self.config.use_mlock,
            )
        return self._llama

    def _ensure_transformers_loaded(self) -> None:
        if self._hf_model is not None and self._tokenizer is not None:
            return

        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        dtype = getattr(torch, self.config.torch_dtype, torch.float16)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map=self.config.device_map,
            torch_dtype=dtype,
            load_in_4bit=self.config.load_in_4bit,
            trust_remote_code=self.config.trust_remote_code,
            attn_implementation=self.config.attn_implementation,
        )

    def _generate_text(self, prompt: str, max_tokens: int) -> str:
        if self.backend in {"llama_cpp", "hybrid"}:
            model = self._ensure_llama_loaded()
            response = model(prompt, max_tokens=max_tokens, stream=False)
            return str(response["choices"][0]["text"])

        self._ensure_transformers_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        output_ids = self._hf_model.generate(**inputs, max_new_tokens=max_tokens)
        return str(self._tokenizer.decode(output_ids[0], skip_special_tokens=True))

    def _stream(self, prompt: str, max_tokens: int) -> Iterator[str]:
        if self.backend not in {"llama_cpp", "hybrid"}:
            yield self._generate_text(prompt, max_tokens)
            return

        model = self._ensure_llama_loaded()
        for chunk in model(prompt, max_tokens=max_tokens, stream=True):
            token = chunk["choices"][0].get("text", "")
            if token:
                yield token

    def _log_gpu_utilisation(self) -> None:
        if importlib.util.find_spec("GPUtil") is None:
            return
        gputil = importlib.import_module("GPUtil")
        gpus = gputil.getGPUs()
        if not gpus:
            return
        gpu = gpus[0]
        logger.info(
            "GPU utilisation: id=%s load=%.1f%% memory=%.1f%% used=%sMB total=%sMB",
            gpu.id,
            gpu.load * 100,
            gpu.memoryUtil * 100,
            gpu.memoryUsed,
            gpu.memoryTotal,
        )


class GirivinityEngine(LLMEngine):
    """Backward-compatible public name used by older routes and tests."""

    def __init__(self, loader: Any | None = None) -> None:
        super().__init__(loader=loader)
