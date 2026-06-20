from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from llm_loader import GirivinityLoader, LLMEngineConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _NativeGeneration:
    text: str


@dataclass(slots=True)
class GenerationMetrics:
    latency_ms: float
    completion_tokens: int
    used_gpu: bool


class GenerationResult(str):
    """String-compatible generation result with structured metadata."""

    def __new__(cls, text: str, metrics: GenerationMetrics | None = None):
        obj = str.__new__(cls, text)
        obj.text = text
        obj.metrics = metrics
        return obj


class LLMEngine:
    """Text-generation engine supporting HuggingFace and native Girivinity models."""

    def __init__(
        self,
        config: LLMEngineConfig | None = None,
        loader: GirivinityLoader | None = None,
    ) -> None:
        self.config = config or LLMEngineConfig()
        self._loader = loader or GirivinityLoader(config=self.config)
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.use_native_model = self.config.use_native_model
        requested_device = self._requested_compute_device()
        if requested_device == "cuda" and not self._cuda_available():
            raise RuntimeError("compute.device is 'cuda' but CUDA is not available")
        self.used_gpu = requested_device != "cpu" and self._cuda_available()
        self.device = "cuda" if self.used_gpu else "cpu"
        self._loaded = False
        self._active_backend = "transformers"
        self._llama: Any | None = None

    @classmethod
    def auto_detect_backend(cls, config: LLMEngineConfig) -> str:
        backend = getattr(config, "backend", "auto")
        if backend != "auto":
            return backend
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            has_gpu_vram = any(getattr(g, "memoryTotal", 0) >= 4096 for g in gpus)
        except Exception:
            has_gpu_vram = False
        gguf_path = str(getattr(config, "gguf_path", "") or getattr(config, "quantised_path", "") or "")
        has_gguf = bool(gguf_path and Path(gguf_path).exists())
        if has_gpu_vram and has_gguf:
            return "llama_cpp"
        if has_gpu_vram:
            return "transformers"
        return "llama_cpp" if has_gguf else "transformers"

    def load(self) -> "LLMEngine":
        backend = self.auto_detect_backend(self.config)
        self._active_backend = backend
        if backend == "llama_cpp":
            return self._load_llama_cpp()
        return self._load_transformers()

    def _load_transformers(self) -> "LLMEngine":
        loaded = self._loader.load()
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.use_native_model = loaded.use_native_model
        self._configure_device()
        self._loaded = True
        return self

    def _load_llama_cpp(self) -> "LLMEngine":
        try:
            from llama_cpp import Llama
        except Exception as exc:
            logger.warning("llama_cpp backend requested but unavailable; falling back to transformers: %s", exc)
            self._active_backend = "transformers"
            return self._load_transformers()
        gguf_path = str(getattr(self.config, "gguf_path", "") or getattr(self.config, "quantised_path", ""))
        if not gguf_path:
            raise RuntimeError("llama_cpp backend requires config.gguf_path or config.quantised_path")
        self._llama = Llama(
            model_path=gguf_path,
            n_gpu_layers=int(getattr(self.config, "n_gpu_layers", -1)),
            n_ctx=int(getattr(self.config, "n_ctx", 4096)),
            n_threads=int(getattr(self.config, "n_threads", 8)),
            verbose=False,
        )
        self.used_gpu = int(getattr(self.config, "n_gpu_layers", -1)) != 0
        self.device = "cuda" if self.used_gpu else "cpu"
        self._loaded = True
        return self

    def get_device_info(self) -> dict[str, Any]:
        try:
            import torch

            if self.used_gpu and torch.cuda.is_available():
                return {
                    "device": "cuda",
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_vram_total_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                    ),
                    "gpu_vram_free_gb": round(torch.cuda.mem_get_info()[0] / 1e9, 1),
                    "inference_mode": "4bit_nf4_gpu",
                }
        except Exception as exc:
            logger.debug("Unable to read CUDA device info: %s", exc)
        return {
            "device": "cpu",
            "gpu_name": None,
            "gpu_vram_total_gb": None,
            "gpu_vram_free_gb": None,
            "inference_mode": "float32_cpu",
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stream: bool = False,
        **generation_kwargs: Any,
    ) -> Iterator[str] | str:
        self._ensure_loaded()
        if getattr(self, "_active_backend", "transformers") == "llama_cpp":
            if stream:
                return self.stream(prompt, max_tokens=max_tokens, **generation_kwargs)
            return self._generate_llama_cpp(prompt, max_tokens=max_tokens, **generation_kwargs)
        if stream:
            return self.stream(prompt, max_tokens=max_tokens, **generation_kwargs)

        if self.use_native_model:
            return self._generate_native(prompt, max_tokens, **generation_kwargs).text
        return self._generate_huggingface(prompt, max_tokens, **generation_kwargs)

    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        **generation_kwargs: Any,
    ) -> Iterator[str]:
        self._ensure_loaded()
        if getattr(self, "_active_backend", "transformers") == "llama_cpp":
            llama = getattr(self, "_llama", None)
            if llama is None:
                return
            max_new = generation_kwargs.get("max_new_tokens", max_tokens)
            temperature = generation_kwargs.get("temperature", getattr(self.config, "temperature", 0.2))
            for chunk in llama(prompt, max_tokens=max_new, temperature=temperature, stream=True):
                token = chunk.get("choices", [{}])[0].get("text", "")
                if token:
                    yield token
            return
        if self.use_native_model:
            text = self._generate_native(prompt, max_tokens, **generation_kwargs).text
            if text:
                yield text
            return

        text = self._generate_huggingface(prompt, max_tokens, **generation_kwargs)
        if text:
            yield text


    def _generate_llama_cpp(self, prompt: str, **overrides: Any) -> GenerationResult:
        llama = getattr(self, "_llama", None)
        if llama is None:
            raise RuntimeError("llama_cpp backend is not loaded")
        start = time.perf_counter()
        max_tokens = int(overrides.get("max_new_tokens", overrides.get("max_tokens", getattr(self.config, "max_new_tokens", 512))))
        temperature = float(overrides.get("temperature", getattr(self.config, "temperature", 0.2)))
        output = llama(prompt, max_tokens=max_tokens, temperature=temperature, echo=False)
        choice = (output.get("choices") or [{}])[0]
        text = str(choice.get("text", ""))
        usage = output.get("usage") or {}
        tokens = int(usage.get("completion_tokens", len(text.split())))
        latency = round((time.perf_counter() - start) * 1000, 3)
        return GenerationResult(text=text, metrics=GenerationMetrics(latency, tokens, self.used_gpu))

    def _generate_native(
        self,
        prompt: str,
        max_tokens: int,
        **generation_kwargs: Any,
    ) -> _NativeGeneration:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Native model and tokenizer must be loaded before generation")

        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = self._move_tensor_to_model_device(encoded["input_ids"])
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            **generation_kwargs,
        )
        completion_ids = output_ids[:, input_ids.shape[-1] :]
        return _NativeGeneration(text=self.tokenizer.decode(completion_ids[0]))

    def _generate_huggingface(
        self,
        prompt: str,
        max_tokens: int,
        **generation_kwargs: Any,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("HuggingFace model and tokenizer must be loaded before generation")

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = self._move_inputs_to_model_device(encoded)
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            **generation_kwargs,
        )
        input_length = encoded["input_ids"].shape[-1]
        completion_ids = output_ids[:, input_length:]
        return self.tokenizer.decode(completion_ids[0], skip_special_tokens=True)


    def get_token_entropy(self, prompt: str = "", text: str | None = None) -> float:
        """Compute next-token entropy from model logits; fallback to lexical entropy offline."""
        sample = text if text is not None else prompt
        if getattr(self, "_active_backend", "transformers") != "llama_cpp":
            try:
                import torch

                self._ensure_loaded()
                inputs = self._tokenize(sample or " ")
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits.float(), dim=-1)
                entropy = -(probs * (probs + 1e-9).log()).sum().item()
                return float(entropy)
            except Exception:
                pass
        import math, re

        tokens = re.findall(r"[A-Za-z0-9_]+", (sample or "").lower())
        if not tokens:
            return 1.0
        counts = {tok: tokens.count(tok) for tok in set(tokens)}
        total = float(len(tokens))
        return -sum((count / total) * math.log(count / total) for count in counts.values())

    def _tokenize(self, prompt: str) -> dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before entropy computation")
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return self._move_inputs_to_model_device(encoded)

    def confidence_from_entropy(self, prompt: str = "", agent_name: str = "default") -> float:
        entropy = self.get_token_entropy(prompt)
        raw = 1.0 / (1.0 + max(0.0, entropy))
        try:
            from app.cognition.calibration import CalibrationManager

            return CalibrationManager.from_config().calibrate(raw, agent_name)
        except Exception:
            return round(min(1.0, max(0.0, raw)), 3)

    def _move_inputs_to_model_device(self, encoded: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self._move_tensor_to_model_device(value) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

    def _move_tensor_to_model_device(self, value: Any) -> Any:
        if self.model is None or not hasattr(value, "to"):
            return value
        device = self._model_device()
        return value.to(device) if device is not None else value

    def _configure_device(self) -> None:
        if self.model is None:
            return
        try:
            import torch

            requested_device = self._requested_compute_device()
            if requested_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("compute.device is 'cuda' but CUDA is not available")
            target_device = torch.device(
                "cuda" if requested_device != "cpu" and torch.cuda.is_available() else "cpu"
            )
            if self.use_native_model:
                self.model.to(target_device)
            model_device = self._model_device()
            self.used_gpu = bool(model_device and str(model_device).startswith("cuda"))
            self.device = "cuda" if self.used_gpu else "cpu"
        except Exception as exc:
            logger.debug("Model device configuration failed; using CPU metadata: %s", exc)
            self.used_gpu = False
            self.device = "cpu"

    def _model_device(self) -> Any | None:
        if self.model is None:
            return None
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        try:
            return next(self.model.parameters()).device
        except Exception:
            return None

    @staticmethod
    def _requested_compute_device() -> str:
        try:
            from pathlib import Path

            import yaml

            cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
            return str((cfg.get("compute", {}) or {}).get("device", "auto"))
        except Exception:
            return "auto"

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()


class GirivinityEngine:
    """Legacy llama.cpp inference wrapper retained for existing callers."""

    def __init__(self, loader: GirivinityLoader | None = None) -> None:
        self._loader = loader or GirivinityLoader.instance()
        self._model = self._loader.get_model()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stream: bool = True,
    ) -> Iterator[str] | str:
        if self._loader.use_native_model:
            engine = LLMEngine(loader=self._loader).load()
            return engine.generate(prompt, max_tokens=max_tokens, stream=stream)
        if stream:
            return self._stream(prompt, max_tokens)
        result = self._model(prompt, max_tokens=max_tokens, stream=False)
        return result["choices"][0]["text"]

    def stream(self, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        return self._stream(prompt, max_tokens)

    def generate_with_context(
        self,
        question: str,
        context: str,
        user_level: int = 1,
        max_tokens: int = 512,
    ) -> Iterator[str] | str:
        prompt = (
            f"User level: {user_level}\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self.generate(prompt, max_tokens=max_tokens, stream=True)

    def _stream(self, prompt: str, max_tokens: int) -> Iterator[str]:
        for chunk in self._model(prompt, max_tokens=max_tokens, stream=True):
            token = chunk["choices"][0].get("text", "")
            if token:
                yield token
