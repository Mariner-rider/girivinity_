from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "models/base"
_DEFAULT_DEVICE_MAP = "auto"
_DEFAULT_TORCH_DTYPE = "float16"
_DEFAULT_LOAD_IN_4BIT = True
_DEFAULT_KV_CACHE = True
_DEFAULT_BASE_MODEL_PATH = "models/base"
_DEFAULT_QUANTISED_PATH = Path("models/girivinity_quantised/model.gguf")
_DEFAULT_N_CTX = 4096
_DEFAULT_N_THREADS = max(1, (os.cpu_count() or 2) - 1)
_DEFAULT_N_GPU_LAYERS = 0
_DEFAULT_USE_NATIVE_MODEL = False
_DEFAULT_NATIVE_MODEL_PATH = ""

_INSTANCE = None
_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}


@dataclass(slots=True)
class ModelConfig:
    """Configuration for loading either a native Girivinity or external model."""

    model_id: str = _DEFAULT_MODEL_ID
    device_map: str = _DEFAULT_DEVICE_MAP
    torch_dtype: str = _DEFAULT_TORCH_DTYPE
    load_in_4bit: bool = _DEFAULT_LOAD_IN_4BIT
    kv_cache: bool = _DEFAULT_KV_CACHE
    base_model_path: str = _DEFAULT_BASE_MODEL_PATH
    quantised_path: Path = _DEFAULT_QUANTISED_PATH
    n_ctx: int = _DEFAULT_N_CTX
    n_threads: int = _DEFAULT_N_THREADS
    n_gpu_layers: int = _DEFAULT_N_GPU_LAYERS
    use_native_model: bool = _DEFAULT_USE_NATIVE_MODEL
    native_model_path: str = _DEFAULT_NATIVE_MODEL_PATH

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config.yaml") -> "ModelConfig":
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        model_cfg = cfg.get("model", {}) or {}
        return cls.from_mapping(model_cfg)

    @classmethod
    def from_mapping(cls, model_cfg: dict[str, Any]) -> "ModelConfig":
        return cls(
            model_id=str(model_cfg.get("model_id", _DEFAULT_MODEL_ID)),
            device_map=str(model_cfg.get("device_map", _DEFAULT_DEVICE_MAP)),
            torch_dtype=str(model_cfg.get("torch_dtype", _DEFAULT_TORCH_DTYPE)),
            load_in_4bit=_as_bool(model_cfg.get("load_in_4bit", _DEFAULT_LOAD_IN_4BIT)),
            kv_cache=_as_bool(model_cfg.get("kv_cache", _DEFAULT_KV_CACHE)),
            base_model_path=str(model_cfg.get("base_model_path", _DEFAULT_BASE_MODEL_PATH)),
            quantised_path=Path(model_cfg.get("quantised_path", _DEFAULT_QUANTISED_PATH)),
            n_ctx=int(model_cfg.get("n_ctx", _DEFAULT_N_CTX)),
            n_threads=int(model_cfg.get("n_threads", _DEFAULT_N_THREADS)),
            n_gpu_layers=int(model_cfg.get("n_gpu_layers", _DEFAULT_N_GPU_LAYERS)),
            use_native_model=_as_bool(
                model_cfg.get("use_native_model", _DEFAULT_USE_NATIVE_MODEL)
            ),
            native_model_path=str(model_cfg.get("native_model_path", _DEFAULT_NATIVE_MODEL_PATH)),
        )

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.model_id,
            self.device_map,
            self.torch_dtype,
            self.load_in_4bit,
            self.kv_cache,
            self.base_model_path,
            str(self.quantised_path),
            self.n_ctx,
            self.n_threads,
            self.n_gpu_layers,
            self.use_native_model,
            self.native_model_path,
        )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(slots=True)
class LLMEngineConfig(ModelConfig):
    """Runtime generation config for LLMEngine.

    Kept separate from ModelConfig so callers can depend on an engine-focused
    name while sharing the same model-loading fields.
    """


GirivinityLoaderConfig = ModelConfig


@dataclass(slots=True)
class LoadedLLM:
    model: Any
    tokenizer: Any | None = None
    use_native_model: bool = False


class GirivinityTokenizerAdapter:
    """Expose GirivinityTokenizer through a HuggingFace-like tokenizer surface."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = self._token_id("[PAD]", default=0)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "GirivinityTokenizerAdapter":
        from app.llm.girivinity_tokenizer import GirivinityTokenizer

        tokenizer = GirivinityTokenizer.from_file(str(path))
        return cls(tokenizer)

    def encode(self, text: str, **_: Any) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: Any, skip_special_tokens: bool = True, **_: Any) -> str:
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu().tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return self.tokenizer.decode(list(ids))

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        input_ids = self.encode(text)
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]

        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([input_ids], dtype=torch.long)}

        if padding:
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        return {"input_ids": input_ids}

    def _token_id(self, token: str, default: int) -> int:
        raw_tokenizer = getattr(self.tokenizer, "tokenizer", None)
        if raw_tokenizer is not None and hasattr(raw_tokenizer, "token_to_id"):
            token_id = raw_tokenizer.token_to_id(token)
            if token_id is not None:
                return int(token_id)
        return default


class GirivinityLoader:
    def __init__(self, config_path: str | Path = "config.yaml", config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_yaml(config_path)
        self.quantised_path = self.config.quantised_path
        self.n_ctx = self.config.n_ctx
        self.n_threads = self.config.n_threads
        self.n_gpu_layers = self.config.n_gpu_layers
        self.use_native_model = self.config.use_native_model
        self.native_model_path = self.config.native_model_path
        self._model = None
        self._loaded_llm: LoadedLLM | None = None

    def load(self) -> LoadedLLM:
        """Load a HuggingFace-compatible or native Girivinity model."""
        if self._loaded_llm is not None:
            return self._loaded_llm

        cache_key = ("native_or_hf", self.config.cache_key())
        if cache_key in _MODEL_CACHE:
            self._loaded_llm = _MODEL_CACHE[cache_key]
            return self._loaded_llm

        if self.use_native_model:
            self._loaded_llm = self._load_native_model()
        else:
            self._loaded_llm = self._load_huggingface_model()
        _MODEL_CACHE[cache_key] = self._loaded_llm
        return self._loaded_llm

    def get_model(self):
        """Return the legacy llama.cpp model used by GirivinityEngine."""
        if self._model is not None:
            return self._model

        if self.use_native_model:
            self._model = self.load().model
            return self._model

        cache_key = ("llama_cpp", self.config.cache_key())
        if cache_key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[cache_key]
            return self._model

        if not self.quantised_path.exists():
            raise FileNotFoundError(
                f"Girivinity model not built yet at {self.quantised_path}. "
                "Build it first: python model/quantise.py"
            )
        from llama_cpp import Llama

        logger.info("Loading model from %s", self.quantised_path)
        self._model = Llama(
            model_path=str(self.quantised_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
        )
        _MODEL_CACHE[cache_key] = self._model
        logger.info("Model loaded successfully")
        return self._model

    def _load_native_model(self) -> LoadedLLM:
        if not self.native_model_path:
            raise ValueError("model.native_model_path is required when use_native_model is true")

        from app.llm.girivinity_architecture import GirivinityModel

        logger.info("Loading native GirivinityModel from %s", self.native_model_path)
        model = GirivinityModel.load_pretrained(self.native_model_path)
        tokenizer = GirivinityTokenizerAdapter.from_pretrained(self.native_model_path)
        logger.info("Native GirivinityModel loaded successfully")
        return LoadedLLM(model=model, tokenizer=tokenizer, use_native_model=True)

    def _load_huggingface_model(self) -> LoadedLLM:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading HuggingFace model", extra={"model_id": self.config.model_id})
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map=self.config.device_map,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=False,
        )
        logger.info("HuggingFace model loaded successfully", extra={"model_id": self.config.model_id})
        return LoadedLLM(model=model, tokenizer=tokenizer, use_native_model=False)

    @classmethod
    def instance(cls) -> "GirivinityLoader":
        global _INSTANCE
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = cls()
        return _INSTANCE

    @classmethod
    def reset_cache(cls) -> None:
        global _INSTANCE
        with _LOCK:
            _INSTANCE = None
            _MODEL_CACHE.clear()
