from __future__ import annotations
import logging
import os
import threading
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_INSTANCE = None
_LOCK = threading.Lock()

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
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        model_cfg = cfg.get("model", {})
        self.quantised_path = Path(
            model_cfg.get(
                "quantised_path",
                "models/girivinity_quantised/model.gguf",
            )
        )
        self.n_ctx = int(model_cfg.get("n_ctx", 4096))
        self.n_threads = int(
            model_cfg.get("n_threads", max(1, os.cpu_count() - 1))
        )
        self.n_gpu_layers = int(model_cfg.get("n_gpu_layers", 0))
        self._model = None

    def get_model(self):
        if self._model is None:
            if not self.quantised_path.exists():
                raise FileNotFoundError(
                    f"Girivinity model not found at {self.quantised_path}. "
                    "Build it first: python model/quantise.py"
                )
            from llama_cpp import Llama

            logger.info("Loading model from %s", self.quantised_path)
            self._model = Llama(
                model_path=str(self.quantised_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            logger.info("Model loaded successfully")
        return self._model

    @classmethod
    def instance(cls) -> "GirivinityLoader":
        global _INSTANCE
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = cls()
        return _INSTANCE
