from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

from llm_loader import GirivinityLoader, LLMEngineConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _NativeGeneration:
    text: str


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
        self._loaded = False

    def load(self) -> "LLMEngine":
        loaded = self._loader.load()
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.use_native_model = loaded.use_native_model
        self._loaded = True
        return self

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stream: bool = False,
        **generation_kwargs: Any,
    ) -> Iterator[str] | str:
        self._ensure_loaded()
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
        if self.use_native_model:
            text = self._generate_native(prompt, max_tokens, **generation_kwargs).text
            if text:
                yield text
            return

        text = self._generate_huggingface(prompt, max_tokens, **generation_kwargs)
        if text:
            yield text

    def _generate_native(
        self,
        prompt: str,
        max_tokens: int,
        **generation_kwargs: Any,
    ) -> _NativeGeneration:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Native model and tokenizer must be loaded before generation")

        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
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

    def _move_inputs_to_model_device(self, encoded: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            return encoded
        device = getattr(self.model, "device", None)
        if device is None:
            return encoded
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

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
