from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterator

import yaml


class GirivinityInference:
    """llama-cpp-python GGUF inference wrapper with streaming support."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        config_path: str | Path = "config.yaml",
        n_ctx: int | None = None,
        n_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else self._model_path_from_config(config_path)
        llama_cpp = importlib.import_module("llama_cpp")
        kwargs: dict[str, object] = {"model_path": str(self.model_path)}
        if n_ctx is not None:
            kwargs["n_ctx"] = n_ctx
        if n_threads is not None:
            kwargs["n_threads"] = n_threads
        self._llm = llama_cpp.Llama(**kwargs)

    def generate(self, prompt: str, max_tokens: int, stream: bool = True) -> str | Iterator[str]:
        completion = self._llm(prompt, max_tokens=max_tokens, stream=stream)
        if stream:
            return self._stream_text(completion)
        return self._collect_text(completion)

    def _stream_text(self, completion) -> Iterator[str]:
        for chunk in completion:
            text = self._extract_text(chunk)
            if text:
                yield text

    def _collect_text(self, completion) -> str:
        if isinstance(completion, dict):
            return self._extract_text(completion)
        return "".join(self._extract_text(chunk) for chunk in completion)

    def _extract_text(self, chunk: dict) -> str:
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        return str(choice.get("text") or choice.get("delta", {}).get("content") or "")

    def _model_path_from_config(self, config_path: str | Path) -> Path:
        path = Path(config_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        raw = raw or {}
        quantisation = (raw.get("model") or {}).get("quantisation") or raw.get("quantisation") or {}
        return Path(quantisation.get("output_model_path", "models/gguf/girivinity-q4_k_m.gguf"))
