"""Girivinity local GGUF inference engine."""

from __future__ import annotations

from typing import Iterator

from llm_loader import GirivinityLoader


class GirivinityEngine:
    def __init__(self, loader: GirivinityLoader) -> None:
        self.model = loader.get_model()

    def generate(self, prompt: str, max_tokens: int = 512, stream: bool = True) -> Iterator[str] | str:
        if stream:
            return self._stream_tokens(prompt, max_tokens=max_tokens)
        result = self.model(prompt, max_tokens=max_tokens)
        return result["choices"][0]["text"]

    def generate_with_context(self, query: str, context: str, user_level: int = 3) -> Iterator[str]:
        prompt = self._prompt_for_level(query=query, context=context, user_level=user_level)
        stream = self.generate(prompt, stream=True)
        if isinstance(stream, str):
            return iter([stream])
        return stream

    def _stream_tokens(self, prompt: str, *, max_tokens: int) -> Iterator[str]:
        for token in self.model(prompt, max_tokens=max_tokens, stream=True):
            choices = token.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text:
                yield text

    def _prompt_for_level(self, *, query: str, context: str, user_level: int) -> str:
        if user_level <= 2:
            return f"Explain simply: {context}\nQuestion: {query}\nAnswer in easy language:"
        if user_level >= 4:
            return f"Technical context: {context}\nQuery: {query}\nDetailed answer:"
        return f"Context: {context}\nQuestion: {query}\nAnswer:"


# Compatibility aliases for older imports. They intentionally use the local Girivinity stack.
LLMEngine = GirivinityEngine
