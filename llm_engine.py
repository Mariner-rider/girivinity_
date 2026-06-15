from __future__ import annotations
import logging
from typing import Iterator

from llm_loader import GirivinityLoader, LLMEngineConfig

logger = logging.getLogger(__name__)


class GirivinityEngine:
    def __init__(self, loader: GirivinityLoader | None = None) -> None:
        self._loader = loader or GirivinityLoader.instance()
        self._model = self._loader.get_model()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stream: bool = True,
    ) -> Iterator[str] | str:
        if stream:
            return self._stream(prompt, max_tokens)
        result = self._model(prompt, max_tokens=max_tokens, stream=False)
        return result["choices"][0]["text"]

    def _stream(self, prompt: str, max_tokens: int) -> Iterator[str]:
        for chunk in self._model(prompt, max_tokens=max_tokens, stream=True):
            token = chunk["choices"][0].get("text", "")
            if token:
                yield token
