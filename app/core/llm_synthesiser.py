from __future__ import annotations
from typing import Iterator
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)
_ = (Path, yaml)

_ENGINE = None
_ENGINE_LOCK = __import__("threading").Lock()


def get_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = _load_engine()
    return _ENGINE


def _load_engine():
    try:
        from llm_loader import GirivinityLoader
        from llm_engine import GirivinityEngine

        loader = GirivinityLoader()
        return GirivinityEngine(loader)
    except FileNotFoundError:
        logger.warning(
            "Girivinity quantised model not found. "
            "Falling back to extraction-only mode."
        )
        return None
    except Exception as exc:
        logger.error("LLM load failed: %s", exc)
        return None


SYSTEM_PROMPT = (
    "You are Girivinity, an intelligent AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Be clear, accurate, and concise. "
    "If the context does not contain enough information, say so honestly. "
    "Never fabricate facts not present in the context."
)


class LLMSynthesiser:
    def synthesise(
        self,
        query: str,
        context: str,
        urls: list[str],
        stream: bool = False,
    ) -> str | Iterator[str]:
        engine = get_engine()

        if engine is None:
            return self._extraction_fallback(query, context, urls)

        prompt = self._build_prompt(query, context)

        try:
            if stream:
                return engine.generate(prompt, max_tokens=512, stream=True)
            return engine.generate(prompt, max_tokens=512, stream=False)
        except Exception as exc:
            logger.error("LLM synthesis failed: %s", exc)
            return self._extraction_fallback(query, context, urls)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"{SYSTEM_PROMPT}\n\n" f"Context:\n{context}\n\n" f"Question: {query}\n\n" f"Answer:"

    def _extraction_fallback(self, query: str, context: str, urls: list[str]) -> str:
        """Used when no LLM is loaded. Returns clean structured answer."""
        if not context:
            return (
                "I could not find verified information on this topic. "
                "The system is searching and will learn more about it shortly."
            )
        lines = [
            f"Based on retrieved information about '{query}':\n",
            context,
        ]
        if urls:
            lines.append("\nSources:")
            for i, url in enumerate(urls[:3], 1):
                lines.append(f"  [{i}] {url}")
        return "\n".join(lines)
