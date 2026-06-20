from __future__ import annotations
from typing import Iterator
import logging
from pathlib import Path
import yaml

from app.core.truth_engine import TruthEngine

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
        web_sources: list[dict] | None = None,
    ) -> str | Iterator[str]:
        engine = get_engine()

        if engine is None:
            raw = self._extraction_fallback(query, context, urls)
        else:
            prompt = self._build_prompt(query, context)
            try:
                if stream:
                    # Stream bypasses truth engine (real-time)
                    return self._stream_with_sources(
                        engine, prompt, urls
                    )
                raw = engine.generate(prompt, max_tokens=512, stream=False)
                if not isinstance(raw, str):
                    raw = "".join(raw)
            except Exception as exc:
                logger.error("LLM synthesis failed: %s", exc)
                raw = self._extraction_fallback(query, context, urls)

        # Run truth verification on non-streaming responses
        try:
            verified = TruthEngine().verify(
                response_text=raw,
                web_sources=web_sources or [],
                query=query,
            )
            return verified.text
        except Exception as exc:
            logger.warning("TruthEngine failed, returning raw: %s", exc)
            return raw

    def _stream_with_sources(
        self,
        engine,
        prompt: str,
        urls: list[str],
    ) -> Iterator[str]:
        yield from engine.generate(prompt, max_tokens=512, stream=True)
        if urls:
            yield "\n\nSources:"
            for i, url in enumerate(urls[:3], 1):
                yield f"\n  [{i}] {url}"

    def _build_prompt(self, query: str, context: str) -> str:
        skill_block = ""
        try:
            from app.core.skill_forge import SkillForge
            skill = SkillForge().get_skill_for_query(query)
            if skill:
                skill_block = f"\n\n{skill.to_prompt_block()}"
        except Exception as exc:
            logger.warning("SkillForge lookup failed: %s", exc)
        return (
            f"{SYSTEM_PROMPT}"
            f"{skill_block}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

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
