from __future__ import annotations
from typing import Iterator
import logging
from pathlib import Path
import yaml

from app.core.truth_engine import TruthEngine
from app.core.citation_engine import CitationEngine
from app.core.teaching_engine import TeachingEngine
from app.core.domain_router import DomainRouter

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
        user_id: str = "anonymous",
    ) -> str | Iterator[str]:
        engine = get_engine()

        if engine is None:
            raw = self._extraction_fallback(query, context, urls)
        else:
            prompt = self._build_prompt(
                query, context, user_id, web_sources
            )
            try:
                if stream:
                    return self._stream_with_sources(
                        engine, prompt, urls, web_sources or []
                    )
                raw = engine.generate(prompt, max_tokens=512, stream=False)
                if not isinstance(raw, str):
                    raw = "".join(raw)
            except Exception as exc:
                logger.error("LLM synthesis failed: %s", exc)
                raw = self._extraction_fallback(query, context, urls)

        try:
            if web_sources:
                citations = CitationEngine().generate_citations(
                    web_sources
                )
                if citations:
                    citation_block = CitationEngine(
                    ).format_citations_block(citations, style="apa")
                    raw = raw + citation_block
        except Exception as exc:
            logger.warning("CitationEngine failed: %s", exc)

        try:
            verified = TruthEngine().verify(
                response_text=raw,
                web_sources=web_sources or [],
                query=query,
            )
            return verified.text
        except Exception as exc:
            logger.warning("TruthEngine failed: %s", exc)
            return raw

    def _stream_with_sources(
        self,
        engine,
        prompt: str,
        urls: list[str],
        web_sources: list[dict],
    ) -> Iterator[str]:
        yield from engine.generate(prompt, max_tokens=512, stream=True)
        if urls:
            yield "\n\nSources:"
            for i, url in enumerate(urls[:3], 1):
                yield f"\n  [{i}] {url}"

    def _build_prompt(
        self,
        query: str,
        context: str,
        user_id: str = "anonymous",
        web_sources: list[dict] | None = None,
    ) -> str:
        parts = [SYSTEM_PROMPT]

        try:
            domain_match = DomainRouter().route(query)
            if domain_match.domain_prompt:
                parts.append(
                    f"\nDomain expertise: {domain_match.domain_prompt}"
                )
        except Exception as exc:
            logger.warning("DomainRouter failed: %s", exc)

        try:
            teaching_injection = TeachingEngine().get_prompt_injection(
                query, user_id
            )
            if teaching_injection:
                parts.append(f"\nTeaching mode: {teaching_injection}")
        except Exception as exc:
            logger.warning("TeachingEngine failed: %s", exc)

        try:
            from app.core.skill_forge import SkillForge
            skill = SkillForge().get_skill_for_query(query)
            if skill:
                parts.append(f"\n{skill.to_prompt_block()}")
        except Exception as exc:
            logger.warning("SkillForge lookup failed: %s", exc)

        if context:
            parts.append(f"\nContext:\n{context}")

        parts.append(f"\nQuestion: {query}\n\nAnswer:")

        return "\n".join(parts)

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
