from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import yaml

from app.core.query_router import get_embedder

logger = logging.getLogger(__name__)


@dataclass
class VerifiedResponse:
    text: str
    confidence: float
    sources: list[dict] = field(default_factory=list)
    unverified_count: int = 0
    total_claims: int = 0
    disclaimer: str = ""


class TruthEngine:
    UNVERIFIED_THRESHOLD = 0.30
    EVIDENCE_SIMILARITY = 0.70

    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        chroma_path: str = cfg["rag"]["chroma_path"]
        client = chromadb.PersistentClient(path=chroma_path)
        self.kb = client.get_or_create_collection("girivinity_knowledge")

    def verify(
        self,
        response_text: str,
        web_sources: list[dict] | None = None,
        query: str = "",
    ) -> VerifiedResponse:
        claims = self._extract_claims(response_text)
        if not claims:
            return VerifiedResponse(
                text=response_text,
                confidence=1.0,
                sources=web_sources or [],
            )

        labelled: list[tuple[str, str, str | None]] = []
        for claim in claims:
            label, url = self._verify_claim(claim, web_sources)
            labelled.append((claim, label, url))

        unverified = [c for c in labelled if c[1] == "UNVERIFIED"]
        unverified_ratio = len(unverified) / len(labelled)

        final_text, inline_sources = self._format_citations(
            response_text, labelled, web_sources or []
        )

        disclaimer = ""
        if unverified_ratio > self.UNVERIFIED_THRESHOLD:
            disclaimer = (
                "I found limited verified information on this topic. "
                "The following is based on retrieved sources and "
                "may be incomplete:\n\n"
            )

        confidence = self._score_confidence(labelled, web_sources or [], unverified_ratio)

        return VerifiedResponse(
            text=disclaimer + final_text,
            confidence=round(confidence, 3),
            sources=inline_sources,
            unverified_count=len(unverified),
            total_claims=len(labelled),
            disclaimer=disclaimer,
        )

    def _extract_claims(self, text: str) -> list[str]:
        """Split response into individual factual sentences."""
        text = text.strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []
        for s in sentences:
            s = s.strip()
            if (
                len(s) > 8
                and not s.startswith("Source")
                and not s.startswith("[")
                and not s.startswith("http")
            ):
                claims.append(s)
        return claims

    def _verify_claim(self, claim: str, web_sources: list[dict] | None) -> tuple[str, str | None]:
        embedder = get_embedder()
        vec = embedder.encode(claim).tolist()

        try:
            results = self.kb.query(
                query_embeddings=[vec],
                n_results=1,
                include=["distances"],
            )
            distances = results["distances"][0] if results["distances"] else []
            if distances:
                score = max(0.0, 1.0 - distances[0] / 2.0)
                if score >= self.EVIDENCE_SIMILARITY:
                    return "KB_SOURCED", None
        except Exception as exc:
            logger.warning("KB verification failed: %s", exc)

        if web_sources:
            for src in web_sources:
                url = src.get("url", "")
                src_score = src.get("score", 0.0)
                if src_score >= 0.45:
                    return "WEB_SOURCED", url

        return "UNVERIFIED", None

    def _format_citations(
        self,
        original_text: str,
        labelled: list[tuple[str, str, str | None]],
        web_sources: list[dict],
    ) -> tuple[str, list[dict]]:
        url_to_index: dict[str, int] = {}
        citation_counter = 1
        inline_sources: list[dict] = []

        for _claim, label, url in labelled:
            if label == "WEB_SOURCED" and url and url not in url_to_index:
                url_to_index[url] = citation_counter
                inline_sources.append(
                    {
                        "index": citation_counter,
                        "url": url,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                citation_counter += 1

        if not inline_sources:
            return original_text, inline_sources

        sources_section = "\n\nSources:"
        for src in inline_sources:
            sources_section += f"\n  [{src['index']}] {src['url']}"

        return original_text + sources_section, inline_sources

    def _score_confidence(
        self,
        labelled: list[tuple[str, str, str | None]],
        web_sources: list[dict],
        unverified_ratio: float,
    ) -> float:
        if not labelled:
            return 0.5

        verified_ratio = 1.0 - unverified_ratio

        recency_bonus = 0.0
        now = datetime.now(timezone.utc)
        for src in web_sources[:3]:
            ts = src.get("timestamp", "")
            if ts:
                try:
                    age_days = (now - datetime.fromisoformat(ts)).days
                    if age_days <= 7:
                        recency_bonus = 0.05
                        break
                except Exception:
                    pass

        raw = (verified_ratio * 0.8) + recency_bonus
        return min(1.0, max(0.0, raw))
