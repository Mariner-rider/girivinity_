from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from core.web_intelligence import WebSearchPipeline

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Evidence:
    text: str
    score: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class ClaimStatus:
    claim: str
    marker: str
    evidence: Evidence | None = None
    citation_number: int | None = None
    source_url: str | None = None
    flagged: bool = False


@dataclass(slots=True)
class TruthEngineResult:
    response: str
    confidence: float
    claims: list[ClaimStatus]
    sources: list[dict[str, Any]]
    unverified_claims: list[str]
    triggered_web_search: bool


class TruthEngine:
    """Verifies generated responses against current-session and ChromaDB evidence.

    The engine never invents citations: inline citation numbers are created only for
    URLs supplied by current-session web retrieval or by ChromaDB metadata attached
    to retrieved evidence.
    """

    def __init__(
        self,
        *,
        chroma_collection: str = "knowledge_base",
        similarity_threshold: float = 0.7,
        unverified_threshold: float = 0.30,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
    ) -> None:
        self.chroma_collection = chroma_collection
        self.similarity_threshold = similarity_threshold
        self.unverified_threshold = unverified_threshold
        self.top_k = top_k
        self._embedder = None
        self._collection = None
        self._model_name = model_name

    def verify_response(
        self,
        response: str,
        *,
        source: str | None = None,
        session_sources: list[dict[str, Any]] | None = None,
        query: str | None = None,
    ) -> TruthEngineResult:
        claims = self.extract_claims(response)
        session_sources = session_sources or []
        citation_registry: dict[str, int] = {}
        formatted_claims: list[str] = []
        statuses: list[ClaimStatus] = []
        triggered_web_search = False

        for claim in claims:
            evidence = self._find_supporting_evidence(claim)
            if evidence is not None:
                status = self._kb_status(claim, evidence, citation_registry)
            else:
                status = self._fallback_source_status(
                    claim,
                    source=source,
                    session_sources=session_sources,
                    citation_registry=citation_registry,
                )
            statuses.append(status)
            formatted_claims.append(self._format_claim(status))

        unverified = [status.claim for status in statuses if status.marker == "UNVERIFIED"]
        if statuses and (len(unverified) / len(statuses)) > self.unverified_threshold:
            web_sources = self._trigger_web_search(query or " ".join(unverified) or response)
            session_sources.extend(web_sources)
            triggered_web_search = True

        formatted_response = " ".join(formatted_claims).strip()
        if triggered_web_search:
            formatted_response = (
                "I found limited verified information on this. Here is what I retrieved: "
                + formatted_response
            ).strip()

        source_list = self._citation_sources(citation_registry, session_sources, statuses)
        if source_list:
            formatted_response = f"{formatted_response}\n\nSources:\n" + "\n".join(
                f"[{source['citation_number']}] {source['url']}" for source in source_list
            )

        confidence = self._confidence(statuses, source_list)
        return TruthEngineResult(
            response=formatted_response,
            confidence=confidence,
            claims=statuses,
            sources=source_list,
            unverified_claims=unverified,
            triggered_web_search=triggered_web_search,
        )

    def wrap_response(self, response: str, **kwargs: Any) -> TruthEngineResult:
        return self.verify_response(response, **kwargs)

    def extract_claims(self, response: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", response).strip()
        if not cleaned:
            return []
        return [claim.strip() for claim in re.split(r"(?<=[.!?])\s+", cleaned) if claim.strip()]

    def _find_supporting_evidence(self, claim: str) -> Evidence | None:
        try:
            collection = self._get_collection()
            embedding = self._embed_claim(claim)
            result = collection.query(
                query_embeddings=[embedding],
                n_results=self.top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("TruthEngine knowledge-base search failed: %s", exc)
            return None

        docs = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        best: Evidence | None = None
        for index, doc in enumerate(docs):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            score = self._similarity_from_distance(distance, metadata)
            if score > self.similarity_threshold and str(doc).strip():
                evidence = Evidence(text=str(doc), score=score, metadata=metadata)
                if best is None or evidence.score > best.score:
                    best = evidence
        return best

    def _get_collection(self):
        if self._collection is None:
            chromadb = importlib.import_module("chromadb")
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=self.chroma_collection)
        return self._collection

    def _embed_claim(self, claim: str) -> list[float]:
        if self._embedder is None:
            sentence_transformers = importlib.import_module("sentence_transformers")
            self._embedder = sentence_transformers.SentenceTransformer(self._model_name)
        vector = self._embedder.encode(claim, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32).reshape(-1).tolist()

    def _similarity_from_distance(self, distance: Any, metadata: dict[str, Any]) -> float:
        if "similarity" in metadata:
            return float(metadata["similarity"])
        if "score" in metadata:
            return float(metadata["score"])
        if distance is None:
            return 0.0
        value = float(distance)
        if 0.0 <= value <= 2.0:
            return max(0.0, min(1.0, 1.0 - value))
        return 1.0 / (1.0 + max(0.0, value))

    def _kb_status(self, claim: str, evidence: Evidence, citation_registry: dict[str, int]) -> ClaimStatus:
        url = self._metadata_url(evidence.metadata)
        citation_number = None
        if url:
            citation_number = self._register_citation(url, citation_registry)
        return ClaimStatus(
            claim=claim,
            marker="KB_SOURCED",
            evidence=evidence,
            citation_number=citation_number,
            source_url=url,
        )

    def _fallback_source_status(
        self,
        claim: str,
        *,
        source: str | None,
        session_sources: list[dict[str, Any]],
        citation_registry: dict[str, int],
    ) -> ClaimStatus:
        source_type = (source or "").lower()
        if source_type == "web":
            url = self._best_session_url(session_sources)
            citation_number = self._register_citation(url, citation_registry) if url else None
            return ClaimStatus(
                claim=claim,
                marker="WEB_SOURCED" if url else "UNVERIFIED",
                citation_number=citation_number,
                source_url=url,
                flagged=url is None,
            )
        if source_type == "knowledge_base":
            return ClaimStatus(claim=claim, marker="KB_SOURCED")
        return ClaimStatus(claim=claim, marker="UNVERIFIED", flagged=True)

    def _format_claim(self, status: ClaimStatus) -> str:
        if status.marker in {"WEB_SOURCED", "KB_SOURCED"} and status.citation_number is not None:
            return f"{status.claim} [{status.citation_number}]"
        if status.marker == "KB_SOURCED":
            return f"[KB_SOURCED] {status.claim}"
        if status.marker == "UNVERIFIED":
            return f"[UNVERIFIED] {status.claim}"
        return status.claim

    def _trigger_web_search(self, query: str) -> list[dict[str, Any]]:
        try:
            result = WebSearchPipeline(query).run()
        except Exception as exc:
            logger.warning("TruthEngine supplemental web search failed: %s", exc)
            return []
        return [source for source in result.get("sources", []) if source.get("url")]

    def _best_session_url(self, session_sources: list[dict[str, Any]]) -> str | None:
        urls = [source for source in session_sources if source.get("url")]
        if not urls:
            return None
        best = max(urls, key=lambda item: float(item.get("score") or 0.0))
        return str(best["url"])

    def _citation_sources(
        self,
        citation_registry: dict[str, int],
        session_sources: list[dict[str, Any]],
        statuses: list[ClaimStatus],
    ) -> list[dict[str, Any]]:
        session_by_url = {str(source.get("url")): source for source in session_sources if source.get("url")}
        status_by_url = {status.source_url: status for status in statuses if status.source_url}
        sources: list[dict[str, Any]] = []
        for url, number in sorted(citation_registry.items(), key=lambda item: item[1]):
            session_source = session_by_url.get(url, {})
            status = status_by_url.get(url)
            score = session_source.get("score")
            if score is None and status and status.evidence:
                score = status.evidence.score
            timestamp = session_source.get("timestamp")
            if timestamp is None and status and status.evidence:
                timestamp = status.evidence.metadata.get("timestamp")
            sources.append(
                {
                    "citation_number": number,
                    "url": url,
                    "title": session_source.get("title", ""),
                    "score": float(score or 0.0),
                    "timestamp": timestamp,
                }
            )
        return sources

    def _confidence(self, statuses: list[ClaimStatus], sources: list[dict[str, Any]]) -> float:
        if not statuses:
            return 0.0
        verified_ratio = sum(status.marker != "UNVERIFIED" for status in statuses) / len(statuses)
        quality_scores = [float(source.get("score") or 0.0) for source in sources]
        source_quality = sum(quality_scores) / len(quality_scores) if quality_scores else verified_ratio
        recency = self._recency_score(sources)
        confidence = (0.60 * verified_ratio) + (0.25 * source_quality) + (0.15 * recency)
        return max(0.0, min(1.0, confidence))

    def _recency_score(self, sources: list[dict[str, Any]]) -> float:
        dated_scores: list[float] = []
        now = datetime.now(timezone.utc)
        for source in sources:
            timestamp = source.get("timestamp")
            if not timestamp:
                continue
            try:
                parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - parsed).total_seconds() / 86_400)
            dated_scores.append(max(0.0, 1.0 - (age_days / 365.0)))
        return sum(dated_scores) / len(dated_scores) if dated_scores else 0.5

    def _metadata_url(self, metadata: dict[str, Any]) -> str | None:
        for key in ("url", "source_url", "source"):
            value = metadata.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
        return None

    def _register_citation(self, url: str, citation_registry: dict[str, int]) -> int:
        if url not in citation_registry:
            citation_registry[url] = len(citation_registry) + 1
        return citation_registry[url]
