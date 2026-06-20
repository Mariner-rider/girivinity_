"""Production-grade knowledge distillation for crawler output."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DistilledRecord:
    source_url: str
    summary: str
    key_facts: list[str]
    quality_score: float
    metadata: dict


class _FallbackLLM:
    """Small deterministic fallback used in tests and minimal local environments."""

    def generate(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.05) -> Any:
        content = prompt.split("Content:", 1)[-1].split("Summary:", 1)[0].strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
        summary = " ".join(sentences[:3])
        return type("LLMResult", (), {"text": summary})()


class _HashingEmbedder:
    """Dependency-free normalized hashing embedder for graceful fallback."""

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * 256
            for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
                vector[hash(token) % len(vector)] += 1.0
            if normalize_embeddings:
                norm = math.sqrt(sum(value * value for value in vector)) or 1.0
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors


class KnowledgeDistillationSystem:
    """
    Production-grade distillation pipeline:
      1. trafilatura.extract() for HTML denoising.
      2. spaCy NER to extract named entities.
      3. LLM-generated 2-3 sentence factual summaries.
      4. Embedding cosine semantic deduplication above a 0.92 threshold.
      5. Multi-factor quality scoring for density, entities, source authority, and claims.
      6. Domain classification metadata for filtered RAG retrieval.
    """

    def __init__(self, llm_engine: Any | None = None, embedder: Any | None = None, nlp_model: str = "en_core_web_sm"):
        self.llm = llm_engine or _FallbackLLM()
        self.embedder = embedder or _HashingEmbedder()
        self.nlp_model = nlp_model
        self.nlp = self._load_spacy_model(nlp_model)
        self._dedup_threshold = 0.92

    def _load_spacy_model(self, nlp_model: str) -> Any | None:
        try:
            import spacy

            try:
                return spacy.load(nlp_model)
            except OSError:
                logger.warning("spaCy model %s is not installed; using blank English NER fallback", nlp_model)
                nlp = spacy.blank("en")
                ruler = nlp.add_pipe("entity_ruler")
                ruler.add_patterns(
                    [
                        {"label": "PRODUCT", "pattern": [{"TEXT": {"REGEX": r"CVE-\d{4}-\d{4,}"}}]},
                        {"label": "ORG", "pattern": "OpenAI"},
                        {"label": "ORG", "pattern": "Microsoft"},
                        {"label": "ORG", "pattern": "Google"},
                    ]
                )
                return nlp
        except Exception as exc:  # pragma: no cover - only used when spaCy is unavailable.
            logger.warning("spaCy unavailable; falling back to regex entity extraction: %s", exc)
            return None

    def extract_clean_text(self, html_or_text: str, url: str = "") -> str:
        """Use trafilatura for HTML. Fall back to raw text if extraction fails or input is plain text."""
        if not html_or_text:
            return ""
        try:
            import trafilatura

            extracted = trafilatura.extract(
                html_or_text,
                url=url or None,
                include_comments=False,
                include_tables=True,
            )
            cleaned = extracted or html_or_text[:5000]
        except Exception as exc:  # pragma: no cover - parser/runtime dependent.
            logger.warning("trafilatura extraction failed; using raw text fallback: %s", exc)
            cleaned = html_or_text[:5000]
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def llm_summarize(self, text: str, domain_hint: str = "general") -> str:
        """Generate a 2-3 sentence factual summary with the configured LLM engine."""
        prompt = f"""Summarize the following {domain_hint} content in exactly 2-3 factual sentences.
Include specific numbers, names, dates, or technical terms if present.
Do not add opinions or speculation. Write only the summary.

Content:
{text[:2500]}

Summary:"""
        result = self.llm.generate(prompt, max_new_tokens=150, temperature=0.05)
        if isinstance(result, str):
            return result.strip()
        return str(getattr(result, "text", result)).strip()

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Backward-compatible summary helper used by older callers/tests."""
        summary = self.llm_summarize(text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        return " ".join(sentences[:max_sentences])

    def extract_named_entities(self, text: str) -> list[str]:
        """Extract people, organizations, locations, products, laws, events, and CVE identifiers."""
        allowed_labels = {"ORG", "PERSON", "GPE", "PRODUCT", "EVENT", "LAW", "LOC", "NORP"}
        entities: set[str] = set(re.findall(r"\bCVE-\d{4}-\d{4,}\b", text, flags=re.IGNORECASE))
        if self.nlp is not None:
            doc = self.nlp(text[:5000])
            entities.update(ent.text.strip() for ent in doc.ents if ent.label_ in allowed_labels and ent.text.strip())
        else:
            entities.update(re.findall(r"\b[A-Z][a-zA-Z0-9&.-]*(?:\s+[A-Z][a-zA-Z0-9&.-]*){0,3}\b", text[:1000]))
        return sorted(entities)

    def extract_key_facts(self, text: str, max_facts: int = 5) -> list[str]:
        """Backward-compatible fact extraction; returns named entities first, then factual sentences."""
        facts = self.extract_named_entities(text)[:max_facts]
        if len(facts) >= max_facts:
            return facts
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        for sentence in sentences:
            if re.search(r"\b(is|are|has|have|was|were|\d+%?|20\d{2}|19\d{2})\b", sentence, re.IGNORECASE):
                facts.append(sentence)
            if len(facts) >= max_facts:
                break
        return self.remove_redundancy(facts)

    def classify_domain(self, text: str) -> str:
        """Classify distilled content for filtered RAG retrieval."""
        text_lower = text.lower()
        if any(k in text_lower for k in ["cve", "vulnerability", "exploit", "malware", "threat", "ransomware"]):
            return "cybersecurity"
        if any(k in text_lower for k in ["revenue", "earnings", "market cap", "stock", "investment"]):
            return "finance"
        if any(k in text_lower for k in ["study", "research", "clinical trial", "gene", "protein", "algorithm"]):
            return "science"
        return "general"

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=False))

    @staticmethod
    def _as_vectors(embeddings: Any) -> list[list[float]]:
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(row) for row in embeddings]

    def semantic_deduplicate(self, texts: list[str]) -> list[str]:
        """Remove near-duplicate texts by normalized embedding cosine similarity."""
        if len(texts) <= 1:
            return texts
        embeddings = self._as_vectors(self.embedder.encode(texts, normalize_embeddings=True))
        keep: list[str] = []
        kept_vectors: list[list[float]] = []
        for text, vector in zip(texts, embeddings, strict=False):
            if any(self._dot(vector, kept) > self._dedup_threshold for kept in kept_vectors):
                continue
            keep.append(text)
            kept_vectors.append(vector)
        return keep

    def remove_redundancy(self, texts: list[str]) -> list[str]:
        """Backward-compatible redundancy removal wrapper around semantic deduplication."""
        normalized: list[str] = []
        seen: set[str] = set()
        for text in texts:
            key = re.sub(r"\s+", " ", text.strip().lower())
            if key and key not in seen:
                normalized.append(text.strip())
                seen.add(key)
        return self.semantic_deduplicate(normalized)

    @staticmethod
    def _source_authority(source_url: str) -> float:
        parsed = urlparse(source_url)
        host = parsed.netloc.lower()
        authority = 0.0
        if parsed.scheme == "https":
            authority += 0.5
        if any(marker in host for marker in [".gov", ".edu", ".ac.", "nist.gov", "mitre.org"]):
            authority += 0.5
        return min(authority, 1.0)

    @staticmethod
    def _factual_claim_density(summary: str) -> float:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        if not sentences:
            return 0.0
        claim_sentences = 0
        for sentence in sentences:
            has_number = bool(re.search(r"\b\d+[\d,.%]*\b", sentence))
            has_date = bool(re.search(r"\b(20\d{2}|19\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", sentence.lower()))
            has_proper_noun = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence))
            if has_number or has_date or has_proper_noun:
                claim_sentences += 1
        return claim_sentences / len(sentences)

    def quality_score(self, summary: str, entities: list, source_url: str = "") -> float:
        """Return a 0.0-1.0 multi-factor quality score."""
        if not summary:
            return 0.0
        words = summary.split()
        unique_terms = len(set(re.findall(r"[a-zA-Z0-9]+", summary.lower())))
        information_density = unique_terms / max(len(words), 1)
        length_score = min(len(words) / 40.0, 1.0) * min(information_density * 1.5, 1.0) * 0.25
        entity_score = min(len(entities) / 5.0, 1.0) * 0.30
        factual_score = self._factual_claim_density(summary) * 0.25
        authority_score = self._source_authority(source_url) * 0.20
        return round(min(length_score + entity_score + factual_score + authority_score, 1.0), 3)

    def distill(self, crawler_output: list[dict], min_quality: float = 0.35) -> list[DistilledRecord]:
        """Full pipeline with LLM summarization, NER, semantic deduplication, and quality gating."""
        summaries: list[str] = []
        meta_list: list[dict[str, Any]] = []
        for doc in crawler_output:
            source_url = str(doc.get("url", ""))
            text = self.extract_clean_text(str(doc.get("text", "")), source_url)
            if len(text) < 40:
                continue
            domain = self.classify_domain(text)
            summary = self.llm_summarize(text, domain_hint=domain)
            entities = self.extract_named_entities(text)
            summaries.append(summary)
            meta_list.append({"doc": doc, "clean_text": text, "entities": entities, "domain": domain})

        unique_summaries = set(self.semantic_deduplicate(summaries))
        result: list[DistilledRecord] = []
        for summary, meta in zip(summaries, meta_list, strict=False):
            if summary not in unique_summaries:
                continue
            doc = meta["doc"]
            source_url = str(doc.get("url", ""))
            score = self.quality_score(summary, meta["entities"], source_url)
            if score < min_quality:
                continue
            result.append(
                DistilledRecord(
                    source_url=source_url,
                    summary=summary,
                    key_facts=meta["entities"],
                    quality_score=score,
                    metadata={
                        "title": doc.get("title", ""),
                        "domain": meta["domain"],
                        "word_count": len(str(doc.get("text", "")).split()),
                    },
                )
            )
        return result

    def store_structured(self, records: list[DistilledRecord], output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as out:
            for row in records:
                out.write(
                    json.dumps(
                        {
                            "source_url": row.source_url,
                            "summary": row.summary,
                            "key_facts": row.key_facts,
                            "quality_score": row.quality_score,
                            "metadata": row.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return str(path)
