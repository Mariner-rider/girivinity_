"""Knowledge distillation system for compressing crawler output into high-quality structured datasets."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DistilledRecord:
    source_url: str
    summary: str
    key_facts: list[str]
    quality_score: float
    metadata: dict


class KnowledgeDistillationSystem:
    def summarize(self, text: str, max_sentences: int = 2) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return " ".join(sentences[:max_sentences]) if sentences else ""

    def remove_redundancy(self, texts: list[str]) -> list[str]:
        seen = set()
        unique = []
        for text in texts:
            norm = re.sub(r"\s+", " ", text.strip().lower())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            unique.append(text.strip())
        return unique

    def extract_key_facts(self, text: str, max_facts: int = 5) -> list[str]:
        candidates = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        fact_like = [c for c in candidates if re.search(r"\b(is|are|has|have|was|were|\d+%?)\b", c.lower())]
        return fact_like[:max_facts]

    def quality_score(self, summary: str, facts: list[str]) -> float:
        if not summary:
            return 0.0
        length_factor = min(len(summary.split()) / 40.0, 1.0)
        fact_factor = min(len(facts) / 5.0, 1.0)
        diversity = len(set(re.findall(r"[a-zA-Z]+", " ".join(facts).lower())))
        diversity_factor = min(diversity / 30.0, 1.0)
        return round(0.4 * length_factor + 0.4 * fact_factor + 0.2 * diversity_factor, 3)

    def distill(self, crawler_output: list[dict], min_quality: float = 0.35) -> list[DistilledRecord]:
        distilled: list[DistilledRecord] = []
        for doc in crawler_output:
            text = str(doc.get("text", ""))
            source_url = str(doc.get("url", ""))
            summary = self.summarize(text)
            facts = self.extract_key_facts(text)
            facts = self.remove_redundancy(facts)
            score = self.quality_score(summary, facts)

            if score < min_quality:
                continue

            distilled.append(
                DistilledRecord(
                    source_url=source_url,
                    summary=summary,
                    key_facts=facts,
                    quality_score=score,
                    metadata={
                        "title": doc.get("title", ""),
                        "word_count": len(text.split()),
                    },
                )
            )
        return distilled

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
