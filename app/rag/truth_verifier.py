from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class RetrievedEvidence:
    document_id: str
    text: str
    score: float
    metadata: dict


class EvidenceSearcher(Protocol):
    def search(self, claim: str, top_k: int = 3) -> list[RetrievedEvidence]:
        ...


@dataclass(slots=True)
class VerificationResult:
    verified_response: str
    citation_mapping: dict[str, list[str]]


class TruthVerificationLayer:
    def __init__(self, searcher: EvidenceSearcher, min_confidence: float = 0.65) -> None:
        self.searcher = searcher
        self.min_confidence = min_confidence

    def extract_claims(self, response: str) -> list[str]:
        claims = [c.strip() for c in re.split(r"(?<=[.!?])\s+", response) if c.strip()]
        return claims

    def validate_source_existence(self, evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
        return [e for e in evidence if e.document_id and e.text.strip()]

    def verify(self, response: str, top_k: int = 3) -> VerificationResult:
        claims = self.extract_claims(response)
        verified_claims: list[str] = []
        citation_mapping: dict[str, list[str]] = {}

        for claim in claims:
            evidence = self.searcher.search(claim, top_k=top_k)
            evidence = self.validate_source_existence(evidence)

            # no source -> remove claim
            if not evidence:
                continue

            best = max(evidence, key=lambda e: e.score)
            citations = [f"source:{item.document_id}" for item in evidence]

            # low confidence -> mark uncertain
            if best.score < self.min_confidence:
                verified_claims.append(f"[UNCERTAIN] {claim}")
            else:
                verified_claims.append(claim)

            citation_mapping[claim] = citations

        verified_response = " ".join(verified_claims).strip()
        return VerificationResult(verified_response=verified_response, citation_mapping=citation_mapping)
