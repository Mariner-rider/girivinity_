from app.rag.truth_verifier import RetrievedEvidence, TruthVerificationLayer


class FakeSearcher:
    def search(self, claim: str, top_k: int = 3) -> list[RetrievedEvidence]:
        _ = top_k
        if "Earth is flat" in claim:
            return []
        if "Mars has water" in claim:
            return [RetrievedEvidence(document_id="doc-low", text="possible traces", score=0.4, metadata={})]
        return [
            RetrievedEvidence(document_id="doc-1", text="verified fact", score=0.91, metadata={}),
            RetrievedEvidence(document_id="doc-2", text="supporting fact", score=0.88, metadata={}),
        ]


def test_truth_verifier_removes_claims_without_sources_and_marks_uncertain():
    verifier = TruthVerificationLayer(searcher=FakeSearcher(), min_confidence=0.65)
    response = "Paris is the capital of France. Earth is flat. Mars has water."

    result = verifier.verify(response)

    assert "Paris is the capital of France." in result.verified_response
    assert "Earth is flat" not in result.verified_response
    assert "[UNCERTAIN] Mars has water." in result.verified_response

    assert "Paris is the capital of France." in result.citation_mapping
    assert "Earth is flat." not in result.citation_mapping
    assert any(src.startswith("source:") for src in result.citation_mapping["Paris is the capital of France."])
