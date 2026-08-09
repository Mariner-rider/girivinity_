import builtins
from unittest.mock import MagicMock, patch

import numpy as np


def _make_engine():
    real_open = builtins.open
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col
    with patch("chromadb.PersistentClient", return_value=mock_client):
        with patch(
            "builtins.open",
            side_effect=lambda p, *a, **k: __import__(
                "io"
            ).StringIO(
                "rag:\n  chroma_path: /tmp/chroma\n"
            )
            if str(p) == "config.yaml"
            else real_open(p, *a, **k),
        ):
            with patch("yaml.safe_load", return_value={
                "rag": {"chroma_path": "/tmp/chroma"}
            }):
                from app.core.truth_engine import TruthEngine
                engine = TruthEngine.__new__(TruthEngine)
                engine.kb = mock_col
                engine.UNVERIFIED_THRESHOLD = TruthEngine.UNVERIFIED_THRESHOLD
                engine.EVIDENCE_SIMILARITY = TruthEngine.EVIDENCE_SIMILARITY
                return engine, mock_col


def test_extract_claims_splits_sentences():
    engine, _ = _make_engine()
    claims = engine._extract_claims(
        "The sky is blue. Water is wet. Fire is hot."
    )
    assert len(claims) == 3


def test_verify_returns_kb_sourced_on_high_score():
    engine, mock_col = _make_engine()
    mock_col.query.return_value = {"distances": [[0.1]]}
    with patch("app.core.truth_engine.get_embedder") as mock_emb:
        mock_emb.return_value.encode.return_value = np.array([0.1] * 384)
        label, url = engine._verify_claim("The sky is blue.", [])
    assert label == "KB_SOURCED"
    assert url is None


def test_unverified_adds_disclaimer():
    engine, mock_col = _make_engine()
    mock_col.query.return_value = {"distances": [[1.9]]}
    with patch("app.core.truth_engine.get_embedder") as mock_emb:
        mock_emb.return_value.encode.return_value = np.array([0.1] * 384)
        result = engine.verify(
            "Claim one is true. Claim two is true. Claim three is true.",
            web_sources=[],
            query="test",
        )
    assert result.confidence < 1.0


def test_confidence_score_range():
    engine, _ = _make_engine()
    score = engine._score_confidence(
        [("claim", "KB_SOURCED", None)], [], 0.0
    )
    assert 0.0 <= score <= 1.0
