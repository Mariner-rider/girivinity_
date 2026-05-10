from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path

import faiss  # type: ignore
import numpy as np
import pytest
import requests

from app.crawler.pipeline import KnowledgeIngestionPipeline


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        assert model_name == "all-MiniLM-L6-v2"

    def encode(self, texts, convert_to_numpy: bool = True):
        if isinstance(texts, str):
            texts = [texts]
        vectors = np.array([[1.0, 0.1, 0.2] for _ in texts], dtype=np.float32)
        return vectors if convert_to_numpy else vectors.tolist()


def test_crawler_pipeline_indexes_example_com(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    index_path = tmp_path / "rag.index"
    db_path = tmp_path / "chunks.db"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "crawler:",
                "  seed_urls:",
                "    - 'https://example.com'",
                "  max_depth: 1",
                "rag:",
                f"  index_path: '{index_path}'",
                f"  chunk_db_path: '{db_path}'",
            ]
        ),
        encoding="utf-8",
    )

    pipeline = KnowledgeIngestionPipeline(config_path=cfg_path)
    try:
        chunk_count = pipeline.run()
    except requests.RequestException as exc:
        pytest.skip(f"Network/proxy limitation prevented fetching https://example.com: {exc}")

    assert chunk_count >= 1
    idx = faiss.read_index(str(index_path))
    assert idx.ntotal >= 1

    with sqlite3.connect(db_path) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert row_count >= 1
