from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


class WebSearchPipeline:
    """Web retrieval pipeline used when knowledge base misses."""

    def __init__(self, query: str) -> None:
        self.query = query

    def fetch(self) -> list[str]:
        import requests

        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": self.query},
            timeout=10,
        )
        response.raise_for_status()
        text = response.text
        if not text.strip():
            return []
        return [text[:1200]]


@dataclass(slots=True)
class SelfTrainer:
    chunks: list[str]

    def queue_for_training(self) -> None:
        from pathlib import Path
        import json

        out = Path("data/training_queue.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"chunks": self.chunks}) + "\n")


class QueryRouter:
    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "sentence-transformers is required. Install with `pip install sentence-transformers`."
            ) from exc

        try:
            import chromadb  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("chromadb is required. Install with `pip install chromadb`.") from exc

        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._chroma_client = chromadb.Client()
        self._collection = self._chroma_client.get_or_create_collection(name="knowledge_base")

    def _embed_query(self, query: str) -> list[float]:
        vec = self._embedder.encode(query, convert_to_numpy=True)
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        return arr.tolist()

    def _score_from_distance(self, distance: float) -> float:
        return 1.0 / (1.0 + max(0.0, float(distance)))

    def route(self, query: str) -> dict[str, Any]:
        embedding = self._embed_query(query)
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=5,
            include=["documents", "distances"],
        )

        docs = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        filtered_chunks: list[str] = []
        for doc, distance in zip(docs, distances):
            score = self._score_from_distance(distance)
            if score >= self.threshold:
                filtered_chunks.append(str(doc))

        if filtered_chunks:
            return {"source": "knowledge_base", "chunks": filtered_chunks}

        web_chunks = WebSearchPipeline(query).fetch()
        response = {"source": "web", "chunks": web_chunks, "trigger_training": True}

        thread = threading.Thread(
            target=SelfTrainer(web_chunks).queue_for_training,
            daemon=True,
            name="self-trainer-queue",
        )
        thread.start()
        return response
