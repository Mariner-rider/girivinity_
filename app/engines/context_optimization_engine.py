"""Context optimization system to maximize useful tokens for LLM prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class ContextChunk:
    text: str
    source: str = ""


@dataclass(slots=True)
class OptimizedPrompt:
    prompt: str
    selected_chunks: list[str]
    dropped_chunks: int


class ContextOptimizationSystem:
    def rank_context_relevance(self, query: str, chunks: list[ContextChunk]) -> list[tuple[float, ContextChunk]]:
        q_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        scored: list[tuple[float, ContextChunk]] = []
        for chunk in chunks:
            c_tokens = set(re.findall(r"[a-zA-Z0-9]+", chunk.text.lower()))
            overlap = len(q_tokens & c_tokens)
            density = overlap / max(len(c_tokens), 1)
            score = (0.7 * overlap) + (0.3 * density)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def compress_context(self, text: str, max_sentences: int = 2) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return " ".join(sentences[:max_sentences])

    def remove_noise(self, text: str) -> str:
        cleaned = re.sub(r"\b(lorem ipsum|click here|subscribe now|advertisement)\b", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def optimize(self, query: str, context_chunks: list[ContextChunk], max_chars: int = 1200) -> OptimizedPrompt:
        ranked = self.rank_context_relevance(query, context_chunks)
        selected: list[str] = []
        total_chars = 0
        for _, chunk in ranked:
            denoised = self.remove_noise(chunk.text)
            compressed = self.compress_context(denoised)
            if not compressed:
                continue
            if total_chars + len(compressed) > max_chars:
                continue
            selected.append(compressed)
            total_chars += len(compressed)

        dropped = max(0, len(context_chunks) - len(selected))
        prompt = (
            f"User Query: {query}\n\n"
            "Relevant Context:\n"
            + "\n".join(f"- {chunk}" for chunk in selected)
            + "\n\nUse only the context above. If insufficient, state uncertainty."
        )
        return OptimizedPrompt(prompt=prompt, selected_chunks=selected, dropped_chunks=dropped)
