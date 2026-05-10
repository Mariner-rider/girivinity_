from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MultimodalPayload:
    text: str | None = None
    image_paths: list[str] = field(default_factory=list)
    audio_paths: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class MultimodalProcessor:
    """Normalizes text/image/audio payloads before routing into RAG or agents."""

    def normalize(self, payload: MultimodalPayload) -> dict:
        return {
            "text": (payload.text or "").strip(),
            "image_paths": list(payload.image_paths),
            "audio_paths": list(payload.audio_paths),
            "metadata": dict(payload.metadata),
        }
