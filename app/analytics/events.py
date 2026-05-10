from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol


@dataclass(slots=True)
class AnalyticsEvent:
    name: str
    payload: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AnalyticsSink(Protocol):
    def emit(self, event: AnalyticsEvent) -> None:
        ...


class InMemoryAnalyticsSink:
    """Small local sink; replace with OTEL/Kafka/warehouse adapters in production."""

    def __init__(self) -> None:
        self.events: list[AnalyticsEvent] = []

    def emit(self, event: AnalyticsEvent) -> None:
        self.events.append(event)
