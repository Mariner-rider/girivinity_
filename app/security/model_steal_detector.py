from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StealDetectionResult:
    is_steal_attempt: bool
    confidence: float
    signals: list[str]
    queries_analysed: int


CAPABILITY_PROBE_PATTERNS = [
    "what can you do", "list your capabilities", "what are you able to", "test your knowledge",
    "how do you handle", "what is your limit", "can you solve", "demonstrate your ability",
    "show me what you know about", "explain how you think", "what topics do you know", "give me your best answer on",
]


class ModelStealDetector:
    def analyse(self, user_id: str, api_key: str) -> StealDetectionResult:
        identifier = api_key if api_key else user_id
        signals = []
        confidence = 0.0
        rows = []

        try:
            from app.core import db

            rows = db.fetchall(
                """
                SELECT query, endpoint, timestamp
                FROM security_events
                WHERE (user_id = %s OR detail LIKE %s)
                  AND event_type = 'request'
                  AND timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC
                LIMIT 500
                """,
                (identifier, f"%{api_key}%"),
            )

            if not rows:
                return StealDetectionResult(False, 0.0, [], 0)

            query_count = len(rows)
            if query_count > 100:
                confidence += 0.3
                signals.append(f"high_volume:{query_count}_queries_per_hour")
            if query_count > 300:
                confidence += 0.2
                signals.append("extreme_volume")

            probe_hits = sum(1 for row in rows if any(p in str(row[0]).lower() for p in CAPABILITY_PROBE_PATTERNS))
            if probe_hits > 5:
                confidence += 0.25
                signals.append(f"capability_probing:{probe_hits}_probe_queries")

            avg_len = sum(len(str(r[0])) for r in rows) / query_count if query_count else 0
            if avg_len < 30:
                confidence += 0.15
                signals.append(f"short_structured_queries:avg_len={avg_len:.0f}")

        except Exception as exc:
            logger.warning("StealDetector DB error: %s", exc)

        is_steal = confidence >= 0.5
        if is_steal:
            logger.warning("Model steal attempt: identifier=%s confidence=%.2f", identifier, confidence)

        return StealDetectionResult(is_steal, min(1.0, round(confidence, 3)), signals, len(rows))
