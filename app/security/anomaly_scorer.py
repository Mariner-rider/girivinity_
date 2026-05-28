from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    score: float
    signals: list[str]
    user_id: str


class AnomalyScorer:
    def score(
        self,
        user_id: str,
        ip: str,
        query: str,
        endpoint: str,
    ) -> AnomalyResult:
        signals: list[str] = []
        score = 0.0

        try:
            from app.core import db
            rows = db.fetchall(
                """
                SELECT ip_address, endpoint, hour_of_day
                FROM security_events
                WHERE user_id = %s
                  AND event_type = 'request'
                ORDER BY timestamp DESC LIMIT 100
                """,
                (user_id,),
            )
            if rows:
                known_ips       = {r[0] for r in rows}
                known_endpoints = {r[1] for r in rows}
                usual_hours     = [r[2] for r in rows if r[2] is not None]

                if ip and ip not in known_ips:
                    score += 0.3
                    signals.append(f"new_ip:{ip}")

                if endpoint not in known_endpoints:
                    score += 0.15
                    signals.append(f"new_endpoint:{endpoint}")

                if usual_hours:
                    current_hour = datetime.now(timezone.utc).hour
                    avg_hour     = sum(usual_hours) / len(usual_hours)
                    if abs(current_hour - avg_hour) > 8:
                        score += 0.2
                        signals.append(f"unusual_hour:{current_hour}")

            db.execute(
                """
                INSERT INTO security_events
                    (user_id, ip_address, endpoint,
                     event_type, hour_of_day, timestamp)
                VALUES (%s, %s, %s, 'request', %s, NOW())
                """,
                (
                    user_id, ip, endpoint,
                    datetime.now(timezone.utc).hour,
                ),
            )
        except Exception as exc:
            logger.warning("AnomalyScorer error: %s", exc)

        return AnomalyResult(
            score=min(1.0, round(score, 3)),
            signals=signals,
            user_id=user_id,
        )
