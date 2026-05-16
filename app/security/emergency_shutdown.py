from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)



class EmergencyShutdown:
    def execute(self, reason: str, triggered_by: str = "system", threat_details: dict | None = None) -> dict:
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "triggered_by": triggered_by,
            "sessions_killed": 0,
            "api_keys_suspended": 0,
            "mode_set": "emergency",
        }

        try:
            from app.core import db

            row = db.fetchone("SELECT COUNT(*) FROM active_sessions WHERE valid = TRUE")
            session_count = int(row[0]) if row else 0
            db.execute("UPDATE active_sessions SET valid = FALSE WHERE valid = TRUE")
            results["sessions_killed"] = session_count

            row2 = db.fetchone("SELECT COUNT(*) FROM tenant_security_configs")
            key_count = int(row2[0]) if row2 else 0
            db.execute(
                """
                UPDATE tenant_security_configs
                SET rate_limit_rpm = 0,
                    updated_at = NOW()
                WHERE rate_limit_rpm > 0
                """
            )
            results["api_keys_suspended"] = key_count

            db.execute(
                """
                INSERT INTO system_security_mode
                    (mode, triggered_by, set_at)
                VALUES ('emergency', %s, NOW())
                """,
                (triggered_by,),
            )

            db.execute(
                """
                INSERT INTO security_events
                    (user_id, ip_address, event_type, threat_type,
                     severity, detail, blocked, timestamp)
                VALUES (%s, 'system', 'emergency_shutdown',
                        'emergency', 'critical', %s, TRUE, NOW())
                """,
                (triggered_by, f"reason={reason} sessions={session_count}"),
            )

        except Exception as exc:
            logger.critical("Emergency shutdown DB error: %s", exc)

        try:
            alerts_log = Path("logs/security_alerts.jsonl")
            alerts_log.parent.mkdir(exist_ok=True)
            with open(alerts_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({**results, "alert": "EMERGENCY_SHUTDOWN_EXECUTED", "threat_details": threat_details or {}}) + "\n")
        except Exception as exc:
            logger.critical("Emergency alert file write failed: %s", exc)

        logger.critical("EMERGENCY SHUTDOWN: %d sessions killed. Reason: %s", results["sessions_killed"], reason)
        return results

    def resolve(self, resolved_by: str = "admin") -> bool:
        try:
            from app.core import db

            db.execute(
                """
                INSERT INTO system_security_mode
                    (mode, triggered_by, set_at)
                VALUES ('observe', %s, NOW())
                """,
                (resolved_by,),
            )
            db.execute(
                """
                UPDATE tenant_security_configs
                SET rate_limit_rpm = 500,
                    updated_at = NOW()
                WHERE rate_limit_rpm = 0
                """
            )
            logger.info("Emergency resolved by %s. System restored.", resolved_by)
            return True
        except Exception as exc:
            logger.error("Emergency resolve failed: %s", exc)
            return False
