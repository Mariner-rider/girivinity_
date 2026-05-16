from __future__ import annotations
import logging
import secrets

logger = logging.getLogger(__name__)


class SessionManager:
    def create_session(
        self, user_id: str, ip: str, api_key: str = ""
    ) -> str:
        token = secrets.token_urlsafe(32)
        try:
            from app.core import db
            db.execute(
                """
                INSERT INTO active_sessions
                    (token, user_id, ip_address, api_key,
                     created_at, expires_at, valid)
                VALUES (%s, %s, %s, %s, NOW(),
                        NOW() + INTERVAL '24 hours', TRUE)
                """,
                (token, user_id, ip, api_key),
            )
        except Exception as exc:
            logger.warning("Session create failed: %s", exc)
        return token

    def validate_session(self, token: str) -> dict | None:
        try:
            from app.core import db
            row = db.fetchone(
                """
                SELECT user_id, ip_address, api_key
                FROM active_sessions
                WHERE token = %s
                  AND valid = TRUE
                  AND expires_at > NOW()
                """,
                (token,),
            )
            if row:
                return {
                    "user_id": row[0],
                    "ip":      row[1],
                    "api_key": row[2],
                }
        except Exception as exc:
            logger.warning("Session validate failed: %s", exc)
        return None

    def invalidate_session(self, token: str) -> None:
        try:
            from app.core import db
            db.execute(
                "UPDATE active_sessions SET valid = FALSE "
                "WHERE token = %s",
                (token,),
            )
        except Exception as exc:
            logger.warning("Session invalidate failed: %s", exc)

    def emergency_invalidate_all(
        self, reason: str = "emergency_mode"
    ) -> int:
        try:
            from app.core import db
            row = db.fetchone(
                "SELECT COUNT(*) FROM active_sessions WHERE valid = TRUE"
            )
            count = int(row[0]) if row else 0
            db.execute(
                "UPDATE active_sessions SET valid = FALSE WHERE valid = TRUE"
            )
            logger.critical(
                "EMERGENCY: %d sessions killed. Reason: %s",
                count, reason,
            )
            return count
        except Exception as exc:
            logger.error("Emergency invalidation failed: %s", exc)
            return 0

    def get_active_count(self) -> int:
        try:
            from app.core import db
            row = db.fetchone(
                "SELECT COUNT(*) FROM active_sessions "
                "WHERE valid = TRUE AND expires_at > NOW()"
            )
            return int(row[0]) if row else 0
        except Exception:
            return 0
