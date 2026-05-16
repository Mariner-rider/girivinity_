from __future__ import annotations
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    allowed: bool
    requests_made: int
    limit: int
    window_seconds: int
    retry_after_seconds: int = 0
    key: str = ""


RATE_LIMITS = {
    "ip":      {"requests": 100, "window_seconds": 60},
    "user":    {"requests": 200, "window_seconds": 60},
    "api_key": {"requests": 500, "window_seconds": 60},
    "auth":    {"requests": 10,  "window_seconds": 300},
    "chat":    {"requests": 50,  "window_seconds": 60},
}


class RateLimiter:
    def check(
        self,
        identifier: str,
        limit_type: str = "ip",
    ) -> RateLimitResult:
        cfg    = RATE_LIMITS.get(limit_type, RATE_LIMITS["ip"])
        limit  = cfg["requests"]
        window = cfg["window_seconds"]
        key    = f"{limit_type}:{identifier}"

        try:
            from app.core import db
            from datetime import datetime, timezone, timedelta
            now          = datetime.now(timezone.utc)
            window_start = now - timedelta(seconds=window)

            db.execute(
                "INSERT INTO rate_limit_buckets "
                "(bucket_key, request_time) VALUES (%s, %s)",
                (key, now.isoformat()),
            )
            db.execute(
                "DELETE FROM rate_limit_buckets "
                "WHERE bucket_key = %s AND request_time < %s",
                (key, window_start.isoformat()),
            )
            row   = db.fetchone(
                "SELECT COUNT(*) FROM rate_limit_buckets "
                "WHERE bucket_key = %s",
                (key,),
            )
            count = int(row[0]) if row else 1
            allowed = count <= limit
            return RateLimitResult(
                allowed=allowed,
                requests_made=count,
                limit=limit,
                window_seconds=window,
                retry_after_seconds=window if not allowed else 0,
                key=key,
            )
        except Exception as exc:
            logger.warning("RateLimiter DB error: %s", exc)
            return RateLimitResult(
                allowed=True,
                requests_made=0,
                limit=limit,
                window_seconds=window,
                key=key,
            )
