from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.security.anomaly_scorer import AnomalyScorer
from app.security.policy_engine import PolicyEngine, SecurityMode
from app.security.rate_limiter import RateLimiter
from app.security.session_manager import SessionManager
from app.security.threat_detector import ThreatDetector

logger = logging.getLogger(__name__)

EXEMPT_PATHS = {"/health", "/health/deep", "/docs", "/openapi.json"}


class CyberShieldMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, **kwargs) -> None:
        super().__init__(app, **kwargs)
        self.detector = ThreatDetector()
        self.limiter = RateLimiter()
        self.anomaly = AnomalyScorer()
        self.policy = PolicyEngine()
        self.sessions = SessionManager()

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        start = time.time()
        ip = self._get_ip(request)
        user_id = self._get_user_id(request)
        api_key = request.headers.get("x-api-key", "")

        query_text = ""
        if request.method == "POST":
            try:
                body = await request.body()
                if body:
                    parsed = json.loads(body)
                    query_text = str(parsed.get("query", ""))
            except Exception:
                pass

        threat = self.detector.scan(
            query=query_text,
            url_path=str(request.url.path),
            headers=dict(request.headers),
            user_agent=request.headers.get("user-agent", ""),
        )

        rate_key = api_key if api_key else ip
        rate_type = "api_key" if api_key else "ip"
        rate_result = self.limiter.check(rate_key, rate_type)

        if "/auth" in str(request.url.path):
            auth_rate = self.limiter.check(ip, "auth")
            if not auth_rate.allowed:
                return self._block_response(
                    "Too many authentication attempts",
                    429,
                    retry_after=auth_rate.retry_after_seconds,
                )

        anomaly = self.anomaly.score(
            user_id=user_id,
            ip=ip,
            query=query_text,
            endpoint=str(request.url.path),
        )

        decision = self.policy.decide(
            threat_score=threat.score,
            threat_type=threat.threat_type.value,
            rate_limited=not rate_result.allowed,
            anomaly_score=anomaly.score,
            user_id=user_id,
        )

        if decision.mode == SecurityMode.EMERGENCY:
            invalidated = self.sessions.emergency_invalidate_all(reason=decision.reason)
            self._log_event(
                user_id=user_id,
                ip=ip,
                event_type="emergency",
                threat_type=threat.threat_type.value,
                severity="critical",
                detail=f"Emergency: {invalidated} sessions cleared. {decision.reason}",
                blocked=True,
            )
            self._alert_admin(decision.reason, user_id, ip)
            return JSONResponse(
                status_code=503,
                content={"error": "Service temporarily restricted", "code": "SECURITY_EMERGENCY"},
            )

        if decision.mode == SecurityMode.CONTAIN:
            self._log_event(
                user_id=user_id,
                ip=ip,
                event_type="contain",
                threat_type=threat.threat_type.value,
                severity="high",
                detail=decision.reason,
                blocked=True,
            )
            return self._block_response("Session suspended due to suspicious activity", 403)

        if decision.block_request:
            self._log_event(
                user_id=user_id,
                ip=ip,
                event_type="blocked",
                threat_type=threat.threat_type.value,
                severity=threat.severity,
                detail=threat.recommendation,
                blocked=True,
            )
            if not rate_result.allowed:
                return self._block_response(
                    "Rate limit exceeded",
                    429,
                    retry_after=rate_result.retry_after_seconds,
                )
            return self._block_response("Request blocked", 403)

        if decision.log_event:
            self._log_event(
                user_id=user_id,
                ip=ip,
                event_type="request",
                threat_type=threat.threat_type.value,
                severity=threat.severity if threat.score > 0 else "none",
                detail=f"score={threat.score:.2f} anomaly={anomaly.score:.2f}",
                blocked=False,
            )

        response = await call_next(request)
        elapsed = round((time.time() - start) * 1000, 2)

        response.headers["X-Response-Time"] = f"{elapsed}ms"
        response.headers["X-Security-Mode"] = decision.mode.value

        return response

    def _get_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _get_user_id(self, request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            session = self.sessions.validate_session(token)
            if session:
                return session.get("user_id", "anonymous")
        return "anonymous"

    def _block_response(self, message: str, status: int, retry_after: int = 0) -> JSONResponse:
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        return JSONResponse(status_code=status, content={"error": message}, headers=headers)

    def _log_event(
        self,
        user_id: str,
        ip: str,
        event_type: str,
        threat_type: str,
        severity: str,
        detail: str,
        blocked: bool,
    ) -> None:
        try:
            from app.core import db

            db.execute(
                """
                INSERT INTO security_events
                    (user_id, ip_address, event_type, threat_type,
                     severity, detail, blocked, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (user_id, ip, event_type, threat_type, severity, detail[:500], blocked),
            )
        except Exception as exc:
            logger.warning("Security event log failed: %s", exc)

    def _alert_admin(self, reason: str, user_id: str, ip: str) -> None:
        try:
            from pathlib import Path
            import json as _json

            alerts_log = Path("logs/security_alerts.jsonl")
            alerts_log.parent.mkdir(exist_ok=True)
            with open(alerts_log, "a", encoding="utf-8") as f:
                f.write(
                    _json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "alert": "SECURITY_EMERGENCY",
                            "reason": reason,
                            "triggered_by_user": user_id,
                            "triggered_by_ip": ip,
                        }
                    )
                    + "\n"
                )
        except Exception as exc:
            logger.error("Admin alert failed: %s", exc)
