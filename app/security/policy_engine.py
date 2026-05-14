from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class SecurityMode(str, Enum):
    OBSERVE = "observe"
    GUARD = "guard"
    CONTAIN = "contain"
    EMERGENCY = "emergency"


@dataclass
class PolicyDecision:
    mode: SecurityMode
    action: str
    reason: str
    block_request: bool
    log_event: bool
    alert_admin: bool


class PolicyEngine:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        se = cfg.get("cyber_shield", {})
        self.observe_threshold = float(se.get("observe_threshold", 0.3))
        self.guard_threshold = float(se.get("guard_threshold", 0.6))
        self.contain_threshold = float(se.get("contain_threshold", 0.9))
        self.current_mode = self._get_system_mode()

    def decide(
        self,
        threat_score: float,
        threat_type: str,
        rate_limited: bool,
        anomaly_score: float,
        user_id: str = "",
    ) -> PolicyDecision:
        if self.current_mode == SecurityMode.EMERGENCY:
            return PolicyDecision(
                mode=SecurityMode.EMERGENCY,
                action="reject_all",
                reason="System in emergency mode",
                block_request=True,
                log_event=True,
                alert_admin=False,
            )

        if threat_type in ("ssrf", "sql_injection") and threat_score > 0.7:
            self._escalate_mode(SecurityMode.EMERGENCY, user_id)
            return PolicyDecision(
                mode=SecurityMode.EMERGENCY,
                action="block_and_escalate",
                reason=f"Critical threat: {threat_type}",
                block_request=True,
                log_event=True,
                alert_admin=True,
            )

        combined = threat_score + (anomaly_score * 0.2) + (0.1 if rate_limited else 0.0)

        if combined >= self.contain_threshold:
            return PolicyDecision(
                mode=SecurityMode.CONTAIN,
                action="block_session",
                reason=f"High combined risk score: {combined:.2f}",
                block_request=True,
                log_event=True,
                alert_admin=True,
            )
        if combined >= self.guard_threshold or threat_score >= 0.5 or anomaly_score >= 0.4:
            return PolicyDecision(
                mode=SecurityMode.GUARD,
                action="throttle_and_warn",
                reason=f"Elevated risk score: {combined:.2f}",
                block_request=rate_limited,
                log_event=True,
                alert_admin=False,
            )
        return PolicyDecision(
            mode=SecurityMode.OBSERVE,
            action="log_and_pass",
            reason="Normal traffic",
            block_request=False,
            log_event=combined > 0.1,
            alert_admin=False,
        )

    def _get_system_mode(self) -> SecurityMode:
        try:
            from app.core import db

            row = db.fetchone("SELECT mode FROM system_security_mode ORDER BY set_at DESC LIMIT 1")
            if row:
                return SecurityMode(row[0])
        except Exception:
            pass
        return SecurityMode.OBSERVE

    def _escalate_mode(self, mode: SecurityMode, triggered_by: str) -> None:
        try:
            from app.core import db

            db.execute(
                """
                INSERT INTO system_security_mode
                    (mode, triggered_by, set_at)
                VALUES (%s, %s, NOW())
                """,
                (mode.value, triggered_by),
            )
            logger.critical("SECURITY MODE ESCALATED TO %s by %s", mode.value, triggered_by)
        except Exception as exc:
            logger.error("Mode escalation failed: %s", exc)
