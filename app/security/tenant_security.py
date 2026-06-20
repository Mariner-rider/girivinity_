from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TenantSecurityConfig:
    api_key: str
    observe_threshold: float = 0.3
    guard_threshold: float = 0.6
    contain_threshold: float = 0.9
    rate_limit_rpm: int = 500
    block_prompt_injection: bool = True
    block_sql_injection: bool = True
    block_xss: bool = True
    block_ssrf: bool = True
    alert_email: str = ""
    custom_blocked_patterns: list[str] | None = None

    def __post_init__(self):
        if self.custom_blocked_patterns is None:
            self.custom_blocked_patterns = []


DEFAULT_CONFIG = TenantSecurityConfig(api_key="default")


class TenantSecurity:
    def get_config(self, api_key: str) -> TenantSecurityConfig:
        if not api_key:
            return DEFAULT_CONFIG
        try:
            from app.core import db
            import json

            row = db.fetchone(
                """
                SELECT observe_threshold, guard_threshold,
                       contain_threshold, rate_limit_rpm,
                       block_prompt_injection, block_sql_injection,
                       block_xss, block_ssrf, alert_email,
                       custom_blocked_patterns
                FROM tenant_security_configs
                WHERE api_key = %s
                """,
                (api_key,),
            )
            if row:
                return TenantSecurityConfig(
                    api_key=api_key,
                    observe_threshold=float(row[0]),
                    guard_threshold=float(row[1]),
                    contain_threshold=float(row[2]),
                    rate_limit_rpm=int(row[3]),
                    block_prompt_injection=bool(row[4]),
                    block_sql_injection=bool(row[5]),
                    block_xss=bool(row[6]),
                    block_ssrf=bool(row[7]),
                    alert_email=str(row[8] or ""),
                    custom_blocked_patterns=json.loads(row[9] or "[]"),
                )
        except Exception as exc:
            logger.warning("TenantSecurity config fetch failed: %s", exc)
        return DEFAULT_CONFIG

    def update_config(self, api_key: str, updates: dict) -> bool:
        try:
            from app.core import db
            import json

            db.execute(
                """
                INSERT INTO tenant_security_configs
                    (api_key, observe_threshold, guard_threshold,
                     contain_threshold, rate_limit_rpm,
                     block_prompt_injection, block_sql_injection,
                     block_xss, block_ssrf, alert_email,
                     custom_blocked_patterns, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (api_key) DO UPDATE SET
                    observe_threshold     = EXCLUDED.observe_threshold,
                    guard_threshold       = EXCLUDED.guard_threshold,
                    contain_threshold     = EXCLUDED.contain_threshold,
                    rate_limit_rpm        = EXCLUDED.rate_limit_rpm,
                    block_prompt_injection= EXCLUDED.block_prompt_injection,
                    block_sql_injection   = EXCLUDED.block_sql_injection,
                    block_xss             = EXCLUDED.block_xss,
                    block_ssrf            = EXCLUDED.block_ssrf,
                    alert_email           = EXCLUDED.alert_email,
                    custom_blocked_patterns=EXCLUDED.custom_blocked_patterns,
                    updated_at            = NOW()
                """,
                (
                    api_key,
                    updates.get("observe_threshold", 0.3),
                    updates.get("guard_threshold", 0.6),
                    updates.get("contain_threshold", 0.9),
                    updates.get("rate_limit_rpm", 500),
                    updates.get("block_prompt_injection", True),
                    updates.get("block_sql_injection", True),
                    updates.get("block_xss", True),
                    updates.get("block_ssrf", True),
                    updates.get("alert_email", ""),
                    json.dumps(updates.get("custom_blocked_patterns", [])),
                ),
            )
            return True
        except Exception as exc:
            logger.error("TenantSecurity update failed: %s", exc)
            return False
