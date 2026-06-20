"""RASP — Runtime Application Self-Protection.

Provides request/response inspection for Girivinity itself and for third-party
applications that use the RASP API endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.security.rasp.rasp_engine import RASPEngine  # noqa: F401

logger = logging.getLogger(__name__)


class ThreatCategory(Enum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    DATA_EXFILTRATION = "data_exfiltration"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    SSRF = "ssrf"
    XXE = "xxe"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    PII_IN_OUTPUT = "pii_in_output"
    CREDENTIAL_LEAK = "credential_leak"
    RATE_ABUSE = "rate_abuse"
    TOKEN_EXHAUSTION = "token_exhaustion"
    ADVERSARIAL_INPUT = "adversarial_input"


class Severity(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class RASPEvent:
    event_id: str
    timestamp: float
    category: ThreatCategory
    severity: Severity
    description: str
    source_ip: str
    client_id: str
    original_input_hash: str
    sanitised_input: str
    recommended_action: str
    blocked: bool
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["RASPEngine", "RASPEvent", "Severity", "ThreatCategory"]
