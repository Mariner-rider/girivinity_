"""Runtime application self-protection package exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.security.rasp.rasp_engine import RASPEngine  # noqa: F401


class ThreatCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    RUNTIME_ANOMALY = "runtime_anomaly"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class RASPEvent:
    category: ThreatCategory
    severity: Severity
    message: str
    evidence: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


__all__ = [
    "RASPEngine",
    "RASPEvent",
    "Severity",
    "ThreatCategory",
]
