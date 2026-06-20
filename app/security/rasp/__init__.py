"""RASP — Runtime Application Self-Protection.

Provides request/response inspection for Girivinity itself and for third-party
applications that use the RASP API endpoints.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

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


class RASPEngine:
    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(?:previous|prior|above)\s+instructions",
        r"disregard\s+(?:all\s+)?(?:prior|previous)\s+(?:instructions|context|training)",
        r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|free|uncensored|unfiltered)",
        r"(?:new|updated|override)\s+(?:system\s+)?prompt[:\s]",
        r"jailbreak|DAN\s+mode|developer\s+mode|god\s+mode|unrestricted\s+mode",
        r"pretend\s+(?:you\s+)?(?:are|have\s+no|don't\s+have)\s+(?:restrictions|rules|guidelines)",
        r"forget\s+(?:you\s+are|your|everything|all)\s+(?:an?\s+AI|instructions|training)",
    ]
    DATA_EXFILTRATION_PATTERNS = [
        r"repeat\s+(?:everything|all|your\s+instructions|the\s+above)\s+(?:above|verbatim)",
        r"show\s+(?:me\s+)?(?:your\s+)?(?:system\s+prompt|instructions|training\s+data|context)",
        r"what\s+(?:are|were)\s+your\s+(?:original\s+)?instructions",
        r"print\s+(?:your\s+)?(?:system\s+message|prompt|context)",
        r"output\s+(?:your\s+)?(?:initial\s+)?(?:prompt|instructions|context)",
    ]
    SQL_INJECTION_PATTERNS = [
        r"(?:union\s+(?:all\s+)?select|select\s+.*\s+from\s+\w+)",
        r"(?:drop\s+table|truncate\s+table|delete\s+from)",
        r"(?:--\s*$|;\s*--|\b(?:or|and)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+)",
        r"(?:sleep\s*\(\s*\d+|benchmark\s*\(|waitfor\s+delay)",
        r"(?:information_schema|sys\.tables|sqlite_master|pg_tables)",
    ]
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript\s*:",
        r"on(?:load|click|error|mouseover|submit|focus|blur)\s*=",
        r"<(?:iframe|object|embed|form)[^>]*>",
        r"(?:document\.cookie|window\.location|document\.write)",
    ]
    PATH_TRAVERSAL_PATTERNS = [r"\.\.[\\/]", r"(?:%2e%2e|%252e%252e)[\\/]", r"(?:/etc/passwd|/etc/shadow|/proc/self|c:\\windows\\system32)"]
    COMMAND_INJECTION_PATTERNS = [
        r"(?:;|\||&&|\$\(|`)\s*(?:ls|cat|rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat)",
        r"(?:\|\s*(?:bash|sh|zsh)|>\s*/(?:dev|tmp|var))",
        r"(?:system\s*\(|exec\s*\(|popen\s*\(|subprocess\.)",
    ]
    SSRF_PATTERNS = [
        r"(?:http|https|ftp)://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|169\.254\.|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.)",
        r"file://",
        r"gopher://",
    ]
    XXE_PATTERNS = [r"<!DOCTYPE\s+[^>]+\[", r"<!ENTITY\s+\w+\s+SYSTEM", r"SYSTEM\s+['\"]file://"]
    DESERIALIZATION_PATTERNS = [r"pickle\.loads?\s*\(", r"yaml\.load\s*\(", r"ObjectInputStream", r"unserialize\s*\("]
    PII_PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "phone_in": re.compile(r"\b[6-9]\d{9}\b"),
        "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }
    CREDENTIAL_PATTERNS = {
        "api_key_generic": re.compile(r"\b(?:sk-|pk-|api[-_]?key[-_:]?\s*)[A-Za-z0-9]{20,}\b", re.I),
        "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "github_token": re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
        "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "password_in_text": re.compile(r"password\s*[:=]\s*\S{8,}", re.I),
    }

    def __init__(self, config: dict[str, Any] | None = None, audit_log_path: str = "logs/rasp_audit.jsonl") -> None:
        self.config = config or {}
        self.audit_log_path = audit_log_path
        Path(audit_log_path).parent.mkdir(parents=True, exist_ok=True)
        self._rate_windows: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1000))
        self._blocked_clients: dict[str, float] = {}
        self.guard = self
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        flags = re.IGNORECASE | re.DOTALL
        self._injection_re = [re.compile(pattern, flags) for pattern in self.PROMPT_INJECTION_PATTERNS]
        self._exfil_re = [re.compile(pattern, flags) for pattern in self.DATA_EXFILTRATION_PATTERNS]
        self._sql_re = [re.compile(pattern, flags) for pattern in self.SQL_INJECTION_PATTERNS]
        self._xss_re = [re.compile(pattern, flags) for pattern in self.XSS_PATTERNS]
        self._path_re = [re.compile(pattern, re.IGNORECASE) for pattern in self.PATH_TRAVERSAL_PATTERNS]
        self._cmd_re = [re.compile(pattern, re.IGNORECASE) for pattern in self.COMMAND_INJECTION_PATTERNS]
        self._ssrf_re = [re.compile(pattern, re.IGNORECASE) for pattern in self.SSRF_PATTERNS]
        self._xxe_re = [re.compile(pattern, flags) for pattern in self.XXE_PATTERNS]
        self._deser_re = [re.compile(pattern, flags) for pattern in self.DESERIALIZATION_PATTERNS]

    def inspect_input(self, text: str, client_id: str = "unknown", source_ip: str = "0.0.0.0", context: str = "ai_input") -> list[RASPEvent]:
        text = text or ""
        events: list[RASPEvent] = []
        rate_event = self._check_rate_limit(client_id, source_ip)
        if rate_event:
            self._log_event(rate_event)
            return [rate_event]
        max_chars = int(self.config.get("max_input_chars", 50000))
        if len(text) > max_chars:
            event = self._make_event(ThreatCategory.TOKEN_EXHAUSTION, Severity.MEDIUM, f"Input length {len(text)} chars exceeds safe limit", client_id, source_ip, text, "block", True)
            self._log_event(event)
            return [event]
        for regex, category, description in (
            (self._injection_re, ThreatCategory.PROMPT_INJECTION, "Prompt injection pattern detected"),
            (self._exfil_re, ThreatCategory.DATA_EXFILTRATION, "Data exfiltration attempt detected"),
        ):
            match = self._first_match(regex, text)
            if match:
                event = self._make_event(category, Severity.HIGH, f"{description}: {match[:80]}", client_id, source_ip, text, "block", True)
                self._log_event(event)
                return [event]
        if context in {"api_input", "web_input"}:
            checks = [
                (self._sql_re, ThreatCategory.SQL_INJECTION, Severity.CRITICAL, "SQL injection pattern detected"),
                (self._xss_re, ThreatCategory.XSS, Severity.HIGH, "XSS pattern detected"),
                (self._path_re, ThreatCategory.PATH_TRAVERSAL, Severity.HIGH, "Path traversal attempt detected"),
                (self._cmd_re, ThreatCategory.COMMAND_INJECTION, Severity.CRITICAL, "Command injection pattern detected"),
                (self._ssrf_re, ThreatCategory.SSRF, Severity.HIGH, "SSRF attempt detected"),
                (self._xxe_re, ThreatCategory.XXE, Severity.CRITICAL, "XXE pattern detected"),
                (self._deser_re, ThreatCategory.INSECURE_DESERIALIZATION, Severity.HIGH, "Insecure deserialization pattern detected"),
            ]
            for regexes, category, severity, description in checks:
                if self._first_match(regexes, text):
                    events.append(self._make_event(category, severity, description, client_id, source_ip, text, "block", True))
        for event in events:
            self._log_event(event)
        return events

    def inspect_output(self, text: str, client_id: str = "unknown", grounded_sources: list[Any] | None = None) -> tuple[str, list[RASPEvent]]:
        text = text or ""
        sanitised = text
        events: list[RASPEvent] = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = pattern.findall(sanitised)
            if matches:
                sanitised = pattern.sub(f"[{pii_type.upper()}_REDACTED]", sanitised)
                events.append(self._make_event(ThreatCategory.PII_IN_OUTPUT, Severity.MEDIUM, f"Redacted {len(matches)} {pii_type} value(s) from output", client_id, "output", text, "allow_sanitised", False))
        for cred_type, pattern in self.CREDENTIAL_PATTERNS.items():
            matches = pattern.findall(sanitised)
            if matches:
                sanitised = pattern.sub(f"[{cred_type.upper()}_REDACTED]", sanitised)
                events.append(self._make_event(ThreatCategory.CREDENTIAL_LEAK, Severity.CRITICAL, f"Credential leak ({cred_type}) blocked from output", client_id, "output", text, "block_and_alert", False))
        for event in events:
            self._log_event(event)
        return sanitised, events

    def _first_match(self, patterns: list[re.Pattern[str]], text: str) -> str:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return ""

    def _check_rate_limit(self, client_id: str, source_ip: str, window_seconds: int | None = None, max_requests: int | None = None) -> RASPEvent | None:
        now = time.time()
        window_seconds = int(window_seconds or self.config.get("rate_window_seconds", 60))
        max_requests = int(max_requests or self.config.get("max_requests_per_window", 60))
        if client_id in self._blocked_clients:
            if now < self._blocked_clients[client_id]:
                return self._make_event(ThreatCategory.RATE_ABUSE, Severity.HIGH, f"Client {client_id} is rate-limited", client_id, source_ip, "", "block", True)
            del self._blocked_clients[client_id]
        window = self._rate_windows[client_id]
        window.append(now)
        recent = sum(1 for timestamp in window if now - timestamp <= window_seconds)
        if recent > max_requests:
            self._blocked_clients[client_id] = now + int(self.config.get("block_seconds", 300))
            return self._make_event(ThreatCategory.RATE_ABUSE, Severity.HIGH, f"Rate limit exceeded: {recent} requests in {window_seconds}s", client_id, source_ip, "", "rate_limit", True)
        return None

    def _make_event(self, category: ThreatCategory, severity: Severity, description: str, client_id: str, source_ip: str, original_input: str, recommended_action: str, blocked: bool) -> RASPEvent:
        return RASPEvent(str(uuid.uuid4()), time.time(), category, severity, description, source_ip, client_id, hashlib.sha256((original_input or "").encode()).hexdigest(), self._sanitise(original_input or "", category), recommended_action, blocked)

    def _sanitise(self, text: str, category: ThreatCategory) -> str:
        hard_block = {ThreatCategory.PROMPT_INJECTION, ThreatCategory.JAILBREAK, ThreatCategory.DATA_EXFILTRATION, ThreatCategory.SQL_INJECTION, ThreatCategory.COMMAND_INJECTION, ThreatCategory.SSRF, ThreatCategory.XXE, ThreatCategory.INSECURE_DESERIALIZATION}
        if category in hard_block:
            return ""
        if category == ThreatCategory.XSS:
            return html.escape(text)
        return text

    def _log_event(self, event: RASPEvent) -> None:
        entry = {"event_id": event.event_id, "timestamp": event.timestamp, "category": event.category.value, "severity": event.severity.value, "description": event.description, "client_id": event.client_id, "source_ip": event.source_ip, "blocked": event.blocked, "recommended_action": event.recommended_action}
        with Path(self.audit_log_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if event.severity in {Severity.HIGH, Severity.CRITICAL}:
            logger.warning("RASP [%s] %s: %s", event.severity.value.upper(), event.category.value, event.description)

    def get_threat_summary(self, hours_back: int = 24) -> dict[str, Any]:
        cutoff = time.time() - (hours_back * 3600)
        counts: dict[str, int] = defaultdict(int)
        severity_counts: dict[str, int] = defaultdict(int)
        blocked = 0
        path = Path(self.audit_log_path)
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("timestamp", 0) >= cutoff:
                        counts[event.get("category", "unknown")] += 1
                        severity_counts[event.get("severity", "unknown")] += 1
                        blocked += int(bool(event.get("blocked")))
        return {"period_hours": hours_back, "total_events": sum(counts.values()), "by_category": dict(counts), "by_severity": dict(severity_counts), "blocked": blocked}


__all__ = ["RASPEngine", "RASPEvent", "Severity", "ThreatCategory"]
