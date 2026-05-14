from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatType(str, Enum):
    SQL_INJECTION = "sql_injection"
    PROMPT_INJECTION = "prompt_injection"
    XSS = "xss"
    SSRF = "ssrf"
    PATH_TRAVERSAL = "path_traversal"
    BRUTE_FORCE = "brute_force"
    ANOMALY = "anomaly"
    CLEAN = "clean"


@dataclass
class ThreatResult:
    threat_type: ThreatType
    severity: str
    score: float
    matched_patterns: list[str] = field(default_factory=list)
    recommendation: str = ""
    block: bool = False


SQL_INJECTION_PATTERNS = [
    r"(\s|^)(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\s",
    r"--\s*$",
    r";\s*(DROP|DELETE|UPDATE|INSERT)",
    r"'\s*(OR|AND)\s*'?\d",
    r"1\s*=\s*1",
    r"'\s*OR\s*'\w+'\s*=\s*'\w+",
    r"SLEEP\s*\(\s*\d+\s*\)",
    r"WAITFOR\s+DELAY",
    r"BENCHMARK\s*\(",
    r"LOAD_FILE\s*\(",
    r"INTO\s+OUTFILE",
    r"xp_cmdshell",
    r"information_schema",
    r"pg_sleep",
    r";\s*--",
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+(a|an|the)\s+\w+",
    r"act\s+as\s+(a|an|the)\s+\w+\s+(without|with\s+no)\s+restrictions?",
    r"(pretend|imagine|roleplay|simulate)\s+(you\s+are|to\s+be)",
    r"do\s+anything\s+now",
    r"DAN\s*:",
    r"jailbreak",
    r"override\s+(your\s+)?(system|safety|content)\s+(prompt|policy|filter)",
    r"bypass\s+(your\s+)?(restrictions?|guidelines?|rules?|filters?)",
    r"system\s*prompt\s*:",
    r"<\s*system\s*>",
    r"\[INST\].*override",
    r"###\s*system\s*###",
    r"print\s+(your|the)\s+(system\s+)?prompt",
    r"reveal\s+(your|the)\s+(system\s+)?instructions?",
    r"what\s+are\s+your\s+(exact\s+)?instructions?",
    r"(sudo|root|admin)\s*:\s*enable",
    r"developer\s+mode\s*:",
    r"enable\s+(developer|god|unrestricted)\s+mode",
]

XSS_PATTERNS = [
    r"<\s*script[^>]*>",
    r"javascript\s*:",
    r"on(load|click|mouseover|error|focus|blur|submit)\s*=",
    r"<\s*iframe[^>]*>",
    r"<\s*img[^>]+onerror\s*=",
    r"document\.(cookie|write|location)",
    r"eval\s*\(",
    r"alert\s*\(",
    r"expression\s*\(",
    r"vbscript\s*:",
]

SSRF_PATTERNS = [
    r"(http|https|ftp)://(localhost|127\.0\.0\.1|0\.0\.0\.0|::1)",
    r"(http|https)://169\.254\.169\.254",
    r"(http|https)://metadata\.google",
    r"file://",
    r"gopher://",
    r"dict://",
    r"(http|https)://10\.\d+\.\d+\.\d+",
    r"(http|https)://172\.(1[6-9]|2\d|3[01])\.\d+\.\d+",
    r"(http|https)://192\.168\.\d+\.\d+",
]

PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e%2f",
    r"%252e%252e%252f",
    r"/etc/passwd",
    r"/etc/shadow",
    r"C:\\Windows\\",
    r"\.\./\.\./",
]


class ThreatDetector:
    def scan(
        self,
        query: str = "",
        url_path: str = "",
        headers: dict | None = None,
        user_agent: str = "",
    ) -> ThreatResult:
        _ = user_agent
        results = [
            self._scan_sql_injection(query),
            self._scan_prompt_injection(query),
            self._scan_xss(query),
            self._scan_ssrf(query + url_path),
            self._scan_path_traversal(url_path),
            self._scan_headers(headers or {}),
        ]
        for threat_type in [
            ThreatType.SQL_INJECTION,
            ThreatType.PROMPT_INJECTION,
            ThreatType.XSS,
            ThreatType.SSRF,
            ThreatType.PATH_TRAVERSAL,
        ]:
            for result in results:
                if result.threat_type == threat_type and result.score > 0:
                    return result
        return ThreatResult(
            threat_type=ThreatType.CLEAN,
            severity="none",
            score=0.0,
            block=False,
        )

    def _scan_sql_injection(self, text: str) -> ThreatResult:
        return self._pattern_scan(
            text.upper(),
            SQL_INJECTION_PATTERNS,
            ThreatType.SQL_INJECTION,
            "high",
            "SQL injection attempt blocked",
        )

    def _scan_prompt_injection(self, text: str) -> ThreatResult:
        return self._pattern_scan(
            text.lower(),
            PROMPT_INJECTION_PATTERNS,
            ThreatType.PROMPT_INJECTION,
            "high",
            "Prompt injection attempt blocked",
        )

    def _scan_xss(self, text: str) -> ThreatResult:
        return self._pattern_scan(
            text.lower(),
            XSS_PATTERNS,
            ThreatType.XSS,
            "medium",
            "XSS attempt blocked",
        )

    def _scan_ssrf(self, text: str) -> ThreatResult:
        return self._pattern_scan(
            text.lower(),
            SSRF_PATTERNS,
            ThreatType.SSRF,
            "critical",
            "SSRF attempt blocked",
        )

    def _scan_path_traversal(self, path: str) -> ThreatResult:
        return self._pattern_scan(
            path.lower(),
            PATH_TRAVERSAL_PATTERNS,
            ThreatType.PATH_TRAVERSAL,
            "high",
            "Path traversal attempt blocked",
        )

    def _scan_headers(self, headers: dict) -> ThreatResult:
        suspicious = [
            "x-forwarded-host",
            "x-original-url",
            "x-rewrite-url",
            "x-override-url",
        ]
        matched = [h for h in suspicious if h in {k.lower() for k in headers}]
        if matched:
            return ThreatResult(
                threat_type=ThreatType.ANOMALY,
                severity="medium",
                score=0.5,
                matched_patterns=matched,
                recommendation="Suspicious headers detected",
                block=False,
            )
        return ThreatResult(
            threat_type=ThreatType.CLEAN,
            severity="none",
            score=0.0,
        )

    def _pattern_scan(
        self,
        text: str,
        patterns: list[str],
        threat_type: ThreatType,
        severity: str,
        recommendation: str,
    ) -> ThreatResult:
        matched = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched.append(pattern[:50])
        if not matched:
            return ThreatResult(
                threat_type=ThreatType.CLEAN,
                severity="none",
                score=0.0,
            )
        score = min(1.0, len(matched) * 0.3 + 0.4)
        return ThreatResult(
            threat_type=threat_type,
            severity=severity,
            score=round(score, 3),
            matched_patterns=matched,
            recommendation=recommendation,
            block=score >= 0.5,
        )
