from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar
from urllib.parse import urlparse

P = ParamSpec("P")
T = TypeVar("T")


class SecurityPolicyError(PermissionError):
    """Raised when a subsystem violates a required safety policy."""


@dataclass(frozen=True, slots=True)
class TrustScore:
    url: str
    score: float
    reasons: tuple[str, ...]

    @property
    def trusted(self) -> bool:
        return self.score >= 0.6


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    score: float


class SecurityGuard:
    """Real enforcement — not just call sites."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?(prior|previous|above)",
        r"you\s+are\s+now\s+\w+",
        r"jailbreak|DAN\s+mode|developer\s+mode",
        r"pretend\s+(you\s+are|to\s+be)",
        r"forget\s+(your|all)\s+(instructions|training|guidelines)",
        r"override\s+(your\s+)?(safety|ethics|alignment)",
    ]

    PII_PATTERNS = {
        "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "pan_card": r"[A-Z]{5}[0-9]{4}[A-Z]{1}",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone_in": r"\b[6-9]\d{9}\b",
    }

    def __init__(
        self,
        audit_path: str | Path = "logs/security_audit.jsonl",
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        self.audit_path = Path(audit_path)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._rate_events: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def validate_prompt(self, prompt: str) -> None:
        if not prompt or not prompt.strip():
            self.audit_log("prompt_rejected", {"reason": "empty"})
            raise SecurityPolicyError("Prompt must be non-empty.")
        matches = self._matches(self.INJECTION_PATTERNS, prompt, flags=re.IGNORECASE)
        if matches:
            self.audit_log("prompt_rejected", {"reason": "prompt_injection", "matches": matches})
            raise SecurityPolicyError("Prompt rejected: prompt-injection pattern detected.")
        pii = self._detect_pii(prompt)
        if pii:
            self.audit_log("prompt_pii_detected", {"types": sorted(pii)})

    def require_grounding(self, sources: list[dict] | list[str], context: str) -> None:
        if not sources:
            self.audit_log("grounding_rejected", {"reason": "missing_sources"})
            raise SecurityPolicyError("Refusing to produce output without RAG/source grounding.")
        if not context.strip():
            self.audit_log("grounding_rejected", {"reason": "missing_context"})
            raise SecurityPolicyError("Refusing to produce output without grounded context.")

    def sanitize_output(self, output: str, allow_pii: bool = False) -> str:
        sanitized = output or ""
        sanitized = re.sub(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", "[removed-script]", sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r"javascript\s*:", "", sanitized, flags=re.IGNORECASE)
        if not allow_pii:
            for pii_type, pattern in self.PII_PATTERNS.items():
                sanitized = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", sanitized)
        findings = self.detect_adversarial_output(sanitized, allow_pii=allow_pii)
        if findings["has_injection"]:
            sanitized = self._neutralize_injection_phrases(sanitized)
        return sanitized

    def rate_limit_check(self, key: str, limit: int | None = None, window_seconds: int | None = None) -> bool:
        limit = limit or self.max_requests
        window_seconds = window_seconds or self.window_seconds
        now = time.monotonic()
        with self._lock:
            events = [ts for ts in self._rate_events.get(key, []) if now - ts < window_seconds]
            if len(events) >= limit:
                self._rate_events[key] = events
                self.audit_log("rate_limit_block", {"key": key, "limit": limit, "window": window_seconds})
                return False
            events.append(now)
            self._rate_events[key] = events
        return True

    def audit_log(self, event: str, details: dict | None = None) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "details": details or {},
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def detect_adversarial_output(self, output: str, allow_pii: bool = False) -> dict:
        injection_matches = self._matches(self.INJECTION_PATTERNS, output or "", flags=re.IGNORECASE)
        pii_matches = {} if allow_pii else self._detect_pii(output or "")
        finding = {
            "has_injection": bool(injection_matches),
            "has_pii": bool(pii_matches),
            "injection_matches": injection_matches,
            "pii_matches": pii_matches,
            "safe": not injection_matches and not pii_matches,
        }
        if not finding["safe"]:
            self.audit_log("adversarial_output_detected", finding)
        return finding

    def score_url_trust(self, url: str) -> TrustScore:
        parsed = urlparse(url)
        reasons: list[str] = []
        score = 0.0

        if parsed.scheme == "https":
            score += 0.45
            reasons.append("https")
        elif parsed.scheme == "http":
            score += 0.2
            reasons.append("http")

        if parsed.netloc and "." in parsed.netloc:
            score += 0.25
            reasons.append("valid_host")

        lowered = parsed.netloc.lower()
        if any(domain in lowered for domain in (".edu", ".gov", ".org")):
            score += 0.2
            reasons.append("reputable_tld")

        if not parsed.fragment:
            score += 0.1
            reasons.append("no_fragment")

        return TrustScore(url=url, score=round(min(score, 1.0), 3), reasons=tuple(reasons))

    def require_trusted_url(self, url: str) -> TrustScore:
        trust = self.score_url_trust(url)
        if not trust.trusted:
            self.audit_log("url_rejected", {"url": url, "score": trust.score, "reasons": trust.reasons})
            raise SecurityPolicyError(f"Crawler URL failed trust scoring: {url}")
        return trust

    def require_validation_dataset(self, validation_dataset_path: str | Path | None) -> None:
        if validation_dataset_path is None:
            raise SecurityPolicyError("Training requires a validation dataset path.")
        path = Path(validation_dataset_path)
        if not path.exists() or path.stat().st_size == 0:
            raise SecurityPolicyError("Training validation dataset must exist and be non-empty.")

    def require_benchmark_improvement(
        self,
        baseline: BenchmarkResult,
        candidate: BenchmarkResult,
        min_delta: float = 0.0,
    ) -> None:
        if candidate.score <= baseline.score + min_delta:
            self.audit_log(
                "model_update_rejected",
                {
                    "baseline": baseline.name,
                    "baseline_score": baseline.score,
                    "candidate": candidate.name,
                    "candidate_score": candidate.score,
                    "min_delta": min_delta,
                },
            )
            raise SecurityPolicyError(
                f"Model update rejected: {candidate.name}={candidate.score} did not improve "
                f"over {baseline.name}={baseline.score} by > {min_delta}."
            )

    def _detect_pii(self, text: str) -> dict[str, list[str]]:
        findings: dict[str, list[str]] = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            flags = 0 if pii_type == "pan_card" else re.IGNORECASE
            matches = re.findall(pattern, text, flags=flags)
            if matches:
                findings[pii_type] = [match if isinstance(match, str) else "".join(match) for match in matches]
        return findings

    def _matches(self, patterns: list[str], text: str, flags: int = 0) -> list[str]:
        matches = []
        for pattern in patterns:
            if re.search(pattern, text, flags=flags):
                matches.append(pattern)
        return matches

    def _neutralize_injection_phrases(self, text: str) -> str:
        neutralized = text
        for pattern in self.INJECTION_PATTERNS:
            neutralized = re.sub(pattern, "[removed-adversarial-instruction]", neutralized, flags=re.IGNORECASE)
        return neutralized


def secure_operation(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Annotate guarded module entrypoints without changing their public signatures."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__security_operation__ = name  # type: ignore[attr-defined]
        return wrapper

    return decorator
