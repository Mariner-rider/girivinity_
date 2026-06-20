from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PoisonScanResult:
    is_poisoned: bool
    confidence: float
    signals: list[str] = field(default_factory=list)
    url: str = ""


POISON_PATTERNS = [
    r"###\s*instruction\s*:", r"###\s*system\s*:", r"<\s*system\s*>", r"\[INST\]", r"<\|im_start\|>",
    r"when\s+asked\s+about\s+.{0,50}(always|never|must|should)\s+say",
    r"from\s+now\s+on\s+(you\s+)?(will|must|should|always)", r"your\s+new\s+(rules?|instructions?|directives?)\s+are",
    r"override\s+your\s+(training|instructions?|behaviour)", r"<script[^>]*>", r"javascript\s*:", r"eval\s*\(",
    r"when\s+you\s+see\s+.{0,50}send\s+to", r"exfiltrate\s+", r"leak\s+(user\s+)?(data|information)",
    r"trigger\s+word\s*:", r"activation\s+phrase\s*:", r"secret\s+command\s*:",
]
SUSPICIOUS_DOMAINS = ["pastebin.com", "hastebin.com", "ghostbin.com"]


class TrainingPoisonGuard:
    def scan_chunk(self, chunk_text: str, url: str, query: str) -> PoisonScanResult:
        signals = []
        confidence = 0.0

        if self._is_url_blacklisted(url):
            return PoisonScanResult(True, 1.0, ["url_permanently_blacklisted"], url)

        for pattern in POISON_PATTERNS:
            if re.search(pattern, chunk_text, re.IGNORECASE):
                signals.append(f"pattern:{pattern[:40]}")
                confidence += 0.35

        if self._is_suspicious_domain(url):
            signals.append(f"suspicious_domain:{url}")
            confidence += 0.25

        if self._has_encoding_anomaly(chunk_text):
            signals.append("encoding_anomaly")
            confidence += 0.2

        if self._has_excessive_instructions(chunk_text):
            signals.append("excessive_instruction_density")
            confidence += 0.3

        confidence = min(1.0, round(confidence, 3))
        is_poisoned = confidence >= 0.35

        if is_poisoned:
            self._handle_poison(chunk_text, url, query, signals, confidence)

        return PoisonScanResult(is_poisoned, confidence, signals, url)

    def scan_chunks_batch(self, chunks: list[dict], query: str) -> list[dict]:
        clean = []
        for chunk in chunks:
            result = self.scan_chunk(chunk.get("text", ""), chunk.get("url", ""), query)
            if not result.is_poisoned:
                clean.append(chunk)
            else:
                logger.warning("Poisoned chunk discarded: url=%s signals=%s", chunk.get("url", ""), result.signals)
        return clean

    def _handle_poison(self, chunk_text: str, url: str, query: str, signals: list[str], confidence: float) -> None:
        _ = chunk_text
        self._blacklist_url(url, signals)
        self._alert_admin(url, query, signals, confidence)

    def _blacklist_url(self, url: str, signals: list[str]) -> None:
        if not url:
            return
        try:
            from app.core import db

            db.execute(
                """
                INSERT INTO blacklisted_urls
                    (url, url_hash, reason, blacklisted_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (url_hash) DO NOTHING
                """,
                (url[:500], hashlib.sha256(url.encode()).hexdigest(), str(signals[:3])),
            )
            logger.warning("URL permanently blacklisted: %s", url)
        except Exception as exc:
            logger.error("URL blacklist failed: %s", exc)

    def _alert_admin(self, url: str, query: str, signals: list[str], confidence: float) -> None:
        try:
            import json

            alerts_log = Path("logs/security_alerts.jsonl")
            alerts_log.parent.mkdir(exist_ok=True)
            with open(alerts_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alert": "TRAINING_POISON_DETECTED",
                    "url": url,
                    "query": query[:100],
                    "signals": signals,
                    "confidence": confidence,
                    "action": "chunk_discarded_url_blacklisted",
                }) + "\n")

            from app.core import db

            db.execute(
                """
                INSERT INTO security_events
                    (user_id, ip_address, event_type, threat_type,
                     severity, detail, blocked, timestamp)
                VALUES ('system', 'crawler', 'poison_detected',
                        'training_poison', 'critical', %s, TRUE, NOW())
                """,
                (f"url={url[:200]} signals={signals[:2]}",),
            )
        except Exception as exc:
            logger.error("Poison alert failed: %s", exc)

    def _is_url_blacklisted(self, url: str) -> bool:
        if not url:
            return False
        try:
            from app.core import db

            url_hash = hashlib.sha256(url.encode()).hexdigest()
            row = db.fetchone("SELECT 1 FROM blacklisted_urls WHERE url_hash = %s", (url_hash,))
            return row is not None
        except Exception:
            return False

    def _is_suspicious_domain(self, url: str) -> bool:
        return any(domain in url.lower() for domain in SUSPICIOUS_DOMAINS)

    def _has_encoding_anomaly(self, text: str) -> bool:
        null_bytes = text.count("\x00")
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
        return null_bytes > 0 or control_chars > 10

    def _has_excessive_instructions(self, text: str) -> bool:
        instruction_words = ["you must", "you should", "you will", "always", "never", "do not", "make sure", "remember to", "from now on"]
        count = sum(1 for w in instruction_words if w in text.lower())
        return count >= 4
