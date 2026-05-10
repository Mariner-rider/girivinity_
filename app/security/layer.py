from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ThreatEvent:
    timestamp: float
    category: str
    severity: str
    message: str
    payload_excerpt: str
    metadata: dict = field(default_factory=dict)


class ThreatIntelligenceProvider:
    def __init__(self, suspicious_patterns: list[str] | None = None) -> None:
        self.suspicious_patterns = suspicious_patterns or [
            "ignore previous instructions",
            "system prompt",
            "jailbreak",
            "exfiltrate",
            "base64 payload",
        ]

    def lookup(self, text: str) -> list[str]:
        lower = text.lower()
        return [pat for pat in self.suspicious_patterns if pat in lower]


class SelfImprovingRulesEngine:
    def __init__(self, min_hits_to_promote: int = 3) -> None:
        self.min_hits_to_promote = min_hits_to_promote
        self.hit_counter: dict[str, int] = {}
        self.dynamic_block_rules: set[str] = set()

    def learn_from_event(self, event: ThreatEvent) -> None:
        token = event.message.lower().strip()
        self.hit_counter[token] = self.hit_counter.get(token, 0) + 1
        if self.hit_counter[token] >= self.min_hits_to_promote:
            self.dynamic_block_rules.add(token)

    def should_block(self, text: str) -> bool:
        lower = text.lower()
        return any(rule in lower for rule in self.dynamic_block_rules)


class SecurityLayer:
    def __init__(
        self,
        threat_intel: ThreatIntelligenceProvider | None = None,
        rules_engine: SelfImprovingRulesEngine | None = None,
        log_path: str = "security_events.jsonl",
    ) -> None:
        self.threat_intel = threat_intel or ThreatIntelligenceProvider()
        self.rules_engine = rules_engine or SelfImprovingRulesEngine()
        self.log_path = Path(log_path)

    def sanitize_input(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
        return text

    def detect_prompt_injection(self, text: str) -> tuple[bool, list[str]]:
        patterns = [
            r"ignore\s+previous\s+instructions",
            r"reveal\s+system\s+prompt",
            r"bypass\s+safety",
            r"act\s+as\s+developer\s+mode",
        ]
        hits = [p for p in patterns if re.search(p, text, flags=re.IGNORECASE)]
        return (len(hits) > 0, hits)

    def detect_anomaly(self, text: str) -> tuple[bool, str]:
        if len(text) > 10000:
            return True, "oversized_payload"
        if len(re.findall(r"[{}<>$]{6,}", text)) > 0:
            return True, "payload_obfuscation_pattern"
        if text.count("\n") > 300:
            return True, "excessive_line_breaks"
        return False, ""

    def log_attack(self, event: ThreatEvent) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as out:
            out.write(
                json.dumps(
                    {
                        "timestamp": event.timestamp,
                        "category": event.category,
                        "severity": event.severity,
                        "message": event.message,
                        "payload_excerpt": event.payload_excerpt,
                        "metadata": event.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self.rules_engine.learn_from_event(event)

    def inspect(self, raw_text: str) -> dict:
        sanitized = self.sanitize_input(raw_text)

        if self.rules_engine.should_block(sanitized):
            event = ThreatEvent(
                timestamp=time.time(),
                category="rules_engine",
                severity="high",
                message="dynamic_block_rule_triggered",
                payload_excerpt=sanitized[:200],
            )
            self.log_attack(event)
            return {"allowed": False, "reason": "dynamic_block_rule_triggered", "sanitized": sanitized}

        prompt_injection, injection_hits = self.detect_prompt_injection(sanitized)
        anomaly, anomaly_kind = self.detect_anomaly(sanitized)
        intel_hits = self.threat_intel.lookup(sanitized)

        if prompt_injection or anomaly or intel_hits:
            reason = "prompt_injection" if prompt_injection else anomaly_kind or "threat_intel_match"
            severity = "high" if prompt_injection else "medium"
            event = ThreatEvent(
                timestamp=time.time(),
                category="input_threat",
                severity=severity,
                message=reason,
                payload_excerpt=sanitized[:200],
                metadata={
                    "injection_hits": injection_hits,
                    "intel_hits": intel_hits,
                    "anomaly": anomaly_kind,
                },
            )
            self.log_attack(event)
            return {"allowed": False, "reason": reason, "sanitized": sanitized}

        return {"allowed": True, "reason": "clean", "sanitized": sanitized}
