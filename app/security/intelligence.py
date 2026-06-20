from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
DOMAIN_RE = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")
HASH_RE = re.compile(r"\b(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b")
URL_RE = re.compile(r"https?://[^\s\]\)\}\"']+")

DEFAULT_IOC_FEEDS: tuple[str, ...] = ()


@dataclass
class InMemoryCyberRAG:
    records: list[dict] = field(default_factory=list)

    def add(self, text: str, metadata: dict | None = None, **kwargs: Any) -> None:
        item = {"text": text, "metadata": metadata or {}}
        item.update(kwargs)
        self.records.append(item)


class NVDClient:
    def __init__(self, api_key: str | None = None, rag_engine: Any | None = None) -> None:
        self.api_key = api_key
        self.rag_engine = rag_engine or InMemoryCyberRAG()
        self._request_times: list[float] = []
        self._lock = threading.Lock()

    def search(self, keyword: str, limit: int = 10) -> list[dict]:
        self._rate_limit()
        import nvdlib
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(wait=wait_exponential(multiplier=1, min=1, max=16), stop=stop_after_attempt(4))
        def _search() -> list[Any]:
            kwargs: dict[str, Any] = {"keywordSearch": keyword, "limit": limit}
            if self.api_key:
                kwargs["key"] = self.api_key
            return list(nvdlib.searchCVE(**kwargs))

        return [self._record_to_dict(record) for record in _search()]

    def get_by_id(self, cve_id: str) -> dict:
        self._rate_limit()
        import nvdlib
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(wait=wait_exponential(multiplier=1, min=1, max=16), stop=stop_after_attempt(4))
        def _get() -> list[Any]:
            kwargs: dict[str, Any] = {"cveId": cve_id}
            if self.api_key:
                kwargs["key"] = self.api_key
            return list(nvdlib.searchCVE(**kwargs))

        records = _get()
        return self._record_to_dict(records[0]) if records else {}

    def sync_recent(self, hours_back: int = 24) -> list[dict]:
        self._rate_limit()
        import nvdlib
        from tenacity import retry, stop_after_attempt, wait_exponential

        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours_back)

        @retry(wait=wait_exponential(multiplier=1, min=1, max=16), stop=stop_after_attempt(4))
        def _sync() -> list[Any]:
            kwargs: dict[str, Any] = {"pubStartDate": start, "pubEndDate": end}
            if self.api_key:
                kwargs["key"] = self.api_key
            return list(nvdlib.searchCVE(**kwargs))

        records = [self._record_to_dict(record) for record in _sync()]
        for record in records:
            self._store_rag(
                text=f"{record.get('id', '')}: {record.get('description', '')}",
                metadata={"source_type": "cve", "cve_id": record.get("id")},
            )
        return records

    def _rate_limit(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._request_times = [ts for ts in self._request_times if now - ts < 30.0]
            if len(self._request_times) >= 6:
                sleep_for = max(0.0, 30.0 - (now - self._request_times[0]))
                time.sleep(sleep_for)
            self._request_times.append(time.monotonic())

    def _record_to_dict(self, record: Any) -> dict:
        descriptions = getattr(record, "descriptions", []) or []
        description = ""
        for item in descriptions:
            if getattr(item, "lang", "") == "en":
                description = getattr(item, "value", "")
                break
        description = description or getattr(record, "description", "") or str(record)
        references = [getattr(ref, "url", str(ref)) for ref in getattr(record, "references", []) or []]
        weaknesses = [getattr(cwe, "value", str(cwe)) for cwe in getattr(record, "weaknesses", []) or []]
        metrics = getattr(record, "metrics", {}) or {}
        cvss = self._extract_cvss(metrics, record)
        affected = [str(item) for item in getattr(record, "configurations", []) or []]
        return {
            "id": getattr(record, "id", getattr(record, "cveId", "")),
            "description": description,
            "published": str(getattr(record, "published", "")),
            "last_modified": str(getattr(record, "lastModified", "")),
            "cvss": cvss,
            "cvss_vector": cvss.get("vector"),
            "severity": cvss.get("severity"),
            "cwe": weaknesses,
            "references": references,
            "affected_products": affected,
            "source_type": "cve",
        }

    def _extract_cvss(self, metrics: Any, record: Any) -> dict:
        candidates = []
        if isinstance(metrics, dict):
            for value in metrics.values():
                if isinstance(value, list):
                    candidates.extend(value)
        for metric in candidates:
            data = getattr(metric, "cvssData", None) or getattr(metric, "cvss_data", None)
            if data:
                return {
                    "score": getattr(data, "baseScore", None),
                    "severity": getattr(metric, "baseSeverity", getattr(data, "baseSeverity", None)),
                    "vector": getattr(data, "vectorString", None),
                }
        return {
            "score": getattr(record, "score", None),
            "severity": getattr(record, "severity", None),
            "vector": getattr(record, "v31vector", None) or getattr(record, "v30vector", None),
        }

    def _store_rag(self, text: str, metadata: dict) -> None:
        if hasattr(self.rag_engine, "add"):
            self.rag_engine.add(text, metadata={**metadata, "source_type": "cybersecurity"})


class MITREClient:
    def __init__(self, version: str = "enterprise-attack", bundle_url: str | None = None) -> None:
        self.version = version
        self.bundle_url = bundle_url or (
            "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        )
        self._data = None
        self._bundle = self._download_bundle()
        self._load_attack_data()

    def search_techniques(self, keyword: str) -> list[dict]:
        needle = keyword.lower()
        return [tech for tech in self._techniques() if needle in f"{tech.get('name', '')} {tech.get('description', '')}".lower()]

    def get_technique(self, technique_id: str) -> dict:
        for technique in self._techniques():
            if technique.get("id") == technique_id:
                return technique
        return {}

    def get_mitigations(self, technique_id: str) -> list[dict]:
        if self._data and hasattr(self._data, "get_mitigations_mitigating_technique"):
            return [self._stix_to_dict(item) for item in self._data.get_mitigations_mitigating_technique(technique_id)]
        mitigations = []
        relationships = [obj for obj in self._objects() if obj.get("type") == "relationship"]
        technique_stix = self._stix_id_for_attack_id(technique_id)
        for rel in relationships:
            if rel.get("relationship_type") == "mitigates" and rel.get("target_ref") == technique_stix:
                mitigation = self._object_by_stix_id(rel.get("source_ref"))
                if mitigation:
                    mitigations.append(self._mitigation_to_dict(mitigation))
        return mitigations

    def get_groups_using_technique(self, technique_id: str) -> list[str]:
        if self._data and hasattr(self._data, "get_groups_using_technique"):
            return [getattr(item, "name", str(item)) for item in self._data.get_groups_using_technique(technique_id)]
        groups = []
        technique_stix = self._stix_id_for_attack_id(technique_id)
        for rel in self._objects():
            if rel.get("type") != "relationship" or rel.get("relationship_type") != "uses":
                continue
            if rel.get("target_ref") != technique_stix:
                continue
            group = self._object_by_stix_id(rel.get("source_ref"))
            if group and group.get("type") == "intrusion-set":
                groups.append(group.get("name", rel.get("source_ref", "")))
        return groups

    def _download_bundle(self) -> dict:
        req = Request(self.bundle_url, headers={"User-Agent": "GirivinitySecurityIntelligence/1.0"})
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _load_attack_data(self) -> None:
        try:
            from mitreattack.stix20 import MitreAttackData
        except ModuleNotFoundError:
            self._data = None
            return
        cache = Path(".cache/girivinity_mitre")
        cache.mkdir(parents=True, exist_ok=True)
        bundle_path = cache / f"{self.version}.json"
        bundle_path.write_text(json.dumps(self._bundle), encoding="utf-8")
        self._data = MitreAttackData(str(bundle_path))

    def _objects(self) -> list[dict]:
        return self._bundle.get("objects", [])

    def _techniques(self) -> list[dict]:
        techniques = []
        for obj in self._objects():
            if obj.get("type") != "attack-pattern" or obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue
            techniques.append(self._technique_to_dict(obj))
        return techniques

    def _technique_to_dict(self, obj: dict) -> dict:
        attack_id = self._attack_id(obj)
        return {
            "id": attack_id,
            "stix_id": obj.get("id"),
            "name": obj.get("name", ""),
            "description": obj.get("description", ""),
            "tactics": [phase.get("phase_name", "") for phase in obj.get("kill_chain_phases", [])],
            "platforms": obj.get("x_mitre_platforms", []),
            "detection": obj.get("x_mitre_detection", ""),
            "url": f"https://attack.mitre.org/techniques/{attack_id.replace('.', '/')}/" if attack_id else "",
        }

    def _mitigation_to_dict(self, obj: dict) -> dict:
        return {
            "id": self._attack_id(obj),
            "name": obj.get("name", ""),
            "description": obj.get("description", ""),
        }

    def _attack_id(self, obj: dict) -> str:
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
                return ref["external_id"]
        return ""

    def _stix_id_for_attack_id(self, attack_id: str) -> str:
        for obj in self._objects():
            if self._attack_id(obj) == attack_id:
                return obj.get("id", "")
        return ""

    def _object_by_stix_id(self, stix_id: str | None) -> dict | None:
        for obj in self._objects():
            if obj.get("id") == stix_id:
                return obj
        return None

    def _stix_to_dict(self, item: Any) -> dict:
        if isinstance(item, dict):
            return item
        return {"id": getattr(item, "id", ""), "name": getattr(item, "name", str(item))}


class IOCManager:
    def __init__(self, feed_urls: list[str] | None = None, refresh_interval: int = 3600) -> None:
        self.feed_urls = feed_urls or list(DEFAULT_IOC_FEEDS)
        self.refresh_interval = refresh_interval
        self._ioc_set: set[str] = set()
        self._sources: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        self.refresh()
        self._schedule_refresh()

    def refresh(self) -> int:
        before = len(self._ioc_set)
        for feed_url in self.feed_urls:
            try:
                payload = self._download(feed_url)
                values = self._parse_feed(payload)
                with self._lock:
                    for value in values:
                        normalized = value.strip().lower()
                        if normalized:
                            self._ioc_set.add(normalized)
                            self._sources.setdefault(normalized, set()).add(feed_url)
            except Exception as exc:
                logger.warning("IOC feed refresh failed for %s: %s", feed_url, exc)
        return max(0, len(self._ioc_set) - before)

    def is_ioc(self, value: str) -> dict:
        normalized = value.strip().lower()
        value_type = self._classify_value(normalized)
        with self._lock:
            found = normalized in self._ioc_set
            sources = sorted(self._sources.get(normalized, set()))
        return {"is_ioc": found, "value_type": value_type, "feed_sources": sources}

    def export_yara_rule(self, rule_name: str) -> str:
        safe_name = re.sub(r"\W+", "_", rule_name).strip("_") or "girivinity_iocs"
        with self._lock:
            values = sorted(self._ioc_set)[:256]
        strings = [f'        $ioc_{idx} = "{value}" nocase' for idx, value in enumerate(values)]
        if not strings:
            strings = ['        $placeholder = "girivinity-no-iocs"']
        return f"rule {safe_name} {{\n    strings:\n" + "\n".join(strings) + "\n    condition:\n        any of them\n}"

    def _schedule_refresh(self) -> None:
        if self.refresh_interval <= 0:
            return

        def _loop() -> None:
            while True:
                time.sleep(self.refresh_interval)
                self.refresh()

        threading.Thread(target=_loop, name="girivinity-ioc-refresh", daemon=True).start()

    def _download(self, feed_url: str) -> str:
        req = Request(feed_url, headers={"User-Agent": "GirivinitySecurityIntelligence/1.0"})
        with urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")

    def _parse_feed(self, payload: str) -> set[str]:
        payload = payload.strip()
        if not payload:
            return set()
        if payload.startswith("{") or payload.startswith("["):
            return self._extract_from_json(json.loads(payload))
        values = set()
        reader = csv.reader(StringIO(payload))
        for row in reader:
            for cell in row:
                values.update(self._extract_iocs(cell))
        return values

    def _extract_from_json(self, obj: Any) -> set[str]:
        values = set()
        if isinstance(obj, dict):
            for value in obj.values():
                values.update(self._extract_from_json(value))
        elif isinstance(obj, list):
            for value in obj:
                values.update(self._extract_from_json(value))
        elif isinstance(obj, str):
            values.update(self._extract_iocs(obj))
        return values

    def _extract_iocs(self, text: str) -> set[str]:
        candidates = set(URL_RE.findall(text)) | set(IP_RE.findall(text)) | set(DOMAIN_RE.findall(text)) | set(HASH_RE.findall(text))
        return {value for value in candidates if self._classify_value(value) != "unknown"}

    def _classify_value(self, value: str) -> str:
        if value.startswith(("http://", "https://")):
            return "url"
        try:
            ipaddress.ip_address(value)
            return "ip"
        except ValueError:
            pass
        if HASH_RE.fullmatch(value):
            return "hash"
        if DOMAIN_RE.fullmatch(value):
            return "domain"
        return "unknown"


class ThreatCorrelator:
    def __init__(
        self,
        nvd_client: NVDClient | None = None,
        mitre_client: MITREClient | None = None,
        ioc_manager: IOCManager | None = None,
    ) -> None:
        self.nvd_client = nvd_client
        self.mitre_client = mitre_client
        self.ioc_manager = ioc_manager or IOCManager(refresh_interval=0)

    def correlate(self, text: str) -> dict:
        cve_ids = sorted({match.upper() for match in CVE_RE.findall(text)})
        technique_ids = sorted({match.upper() for match in TECHNIQUE_RE.findall(text)})
        observable_values = sorted(
            set(URL_RE.findall(text)) | set(IP_RE.findall(text)) | set(DOMAIN_RE.findall(text)) | set(HASH_RE.findall(text))
        )
        cves = [self._enrich_cve(cve_id) for cve_id in cve_ids]
        techniques = [self._enrich_technique(technique_id) for technique_id in technique_ids]
        iocs = [self.ioc_manager.is_ioc(value) | {"value": value} for value in observable_values]
        return {
            "cves": cves,
            "techniques": techniques,
            "iocs": iocs,
            "risk_score": self.risk_score(cves, techniques, iocs),
        }

    def risk_score(self, cves: list[dict], techniques: list[dict], iocs: list[dict]) -> float:
        score = 0.0
        for cve in cves:
            severity = str(cve.get("severity") or cve.get("cvss", {}).get("severity") or "").lower()
            if severity == "critical":
                score += 3.0
            elif severity == "high":
                score += 2.0
        score += 2.0 * sum(1 for ioc in iocs if ioc.get("is_ioc"))
        score += 2.0 * sum(1 for technique in techniques if technique.get("apt_groups"))
        return round(min(score, 10.0), 2)

    def _enrich_cve(self, cve_id: str) -> dict:
        if not self.nvd_client:
            return {"id": cve_id}
        try:
            return self.nvd_client.get_by_id(cve_id) or {"id": cve_id}
        except Exception as exc:
            logger.warning("CVE enrichment failed for %s: %s", cve_id, exc)
            return {"id": cve_id, "error": str(exc)}

    def _enrich_technique(self, technique_id: str) -> dict:
        if not self.mitre_client:
            return {"id": technique_id, "apt_groups": []}
        try:
            technique = self.mitre_client.get_technique(technique_id) or {"id": technique_id}
            technique["mitigations"] = self.mitre_client.get_mitigations(technique_id)
            technique["apt_groups"] = self.mitre_client.get_groups_using_technique(technique_id)
            return technique
        except Exception as exc:
            logger.warning("Technique enrichment failed for %s: %s", technique_id, exc)
            return {"id": technique_id, "apt_groups": [], "error": str(exc)}


class SecurityIntelligenceEngine:
    def __init__(
        self,
        rag_engine: Any | None = None,
        nvd_api_key: str | None = None,
        ioc_feeds: list[str] | None = None,
        mitre_version: str = "enterprise-attack",
        enable_mitre: bool = True,
    ) -> None:
        self.rag_engine = rag_engine or InMemoryCyberRAG()
        self.nvd = NVDClient(api_key=nvd_api_key, rag_engine=self.rag_engine)
        self.iocs = IOCManager(feed_urls=ioc_feeds, refresh_interval=3600)
        self.mitre = MITREClient(version=mitre_version) if enable_mitre else None
        self.correlator = ThreatCorrelator(self.nvd, self.mitre, self.iocs)

    def enrich_text(self, text: str) -> dict:
        return self.correlator.correlate(text)

    def sync_all(self) -> dict:
        cves = self.nvd.sync_recent()
        iocs_added = self.iocs.refresh()
        for cve in cves:
            if hasattr(self.rag_engine, "add"):
                self.rag_engine.add(
                    f"{cve.get('id', '')}: {cve.get('description', '')}",
                    metadata={"source_type": "cybersecurity", "subtype": "cve", "cve_id": cve.get("id")},
                )
        return {"cves_synced": len(cves), "iocs_added": iocs_added}

    def generate_report(self, enriched: dict) -> str:
        lines = ["# Girivinity Cybersecurity Threat Report", ""]
        lines.append(f"**Risk score:** {enriched.get('risk_score', 0.0)}/10.0")
        lines.append("")
        lines.append("## CVEs")
        for cve in enriched.get("cves", []):
            lines.append(
                f"- **{cve.get('id', 'unknown')}** "
                f"({cve.get('severity') or cve.get('cvss', {}).get('severity', 'unknown')}): "
                f"{cve.get('description', '')}"
            )
        lines.append("")
        lines.append("## ATT&CK Techniques")
        for technique in enriched.get("techniques", []):
            groups = ", ".join(technique.get("apt_groups", [])) or "none linked"
            lines.append(f"- **{technique.get('id', '')} {technique.get('name', '')}** — APT groups: {groups}")
        lines.append("")
        lines.append("## Indicators of Compromise")
        for ioc in enriched.get("iocs", []):
            marker = "MATCH" if ioc.get("is_ioc") else "observed"
            lines.append(f"- `{ioc.get('value')}` ({ioc.get('value_type')}): {marker}")
        return "\n".join(lines)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
