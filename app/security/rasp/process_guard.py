from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessThreat:
    threat_type: str
    severity: str
    description: str
    pid: int = 0
    process_name: str = ""
    file_path: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


SUSPICIOUS_PROCESS_NAMES = ["nc", "ncat", "netcat", "nmap", "masscan", "msfconsole", "metasploit", "hydra", "john", "hashcat", "aircrack", "tcpdump", "wireshark", "sqlmap", "nikto", "dirb", "gobuster", "mimikatz", "powersploit", "empire"]

INTEGRITY_FILES = [
    "app/core/cyber_shield.py",
    "app/core/query_router.py",
    "app/core/self_trainer.py",
    "app/core/db.py",
    "app/core/migrations.py",
    "app/security/rasp/rasp_engine.py",
    "config.yaml",
]


class ProcessGuard:
    def __init__(self) -> None:
        self._integrity_db: dict[str, str] = {}
        self._baseline_built = False

    def build_integrity_baseline(self) -> None:
        self._integrity_db = {}
        for file_path in INTEGRITY_FILES:
            path = Path(file_path)
            if path.exists():
                self._integrity_db[file_path] = self._hash_file(path)
        self._baseline_built = True
        logger.info("ProcessGuard: integrity baseline for %d files", len(self._integrity_db))

    def check_integrity(self) -> list[ProcessThreat]:
        threats = []
        if not self._baseline_built:
            return threats
        for file_path, expected_hash in self._integrity_db.items():
            path = Path(file_path)
            if not path.exists():
                threats.append(ProcessThreat("file_deleted", "critical", f"Core file deleted: {file_path}", file_path=file_path))
                continue
            current = self._hash_file(path)
            if current != expected_hash:
                threats.append(ProcessThreat("file_tampered", "critical", f"Core file tampered: {file_path} — hash mismatch detected", file_path=file_path))
                logger.critical("FILE TAMPERING DETECTED: %s", file_path)
        return threats

    def scan_processes(self) -> list[ProcessThreat]:
        threats = []
        try:
            import psutil

            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    name = (proc.info["name"] or "").lower()
                    cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                    for suspicious in SUSPICIOUS_PROCESS_NAMES:
                        if suspicious in name or suspicious in cmdline:
                            threats.append(ProcessThreat("suspicious_process", "high", f"Suspicious process detected: {name} (PID {proc.info['pid']})", pid=proc.info["pid"], process_name=name))
                            logger.warning("Suspicious process: %s PID=%d", name, proc.info["pid"])
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as exc:
            logger.warning("Process scan failed: %s", exc)
        return threats

    def _hash_file(self, path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
