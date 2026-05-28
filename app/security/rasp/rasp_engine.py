from __future__ import annotations

import json
import logging
import multiprocessing
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class RASPEngine:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        rasp = cfg.get("rasp", {})
        self.check_interval_s = int(rasp.get("check_interval_seconds", 30))
        self.enable_interceptor = bool(rasp.get("enable_runtime_interceptor", True))
        self.enable_process_scan = bool(rasp.get("enable_process_scan", True))
        self.enable_integrity_check = bool(rasp.get("enable_integrity_check", True))

        from app.security.rasp.hardware_monitor import HardwareMonitor
        from app.security.rasp.process_guard import ProcessGuard
        from app.security.rasp.runtime_interceptor import RuntimeInterceptor
        from app.security.rasp.self_healer import SelfHealer

        self.hw_monitor = HardwareMonitor()
        self.proc_guard = ProcessGuard()
        self.interceptor = RuntimeInterceptor()
        self.self_healer = SelfHealer()

    @classmethod
    def start(cls) -> multiprocessing.Process:
        engine = cls()
        if engine.enable_interceptor:
            engine.interceptor.activate()
        engine.proc_guard.build_integrity_baseline()
        engine._backup_core_files()

        p = multiprocessing.Process(target=engine._run_daemon, daemon=True)
        p.start()
        Path(".rasp_engine.pid").write_text(str(p.pid))
        logger.info("RASPEngine daemon started PID=%s", p.pid)
        return p

    def get_status(self) -> dict:
        snap = self.hw_monitor.snapshot()
        return {
            "interceptor_active": self.interceptor.is_active,
            "cpu_percent": snap.cpu_percent,
            "memory_percent": snap.memory_percent,
            "open_connections": snap.open_connections,
            "integrity_files_monitored": len(self.proc_guard._integrity_db),
        }

    def _run_daemon(self) -> None:
        logger.info("RASPEngine daemon running, interval=%ds", self.check_interval_s)
        while True:
            try:
                self._check_cycle()
            except Exception as exc:
                logger.error("RASPEngine daemon error: %s", exc)
            time.sleep(self.check_interval_s)

    def _check_cycle(self) -> None:
        all_threats = []
        snap = self.hw_monitor.snapshot()
        all_threats.extend(self.hw_monitor.analyse(snap))
        if self.enable_integrity_check:
            all_threats.extend(self.proc_guard.check_integrity())
        if self.enable_process_scan:
            all_threats.extend(self.proc_guard.scan_processes())
        for threat in all_threats:
            self._handle_threat(threat)
        if not all_threats:
            logger.debug("RASP cycle clean: CPU=%.1f%% MEM=%.1f%%", snap.cpu_percent, snap.memory_percent)

    def _handle_threat(self, threat) -> None:
        logger.warning("RASP threat: type=%s severity=%s desc=%s", threat.threat_type, threat.severity, threat.description)
        try:
            from app.core import db

            db.execute(
                """
                INSERT INTO security_events
                    (user_id, ip_address, event_type, threat_type,
                     severity, detail, blocked, timestamp)
                VALUES ('rasp', 'system', 'rasp_detection',
                        %s, %s, %s, %s, NOW())
                """,
                (threat.threat_type, threat.severity, threat.description[:400], threat.severity == "critical"),
            )
        except Exception as exc:
            logger.warning("RASP DB log failed: %s", exc)

        self._alert_admin(threat)
        if threat.severity in ("critical", "high"):
            self.self_healer.heal(threat_type=threat.threat_type, severity=threat.severity, description=threat.description)

    def _alert_admin(self, threat) -> None:
        try:
            log = Path("logs/security_alerts.jsonl")
            log.parent.mkdir(exist_ok=True)
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "alert": "RASP_THREAT_DETECTED", "threat": threat.threat_type, "severity": threat.severity, "description": threat.description, "source": "rasp_engine"}) + "\n")
        except Exception as exc:
            logger.error("RASP alert failed: %s", exc)

    def _backup_core_files(self) -> None:
        import shutil
        from app.security.rasp.process_guard import INTEGRITY_FILES

        backup_dir = Path("data/file_backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for file_path in INTEGRITY_FILES:
            src = Path(file_path)
            if src.exists():
                dst = backup_dir / f"{src.name}.bak"
                try:
                    shutil.copy2(str(src), str(dst))
                except Exception as exc:
                    logger.warning("Backup failed for %s: %s", file_path, exc)
        logger.info("RASPEngine: core file backups created in %s", backup_dir)
