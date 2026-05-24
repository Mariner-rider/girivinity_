from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class SelfHealer:
    HEAL_LOG = Path("logs/self_healing.jsonl")

    def heal(self, threat_type: str, severity: str, description: str) -> str:
        action = self._decide_action(threat_type, severity)
        result = self._execute(action, threat_type, description)
        self._log(threat_type, severity, action, result)
        return result

    def _decide_action(self, threat_type: str, severity: str) -> str:
        critical_types = {
            "file_tampered": "revert_and_lockdown",
            "file_deleted": "revert_and_lockdown",
            "network_exfiltration": "isolate_network",
            "disk_exfiltration": "isolate_network",
            "connection_flood": "rate_lockdown",
            "suspicious_process": "kill_process",
            "memory_critical": "restart_daemons",
            "cpu_critical": "rate_lockdown",
        }
        if severity == "critical":
            return critical_types.get(threat_type, "lockdown")
        if severity == "high":
            return critical_types.get(threat_type, "isolate")
        return "log_and_monitor"

    def _execute(self, action: str, threat_type: str, description: str) -> str:
        logger.critical("SelfHealer executing '%s' for threat '%s'", action, threat_type)
        if action == "lockdown":
            return self._full_lockdown(description)
        if action == "revert_and_lockdown":
            self._revert_files()
            return self._full_lockdown(f"File integrity breach: {description}")
        if action == "isolate_network":
            return self._isolate_network(description)
        if action == "rate_lockdown":
            return self._rate_lockdown(description)
        if action == "restart_daemons":
            return self._restart_daemons()
        if action == "kill_process":
            return self._kill_suspicious()
        return f"Logged threat: {description}"

    def _full_lockdown(self, reason: str) -> str:
        try:
            from app.security.emergency_shutdown import EmergencyShutdown

            result = EmergencyShutdown().execute(
                reason=f"RASP triggered lockdown: {reason}",
                triggered_by="rasp_self_healer",
                threat_details={"rasp_action": "full_lockdown"},
            )
            sessions = result.get("sessions_killed", 0)
            return f"RASP full lockdown executed. {sessions} sessions terminated."
        except Exception as exc:
            logger.error("Full lockdown failed: %s", exc)
            return f"Lockdown attempted: {exc}"

    def _revert_files(self) -> None:
        backup_dir = Path("data/file_backups")
        if not backup_dir.exists():
            logger.warning("SelfHealer: no backup dir found for revert")
            return
        for backup in backup_dir.glob("*.bak"):
            target = Path(backup.stem)
            if target.exists():
                import shutil

                shutil.copy2(str(backup), str(target))
                logger.info("SelfHealer: reverted %s from backup", target)

    def _isolate_network(self, reason: str) -> str:
        try:
            from app.core import db

            db.execute("INSERT INTO system_security_mode (mode, triggered_by, set_at) VALUES ('contain', 'rasp_self_healer', NOW())")
        except Exception:
            pass
        return f"Network isolated — mode set to CONTAIN: {reason}"

    def _rate_lockdown(self, reason: str) -> str:
        try:
            from app.core import db

            db.execute("UPDATE tenant_security_configs SET rate_limit_rpm = 10 WHERE rate_limit_rpm > 10")
        except Exception:
            pass
        return f"Rate lockdown applied: {reason}"

    def _restart_daemons(self) -> str:
        restarted = []
        try:
            from app.core.self_trainer import SelfTrainer

            SelfTrainer.start()
            restarted.append("SelfTrainer")
        except Exception as exc:
            logger.warning("SelfTrainer restart failed: %s", exc)
        try:
            from app.core.successor_engine import SuccessorEngine

            SuccessorEngine.start()
            restarted.append("SuccessorEngine")
        except Exception as exc:
            logger.warning("SuccessorEngine restart failed: %s", exc)
        return f"Restarted daemons: {restarted}"

    def _kill_suspicious(self) -> str:
        try:
            import psutil
            from app.security.rasp.process_guard import SUSPICIOUS_PROCESS_NAMES

            killed = []
            for proc in psutil.process_iter(["pid", "name"]):
                name = (proc.info.get("name") or "").lower()
                if any(s in name for s in SUSPICIOUS_PROCESS_NAMES):
                    try:
                        proc.kill()
                        killed.append(f"{name}(PID {proc.info['pid']})")
                        logger.critical("RASP killed suspicious process: %s", name)
                    except Exception:
                        pass
            return f"Killed processes: {killed}" if killed else "No suspicious processes to kill"
        except Exception as exc:
            return f"Process kill failed: {exc}"

    def _log(self, threat_type: str, severity: str, action: str, result: str) -> None:
        self.HEAL_LOG.parent.mkdir(exist_ok=True)
        with open(self.HEAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "threat": threat_type, "severity": severity, "action": action, "result": result}) + "\n")
