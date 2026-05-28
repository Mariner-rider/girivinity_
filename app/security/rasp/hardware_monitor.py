from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class HardwareThreat:
    threat_type: str
    severity: str
    metric: str
    current_value: float
    threshold: float
    description: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class HardwareSnapshot:
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    net_sent_mb_s: float
    net_recv_mb_s: float
    open_connections: int
    process_count: int
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


DEFAULT_THRESHOLDS = {
    "cpu_spike": 85.0,
    "cpu_critical": 95.0,
    "memory_high": 80.0,
    "memory_critical": 90.0,
    "disk_write_high": 50.0,
    "net_sent_high": 100.0,
    "connections_high": 500,
    "connections_critical": 1000,
}


class HardwareMonitor:
    def __init__(self, thresholds: dict | None = None) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self._prev_disk = None
        self._prev_net = None
        self._prev_time = None

    def snapshot(self) -> HardwareSnapshot:
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            disk = psutil.disk_io_counters()
            net = psutil.net_io_counters()
            conns = len(psutil.net_connections())
            procs = len(psutil.pids())

            now = time.time()
            disk_read_s = 0.0
            disk_write_s = 0.0
            net_sent_s = 0.0
            net_recv_s = 0.0

            if self._prev_disk and self._prev_time and self._prev_net:
                dt = max(now - self._prev_time, 0.001)
                disk_read_s = (disk.read_bytes - self._prev_disk.read_bytes) / dt / 1_048_576
                disk_write_s = (disk.write_bytes - self._prev_disk.write_bytes) / dt / 1_048_576
                net_sent_s = (net.bytes_sent - self._prev_net.bytes_sent) / dt / 1_048_576
                net_recv_s = (net.bytes_recv - self._prev_net.bytes_recv) / dt / 1_048_576

            self._prev_disk = disk
            self._prev_net = net
            self._prev_time = now

            return HardwareSnapshot(
                cpu_percent=round(cpu, 2),
                memory_percent=round(mem.percent, 2),
                memory_used_mb=round(mem.used / 1_048_576, 2),
                disk_read_mb_s=round(max(0, disk_read_s), 3),
                disk_write_mb_s=round(max(0, disk_write_s), 3),
                net_sent_mb_s=round(max(0, net_sent_s), 3),
                net_recv_mb_s=round(max(0, net_recv_s), 3),
                open_connections=conns,
                process_count=procs,
            )
        except Exception as exc:
            logger.warning("HardwareMonitor snapshot failed: %s", exc)
            return HardwareSnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    def analyse(self, snap: HardwareSnapshot) -> list[HardwareThreat]:
        threats = []
        if snap.cpu_percent >= self.thresholds["cpu_critical"]:
            threats.append(HardwareThreat("cpu_critical", "critical", "cpu_percent", snap.cpu_percent, self.thresholds["cpu_critical"], f"CPU at {snap.cpu_percent}% — possible DoS attack or crypto mining"))
        elif snap.cpu_percent >= self.thresholds["cpu_spike"]:
            threats.append(HardwareThreat("cpu_spike", "high", "cpu_percent", snap.cpu_percent, self.thresholds["cpu_spike"], f"CPU spike at {snap.cpu_percent}%"))

        if snap.memory_percent >= self.thresholds["memory_critical"]:
            threats.append(HardwareThreat("memory_critical", "critical", "memory_percent", snap.memory_percent, self.thresholds["memory_critical"], f"Memory at {snap.memory_percent}% — possible memory injection or leak attack"))
        elif snap.memory_percent >= self.thresholds["memory_high"]:
            threats.append(HardwareThreat("memory_high", "high", "memory_percent", snap.memory_percent, self.thresholds["memory_high"], f"High memory: {snap.memory_percent}%"))

        if snap.disk_write_mb_s >= self.thresholds["disk_write_high"]:
            threats.append(HardwareThreat("disk_exfiltration", "critical", "disk_write_mb_s", snap.disk_write_mb_s, self.thresholds["disk_write_high"], f"Disk write spike {snap.disk_write_mb_s:.1f} MB/s — possible data exfiltration"))

        if snap.net_sent_mb_s >= self.thresholds["net_sent_high"]:
            threats.append(HardwareThreat("network_exfiltration", "critical", "net_sent_mb_s", snap.net_sent_mb_s, self.thresholds["net_sent_high"], f"Network spike {snap.net_sent_mb_s:.1f} MB/s — possible data exfiltration or C2 communication"))

        if snap.open_connections >= self.thresholds["connections_critical"]:
            threats.append(HardwareThreat("connection_flood", "critical", "open_connections", snap.open_connections, self.thresholds["connections_critical"], f"{snap.open_connections} open connections — possible DDoS"))

        return threats
