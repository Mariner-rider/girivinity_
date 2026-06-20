"""Runtime application self-protection engine for hardware/runtime signals."""

from __future__ import annotations

import os
import platform
import resource
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeSignal:
    """A low-level runtime signal observed by the RASP engine."""

    name: str
    value: Any
    severity: str = "info"
    metadata: dict[str, Any] = field(default_factory=dict)


class RASPEngine:
    """Hardware/runtime-level RASP engine.

    This class intentionally lives outside ``app.security.rasp.__init__`` so
    package imports always resolve to the canonical engine implementation.
    """

    def __init__(
        self,
        max_memory_mb: int = 4096,
        max_cpu_load: float = 0.95,
        allowed_process_names: set[str] | None = None,
    ) -> None:
        self.max_memory_mb = max_memory_mb
        self.max_cpu_load = max_cpu_load
        self.allowed_process_names = allowed_process_names or set()
        self.started_at = time.time()
        self.events: list[RuntimeSignal] = []

    def inspect_runtime(self) -> dict[str, Any]:
        """Collect runtime and host-level process signals."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = usage.ru_maxrss / 1024
        if platform.system().lower() == "darwin":
            memory_mb = usage.ru_maxrss / (1024 * 1024)

        load_average = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        cpu_count = os.cpu_count() or 1
        normalized_load = min(1.0, load_average / cpu_count)

        signals = [
            RuntimeSignal("memory_mb", round(memory_mb, 2), self._severity(memory_mb, self.max_memory_mb)),
            RuntimeSignal("cpu_load", round(normalized_load, 4), self._severity(normalized_load, self.max_cpu_load)),
            RuntimeSignal("pid", os.getpid(), "info"),
            RuntimeSignal("uptime_seconds", round(time.time() - self.started_at, 2), "info"),
        ]
        self.events.extend(signals)
        return {
            "allowed": all(signal.severity != "critical" for signal in signals),
            "signals": [signal.__dict__ for signal in signals],
        }

    def protect(self) -> dict[str, Any]:
        """Return allow/block guidance based on current runtime signals."""
        result = self.inspect_runtime()
        critical = [s for s in result["signals"] if s["severity"] == "critical"]
        if critical:
            return {"allowed": False, "reason": "critical_runtime_signal", "signals": result["signals"]}
        return {"allowed": True, "reason": "runtime_within_limits", "signals": result["signals"]}

    def recent_events(self, limit: int = 50) -> list[RuntimeSignal]:
        """Return the most recent runtime signals."""
        return self.events[-limit:]

    @staticmethod
    def _severity(value: float, threshold: float) -> str:
        if value >= threshold:
            return "critical"
        if value >= threshold * 0.8:
            return "warning"
        return "info"
