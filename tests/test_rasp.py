import sys
from unittest.mock import MagicMock

sys.modules.setdefault("psutil", MagicMock())
sys.modules.setdefault("yaml", MagicMock())


def test_hardware_monitor_analyses_cpu_spike():
    from app.security.rasp.hardware_monitor import HardwareMonitor, HardwareSnapshot

    monitor = HardwareMonitor()
    snap = HardwareSnapshot(96.0, 50.0, 2000.0, 1.0, 1.0, 1.0, 1.0, 10, 50)
    threats = monitor.analyse(snap)
    types = [t.threat_type for t in threats]
    assert "cpu_critical" in types


def test_hardware_monitor_clean_snapshot():
    from app.security.rasp.hardware_monitor import HardwareMonitor, HardwareSnapshot

    monitor = HardwareMonitor()
    snap = HardwareSnapshot(20.0, 30.0, 1000.0, 1.0, 1.0, 1.0, 1.0, 10, 50)
    threats = monitor.analyse(snap)
    assert len(threats) == 0


def test_hardware_monitor_detects_memory_critical():
    from app.security.rasp.hardware_monitor import HardwareMonitor, HardwareSnapshot

    monitor = HardwareMonitor()
    snap = HardwareSnapshot(20.0, 92.0, 7500.0, 0.0, 0.0, 0.0, 0.0, 10, 50)
    threats = monitor.analyse(snap)
    types = [t.threat_type for t in threats]
    assert "memory_critical" in types


def test_hardware_monitor_detects_connection_flood():
    from app.security.rasp.hardware_monitor import HardwareMonitor, HardwareSnapshot

    monitor = HardwareMonitor()
    snap = HardwareSnapshot(20.0, 30.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1200, 50)
    threats = monitor.analyse(snap)
    types = [t.threat_type for t in threats]
    assert "connection_flood" in types


def test_process_guard_hash_consistency():
    import tempfile
    from pathlib import Path

    from app.security.rasp.process_guard import ProcessGuard

    guard = ProcessGuard()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('test')")
        path = Path(f.name)
    h1 = guard._hash_file(path)
    h2 = guard._hash_file(path)
    assert h1 == h2
    path.unlink()


def test_runtime_interceptor_activate_deactivate():
    import builtins

    from app.security.rasp.runtime_interceptor import RuntimeInterceptor

    interceptor = RuntimeInterceptor()
    original_eval = builtins.eval
    interceptor.activate()
    assert interceptor.is_active is True
    interceptor.deactivate()
    assert interceptor.is_active is False
    assert builtins.eval is original_eval
