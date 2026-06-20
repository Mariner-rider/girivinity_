from fastapi import APIRouter

router = APIRouter(prefix="/security/rasp", tags=["rasp"])


@router.get("/status")
async def rasp_status():
    from app.security.rasp.rasp_engine import RASPEngine

    try:
        engine = RASPEngine()
        return {"rasp_active": True, **engine.get_status()}
    except Exception as exc:
        return {"rasp_active": False, "error": str(exc)}


@router.get("/intercept-log")
async def intercept_log():
    from app.security.rasp.runtime_interceptor import RuntimeInterceptor

    return {"log": RuntimeInterceptor().get_intercept_log()}


@router.get("/integrity-check")
async def integrity_check():
    from app.security.rasp.process_guard import ProcessGuard

    guard = ProcessGuard()
    guard.build_integrity_baseline()
    threats = guard.check_integrity()
    return {
        "status": "clean" if not threats else "compromised",
        "threats": [{"type": t.threat_type, "severity": t.severity, "file": t.file_path, "desc": t.description} for t in threats],
    }


@router.get("/hardware")
async def hardware_snapshot():
    from app.security.rasp.hardware_monitor import HardwareMonitor

    monitor = HardwareMonitor()
    snap = monitor.snapshot()
    threats = monitor.analyse(snap)
    return {
        "snapshot": {
            "cpu_percent": snap.cpu_percent,
            "memory_percent": snap.memory_percent,
            "memory_used_mb": snap.memory_used_mb,
            "disk_write_mb_s": snap.disk_write_mb_s,
            "net_sent_mb_s": snap.net_sent_mb_s,
            "open_connections": snap.open_connections,
            "process_count": snap.process_count,
        },
        "threats": [{"type": t.threat_type, "severity": t.severity, "metric": t.metric, "value": t.current_value, "desc": t.description} for t in threats],
    }


@router.post("/self-heal/{threat_type}")
async def manual_self_heal(threat_type: str):
    from app.security.rasp.self_healer import SelfHealer

    result = SelfHealer().heal(threat_type=threat_type, severity="high", description=f"Manual self-heal triggered for: {threat_type}")
    return {"action": "self_heal", "result": result}
