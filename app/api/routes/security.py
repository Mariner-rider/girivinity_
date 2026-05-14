from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/security", tags=["security"])


@router.get("/status")
async def security_status():
    from app.core import db
    from app.security.policy_engine import PolicyEngine
    from app.security.session_manager import SessionManager

    engine = PolicyEngine()
    sessions = SessionManager()
    row = db.fetchone(
        "SELECT COUNT(*) FROM security_events "
        "WHERE blocked = TRUE AND timestamp > NOW() - INTERVAL '1 hour'"
    )
    blocked_last_hour = int(row[0]) if row else 0
    return {
        "current_mode": engine.current_mode.value,
        "active_sessions": sessions.get_active_count(),
        "blocked_last_hour": blocked_last_hour,
    }


@router.get("/events")
async def recent_events(limit: int = 50):
    from app.core import db

    rows = db.fetchall(
        """
        SELECT user_id, ip_address, event_type, threat_type,
               severity, detail, blocked, timestamp
        FROM security_events
        ORDER BY timestamp DESC LIMIT %s
        """,
        (min(limit, 200),),
    )
    return {
        "events": [
            {
                "user_id": r[0],
                "ip": r[1],
                "event_type": r[2],
                "threat": r[3],
                "severity": r[4],
                "detail": r[5],
                "blocked": r[6],
                "timestamp": str(r[7]),
            }
            for r in rows
        ]
    }


@router.post("/mode/{mode}")
async def set_security_mode(mode: str):
    from app.core import db
    from app.security.policy_engine import SecurityMode

    valid = [m.value for m in SecurityMode]
    if mode not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Choose: {valid}")

    db.execute(
        "INSERT INTO system_security_mode (mode, triggered_by, set_at) VALUES (%s, 'admin', NOW())",
        (mode,),
    )

    if mode == "emergency":
        from app.security.session_manager import SessionManager

        count = SessionManager().emergency_invalidate_all(reason="admin_manual_emergency")
        return {"mode": mode, "sessions_cleared": count, "status": "emergency_activated"}

    return {"mode": mode, "status": "mode_updated"}


@router.post("/emergency/resolve")
async def resolve_emergency():
    from app.core import db

    db.execute(
        "INSERT INTO system_security_mode (mode, triggered_by, set_at) "
        "VALUES ('observe', 'admin', NOW())"
    )
    return {"status": "emergency_resolved", "new_mode": "observe"}


@router.get("/threat-summary")
async def threat_summary():
    from app.core import db

    rows = db.fetchall(
        """
        SELECT threat_type, COUNT(*), AVG(CASE WHEN blocked THEN 1 ELSE 0 END)
        FROM security_events
        WHERE timestamp > NOW() - INTERVAL '24 hours'
        GROUP BY threat_type
        ORDER BY COUNT(*) DESC
        """
    )
    return {
        "summary": [
            {
                "threat_type": r[0],
                "count": r[1],
                "block_rate": round(float(r[2] or 0) * 100, 1),
            }
            for r in rows
        ]
    }
