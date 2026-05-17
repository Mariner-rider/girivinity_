from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.security.tenant_security import TenantSecurity

router = APIRouter(prefix="/tenant/security", tags=["tenant-security"])


class SecurityConfigUpdate(BaseModel):
    observe_threshold: float = 0.3
    guard_threshold: float = 0.6
    contain_threshold: float = 0.9
    rate_limit_rpm: int = 500
    block_prompt_injection: bool = True
    block_sql_injection: bool = True
    block_xss: bool = True
    block_ssrf: bool = True
    alert_email: str = ""
    custom_blocked_patterns: list[str] = []


@router.get("/config")
async def get_my_config(x_api_key: str = Header(...)):
    config = TenantSecurity().get_config(x_api_key)
    return {
        "api_key": x_api_key,
        "observe_threshold": config.observe_threshold,
        "guard_threshold": config.guard_threshold,
        "contain_threshold": config.contain_threshold,
        "rate_limit_rpm": config.rate_limit_rpm,
        "block_prompt_injection": config.block_prompt_injection,
        "block_sql_injection": config.block_sql_injection,
        "block_xss": config.block_xss,
        "block_ssrf": config.block_ssrf,
    }


@router.put("/config")
async def update_my_config(updates: SecurityConfigUpdate, x_api_key: str = Header(...)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    ok = TenantSecurity().update_config(x_api_key, updates.model_dump())
    if not ok:
        raise HTTPException(status_code=500, detail="Config update failed")
    return {"status": "updated", "api_key": x_api_key}
