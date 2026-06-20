"""RASP API — third-party API for Runtime Application Self-Protection."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl

from app.security.rasp import RASPEngine, RASPEvent

rasp_router = APIRouter(prefix="/rasp/v1", tags=["RASP"])


class InspectInputRequest(BaseModel):
    text: str
    context: str = Field(default="ai_input", pattern="^(ai_input|api_input|web_input)$")
    client_id: str | None = "anonymous"


class InspectOutputRequest(BaseModel):
    text: str
    client_id: str | None = "anonymous"
    grounded_sources: list[Any] | None = None


class InspectResponse(BaseModel):
    clean: bool
    sanitised_text: str
    threat_count: int
    threats: list[dict[str, Any]]
    blocked: bool
    processing_ms: float


class WebhookRegistration(BaseModel):
    url: HttpUrl
    secret: str
    severity_filter: list[str] = Field(default_factory=lambda: ["high", "critical"])
    event_types: list[str] = Field(default_factory=list)


_WEBHOOKS: list[WebhookRegistration] = []


def get_rasp_engine(request: Request) -> RASPEngine:
    engine = getattr(request.app.state, "rasp_engine", None) or getattr(request.app.state, "rasp", None)
    if engine is None:
        engine = RASPEngine()
        request.app.state.rasp_engine = engine
        request.app.state.rasp = engine
    return engine


def validate_api_key(x_rasp_key: str | None = Header(default=None, alias="X-RASP-Key")) -> str:
    valid_keys = [key.strip() for key in os.environ.get("RASP_API_KEYS", "").split(",") if key.strip()]
    if valid_keys and x_rasp_key in valid_keys:
        return str(x_rasp_key)
    if not valid_keys and os.environ.get("ENVIRONMENT", "development").lower() in {"dev", "development", "test"}:
        return x_rasp_key or "development"
    raise HTTPException(status_code=401, detail="Invalid RASP API key")


def _event_payload(event: RASPEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "category": event.category.value,
        "severity": event.severity.value,
        "description": event.description,
        "blocked": event.blocked,
        "recommended_action": event.recommended_action,
        "source_ip": event.source_ip,
        "client_id": event.client_id,
    }


def _response(text: str, events: list[RASPEvent], blocked: bool, processing_ms: float) -> InspectResponse:
    return InspectResponse(clean=not events, sanitised_text=text, threat_count=len(events), threats=[_event_payload(event) for event in events], blocked=blocked, processing_ms=processing_ms)


@rasp_router.post("/inspect/input", response_model=InspectResponse)
async def inspect_input(body: InspectInputRequest, request: Request, api_key: str | None = Header(default=None, alias="X-RASP-Key")) -> InspectResponse:
    key = validate_api_key(api_key)
    rasp = get_rasp_engine(request)
    source_ip = request.client.host if request.client else "unknown"
    start = time.perf_counter()
    events = rasp.inspect_input(body.text, client_id=body.client_id or key, source_ip=source_ip, context=body.context)
    processing_ms = round((time.perf_counter() - start) * 1000, 2)
    blocked = any(event.blocked for event in events)
    sanitised = "" if blocked else body.text
    if not blocked and events:
        sanitised = events[-1].sanitised_input or body.text
    return _response(sanitised, events, blocked, processing_ms)


@rasp_router.post("/inspect/output", response_model=InspectResponse)
async def inspect_output(body: InspectOutputRequest, request: Request, api_key: str | None = Header(default=None, alias="X-RASP-Key")) -> InspectResponse:
    key = validate_api_key(api_key)
    rasp = get_rasp_engine(request)
    start = time.perf_counter()
    sanitised_text, events = rasp.inspect_output(body.text, client_id=body.client_id or key, grounded_sources=body.grounded_sources)
    processing_ms = round((time.perf_counter() - start) * 1000, 2)
    return _response(sanitised_text, events, False, processing_ms)


@rasp_router.post("/webhooks")
async def register_webhook(body: WebhookRegistration, api_key: str | None = Header(default=None, alias="X-RASP-Key")) -> dict[str, Any]:
    validate_api_key(api_key)
    _WEBHOOKS.append(body)
    return {"registered": True, "webhook_count": len(_WEBHOOKS), "severity_filter": body.severity_filter, "event_types": body.event_types}


@rasp_router.get("/threats/summary")
async def threat_summary(request: Request, hours_back: int = 24, api_key: str | None = Header(default=None, alias="X-RASP-Key")) -> dict[str, Any]:
    validate_api_key(api_key)
    rasp = get_rasp_engine(request)
    return rasp.get_threat_summary(hours_back)


@rasp_router.get("/threats/stream")
async def threat_stream(request: Request, api_key: str | None = Header(default=None, alias="X-RASP-Key")) -> StreamingResponse:
    validate_api_key(api_key)
    rasp = get_rasp_engine(request)

    async def generate():
        last_pos = 0
        while True:
            path = Path(rasp.audit_log_path)
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    handle.seek(last_pos)
                    lines = handle.readlines()
                    last_pos = handle.tell()
                for line in lines:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield f"data: {line.strip()}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


__all__ = ["rasp_router", "get_rasp_engine", "validate_api_key"]
