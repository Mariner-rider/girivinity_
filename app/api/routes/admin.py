from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.successor_engine import (
    approve_successor,
    list_model_versions,
    read_notifications,
    reject_successor,
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/notifications")
def notifications() -> dict[str, object]:
    return {"notifications": read_notifications()}


@router.post("/approve-successor/{version}")
def approve(version: str) -> dict[str, object]:
    try:
        return approve_successor(version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/reject-successor/{version}")
def reject(version: str) -> dict[str, object]:
    try:
        return reject_successor(version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/model-versions")
def model_versions() -> dict[str, object]:
    return {"versions": list_model_versions()}
