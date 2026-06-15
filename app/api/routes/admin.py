from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.successor_engine import SuccessorEngine

router = APIRouter(prefix="/admin", tags=["admin"])


class FeedbackRequest(BaseModel):
    user_id: str
    score: float


@router.get("/notifications")
async def get_notifications():
    return {"notifications": SuccessorEngine().get_notifications()}


@router.post("/approve-successor/{version}")
async def approve_successor(version: str):
    ok = SuccessorEngine().approve_successor(version)
    if not ok:
        raise HTTPException(
            status_code=404, detail=f"Version {version} not found"
        )
    return {"status": "approved", "version": version}


@router.post("/reject-successor/{version}")
async def reject_successor(version: str):
    SuccessorEngine().reject_successor(version)
    return {"status": "rejected", "version": version}


@router.get("/model-versions")
async def list_model_versions():
    from pathlib import Path
    import yaml
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    versions_dir = Path(cfg["successor_engine"]["versions_dir"])
    if not versions_dir.exists():
        return {"versions": []}
    versions = sorted(
        [d.name for d in versions_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    return {"versions": versions}


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    if not 1.0 <= req.score <= 5.0:
        raise HTTPException(
            status_code=400, detail="Score must be between 1.0 and 5.0"
        )
    SuccessorEngine().log_feedback(req.user_id, req.score)
    return {"status": "recorded"}
