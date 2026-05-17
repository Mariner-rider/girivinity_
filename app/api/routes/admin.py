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


@router.get("/generation-roadmap")
async def generation_roadmap():
    from app.core.successor_engine import (
        GENERATION_CONFIGS,
        CHUNK_THRESHOLDS_FOR_GENERATION,
        SuccessorEngine,
    )
    engine        = SuccessorEngine()
    total_trained = engine._count_trained_chunks()
    current_gen   = engine._get_current_generation()

    roadmap = []
    for gen, cfg in GENERATION_CONFIGS.items():
        threshold = CHUNK_THRESHOLDS_FOR_GENERATION.get(gen, 0)
        roadmap.append({
            "generation":       gen,
            "description":      cfg["description"],
            "dim":              cfg["dim"],
            "n_layers":         cfg["n_layers"],
            "chunks_required":  threshold,
            "chunks_trained":   total_trained,
            "progress_pct": round(
                min(100.0, total_trained / max(threshold, 1) * 100), 1
            ) if threshold > 0 else 100.0,
            "status": (
                "active"   if gen == current_gen else
                "complete" if gen < current_gen  else
                "pending"
            ),
        })

    return {
        "current_generation": current_gen,
        "total_chunks_trained": total_trained,
        "roadmap": roadmap,
    }
