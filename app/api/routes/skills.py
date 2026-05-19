from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.skill_forge import SkillForge

router = APIRouter(prefix="/skills", tags=["skills"])


class FeedbackRequest(BaseModel):
    skill_slug: str
    user_id: str
    score: float


@router.get("/")
async def list_skills():
    return {"skills": SkillForge().list_skills()}


@router.get("/{slug}")
async def get_skill(slug: str):
    skill = SkillForge()._load_skill(slug)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {
        "slug": skill.slug,
        "topic": skill.topic,
        "version": skill.version,
        "confidence": skill.confidence,
        "usage_count": skill.usage_count,
        "avg_feedback": skill.avg_feedback,
        "instructions": skill.instructions,
        "examples": skill.examples,
        "edge_cases": skill.edge_cases,
        "source_urls": skill.source_urls,
    }


@router.post("/feedback")
async def submit_skill_feedback(req: FeedbackRequest):
    if not 1.0 <= req.score <= 5.0:
        raise HTTPException(
            status_code=400, detail="Score must be 1.0-5.0"
        )
    SkillForge().update_skill_feedback(req.skill_slug, req.score)
    return {"status": "recorded"}


@router.post("/{slug}/evaluate")
async def evaluate_skill(slug: str):
    forge = SkillForge()
    skill = forge._load_skill(slug)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    result = forge.evaluate_skill(skill)
    return {
        "skill_slug": result.skill_slug,
        "score": result.score,
        "passed": result.passed,
        "total": result.total,
        "failures": result.failures,
        "improvement_vs_baseline": result.improvement_vs_baseline,
    }


@router.delete("/{slug}")
async def delete_skill(slug: str):
    from pathlib import Path
    import shutil
    skill_path = Path("skills") / slug
    if not skill_path.exists():
        raise HTTPException(status_code=404, detail="Skill not found")
    shutil.rmtree(skill_path)
    return {"status": "deleted", "slug": slug}
