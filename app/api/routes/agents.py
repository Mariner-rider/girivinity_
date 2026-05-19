from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentRequest(BaseModel):
    request: str
    user_id: str = "anonymous"


@router.post("/run")
async def run_agent(req: AgentRequest):
    if not req.request.strip():
        raise HTTPException(status_code=400, detail="Agent request cannot be empty")

    from app.core.agent_orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    if not orchestrator.is_agent_request(req.request):
        return {
            "status": "not_agent_request",
            "message": "This doesn't look like an agent request. Try: 'Create an agent to research X' or 'Build an agent that monitors Y'.",
        }

    result = orchestrator.orchestrate(req.request, req.user_id)
    return {
        "success": result.success,
        "agent_id": result.agent_id,
        "agent_name": result.agent_name,
        "agent_type": result.agent_type,
        "action_taken": result.action_taken,
        "output": result.output,
        "steps_completed": result.steps_completed,
        "learned_chunks": result.learned_chunks,
        "execution_time_s": result.execution_time_s,
        "sources_used": len(result.sources),
        "error": result.error,
    }


@router.get("/list")
async def list_agents():
    from app.core.agent_registry import AgentRegistry

    agents = AgentRegistry().list_agents()
    return {"agents": agents, "total": len(agents)}


@router.get("/{agent_id}")
async def get_agent(agent_id: str):
    from app.core.agent_registry import AgentRegistry

    agent = AgentRegistry().load(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return asdict(agent)


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    from app.core.agent_registry import AgentRegistry

    ok = AgentRegistry().delete(agent_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted", "agent_id": agent_id}


@router.post("/{agent_id}/run")
async def run_existing_agent(agent_id: str, req: AgentRequest):
    from app.core.agent_registry import AgentRegistry
    from app.core.agent_runner import AgentRunner

    agent = AgentRegistry().load(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    result = AgentRunner().run(agent, req.user_id)
    return {
        "success": result.success,
        "output": result.output,
        "steps_completed": result.steps_completed,
        "learned_chunks": result.learned_chunks,
        "execution_time_s": result.execution_time_s,
        "error": result.error,
    }
