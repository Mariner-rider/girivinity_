from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrchestrationResult:
    output: str
    agent_id: str
    agent_name: str
    agent_type: str
    action_taken: str
    steps_completed: int
    learned_chunks: int
    execution_time_s: float
    sources: list[dict]
    success: bool
    error: str = ""


class AgentOrchestrator:
    AGENT_REQUEST_TRIGGERS = [
        "create an agent",
        "build an agent",
        "make an agent",
        "i need an agent",
        "agent to",
        "agent that",
        "automate",
        "automatically",
        "run agent",
        "use agent",
        "deploy agent",
        "launch agent",
        "set up agent",
        "create a bot",
        "make a bot",
    ]

    def is_agent_request(self, query: str) -> bool:
        q = query.lower()
        return any(trigger in q for trigger in self.AGENT_REQUEST_TRIGGERS)

    def orchestrate(self, request: str, user_id: str = "anonymous") -> OrchestrationResult:
        from app.core.agent_forge import AgentForge
        from app.agents.agent_registry import AgentRegistry
        from app.core.agent_runner import AgentRunner

        forge = AgentForge()
        registry = AgentRegistry()
        runner = AgentRunner()
        similar, score = registry.find_similar(request, threshold=0.5)

        if similar and score >= 0.8:
            agent = forge.adapt(similar, request)
            action_taken = "reused"
        elif similar and score >= 0.5:
            agent = forge.adapt(similar, request)
            action_taken = "adapted"
        else:
            agent = forge.forge(request, user_id)
            action_taken = "created"

        result = runner.run(agent, user_id)
        registry.save(agent)
        return OrchestrationResult(
            output=result.output,
            agent_id=agent.agent_id,
            agent_name=agent.name,
            agent_type=agent.agent_type.value,
            action_taken=action_taken,
            steps_completed=result.steps_completed,
            learned_chunks=result.learned_chunks,
            execution_time_s=result.execution_time_s,
            sources=result.sources,
            success=result.success,
            error=result.error,
        )
