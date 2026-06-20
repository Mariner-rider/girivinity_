"""Agent package exports."""

__all__ = ["AgentController", "AgentRegistry", "AgentTypeDefinition", "CapabilityMerger", "AdaptiveAgentExecutor"]


def __getattr__(name: str):
    if name == "AgentController":
        from agent_controller import AgentController

        return AgentController
    if name in {"AgentRegistry", "AgentTypeDefinition"}:
        from app.agents.agent_registry import AgentRegistry, AgentTypeDefinition

        return {"AgentRegistry": AgentRegistry, "AgentTypeDefinition": AgentTypeDefinition}[name]
    if name == "CapabilityMerger":
        from app.agents.capability_merger import CapabilityMerger

        return CapabilityMerger
    if name == "AdaptiveAgentExecutor":
        from app.agents.adaptive_agent_executor import AdaptiveAgentExecutor

        return AdaptiveAgentExecutor
    raise AttributeError(name)
