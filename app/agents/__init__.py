"""Agent package exports."""

__all__ = ["AgentController"]


def __getattr__(name: str):
    if name == "AgentController":
        from agent_controller import AgentController

        return AgentController
    raise AttributeError(name)
