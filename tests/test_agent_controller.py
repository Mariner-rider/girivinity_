from agent_controller import AgentController
from app.core import query_router


def test_task_routing_includes_all_agents():
    controller = AgentController()
    route = controller.route_task("research quantum computing")
    names = [agent.name for agent in route]
    assert names == ["research_agent", "reasoning_agent", "critic_agent", "memory_agent"]


def test_execute_supports_shared_memory_and_inter_agent_messages(monkeypatch):
    class FakeQueryRouter:
        def route(self, task: str):
            return {
                "context_string": f"Context:\n[1] researched {task}\n\nSources: https://example.org/research",
                "confidence": 0.88,
                "urls": ["https://example.org/research"],
            }

    monkeypatch.setattr(query_router, "QueryRouter", FakeQueryRouter)
    controller = AgentController()
    result = controller.execute("build roadmap for retrieval system")

    result_agents = [entry["agent"] for entry in result["results"]]
    assert result_agents[-1] == "memory_agent"
    assert len(result["shared_memory"]["facts"]) >= 1
    assert "memory_summary" in result["shared_memory"]["notes"]
    assert len(result["inter_agent_messages"]) >= 2
