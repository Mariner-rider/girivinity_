import tempfile
from pathlib import Path


def test_agent_forge_classifies_research():
    from app.core.agent_forge import AgentForge, AgentType

    forge = AgentForge()
    result = forge.classify_request("research everything about ISRO Chandrayaan missions")
    assert result == AgentType.RESEARCH


def test_agent_forge_classifies_code():
    from app.core.agent_forge import AgentForge, AgentType

    forge = AgentForge()
    result = forge.classify_request("build a Python script to parse CSV files")
    assert result == AgentType.CODE


def test_agent_forge_classifies_legal():
    from app.core.agent_forge import AgentForge, AgentType

    forge = AgentForge()
    result = forge.classify_request("legal research on section 302 BNS murder case")
    assert result == AgentType.LEGAL


def test_agent_forge_creates_definition():
    from app.core.agent_forge import AgentForge

    forge = AgentForge()
    agent = forge.forge("research ISRO missions", "user1")
    assert agent.agent_id != ""
    assert agent.name != ""
    assert len(agent.steps) > 0
    assert len(agent.tools) > 0


def test_agent_forge_adapts_existing():
    from app.core.agent_forge import AgentForge

    forge = AgentForge()
    original = forge.forge("research NASA missions", "user1")
    adapted = forge.adapt(original, "research ESA missions")
    assert adapted.version == 2
    assert "ESA" in adapted.created_for


def test_agent_registry_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        from app.core.agent_forge import AgentForge
        from app.agents.agent_registry import AgentRegistry

        registry = AgentRegistry()
        registry.agents_dir = Path(tmp)
        agent = AgentForge().forge("test request", "user1")
        registry.save(agent)
        loaded = registry.load(agent.agent_id)
        assert loaded is not None
        assert loaded.agent_id == agent.agent_id
        assert loaded.name == agent.name


def test_agent_registry_find_similar():
    with tempfile.TemporaryDirectory() as tmp:
        from app.core.agent_forge import AgentForge
        from app.agents.agent_registry import AgentRegistry

        registry = AgentRegistry()
        registry.agents_dir = Path(tmp)
        agent = AgentForge().forge("research ISRO space missions India", "user1")
        registry.save(agent)
        similar, score = registry.find_similar("research ISRO Chandrayaan missions India")
        assert similar is not None
        assert score > 0.0


def test_orchestrator_detects_agent_request():
    from app.core.agent_orchestrator import AgentOrchestrator

    orch = AgentOrchestrator()
    assert orch.is_agent_request("create an agent to monitor stock prices") is True
    assert orch.is_agent_request("what is machine learning") is False
    assert orch.is_agent_request("build an agent that tracks news") is True


def test_agent_list():
    with tempfile.TemporaryDirectory() as tmp:
        from app.core.agent_forge import AgentForge
        from app.agents.agent_registry import AgentRegistry

        registry = AgentRegistry()
        registry.agents_dir = Path(tmp)
        for i in range(3):
            agent = AgentForge().forge(f"test request {i}", "user1")
            registry.save(agent)
        agents = registry.list_agents()
        assert len(agents) == 3
