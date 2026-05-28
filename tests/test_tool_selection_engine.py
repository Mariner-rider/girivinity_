from app.engines.tool_selection_engine import ToolSelectionEngine


def test_selects_api_call_for_realtime_queries():
    engine = ToolSelectionEngine()
    result = engine.select("What is the latest weather in New York today?")
    assert result.decision == "API call"


def test_selects_rag_for_knowledge_base_queries():
    engine = ToolSelectionEngine()
    result = engine.select("According to our policy document, what are refund terms? cite source")
    assert result.decision == "RAG"


def test_selects_agent_workflow_for_complex_multistep_tasks():
    engine = ToolSelectionEngine()
    result = engine.select("Plan and orchestrate a multi-step migration workflow with risk analysis")
    assert result.decision == "agent workflow"


def test_selects_llm_only_for_simple_generation():
    engine = ToolSelectionEngine()
    result = engine.select("Summarize this paragraph in simple English")
    assert result.decision == "LLM only"
