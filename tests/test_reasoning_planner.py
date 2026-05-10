import json

from agent_controller import AgentController
from app.reasoning_planner import ReasoningPlanner


def test_reasoning_planner_outputs_structured_json_plan():
    planner = ReasoningPlanner()
    plan = planner.build_plan("Build a secure analytics API")
    plan_json = plan.to_json()
    data = json.loads(plan_json)

    assert "parsed_query" in data
    assert "intent" in data
    assert "sub_tasks" in data
    assert "execution_plan" in data
    assert isinstance(data["execution_plan"], list)


def test_agent_controller_integrates_reasoning_planner_without_exposing_chain_of_thought():
    controller = AgentController()
    result = controller.execute("Create an API integration plan")

    assert "reasoning_plan" not in result
    assert isinstance(result["final"], str)
