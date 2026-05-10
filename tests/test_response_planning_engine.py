import json

from response_planning_engine import ResponsePlanningSystem


def test_response_planner_code_blueprint():
    planner = ResponsePlanningSystem()
    blueprint = planner.build_blueprint("Implement a Python function to parse CSV")

    assert blueprint.query_type == "technical_implementation"
    assert blueprint.response_format == "code"
    assert any(section["title"] == "Implementation" for section in blueprint.sections)


def test_response_planner_report_blueprint_json_output():
    planner = ResponsePlanningSystem()
    blueprint = planner.build_blueprint("Create a report with findings from sales analysis")
    payload = json.loads(blueprint.to_json())

    assert payload["response_format"] == "report"
    assert len(payload["sections"]) >= 3


def test_response_planner_step_by_step_format():
    planner = ResponsePlanningSystem()
    blueprint = planner.build_blueprint("How to deploy this service? give steps")

    assert blueprint.response_format == "step-by-step"
