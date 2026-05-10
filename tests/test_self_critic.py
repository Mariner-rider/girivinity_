from app.agents.self_critic import SelfCritic


def test_self_critic_improves_incomplete_response():
    critic = SelfCritic()
    response = "This is the answer therefore use it TODO"
    result = critic.critique(response)

    assert result.confidence < 1.0
    assert "missing_structured_steps" in result.issues
    assert "Step 1:" in result.improved_response
    assert "TODO" not in result.improved_response


def test_self_critic_flags_contradictions():
    critic = SelfCritic()
    response = "This always works and it never fails."
    result = critic.critique(response)

    assert result.flagged_error is True
    assert any(issue.startswith("contradiction:") for issue in result.issues)
    assert 0.0 <= result.confidence <= 1.0
