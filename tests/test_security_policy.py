import pytest

from app.finetune.update_gate import approve_model_update
from app.security.policy import SecurityGuard, SecurityPolicyError


def test_security_guard_requires_grounding():
    guard = SecurityGuard()
    with pytest.raises(SecurityPolicyError):
        guard.require_grounding(sources=[], context="")


def test_security_guard_scores_trusted_urls():
    guard = SecurityGuard()
    trusted = guard.score_url_trust("https://example.org/page")
    untrusted = guard.score_url_trust("ftp://localhost")

    assert trusted.trusted is True
    assert untrusted.trusted is False


def test_model_update_requires_benchmark_improvement():
    assert approve_model_update(0.72, 0.74, min_delta=0.01) is True
    with pytest.raises(SecurityPolicyError):
        approve_model_update(0.72, 0.721, min_delta=0.01)
