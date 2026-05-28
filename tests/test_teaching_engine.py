import sys
from unittest.mock import MagicMock

sys.modules.setdefault("yaml", MagicMock())

from app.core.teaching_engine import TeachingEngine


def test_detects_teaching_request():
    engine = TeachingEngine()
    assert engine.is_teaching_request("teach me Python") is True
    assert engine.is_teaching_request("explain to me how ML works") is True
    assert engine.is_teaching_request("weather update") is False


def test_detects_beginner_level():
    engine = TeachingEngine()
    profile = engine.build_learner_profile(
        "I am a complete beginner teach me Python", "user1"
    )
    assert profile.level in ("absolute_beginner", "beginner")


def test_detects_subject():
    engine = TeachingEngine()
    profile = engine.build_learner_profile(
        "teach me machine learning from scratch", "user1"
    )
    assert "machine" in profile.subject.lower() or "learning" in profile.subject.lower()


def test_system_injection_not_empty_for_teaching():
    engine = TeachingEngine()
    injection = engine.get_prompt_injection(
        "teach me how to code Python", "user1"
    )
    assert len(injection) > 20
    assert "teach" in injection.lower() or "learner" in injection.lower()


def test_non_teaching_returns_empty_injection():
    engine = TeachingEngine()
    injection = engine.get_prompt_injection(
        "capital of France", "user1"
    )
    assert injection == ""
