import sys
from unittest.mock import MagicMock, patch

sys.modules.setdefault("yaml", MagicMock())

from app.security.policy_engine import PolicyEngine, SecurityMode


def _engine() -> PolicyEngine:
    with patch("app.security.policy_engine.PolicyEngine._get_system_mode",
               return_value=SecurityMode.OBSERVE):
        with patch("yaml.safe_load", return_value={"cyber_shield": {}}):
            e = PolicyEngine.__new__(PolicyEngine)
            e.observe_threshold  = 0.3
            e.guard_threshold    = 0.6
            e.contain_threshold  = 0.9
            e.current_mode = SecurityMode.OBSERVE
            return e


def test_clean_traffic_observe():
    e = _engine()
    d = e.decide(0.0, "clean", False, 0.0)
    assert d.mode == SecurityMode.OBSERVE
    assert d.block_request is False


def test_moderate_threat_guard():
    e = _engine()
    d = e.decide(0.5, "xss", False, 0.3)
    assert d.mode == SecurityMode.GUARD


def test_high_threat_contain():
    e = _engine()
    d = e.decide(0.8, "xss", True, 0.5)
    assert d.mode == SecurityMode.CONTAIN
    assert d.block_request is True


def test_critical_ssrf_triggers_emergency():
    e = _engine()
    with patch.object(e, "_escalate_mode"):
        d = e.decide(0.9, "ssrf", False, 0.0)
    assert d.mode == SecurityMode.EMERGENCY
    assert d.alert_admin is True


def test_emergency_system_mode_blocks_all():
    e = _engine()
    e.current_mode = SecurityMode.EMERGENCY
    d = e.decide(0.0, "clean", False, 0.0)
    assert d.mode == SecurityMode.EMERGENCY
    assert d.block_request is True
