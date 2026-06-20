import sys
from unittest.mock import MagicMock, patch

sys.modules.setdefault("yaml", MagicMock())
sys.modules.setdefault("app.core.db", MagicMock())

import app.security.emergency_shutdown as es
from app.security.emergency_shutdown import EmergencyShutdown
es.db = MagicMock()


def test_execute_returns_results_dict():
    shutdown = EmergencyShutdown()
    with patch("app.security.emergency_shutdown.Path") as mock_path:
        mock_path.return_value.parent.mkdir = MagicMock()
        mock_path.return_value.__truediv__ = MagicMock()
        with patch("app.security.emergency_shutdown.db.fetchone", return_value=(5,)):
            with patch("app.security.emergency_shutdown.db.execute"):
                with patch("builtins.open", MagicMock()):
                    result = shutdown.execute(
                        reason="test_threat",
                        triggered_by="test",
                    )
    assert "sessions_killed" in result
    assert result["mode_set"] == "emergency"
    assert result["reason"] == "test_threat"
