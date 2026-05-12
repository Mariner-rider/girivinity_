import tempfile
from pathlib import Path
from unittest.mock import patch


def _make_engine(tmp: str):
    cfg_path = Path(tmp) / "config.yaml"
    import yaml
    cfg = {
        "successor_engine": {
            "check_interval_seconds": 86400,
            "knowledge_base_threshold": 100000,
            "quality_score_threshold": 3.5,
            "versions_dir": f"{tmp}/versions",
            "notifications_path": f"{tmp}/notifications.jsonl",
            "corpus_dir": f"{tmp}/corpus",
        },
        "training": {"queue_db": f"{tmp}/queue.db"},
    }
    cfg_path.write_text(yaml.dump(cfg))
    with patch("builtins.open", side_effect=lambda p, *a, **k:
               open(p, *a, **k) if str(p) != "config.yaml"
               else open(cfg_path, *a, **k)):
        pass
    import os
    os.chdir(tmp)
    from app.core.successor_engine import SuccessorEngine
    return SuccessorEngine()


def test_write_and_read_notification():
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(tmp)
        engine._write_notification(
            version="20240101_000000",
            previous_version="none",
            improvement_percent=10.0,
            trained_on_chunks=500,
            perplexity=45.2,
        )
        notes = engine.get_notifications()
        assert len(notes) == 1
        assert notes[0]["version"] == "20240101_000000"
        assert notes[0]["status"] == "awaiting_admin_approval"


def test_approve_updates_status():
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(tmp)
        version = "20240101_000000"
        version_dir = Path(tmp) / "versions" / version
        version_dir.mkdir(parents=True)
        engine._write_notification(
            version=version, previous_version="none",
            improvement_percent=5.0, trained_on_chunks=100,
            perplexity=40.0,
        )
        ok = engine.approve_successor(version)
        assert ok is True
        notes = engine.get_notifications()
        assert notes[0]["status"] == "approved"


def test_no_trigger_below_thresholds():
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(tmp)
        with patch.object(engine, "_build_successor") as mock_build:
            with patch.object(engine, "_count_trained_chunks",
                              return_value=0):
                with patch.object(engine, "_rolling_quality_score",
                                  return_value=0.0):
                    engine._check_thresholds()
        mock_build.assert_not_called()
