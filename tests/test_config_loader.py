import textwrap

import pytest

from app.core.config_loader import ConfigLoader
from app.core.config_schema import ConfigValidationError, validate_config


BASE_CONFIG = """
app:
  name: Girivinity
  environment: test
model:
  model_id: tiny-model
crawler:
  max_depth: 2
  concurrent_requests: 4
  download_timeout_seconds: 10
  trust_threshold: 0.7
training:
  min_validation_samples: 2
  min_benchmark_delta: 0.01
security:
  require_grounding: true
  prompt_max_chars: 1000
feature_flags:
  enable_rag: true
"""


def test_config_loader_reads_yaml_and_feature_flags(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(textwrap.dedent(BASE_CONFIG), encoding="utf-8")

    loader = ConfigLoader(config_path)
    config = loader.load()

    assert config.model.model_id == "tiny-model"
    assert config.crawler.trust_threshold == 0.7
    assert loader.feature_enabled("enable_rag") is True


def test_config_loader_environment_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(textwrap.dedent(BASE_CONFIG), encoding="utf-8")
    monkeypatch.setenv("GIRIVINITY__MODEL__MODEL_ID", "env-model")
    monkeypatch.setenv("GIRIVINITY__FEATURE_FLAGS__ENABLE_TRAINING", "true")

    config = ConfigLoader(config_path).load()

    assert config.model.model_id == "env-model"
    assert config.feature_flags["enable_training"] is True


def test_config_loader_reload_if_changed(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(textwrap.dedent(BASE_CONFIG), encoding="utf-8")
    loader = ConfigLoader(config_path)
    assert loader.load().model.model_id == "tiny-model"

    updated = BASE_CONFIG.replace("tiny-model", "reloaded-model")
    config_path.write_text(textwrap.dedent(updated), encoding="utf-8")

    assert loader.reload().model.model_id == "reloaded-model"


def test_validation_schema_rejects_invalid_thresholds():
    with pytest.raises(ConfigValidationError):
        validate_config(
            {
                "app": {},
                "model": {"model_id": "x"},
                "crawler": {"trust_threshold": 2.0},
                "training": {},
                "security": {},
            }
        )
