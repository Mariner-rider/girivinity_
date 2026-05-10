import textwrap

from llm_engine import LLMEngineConfig


def test_llm_engine_config_loads_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            app:
              name: test
            model:
              model_id: tiny-model
              load_in_4bit: true
              kv_cache: true
            crawler:
              max_depth: 1
              concurrent_requests: 1
              download_timeout_seconds: 1
              trust_threshold: 0.6
            training:
              min_validation_samples: 1
              max_training_epochs: 1
            security:
              prompt_max_chars: 1000
            feature_flags:
              enable_rag: true
            """
        ),
        encoding="utf-8",
    )

    config = LLMEngineConfig.from_yaml(config_path)

    assert config.model_id == "tiny-model"
    assert config.device_map == "auto"
    assert config.load_in_4bit is True
    assert config.kv_cache is True
