import pytest
from model.base_model import GirivinityBaseModel, BASE_MODEL_REGISTRY


def test_registry_contains_required_models():
    required = [
        "llama-3.1-70b",
        "llama-3.1-8b",
        "qwen-2.5-72b",
        "mistral-small-22b",
        "girivinity-360m",
    ]
    for key in required:
        assert key in BASE_MODEL_REGISTRY, f"Missing: {key}"


def test_list_models_returns_all_keys():
    models = GirivinityBaseModel.list_models()
    keys = [m["key"] for m in models]
    assert "girivinity-360m" in keys
    assert "llama-3.1-70b" in keys
    assert len(models) >= 4


def test_girivinity_360m_has_no_hf_id():
    info = BASE_MODEL_REGISTRY["girivinity-360m"]
    assert info["hf_id"] is None


def test_girivinity_360m_vram_under_1gb():
    info = BASE_MODEL_REGISTRY["girivinity-360m"]
    assert info["vram_4bit_gb"] < 1.0


def test_llama_70b_vram_requires_gpu():
    info = BASE_MODEL_REGISTRY["llama-3.1-70b"]
    assert info["vram_4bit_gb"] > 30


def test_invalid_model_key_raises_value_error():
    import pytest
    from unittest.mock import patch
    with patch("yaml.safe_load", return_value={
        "base_model": {
            "model_key": "nonexistent-model-xyz",
            "load_in_4bit": False,
            "hf_token": "",
            "cache_dir": "models/hf_cache",
        }
    }):
        with pytest.raises(ValueError, match="Unknown model key"):
            GirivinityBaseModel()


def test_all_registry_entries_have_required_fields():
    required_fields = [
        "hf_id", "params", "vram_fp16_gb",
        "vram_4bit_gb", "context_len", "recommended_gpu",
    ]
    for key, info in BASE_MODEL_REGISTRY.items():
        for field in required_fields:
            assert field in info, (
                f"Model '{key}' missing field '{field}'"
            )
