from __future__ import annotations

import sys
import textwrap
import types

import pytest

from llm_engine import GirivinityEngine
from llm_loader import GirivinityLoader, GirivinityLoaderConfig


class FakeLlama:
    init_calls = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        FakeLlama.init_calls.append(kwargs)

    def __call__(self, prompt: str, max_tokens: int = 512, stream: bool = False):
        if stream:
            return iter(
                [
                    {"choices": [{"text": "hello "}]},
                    {"choices": [{"text": "world"}]},
                ]
            )
        return {"choices": [{"text": f"answer:{prompt}:{max_tokens}"}]}


def _config(tmp_path):
    model_path = tmp_path / "models" / "girivinity_quantised" / "model.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"gguf")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            model:
              quantised_path: "{model_path}"
              n_ctx: 2048
              n_threads: 3
              n_gpu_layers: 0
            """
        ),
        encoding="utf-8",
    )
    return config_path, model_path


def test_girivinity_loader_config_loads_from_yaml(tmp_path):
    config_path, model_path = _config(tmp_path)

    config = GirivinityLoaderConfig.from_yaml(config_path)

    assert config.quantised_path == model_path
    assert config.n_ctx == 2048
    assert config.n_threads == 3
    assert config.n_gpu_layers == 0


def test_loader_singleton_loads_local_llama_once(tmp_path, monkeypatch):
    config_path, model_path = _config(tmp_path)
    GirivinityLoader.reset_cache()
    FakeLlama.init_calls.clear()
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))

    first = GirivinityLoader(config_path).get_model()
    second = GirivinityLoader(config_path).get_model()

    assert first is second
    assert len(FakeLlama.init_calls) == 1
    assert FakeLlama.init_calls[0] == {
        "model_path": str(model_path),
        "n_ctx": 2048,
        "n_threads": 3,
        "n_gpu_layers": 0,
    }


def test_loader_raises_when_quantised_model_missing(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'model:\n  quantised_path: "missing/model.gguf"\n',
        encoding="utf-8",
    )
    GirivinityLoader.reset_cache()

    with pytest.raises(FileNotFoundError, match="Girivinity model not built yet"):
        GirivinityLoader(config_path).get_model()


def test_girivinity_engine_generate_stream_and_context(tmp_path, monkeypatch):
    config_path, _model_path = _config(tmp_path)
    GirivinityLoader.reset_cache()
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    engine = GirivinityEngine(GirivinityLoader(config_path))

    assert "".join(engine.generate("prompt", max_tokens=5, stream=True)) == "hello world"
    assert engine.generate("prompt", max_tokens=5, stream=False) == "answer:prompt:5"
    assert "".join(engine.generate_with_context("What?", "Facts", user_level=1)) == "hello world"
