from __future__ import annotations

import sys
import types

from model.inference import GirivinityInference


class FakeLlama:
    def __init__(self, **kwargs) -> None:
        assert kwargs["model_path"] == "models/gguf/girivinity-q4_k_m.gguf"
        self.kwargs = kwargs

    def __call__(self, prompt: str, max_tokens: int, stream: bool = True):
        assert prompt == "Namaste, tell me about India"
        assert max_tokens == 50
        if stream:
            return iter(
                [
                    {"choices": [{"text": "India "}]},
                    {"choices": [{"text": "is culturally rich."}]},
                ]
            )
        return {"choices": [{"text": "India is culturally rich."}]}


def test_load_model_and_generate_50_tokens_streaming(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))

    inference = GirivinityInference("models/gguf/girivinity-q4_k_m.gguf")
    output = "".join(inference.generate("Namaste, tell me about India", 50, stream=True))

    assert isinstance(output, str)
    assert output.strip()
