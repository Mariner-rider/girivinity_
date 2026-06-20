from __future__ import annotations
import logging
from typing import Iterator

        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        dtype = getattr(torch, self.config.torch_dtype, torch.float16)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map=self.config.device_map,
            torch_dtype=dtype,
            load_in_4bit=self.config.load_in_4bit,
            trust_remote_code=self.config.trust_remote_code,
            attn_implementation=self.config.attn_implementation,
        )

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class GirivinityEngine:
    def __init__(self, loader: GirivinityLoader | None = None) -> None:
        self._loader = loader or GirivinityLoader.instance()
        self._model = self._loader.get_model()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stream: bool = True,
    ) -> Iterator[str] | str:
        if stream:
            return self._stream(prompt, max_tokens)
        result = self._model(prompt, max_tokens=max_tokens, stream=False)
        return result["choices"][0]["text"]

    def _stream(self, prompt: str, max_tokens: int) -> Iterator[str]:
        for chunk in self._model(prompt, max_tokens=max_tokens, stream=True):
            token = chunk["choices"][0].get("text", "")
            if token:
                yield token
