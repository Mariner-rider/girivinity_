"""Local Girivinity GGUF loader backed by llama-cpp-python only."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml


_NOT_BUILT_MESSAGE = "Girivinity model not built yet. Run: python model/quantise.py first"


def _default_threads() -> int:
    return max((os.cpu_count() or 2) - 1, 1)


def _int_or_default(value: Any, default: int) -> int:
    return default if value is None else int(value)


@dataclass(frozen=True, slots=True)
class GirivinityLoaderConfig:
    quantised_path: Path = Path("models/girivinity_quantised/girivinity-q4_k_m.gguf")
    n_ctx: int = 4096
    n_threads: int = _default_threads()
    n_gpu_layers: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "GirivinityLoaderConfig":
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        raw = raw or {}
        model = raw.get("model") or {}
        quantisation = model.get("quantisation") or {}
        return cls(
            quantised_path=Path(
                model.get(
                    "quantised_path",
                    quantisation.get(
                        "output_model_path",
                        "models/girivinity_quantised/girivinity-q4_k_m.gguf",
                    ),
                )
            ),
            n_ctx=_int_or_default(model.get("n_ctx"), 4096),
            n_threads=_int_or_default(model.get("n_threads"), _default_threads()),
            n_gpu_layers=_int_or_default(model.get("n_gpu_layers"), 0),
        )


class GirivinityLoader:
    """Singleton loader for the local quantised Girivinity GGUF model."""

    _model: ClassVar[Any | None] = None
    _model_path: ClassVar[Path | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = GirivinityLoaderConfig.from_yaml(config_path)

    def get_model(self):
        if not self.config.quantised_path.exists():
            raise FileNotFoundError(_NOT_BUILT_MESSAGE)

        with self._lock:
            if self.__class__._model is None or self.__class__._model_path != self.config.quantised_path:
                from llama_cpp import Llama

                self.__class__._model = Llama(
                    model_path=str(self.config.quantised_path),
                    n_ctx=self.config.n_ctx,
                    n_threads=self.config.n_threads,
                    n_gpu_layers=self.config.n_gpu_layers,
                )
                self.__class__._model_path = self.config.quantised_path
            return self.__class__._model

    @classmethod
    def reset_cache(cls) -> None:
        """Clear singleton state for tests or explicit reloads."""
        with cls._lock:
            cls._model = None
            cls._model_path = None


# Compatibility helper for older callers that imported load_from_yaml().
def load_from_yaml(path: str | Path = "config.yaml"):
    return GirivinityLoader(path).get_model()
