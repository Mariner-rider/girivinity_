from __future__ import annotations

import argparse
import importlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class QuantiseConfig:
    input_model_path: Path = Path("models/gguf/girivinity-f16.gguf")
    output_model_path: Path = Path("models/gguf/girivinity-q4_k_m.gguf")
    quantization: str = "Q4_K_M"

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config.yaml") -> QuantiseConfig:
        path = Path(config_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        raw = raw or {}
        section = (raw.get("model") or {}).get("quantisation") or raw.get("quantisation") or {}
        return cls(
            input_model_path=Path(section.get("input_model_path", "models/gguf/girivinity-f16.gguf")),
            output_model_path=Path(section.get("output_model_path", "models/gguf/girivinity-q4_k_m.gguf")),
            quantization=str(section.get("quantization", "Q4_K_M")),
        )


class GGUFQuantiser:
    """Quantise a trained GGUF checkpoint with llama-cpp-python / llama.cpp Q4_K_M."""

    def __init__(self, config: QuantiseConfig | None = None) -> None:
        self.config = config or QuantiseConfig.from_yaml()

    def quantise(self) -> Path:
        self.config.output_model_path.parent.mkdir(parents=True, exist_ok=True)
        llama_cpp = importlib.import_module("llama_cpp")
        if self._has_python_quantize_api(llama_cpp):
            self._quantise_with_python_bindings(llama_cpp)
        else:
            self._quantise_with_llama_cpp_binary(llama_cpp)
        return self.config.output_model_path

    def _has_python_quantize_api(self, llama_cpp) -> bool:
        return all(
            hasattr(llama_cpp, name)
            for name in (
                "llama_model_quantize",
                "llama_model_quantize_default_params",
                "LLAMA_FTYPE_MOSTLY_Q4_K_M",
            )
        )

    def _quantise_with_python_bindings(self, llama_cpp) -> None:
        params = llama_cpp.llama_model_quantize_default_params()
        params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M
        rc = llama_cpp.llama_model_quantize(
            str(self.config.input_model_path).encode("utf-8"),
            str(self.config.output_model_path).encode("utf-8"),
            params,
        )
        if rc != 0:
            raise RuntimeError(f"llama.cpp quantisation failed with status code {rc}")

    def _quantise_with_llama_cpp_binary(self, llama_cpp) -> None:
        package_dir = Path(llama_cpp.__file__).resolve().parent
        candidates = [package_dir / "llama-quantize", package_dir / "bin" / "llama-quantize"]
        quantize_binary = next((candidate for candidate in candidates if candidate.exists()), None)
        if quantize_binary is None:
            raise RuntimeError(
                "llama.cpp quantise bindings/binary were not found in llama-cpp-python. "
                "Install llama-cpp-python with llama.cpp quantisation support."
            )
        subprocess.run(
            [
                str(quantize_binary),
                str(self.config.input_model_path),
                str(self.config.output_model_path),
                self.config.quantization,
            ],
            check=True,
        )


def quantise(config: QuantiseConfig | None = None) -> Path:
    return GGUFQuantiser(config).quantise()


def quantize(config: QuantiseConfig | None = None) -> Path:
    return quantise(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantise Girivinity model to GGUF Q4_K_M.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    output = GGUFQuantiser(QuantiseConfig.from_yaml(args.config)).quantise()
    print(output)
