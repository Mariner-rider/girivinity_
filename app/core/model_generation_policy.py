from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelGeneration:
    generation: int
    name: str
    target_params_billions: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    max_seq_len: int
    min_training_chunks: int
    min_training_tokens: int
    estimated_gpu_hours: float
    description: str


GENERATION_ROADMAP: list[ModelGeneration] = [
    ModelGeneration(
        generation=1,
        name="Girivinity-3B",
        target_params_billions=3.0,
        hidden_dim=2560,
        num_layers=32,
        num_heads=20,
        ffn_dim=6912,
        max_seq_len=32768,
        min_training_chunks=100_000,
        min_training_tokens=2_000_000_000,
        estimated_gpu_hours=1_000,
        description="First production-scale Girivinity generation. 3B parameters.",
    ),
    ModelGeneration(
        generation=2,
        name="Girivinity-7B",
        target_params_billions=7.0,
        hidden_dim=4096,
        num_layers=32,
        num_heads=32,
        ffn_dim=11008,
        max_seq_len=65536,
        min_training_chunks=500_000,
        min_training_tokens=10_000_000_000,
        estimated_gpu_hours=4_000,
        description="Scaled general-purpose Girivinity generation. 7B parameters.",
    ),
    ModelGeneration(
        generation=3,
        name="Girivinity-13B",
        target_params_billions=13.0,
        hidden_dim=5120,
        num_layers=40,
        num_heads=40,
        ffn_dim=13824,
        max_seq_len=131072,
        min_training_chunks=2_500_000,
        min_training_tokens=50_000_000_000,
        estimated_gpu_hours=16_000,
        description="Higher-capacity reasoning Girivinity generation. 13B parameters.",
    ),
    ModelGeneration(
        generation=4,
        name="Girivinity-34B",
        target_params_billions=34.0,
        hidden_dim=8192,
        num_layers=60,
        num_heads=64,
        ffn_dim=22016,
        max_seq_len=262144,
        min_training_chunks=12_500_000,
        min_training_tokens=250_000_000_000,
        estimated_gpu_hours=64_000,
        description="Large-scale expert Girivinity generation. 34B parameters.",
    ),
    ModelGeneration(
        generation=5,
        name="Girivinity-70B",
        target_params_billions=70.0,
        hidden_dim=12288,
        num_layers=80,
        num_heads=96,
        ffn_dim=32768,
        max_seq_len=524288,
        min_training_chunks=62_500_000,
        min_training_tokens=1_250_000_000_000,
        estimated_gpu_hours=256_000,
        description="Last hand-tuned roadmap generation before automatic extrapolation. 70B parameters.",
    ),
]


class GenerationPolicy:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config_path = Path(config_path)

    def get_current_generation(self) -> ModelGeneration:
        generation_number = self._configured_generation_number()
        return self._generation_for_number(generation_number)

    def get_next_generation(self) -> ModelGeneration:
        current = self.get_current_generation()
        last_defined = GENERATION_ROADMAP[-1]
        if current.generation < last_defined.generation:
            return self._generation_for_number(current.generation + 1)

        next_generation = self._extrapolate_next_generation(current)
        logger.info(
            "Beyond defined roadmap — extrapolating next generation: %s",
            next_generation.name,
        )
        return next_generation

    def status_report(self) -> dict[str, Any]:
        current = self.get_current_generation()
        next_generation = self.get_next_generation()
        report: dict[str, Any] = {
            "current_generation": asdict(current),
            "next_generation": asdict(next_generation),
            "roadmap": "extrapolated" if self._is_extrapolated(next_generation) else "defined",
        }
        if self._is_extrapolated(next_generation):
            report["next_generation_is_extrapolated"] = True
        return report

    def _extrapolate_next_generation(self, last_gen: ModelGeneration) -> ModelGeneration:
        next_gen_number = last_gen.generation + 1
        next_params = round(last_gen.target_params_billions * 2.0, 1)
        hidden_dim = _round_to_multiple(int(last_gen.hidden_dim * 1.4), 128)
        num_layers = _round_to_even(int(last_gen.num_layers * 1.25))
        num_heads = min(hidden_dim // 128, 128)
        ffn_dim = _round_to_multiple(int(hidden_dim * 2.67), 256)
        max_seq_len = min(last_gen.max_seq_len * 2, 1_000_000)
        rounded_params = round(next_params)
        return ModelGeneration(
            generation=next_gen_number,
            name=f"Girivinity-{rounded_params}B",
            target_params_billions=next_params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            min_training_chunks=last_gen.min_training_chunks * 5,
            min_training_tokens=last_gen.min_training_tokens * 5,
            estimated_gpu_hours=last_gen.estimated_gpu_hours * 4,
            description=(
                f"Auto-generated generation {next_gen_number}. "
                f"{rounded_params}B parameters."
            ),
        )

    def _configured_generation_number(self) -> int:
        try:
            text = self.config_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return 1
        in_successor_engine = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not line.startswith(" ") and stripped.endswith(":"):
                in_successor_engine = stripped == "successor_engine:"
                continue
            if in_successor_engine and stripped.startswith("current_generation:"):
                value = stripped.split(":", 1)[1].split("#", 1)[0].strip()
                return int(value or 1)
        return 1

    def _generation_for_number(self, generation_number: int) -> ModelGeneration:
        if generation_number < 1:
            raise ValueError("generation number must be >= 1")
        for generation in GENERATION_ROADMAP:
            if generation.generation == generation_number:
                return generation

        generation = GENERATION_ROADMAP[-1]
        while generation.generation < generation_number:
            generation = self._extrapolate_next_generation(generation)
        return generation

    @staticmethod
    def _is_extrapolated(generation: ModelGeneration) -> bool:
        return generation.generation > GENERATION_ROADMAP[-1].generation


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _round_to_even(value: int) -> int:
    rounded = round(value / 2) * 2
    return max(2, rounded)
