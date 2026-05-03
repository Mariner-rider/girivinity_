import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.core.config import Settings
from app.monitoring.metrics import MODEL_LOAD_SECONDS

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(slots=True)
class LoadedLLM:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM


class QuantizedLLMLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _build_bnb_config(self) -> BitsAndBytesConfig:
        dtype = _DTYPE_MAP.get(self.settings.model_compute_dtype.lower(), torch.float16)
        return BitsAndBytesConfig(
            load_in_4bit=self.settings.model_load_in_4bit,
            bnb_4bit_use_double_quant=self.settings.model_use_double_quant,
            bnb_4bit_quant_type=self.settings.model_quant_type,
            bnb_4bit_compute_dtype=dtype,
        )

    def load(self) -> LoadedLLM:
        logger.info("Loading quantized model", extra={"model_id": self.settings.model_id})

        with MODEL_LOAD_SECONDS.time():
            quantization_config = self._build_bnb_config()
            tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_id,
                quantization_config=quantization_config,
                device_map=self.settings.model_device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )

        logger.info("Model loaded successfully", extra={"model_id": self.settings.model_id})
        return LoadedLLM(tokenizer=tokenizer, model=model)
