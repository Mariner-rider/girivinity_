from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = Field(default="Girivinity", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    structured_logging: bool = Field(default=True, alias="STRUCTURED_LOGGING")
    auto_load_model: bool = Field(default=False, alias="AUTO_LOAD_MODEL")

    model_id: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", alias="MODEL_ID")
    model_device_map: str = Field(default="auto", alias="MODEL_DEVICE_MAP")
    model_load_in_4bit: bool = Field(default=True, alias="MODEL_LOAD_IN_4BIT")
    model_use_double_quant: bool = Field(default=True, alias="MODEL_USE_DOUBLE_QUANT")
    model_quant_type: str = Field(default="nf4", alias="MODEL_QUANT_TYPE")
    model_compute_dtype: str = Field(default="float16", alias="MODEL_COMPUTE_DTYPE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
