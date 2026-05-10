from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when centralized YAML configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ModelSettings:
    model_id: str
    device_map: str = "auto"
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    kv_cache: bool = True


@dataclass(frozen=True, slots=True)
class CrawlerLimits:
    max_depth: int = 2
    concurrent_requests: int = 32
    download_timeout_seconds: int = 15
    trust_threshold: float = 0.6
    obey_robots_txt: bool = True


@dataclass(frozen=True, slots=True)
class TrainingThresholds:
    min_validation_samples: int = 1
    min_benchmark_delta: float = 0.0
    max_training_epochs: int = 1
    require_validation_dataset: bool = True
    require_benchmark_improvement: bool = True


@dataclass(frozen=True, slots=True)
class SecurityPolicies:
    require_grounding: bool = True
    require_trusted_crawler_urls: bool = True
    prompt_max_chars: int = 12000
    allow_remote_code: bool = False


@dataclass(frozen=True, slots=True)
class AppSettings:
    name: str = "Girivinity"
    environment: str = "dev"
    log_level: str = "INFO"
    structured_logging: bool = True
    auto_load_model: bool = False


@dataclass(frozen=True, slots=True)
class CentralConfig:
    app: AppSettings
    model: ModelSettings
    crawler: CrawlerLimits
    training: TrainingThresholds
    security: SecurityPolicies
    feature_flags: dict[str, bool] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _coerce_dataclass(cls, values: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigValidationError(f"Missing or invalid mapping: {key}")
    return value


def _validate_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ConfigValidationError(f"{name} must be >= 1")


def _validate_ratio(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ConfigValidationError(f"{name} must be between 0.0 and 1.0")


def validate_config(raw: dict[str, Any]) -> CentralConfig:
    app_raw = _require_mapping(raw, "app")
    model_raw = _require_mapping(raw, "model")
    crawler_raw = _require_mapping(raw, "crawler")
    training_raw = _require_mapping(raw, "training")
    security_raw = _require_mapping(raw, "security")
    flags_raw = raw.get("feature_flags", {})

    if not isinstance(flags_raw, dict):
        raise ConfigValidationError("feature_flags must be a mapping")

    model = _coerce_dataclass(ModelSettings, model_raw)
    crawler = _coerce_dataclass(CrawlerLimits, crawler_raw)
    training = _coerce_dataclass(TrainingThresholds, training_raw)
    security = _coerce_dataclass(SecurityPolicies, security_raw)
    app = _coerce_dataclass(AppSettings, app_raw)
    feature_flags = {str(key): bool(value) for key, value in flags_raw.items()}

    if not model.model_id.strip():
        raise ConfigValidationError("model.model_id must be non-empty")
    _validate_positive_int("crawler.max_depth", crawler.max_depth)
    _validate_positive_int("crawler.concurrent_requests", crawler.concurrent_requests)
    _validate_positive_int("crawler.download_timeout_seconds", crawler.download_timeout_seconds)
    _validate_ratio("crawler.trust_threshold", crawler.trust_threshold)
    _validate_positive_int("training.min_validation_samples", training.min_validation_samples)
    _validate_positive_int("training.max_training_epochs", training.max_training_epochs)
    if security.prompt_max_chars < 100:
        raise ConfigValidationError("security.prompt_max_chars must be >= 100")

    return CentralConfig(
        app=app,
        model=model,
        crawler=crawler,
        training=training,
        security=security,
        feature_flags=feature_flags,
        raw=raw,
    )
