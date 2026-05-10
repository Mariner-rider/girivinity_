from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config_loader import ConfigLoader


class SystemConfigError(ValueError):
    """Raised when base YAML configuration is missing required sections."""


REQUIRED_MODULES = (
    "core",
    "llm",
    "memory",
    "agents",
    "crawler",
    "rag",
    "security",
    "analytics",
    "multimodal",
    "training",
)


def load_system_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    raw = ConfigLoader(config_path).load().raw
    modules = raw.get("modules", {})
    missing = [name for name in REQUIRED_MODULES if name not in modules]
    if missing:
        raise SystemConfigError(f"Missing module config sections: {', '.join(missing)}")
    return raw


@lru_cache(maxsize=1)
def get_system_config(path: str = "config.yaml") -> dict[str, Any]:
    return load_system_config(path)
