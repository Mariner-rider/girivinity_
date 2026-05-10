from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from app.core.config_schema import CentralConfig, validate_config


class ConfigLoader:
    """YAML configuration loader with env overrides and mtime-based reload support."""

    def __init__(self, path: str | Path = "config.yaml", env_prefix: str = "GIRIVINITY") -> None:
        self.path = Path(path)
        self.env_prefix = env_prefix
        self._mtime_ns: int | None = None
        self._config: CentralConfig | None = None

    def load(self) -> CentralConfig:
        raw = self._load_yaml()
        raw = self._apply_environment_overrides(raw)
        config = validate_config(raw)
        self._mtime_ns = self.path.stat().st_mtime_ns
        self._config = config
        return config

    def get(self) -> CentralConfig:
        if self._config is None:
            return self.load()
        return self._config

    def reload_if_changed(self) -> CentralConfig:
        current_mtime = self.path.stat().st_mtime_ns
        if self._config is None or self._mtime_ns != current_mtime:
            return self.load()
        return self._config

    def reload(self) -> CentralConfig:
        return self.load()

    def feature_enabled(self, flag_name: str, default: bool = False) -> bool:
        return self.get().feature_flags.get(flag_name, default)

    def _load_yaml(self) -> dict[str, Any]:
        return _parse_yaml_mapping(self.path.read_text(encoding="utf-8"))

    def _apply_environment_overrides(self, raw: dict[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(raw)
        prefix = f"{self.env_prefix}__"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            path = key.removeprefix(prefix).lower().split("__")
            self._set_nested(merged, path, self._parse_env_value(value))
        return merged

    def _set_nested(self, target: dict[str, Any], path: list[str], value: Any) -> None:
        cursor = target
        for part in path[:-1]:
            next_value = cursor.setdefault(part, {})
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[part] = next_value
            cursor = next_value
        cursor[path[-1]] = value

    def _parse_env_value(self, value: str) -> Any:
        return _parse_scalar(value)


def _parse_yaml_mapping(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    pending_list_key: tuple[int, dict[str, Any], str] | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        if stripped.startswith("- "):
            if pending_list_key is None:
                raise ValueError("YAML list item found without a parent key")
            list_indent, parent, key = pending_list_key
            if indent <= list_indent:
                raise ValueError("YAML list item indentation is invalid")
            if not isinstance(parent.get(key), list):
                parent[key] = []
            parent[key].append(_parse_scalar(stripped[2:]))
            continue

        pending_list_key = None
        while stack and indent <= stack[-1][0]:
            stack.pop()

        key, separator, value = stripped.partition(":")
        if not separator:
            raise ValueError(f"Invalid YAML line: {raw_line}")
        key = key.strip()
        value = value.strip()
        parent = stack[-1][1]

        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            pending_list_key = (indent, parent, key)
        else:
            parent[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if _is_int(value):
        return int(value)
    if _is_float(value):
        return float(value)
    return value


def _is_int(value: str) -> bool:
    return value.replace("-", "", 1).isdigit()


def _is_float(value: str) -> bool:
    if not value or value.count(".") != 1:
        return False
    left, right = value.split(".", maxsplit=1)
    return left.replace("-", "", 1).isdigit() and right.isdigit()
