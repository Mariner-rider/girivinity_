"""Confidence calibration utilities for Girivinity agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class CalibrationRecord:
    offset: float = 0.0
    n_samples: int = 0


class CalibrationManager:
    """
    Maintains per-agent calibration offsets based on historical accuracy.
    Loaded from data/calibration.json at startup and updated online with a
    running mean of (actual_correct - predicted_confidence).
    """

    def __init__(self, path: str = "data/calibration.json") -> None:
        self.path = Path(path)
        self._data: dict[str, dict[str, Any]] = self._load()

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "CalibrationManager":
        cfg_path = Path(path)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        cal_path = ((cfg or {}).get("cognition", {}) or {}).get("calibration_path", "data/calibration.json")
        return cls(cal_path)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def calibrate(self, raw_confidence: float, agent_name: str) -> float:
        offset = float(self._data.get(agent_name, {}).get("offset", 0.0))
        return round(min(1.0, max(0.0, float(raw_confidence) + offset)), 3)

    def update(self, agent_name: str, predicted: float, actual_correct: bool) -> None:
        record = self._data.setdefault(agent_name, {"offset": 0.0, "n_samples": 0})
        n = int(record.get("n_samples", 0))
        offset = float(record.get("offset", 0.0))
        error = (1.0 if actual_correct else 0.0) - float(predicted)
        record["offset"] = round(((offset * n) + error) / (n + 1), 6)
        record["n_samples"] = n + 1
        self._save()
