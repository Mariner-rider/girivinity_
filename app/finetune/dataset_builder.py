from __future__ import annotations

import json
from pathlib import Path


def build_dataset_from_logs(log_path: str | Path, output_path: str | Path, min_chars: int = 8) -> int:
    """Build supervised finetuning dataset from JSONL logs.

    Input log format (JSONL):
    {"prompt": "...", "response": "..."}
    """
    log_path = Path(log_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with log_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for raw in src:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if len(prompt) < min_chars or len(response) < min_chars:
                continue
            dst.write(json.dumps({"instruction": prompt, "output": response}, ensure_ascii=False) + "\n")
            kept += 1
    return kept
