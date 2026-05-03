import json

from app.finetune.dataset_builder import build_dataset_from_logs


def test_build_dataset_from_logs(tmp_path):
    log_file = tmp_path / "logs.jsonl"
    out_file = tmp_path / "sft.jsonl"

    rows = [
        {"prompt": "Explain vector search", "response": "It finds nearest vectors."},
        {"prompt": "x", "response": "y"},
    ]
    with log_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    kept = build_dataset_from_logs(log_file, out_file)
    assert kept == 1
    assert out_file.exists()
