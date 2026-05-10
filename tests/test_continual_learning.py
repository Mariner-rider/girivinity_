import json
from pathlib import Path

from app.finetune.continual_learning import ContinualLearningConfig, ContinualLearningSystem


def test_continual_learning_promotes_candidate(tmp_path: Path):
    logs = tmp_path / "logs.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    candidate_dir = tmp_path / "candidate"
    prod_dir = tmp_path / "prod"
    prod_dir.mkdir()

    rows = [
        {"prompt": "Explain RAG", "response": "Retrieval augmented generation", "quality_score": 0.9},
        {"prompt": "x", "response": "bad", "quality_score": 0.1},
    ]
    logs.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cfg = ContinualLearningConfig(
        logs_path=str(logs),
        dataset_output_path=str(dataset),
        adapter_output_dir=str(candidate_dir),
        production_adapter_dir=str(prod_dir),
        min_quality_score=0.6,
        min_delta=0.01,
    )
    system = ContinualLearningSystem(cfg)

    def fake_train(dataset_path: str, output_dir: str) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "adapter.bin").write_text(dataset_path, encoding="utf-8")
        return output_dir

    def fake_eval(prod_adapter: str, candidate_adapter: str) -> tuple[float, float]:
        _ = (prod_adapter, candidate_adapter)
        return 0.60, 0.75

    result = system.run(fake_train, fake_eval)
    assert result.promoted is True
    assert result.rollback_triggered is False
    assert result.kept_samples == 1


def test_continual_learning_rolls_back_on_regression(tmp_path: Path):
    logs = tmp_path / "logs.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    candidate_dir = tmp_path / "candidate"
    prod_dir = tmp_path / "prod"
    prod_dir.mkdir()

    rows = [{"prompt": "Explain embeddings", "response": "vector representations", "quality_score": 0.95}]
    logs.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cfg = ContinualLearningConfig(
        logs_path=str(logs),
        dataset_output_path=str(dataset),
        adapter_output_dir=str(candidate_dir),
        production_adapter_dir=str(prod_dir),
        min_quality_score=0.5,
        min_delta=0.05,
    )
    system = ContinualLearningSystem(cfg)

    def fake_train(dataset_path: str, output_dir: str) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "adapter.bin").write_text(dataset_path, encoding="utf-8")
        return output_dir

    def fake_eval(prod_adapter: str, candidate_adapter: str) -> tuple[float, float]:
        _ = (prod_adapter, candidate_adapter)
        return 0.80, 0.75

    result = system.run(fake_train, fake_eval)
    assert result.promoted is False
    assert result.rollback_triggered is True
    assert not candidate_dir.exists()
