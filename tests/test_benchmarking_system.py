from pathlib import Path

from app.training.benchmarking import BenchmarkingSystem


def test_benchmark_scores_per_category_and_persistence(tmp_path: Path):
    history = tmp_path / "bench.jsonl"
    system = BenchmarkingSystem(history_path=str(history))

    def evaluator(category: str) -> float:
        table = {
            "reasoning": 0.80,
            "coding": 0.76,
            "factual_qa": 0.88,
            "hallucination_detection": 0.91,
        }
        return table[category]

    report = system.evaluate_after_update(evaluator)
    assert report.current.reasoning == 0.8
    assert report.current.coding == 0.76
    assert report.current.factual_qa == 0.88
    assert report.current.hallucination_detection == 0.91
    assert history.exists()


def test_regression_tracking_detects_degradation(tmp_path: Path):
    history = tmp_path / "bench.jsonl"
    system = BenchmarkingSystem(history_path=str(history))

    def eval_v1(category: str) -> float:
        return {"reasoning": 0.9, "coding": 0.9, "factual_qa": 0.9, "hallucination_detection": 0.9}[category]

    def eval_v2(category: str) -> float:
        return {"reasoning": 0.85, "coding": 0.92, "factual_qa": 0.86, "hallucination_detection": 0.93}[category]

    system.evaluate_after_update(eval_v1)
    report = system.evaluate_after_update(eval_v2)

    assert "reasoning" in report.regressed_categories
    assert "factual_qa" in report.regressed_categories
    assert report.deltas["coding"] > 0
