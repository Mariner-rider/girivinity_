from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(slots=True)
class BenchmarkScores:
    reasoning: float
    coding: float
    factual_qa: float
    hallucination_detection: float

    @property
    def overall(self) -> float:
        return round(
            (self.reasoning + self.coding + self.factual_qa + self.hallucination_detection) / 4.0,
            4,
        )


@dataclass(slots=True)
class RegressionReport:
    previous: BenchmarkScores | None
    current: BenchmarkScores
    deltas: dict[str, float]
    regressed_categories: list[str]


class BenchmarkingSystem:
    def __init__(self, history_path: str = "benchmark_history.jsonl") -> None:
        self.history_path = Path(history_path)

    def run_reasoning_tasks(self, evaluator) -> float:
        return round(float(evaluator("reasoning")), 4)

    def run_coding_tasks(self, evaluator) -> float:
        return round(float(evaluator("coding")), 4)

    def run_factual_qa_tasks(self, evaluator) -> float:
        return round(float(evaluator("factual_qa")), 4)

    def run_hallucination_detection(self, evaluator) -> float:
        return round(float(evaluator("hallucination_detection")), 4)

    def evaluate(self, evaluator) -> BenchmarkScores:
        return BenchmarkScores(
            reasoning=self.run_reasoning_tasks(evaluator),
            coding=self.run_coding_tasks(evaluator),
            factual_qa=self.run_factual_qa_tasks(evaluator),
            hallucination_detection=self.run_hallucination_detection(evaluator),
        )

    def _read_latest(self) -> BenchmarkScores | None:
        if not self.history_path.exists() or self.history_path.stat().st_size == 0:
            return None
        last = self.history_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        data = json.loads(last)
        return BenchmarkScores(
            reasoning=data["reasoning"],
            coding=data["coding"],
            factual_qa=data["factual_qa"],
            hallucination_detection=data["hallucination_detection"],
        )

    def track_regression(self, current: BenchmarkScores) -> RegressionReport:
        previous = self._read_latest()
        deltas = {
            "reasoning": 0.0,
            "coding": 0.0,
            "factual_qa": 0.0,
            "hallucination_detection": 0.0,
            "overall": 0.0,
        }
        regressed = []

        if previous is not None:
            deltas = {
                "reasoning": round(current.reasoning - previous.reasoning, 4),
                "coding": round(current.coding - previous.coding, 4),
                "factual_qa": round(current.factual_qa - previous.factual_qa, 4),
                "hallucination_detection": round(current.hallucination_detection - previous.hallucination_detection, 4),
                "overall": round(current.overall - previous.overall, 4),
            }
            regressed = [k for k, v in deltas.items() if k != "overall" and v < 0]

        return RegressionReport(previous=previous, current=current, deltas=deltas, regressed_categories=regressed)

    def persist_scores(self, scores: BenchmarkScores) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(asdict(scores)) + "\n")

    def evaluate_after_update(self, evaluator) -> RegressionReport:
        """Automatic evaluation hook intended to run after each model/code update."""
        current = self.evaluate(evaluator)
        report = self.track_regression(current)
        self.persist_scores(current)
        return report
