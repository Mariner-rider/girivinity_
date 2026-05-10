from __future__ import annotations

from dataclasses import dataclass, field

from app.security.policy import SecurityGuard, SecurityPolicyError, secure_operation


@dataclass(slots=True)
class EvaluationMetrics:
    accuracy: float
    hallucination_rate: float
    notes: dict = field(default_factory=dict)


@dataclass(slots=True)
class EvolutionResult:
    weaknesses: list[str]
    training_tasks: list[dict]
    baseline: EvaluationMetrics
    candidate: EvaluationMetrics
    deployed: bool
    notification_message: str


class NotificationSystem:
    def __init__(self) -> None:
        self.sent_messages: list[str] = []

    def send(self, message: str) -> None:
        self.sent_messages.append(message)


class ModelEvolutionSystem:
    def __init__(
        self,
        security_guard: SecurityGuard | None = None,
        notifier: NotificationSystem | None = None,
    ) -> None:
        self.security_guard = security_guard or SecurityGuard()
        self.notifier = notifier or NotificationSystem()

    def identify_weaknesses(self, metrics: EvaluationMetrics) -> list[str]:
        weaknesses: list[str] = []
        if metrics.accuracy < 0.8:
            weaknesses.append("accuracy_gap")
        if metrics.hallucination_rate > 0.1:
            weaknesses.append("hallucination_risk")
        for k, v in metrics.notes.items():
            if isinstance(v, (int, float)) and v < 0.5:
                weaknesses.append(f"weak_{k}")
        return weaknesses

    def generate_training_tasks(self, weaknesses: list[str]) -> list[dict]:
        task_map = {
            "accuracy_gap": {"task": "hard-example-mining", "objective": "improve_accuracy"},
            "hallucination_risk": {"task": "grounded-rag-supervision", "objective": "reduce_hallucination"},
        }
        tasks = [task_map[w] for w in weaknesses if w in task_map]
        for w in weaknesses:
            if w.startswith("weak_"):
                tasks.append({"task": "targeted-eval-repair", "objective": w})
        return tasks

    def should_deploy(self, baseline: EvaluationMetrics, candidate: EvaluationMetrics) -> bool:
        return candidate.accuracy > baseline.accuracy and candidate.hallucination_rate < baseline.hallucination_rate

    @secure_operation("training.model_evolution")
    def run(
        self,
        evaluate_current_fn,
        finetune_candidate_fn,
        benchmark_fn,
    ) -> EvolutionResult:
        # 1) evaluate current model
        baseline = evaluate_current_fn()

        # 2) identify weaknesses
        weaknesses = self.identify_weaknesses(baseline)

        # 3) generate training tasks
        tasks = self.generate_training_tasks(weaknesses)

        # 4) fine-tune candidate model (parameter-efficient expected by caller)
        candidate_model_ref = finetune_candidate_fn(tasks)

        # 5) benchmark vs previous
        candidate = benchmark_fn(candidate_model_ref)

        deployed = self.should_deploy(baseline, candidate)
        if not deployed:
            # policy-level rejection for evolution gate to make failure explicit
            raise_message = (
                f"Deployment blocked: candidate accuracy={candidate.accuracy} (baseline={baseline.accuracy}), "
                f"hallucination={candidate.hallucination_rate} (baseline={baseline.hallucination_rate})"
            )
            self.notifier.send("Model evolution blocked. " + raise_message)
            return EvolutionResult(
                weaknesses=weaknesses,
                training_tasks=tasks,
                baseline=baseline,
                candidate=candidate,
                deployed=False,
                notification_message=self.notifier.sent_messages[-1],
            )

        self.notifier.send(
            f"Model deployed successfully: accuracy {baseline.accuracy:.3f}->{candidate.accuracy:.3f}, "
            f"hallucination {baseline.hallucination_rate:.3f}->{candidate.hallucination_rate:.3f}."
        )
        return EvolutionResult(
            weaknesses=weaknesses,
            training_tasks=tasks,
            baseline=baseline,
            candidate=candidate,
            deployed=True,
            notification_message=self.notifier.sent_messages[-1],
        )
