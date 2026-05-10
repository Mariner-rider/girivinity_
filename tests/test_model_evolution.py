from app.training.model_evolution import EvaluationMetrics, ModelEvolutionSystem, NotificationSystem


def test_model_evolution_deploys_when_accuracy_up_and_hallucination_down():
    notifier = NotificationSystem()
    system = ModelEvolutionSystem(notifier=notifier)

    def eval_current():
        return EvaluationMetrics(accuracy=0.78, hallucination_rate=0.15, notes={"reasoning": 0.4})

    def finetune(tasks):
        assert tasks
        return "candidate-lora-v2"

    def benchmark(model_ref):
        assert model_ref == "candidate-lora-v2"
        return EvaluationMetrics(accuracy=0.84, hallucination_rate=0.08)

    result = system.run(eval_current, finetune, benchmark)
    assert result.deployed is True
    assert "Model deployed successfully" in result.notification_message


def test_model_evolution_blocks_when_hallucination_not_reduced():
    notifier = NotificationSystem()
    system = ModelEvolutionSystem(notifier=notifier)

    def eval_current():
        return EvaluationMetrics(accuracy=0.82, hallucination_rate=0.10)

    def finetune(tasks):
        return "candidate-lora-v3"

    def benchmark(model_ref):
        return EvaluationMetrics(accuracy=0.86, hallucination_rate=0.12)

    result = system.run(eval_current, finetune, benchmark)
    assert result.deployed is False
    assert "blocked" in result.notification_message.lower()
