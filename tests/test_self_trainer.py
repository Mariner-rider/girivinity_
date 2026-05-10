from __future__ import annotations

import json

from core.self_trainer import SelfTrainer, SelfTrainingConfig, TrainingResult


class FakeCollection:
    def __init__(self, *, count: int = 0, batch: dict | None = None) -> None:
        self._count = count
        self._batch = batch or {}
        self.added = None
        self.deleted_ids = None

    def count(self) -> int:
        return self._count

    def get(self, **kwargs):
        return self._batch

    def add(self, **kwargs) -> None:
        self.added = kwargs

    def delete(self, **kwargs) -> None:
        self.deleted_ids = kwargs["ids"]


def test_check_once_formats_trains_moves_and_logs(tmp_path, monkeypatch):
    pending = FakeCollection(
        count=2,
        batch={
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["first chunk", "second chunk"],
            "metadatas": [{"query": "AI safety"}, {"query": "robotics"}],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        },
    )
    trained = FakeCollection()
    config = SelfTrainingConfig(
        interval_seconds=1,
        chunk_threshold=2,
        training_queue_dir=tmp_path / "queue",
        event_log_path=tmp_path / "logs" / "self_training.jsonl",
        alerts_log_path=tmp_path / "logs" / "alerts.jsonl",
        base_model_path=tmp_path / "models" / "base",
        latest_adapter_path=tmp_path / "models" / "adapters" / "latest",
        adapters_dir=tmp_path / "models" / "adapters",
    )
    trainer = SelfTrainer(config=config)

    def fake_get_collection(name: str):
        return {"pending_training": pending, "trained": trained}[name]

    monkeypatch.setattr(trainer, "_get_collection", fake_get_collection)
    monkeypatch.setattr(
        trainer,
        "_run_training_process",
        lambda dataset_path: TrainingResult("adapter-v1", "models/adapters/adapter-v1", 1.25),
    )

    assert trainer.check_once() is True

    dataset_files = list((tmp_path / "queue").glob("*.jsonl"))
    assert len(dataset_files) == 1
    records = [json.loads(line) for line in dataset_files[0].read_text().splitlines()]
    assert records == [
        {"instruction": "What do you know about AI safety?", "response": "first chunk"},
        {"instruction": "What do you know about robotics?", "response": "second chunk"},
    ]
    assert trained.added["ids"] == ["chunk-1", "chunk-2"]
    assert trained.added["documents"] == ["first chunk", "second chunk"]
    assert trained.added["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
    assert pending.deleted_ids == ["chunk-1", "chunk-2"]

    [event] = [json.loads(line) for line in config.event_log_path.read_text().splitlines()]
    assert event["chunks_trained"] == 2
    assert event["adapter_version"] == "adapter-v1"
    assert event["loss"] == 1.25


def test_check_once_waits_until_threshold(tmp_path, monkeypatch):
    pending = FakeCollection(count=1)
    config = SelfTrainingConfig(
        chunk_threshold=2,
        training_queue_dir=tmp_path / "queue",
        event_log_path=tmp_path / "logs" / "self_training.jsonl",
        alerts_log_path=tmp_path / "logs" / "alerts.jsonl",
    )
    trainer = SelfTrainer(config=config)
    monkeypatch.setattr(trainer, "_get_collection", lambda name: pending)

    assert trainer.check_once() is False
    assert not (tmp_path / "queue").exists()


def test_lora_trainer_alerts_and_aborts_on_high_loss(tmp_path, monkeypatch):
    import sys
    import types

    from core.self_trainer import LoRATrainer

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

        def __call__(self, prompts, truncation=True, max_length=1024):
            return {"input_ids": [[1, 2, 3] for _prompt in prompts]}

        def save_pretrained(self, path: str) -> None:
            raise AssertionError("adapter should not be saved when loss is too high")

    class FakeModel:
        def save_pretrained(self, path: str) -> None:
            raise AssertionError("adapter should not be saved when loss is too high")

    class FakeDataset:
        column_names = ["instruction", "response"]

        def map(self, fn, batched=True, remove_columns=None):
            fn({"instruction": ["What do you know about AI?"], "response": ["AI text"]})
            return self

    class FakeTrainOutput:
        metrics = {"train_loss": 2.5}

    class FakeTrainer:
        def __init__(self, **kwargs) -> None:
            pass

        def train(self):
            return FakeTrainOutput()

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"instruction":"i","response":"r"}\n')
    config = SelfTrainingConfig(
        loss_abort_threshold=2.0,
        training_queue_dir=tmp_path / "queue",
        event_log_path=tmp_path / "logs" / "self_training.jsonl",
        alerts_log_path=tmp_path / "logs" / "alerts.jsonl",
        base_model_path=tmp_path / "models" / "base",
        latest_adapter_path=tmp_path / "models" / "adapters" / "latest",
        adapters_dir=tmp_path / "models" / "adapters",
    )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: FakeTokenizer()),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel()),
            TrainingArguments=lambda **kwargs: types.SimpleNamespace(**kwargs),
            Trainer=FakeTrainer,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *args, **kwargs: FakeDataset()),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(
            LoraConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
            get_peft_model=lambda base_model, lora_config: base_model,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            float16="float16",
            float32="float32",
        ),
    )

    try:
        LoRATrainer(dataset_path, config=config).train()
    except RuntimeError as exc:
        assert "exceeded threshold" in str(exc)
    else:
        raise AssertionError("high-loss training should abort")

    [alert] = [json.loads(line) for line in config.alerts_log_path.read_text().splitlines()]
    assert alert["loss"] == 2.5
    assert alert["message"] == "Self-training aborted because training loss exceeded threshold."
