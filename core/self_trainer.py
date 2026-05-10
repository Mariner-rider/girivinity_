from __future__ import annotations

import hashlib
import importlib
import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from threading import Event, Thread
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SelfTrainingConfig:
    interval_seconds: int = 30 * 60
    chunk_threshold: int = 50
    pending_collection: str = "pending_training"
    trained_collection: str = "trained"
    chroma_path: str | None = None
    training_queue_dir: Path = Path("data/training_queue")
    event_log_path: Path = Path("logs/self_training.jsonl")
    alerts_log_path: Path = Path("logs/alerts.jsonl")
    base_model_path: Path = Path("models/base")
    latest_adapter_path: Path = Path("models/adapters/latest")
    adapters_dir: Path = Path("models/adapters")
    epochs: int = 5
    learning_rate: float = 2e-4
    loss_abort_threshold: float = 2.0
    batch_size: int = 2
    max_length: int = 1024

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config.yaml") -> SelfTrainingConfig:
        path = Path(config_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        raw = raw or {}
        section = raw.get("self_training") or (raw.get("training") or {}).get("self_training") or {}

        return cls(
            interval_seconds=int(section.get("interval_seconds", section.get("check_interval_seconds", 30 * 60))),
            chunk_threshold=int(section.get("chunk_threshold", 50)),
            pending_collection=str(section.get("pending_collection", "pending_training")),
            trained_collection=str(section.get("trained_collection", "trained")),
            chroma_path=section.get("chroma_path"),
            training_queue_dir=Path(section.get("training_queue_dir", "data/training_queue")),
            event_log_path=Path(section.get("event_log_path", "logs/self_training.jsonl")),
            alerts_log_path=Path(section.get("alerts_log_path", "logs/alerts.jsonl")),
            base_model_path=Path(section.get("base_model_path", "models/base")),
            latest_adapter_path=Path(section.get("latest_adapter_path", "models/adapters/latest")),
            adapters_dir=Path(section.get("adapters_dir", "models/adapters")),
            epochs=int(section.get("epochs", 5)),
            learning_rate=float(section.get("learning_rate", 2e-4)),
            loss_abort_threshold=float(section.get("loss_abort_threshold", 2.0)),
            batch_size=int(section.get("batch_size", 2)),
            max_length=int(section.get("max_length", 1024)),
        )


@dataclass(slots=True)
class TrainingResult:
    adapter_version: str
    adapter_path: str
    loss: float


class LoRATrainer:
    """LoRA trainer used by the background self-training worker process."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        config: SelfTrainingConfig | None = None,
        config_path: str | Path = "config.yaml",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.config = config or SelfTrainingConfig.from_yaml(config_path)

    def train(self) -> TrainingResult:
        """Train a new LoRA adapter and atomically promote it when loss is acceptable."""
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        adapter_path = self.config.adapters_dir / version
        adapter_path.mkdir(parents=True, exist_ok=True)

        transformers = importlib.import_module("transformers")
        datasets = importlib.import_module("datasets")
        peft = importlib.import_module("peft")
        torch = importlib.import_module("torch")

        tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.config.base_model_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            str(self.config.base_model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        if self.config.latest_adapter_path.exists():
            model = peft.PeftModel.from_pretrained(
                base_model,
                str(self.config.latest_adapter_path),
                is_trainable=True,
            )
        else:
            lora_config = peft.LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            )
            model = peft.get_peft_model(base_model, lora_config)

        dataset = datasets.load_dataset("json", data_files=str(self.dataset_path), split="train")

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            prompts = [
                f"Instruction: {instruction}\nAnswer: {response}"
                for instruction, response in zip(batch["instruction"], batch["response"])
            ]
            tokens = tokenizer(prompts, truncation=True, max_length=self.config.max_length)
            tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
            return tokens

        tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        training_args = transformers.TrainingArguments(
            output_dir=str(adapter_path),
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.epochs,
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            bf16=False,
            report_to=[],
        )
        trainer = transformers.Trainer(model=model, args=training_args, train_dataset=tokenized)
        train_output = trainer.train()
        loss = self._extract_loss(train_output)

        if loss > self.config.loss_abort_threshold:
            self._write_alert(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "dataset_path": str(self.dataset_path),
                    "adapter_version": version,
                    "loss": loss,
                    "message": "Self-training aborted because training loss exceeded threshold.",
                }
            )
            raise RuntimeError(
                f"Training loss {loss:.4f} exceeded threshold {self.config.loss_abort_threshold:.4f}"
            )

        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        self._promote_latest_adapter(adapter_path)
        return TrainingResult(adapter_version=version, adapter_path=str(adapter_path), loss=loss)

    def _extract_loss(self, train_output: Any) -> float:
        metrics = getattr(train_output, "metrics", {}) or {}
        if "train_loss" in metrics:
            return float(metrics["train_loss"])
        if "loss" in metrics:
            return float(metrics["loss"])
        return 0.0

    def _promote_latest_adapter(self, adapter_path: Path) -> None:
        latest_path = self.config.latest_adapter_path
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_link = latest_path.parent / f".{latest_path.name}.tmp"
        if temporary_link.exists() or temporary_link.is_symlink():
            temporary_link.unlink()
        os.symlink(adapter_path.resolve(), temporary_link)
        if latest_path.exists() or latest_path.is_symlink():
            if latest_path.is_dir() and not latest_path.is_symlink():
                import shutil

                shutil.rmtree(latest_path)
            else:
                latest_path.unlink()
        os.replace(temporary_link, latest_path)

    def _write_alert(self, event: dict[str, Any]) -> None:
        self.config.alerts_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.alerts_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


class SelfTrainer(Thread):
    """Daemon thread that promotes pending ChromaDB chunks into LoRA updates."""

    def __init__(
        self,
        *,
        config_path: str | Path = "config.yaml",
        config: SelfTrainingConfig | None = None,
    ) -> None:
        super().__init__(name="self-trainer", daemon=True)
        self.config_path = Path(config_path)
        self.config = config or SelfTrainingConfig.from_yaml(config_path)
        self._stop_event = Event()
        self._training_process: mp.Process | None = None

    def queue(self, query: str, chunks: list[Any]) -> None:
        """Queue freshly retrieved chunks in ChromaDB pending_training without blocking callers."""
        if not chunks:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        for index, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                text = str(chunk.get("text") or chunk.get("chunk") or "").strip()
                url = str(chunk.get("url") or "")
                relevance_score = float(chunk.get("score") or chunk.get("relevance_score") or 0.0)
            else:
                text = str(chunk).strip()
                url = ""
                relevance_score = 0.0
            if not text:
                continue
            documents.append(text)
            metadatas.append(
                {
                    "url": url,
                    "timestamp": timestamp,
                    "query": query,
                    "relevance_score": relevance_score,
                }
            )
            payload = f"{query}\0{timestamp}\0{index}\0{text}"
            ids.append("queued-" + hashlib.sha256(payload.encode("utf-8")).hexdigest())
        if not documents:
            return
        collection = self._get_collection(self.config.pending_collection)
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.check_once()
            except Exception as exc:
                logger.exception("Self-training check failed: %s", exc)
            self._stop_event.wait(self.config.interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()

    def check_once(self) -> bool:
        """Run one pending-training check. Returns True when training was started."""
        if self._training_process and self._training_process.is_alive():
            logger.info("Skipping self-training check because a training process is already running.")
            return False

        pending_collection = self._get_collection(self.config.pending_collection)
        count = int(pending_collection.count())
        if count < self.config.chunk_threshold:
            return False

        batch = pending_collection.get(include=["documents", "metadatas", "embeddings"])
        ids = [str(value) for value in batch.get("ids", [])]
        documents = [str(value) for value in batch.get("documents", [])]
        metadatas = [dict(value or {}) for value in batch.get("metadatas", [])]
        embeddings = batch.get("embeddings")
        if not ids or not documents:
            return False

        timestamp = datetime.now(timezone.utc).isoformat()
        dataset_path = self._write_instruction_dataset(documents, metadatas, timestamp)
        result = self._run_training_process(dataset_path)
        if result is None:
            return False

        self._move_chunks_to_trained(ids, documents, metadatas, embeddings)
        pending_collection.delete(ids=ids)
        self._log_training_event(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chunks_trained": len(documents),
                "adapter_version": result.adapter_version,
                "loss": result.loss,
            }
        )
        return True

    def _get_collection(self, name: str):
        chromadb = importlib.import_module("chromadb")
        if self.config.chroma_path:
            client = chromadb.PersistentClient(path=self.config.chroma_path)
        else:
            client = chromadb.Client()
        return client.get_or_create_collection(name=name)

    def _write_instruction_dataset(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        timestamp: str,
    ) -> Path:
        safe_timestamp = timestamp.replace(":", "").replace("+", "_")
        output_path = self.config.training_queue_dir / f"{safe_timestamp}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for document, metadata in zip(documents, metadatas):
                query = str(metadata.get("query") or "this topic").strip() or "this topic"
                record = {
                    "instruction": f"What do you know about {query}?",
                    "response": document,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output_path

    def _run_training_process(self, dataset_path: Path) -> TrainingResult | None:
        result_queue: mp.Queue = mp.Queue(maxsize=1)
        process = mp.Process(
            target=_training_process_entrypoint,
            args=(str(dataset_path), self.config, result_queue),
            daemon=False,
            name="lora-self-training",
        )
        self._training_process = process
        process.start()
        process.join()

        try:
            payload = result_queue.get_nowait()
        except Empty:
            logger.error("Training process exited without returning a result.")
            return None

        if not payload.get("ok"):
            logger.error("Training process failed: %s", payload.get("error"))
            return None

        return TrainingResult(
            adapter_version=str(payload["adapter_version"]),
            adapter_path=str(payload["adapter_path"]),
            loss=float(payload["loss"]),
        )

    def _move_chunks_to_trained(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: Any,
    ) -> None:
        trained_collection = self._get_collection(self.config.trained_collection)
        trained_metadatas = [
            {**metadata, "trained_at": datetime.now(timezone.utc).isoformat()}
            for metadata in metadatas
        ]
        add_kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": trained_metadatas,
        }
        if embeddings is not None:
            add_kwargs["embeddings"] = embeddings
        trained_collection.add(**add_kwargs)

    def _log_training_event(self, event: dict[str, Any]) -> None:
        self.config.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def _training_process_entrypoint(
    dataset_path: str,
    config: SelfTrainingConfig,
    result_queue: mp.Queue,
) -> None:
    try:
        result = LoRATrainer(dataset_path, config=config).train()
        result_queue.put(
            {
                "ok": True,
                "adapter_version": result.adapter_version,
                "adapter_path": result.adapter_path,
                "loss": result.loss,
            }
        )
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})
