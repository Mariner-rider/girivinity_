from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from model.architecture import GirivinityConfig, GirivinityModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SuccessorConfig:
    check_interval_seconds: int = 24 * 60 * 60
    chunk_threshold: int = 100_000
    quality_threshold: float = 3.5
    feedback_window: int = 1_000
    trained_collection: str = "trained"
    chroma_path: str | None = None
    feedback_db_path: Path = Path("data/user_feedback.sqlite3")
    training_root: Path = Path("data/successor_training")
    versions_dir: Path = Path("models/versions")
    active_model_symlink: Path = Path("models/active")
    notifications_path: Path = Path("admin_notifications.jsonl")
    state_path: Path = Path("data/successor_state.json")
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 2e-4
    train_seq_len: int = 256
    eval_split: float = 0.1
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config.yaml") -> SuccessorConfig:
        path = Path(config_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        raw = raw or {}
        section = raw.get("successor_engine") or {}
        return cls(
            check_interval_seconds=int(section.get("check_interval_seconds", 24 * 60 * 60)),
            chunk_threshold=int(section.get("chunk_threshold", 100_000)),
            quality_threshold=float(section.get("quality_threshold", 3.5)),
            feedback_window=int(section.get("feedback_window", 1_000)),
            trained_collection=str(section.get("trained_collection", "trained")),
            chroma_path=section.get("chroma_path"),
            feedback_db_path=Path(section.get("feedback_db_path", "data/user_feedback.sqlite3")),
            training_root=Path(section.get("training_root", "data/successor_training")),
            versions_dir=Path(section.get("versions_dir", "models/versions")),
            active_model_symlink=Path(section.get("active_model_symlink", "models/active")),
            notifications_path=Path(section.get("notifications_path", "admin_notifications.jsonl")),
            state_path=Path(section.get("state_path", "data/successor_state.json")),
            batch_size=int(section.get("batch_size", 2)),
            epochs=int(section.get("epochs", 1)),
            learning_rate=float(section.get("learning_rate", 2e-4)),
            train_seq_len=int(section.get("train_seq_len", 256)),
            eval_split=float(section.get("eval_split", 0.1)),
            device=str(section.get("device", "cpu")),
        )


@dataclass(slots=True)
class TrainingSummary:
    version: str
    model_path: str
    perplexity: float
    previous_version: str | None
    previous_perplexity: float
    trained_on_chunks: int


class ByteTokenizer:
    """Deterministic byte-level tokenizer for original full-weight training."""

    eos_id = 0

    def encode(self, text: str) -> list[int]:
        return [byte + 1 for byte in text.encode("utf-8", errors="ignore")] + [self.eos_id]


class FullModelTrainer:
    """Real PyTorch full-weight training pipeline for successor models."""

    def __init__(self, config: SuccessorConfig, model_config: GirivinityConfig | None = None) -> None:
        self.config = config
        self.model_config = model_config or GirivinityConfig.from_yaml()
        self.tokenizer = ByteTokenizer()

    def train_and_evaluate(
        self,
        corpus_path: Path,
        version: str,
        previous_version: str | None,
        trained_on_chunks: int,
    ) -> TrainingSummary:
        output_dir = self.config.versions_dir / version
        with tempfile.TemporaryDirectory(prefix=f"{version}-", dir=str(self._tmp_parent())) as tmp:
            tmp_output_dir = Path(tmp) / "model"
            tmp_output_dir.mkdir(parents=True, exist_ok=True)
            train_dataset, eval_dataset = self._build_datasets(corpus_path)
            model = GirivinityModel(self.model_config).to(self.config.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
            loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            model.train()
            for _epoch in range(self.config.epochs):
                for input_ids, labels in loader:
                    input_ids = input_ids.to(self.config.device)
                    labels = labels.to(self.config.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss.backward()
                    optimizer.step()

            new_perplexity = self._evaluate_perplexity(model, eval_dataset)
            previous_perplexity = self._evaluate_previous(previous_version, eval_dataset)
            if new_perplexity >= previous_perplexity:
                return TrainingSummary(
                    version=version,
                    model_path=str(tmp_output_dir),
                    perplexity=new_perplexity,
                    previous_version=previous_version,
                    previous_perplexity=previous_perplexity,
                    trained_on_chunks=trained_on_chunks,
                )

            self._save_model(model, tmp_output_dir, new_perplexity)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_output_dir), output_dir)
            return TrainingSummary(
                version=version,
                model_path=str(output_dir),
                perplexity=new_perplexity,
                previous_version=previous_version,
                previous_perplexity=previous_perplexity,
                trained_on_chunks=trained_on_chunks,
            )

    def _tmp_parent(self) -> Path:
        self.config.versions_dir.mkdir(parents=True, exist_ok=True)
        return self.config.versions_dir

    def _build_datasets(self, corpus_path: Path) -> tuple[TensorDataset, TensorDataset]:
        token_ids: list[int] = []
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                token_ids.extend(self.tokenizer.encode(str(record.get("text") or "")))

        min_tokens = self.config.train_seq_len + 2
        if len(token_ids) < min_tokens:
            raise ValueError(f"Corpus must contain at least {min_tokens} byte tokens for successor training")

        x_rows: list[list[int]] = []
        y_rows: list[list[int]] = []
        stride = self.config.train_seq_len
        for start in range(0, len(token_ids) - self.config.train_seq_len, stride):
            window = token_ids[start : start + self.config.train_seq_len + 1]
            if len(window) == self.config.train_seq_len + 1:
                x_rows.append(window[:-1])
                y_rows.append(window[1:])

        inputs = torch.tensor(x_rows, dtype=torch.long)
        labels = torch.tensor(y_rows, dtype=torch.long)
        eval_count = max(1, int(len(inputs) * self.config.eval_split))
        if len(inputs) == 1:
            return TensorDataset(inputs, labels), TensorDataset(inputs, labels)
        eval_count = min(eval_count, len(inputs) - 1)
        train_inputs = inputs[:-eval_count]
        train_labels = labels[:-eval_count]
        eval_inputs = inputs[-eval_count:]
        eval_labels = labels[-eval_count:]
        return TensorDataset(train_inputs, train_labels), TensorDataset(eval_inputs, eval_labels)

    def _evaluate_perplexity(self, model: GirivinityModel, dataset: TensorDataset) -> float:
        loader = DataLoader(dataset, batch_size=self.config.batch_size)
        losses: list[float] = []
        model.eval()
        with torch.no_grad():
            for input_ids, labels in loader:
                input_ids = input_ids.to(self.config.device)
                labels = labels.to(self.config.device)
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                losses.append(float(loss.detach().cpu()))
        return math.exp(sum(losses) / max(1, len(losses)))

    def _evaluate_previous(self, previous_version: str | None, dataset: TensorDataset) -> float:
        if not previous_version:
            return math.inf
        model_dir = self.config.versions_dir / previous_version
        state_path = model_dir / "model.pt"
        config_path = model_dir / "config.json"
        if not state_path.exists() or not config_path.exists():
            return math.inf
        model_config = GirivinityConfig(**json.loads(config_path.read_text(encoding="utf-8")))
        model = GirivinityModel(model_config).to(self.config.device)
        model.load_state_dict(torch.load(state_path, map_location=self.config.device))
        return self._evaluate_perplexity(model, dataset)

    def _save_model(self, model: GirivinityModel, output_dir: Path, perplexity: float) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "model.pt")
        (output_dir / "config.json").write_text(
            json.dumps(asdict(self.model_config), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps({"perplexity": perplexity}, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class SuccessorEngine(Thread):
    """24-hour monitor that trains successor base models and only notifies admins."""

    def __init__(
        self,
        *,
        config_path: str | Path = "config.yaml",
        config: SuccessorConfig | None = None,
        trainer: FullModelTrainer | None = None,
    ) -> None:
        super().__init__(name="successor-engine", daemon=True)
        self.config_path = Path(config_path)
        self.config = config or SuccessorConfig.from_yaml(config_path)
        self.trainer = trainer or FullModelTrainer(self.config)
        self._stop_event = Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.check_once()
            except Exception as exc:
                logger.exception("Successor engine check failed: %s", exc)
            self._stop_event.wait(self.config.check_interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()

    def check_once(self) -> TrainingSummary | None:
        if any(
            notification.get("status") == "awaiting_admin_approval"
            for notification in read_notifications(self.config)
        ):
            logger.info("Skipping successor training because a successor is awaiting admin approval.")
            return None

        trained_count = self._trained_collection_count()
        quality_score = self._rolling_quality_score()
        state = self._read_state()
        chunks_since_last = trained_count - int(state.get("last_model_chunk_count", 0))
        chunk_triggered = chunks_since_last >= self.config.chunk_threshold
        quality_triggered = quality_score is not None and quality_score < self.config.quality_threshold
        if not chunk_triggered and not quality_triggered:
            return None

        previous_version = state.get("active_version") or self._active_version()
        new_version = self._new_version()
        corpus_path = self._export_trained_corpus(new_version)
        summary = self.trainer.train_and_evaluate(
            corpus_path,
            new_version,
            previous_version,
            trained_count,
        )
        if summary.perplexity < summary.previous_perplexity:
            self._write_successor_notification(summary)
            state.update(
                {
                    "last_candidate_version": summary.version,
                    "last_candidate_chunk_count": trained_count,
                    "last_check_timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._write_state(state)
            return summary
        return None

    def _trained_collection(self):
        import chromadb  # type: ignore

        if self.config.chroma_path:
            client = chromadb.PersistentClient(path=self.config.chroma_path)
        else:
            client = chromadb.Client()
        return client.get_or_create_collection(name=self.config.trained_collection)

    def _trained_collection_count(self) -> int:
        return int(self._trained_collection().count())

    def _export_trained_corpus(self, version: str) -> Path:
        collection = self._trained_collection()
        batch = collection.get(include=["documents", "metadatas"])
        ids = [str(value) for value in batch.get("ids", [])]
        documents = [str(value) for value in batch.get("documents", [])]
        metadatas = [dict(value or {}) for value in batch.get("metadatas", [])]
        version_dir = self.config.training_root / version
        version_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = version_dir / "corpus.jsonl"
        with corpus_path.open("w", encoding="utf-8") as handle:
            for index, document in enumerate(documents):
                record = {
                    "id": ids[index] if index < len(ids) else str(index),
                    "text": document,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return corpus_path

    def _rolling_quality_score(self) -> float | None:
        if not self.config.feedback_db_path.exists():
            return None
        try:
            with sqlite3.connect(self.config.feedback_db_path) as conn:
                rows = conn.execute(
                    "SELECT score FROM user_feedback ORDER BY created_at DESC LIMIT ?",
                    (self.config.feedback_window,),
                ).fetchall()
        except sqlite3.Error:
            return None
        if not rows:
            return None
        return sum(float(row[0]) for row in rows) / len(rows)

    def _write_successor_notification(self, summary: TrainingSummary) -> None:
        previous = summary.previous_perplexity
        improvement = 100.0 if math.isinf(previous) else ((previous - summary.perplexity) / previous) * 100.0
        notification = {
            "type": "successor_ready",
            "version": summary.version,
            "previous_version": summary.previous_version,
            "improvement_percent": improvement,
            "trained_on_chunks": summary.trained_on_chunks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "awaiting_admin_approval",
        }
        self.config.notifications_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.notifications_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(notification, sort_keys=True) + "\n")

    def _read_state(self) -> dict[str, Any]:
        if not self.config.state_path.exists():
            return {}
        return json.loads(self.config.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict[str, Any]) -> None:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _active_version(self) -> str | None:
        if self.config.active_model_symlink.is_symlink():
            return self.config.active_model_symlink.resolve().name
        if self.config.active_model_symlink.exists():
            return self.config.active_model_symlink.name
        return None

    def _new_version(self) -> str:
        return datetime.now(timezone.utc).strftime("successor-%Y%m%dT%H%M%SZ")


def read_notifications(config: SuccessorConfig | None = None) -> list[dict[str, Any]]:
    cfg = config or SuccessorConfig.from_yaml()
    if not cfg.notifications_path.exists():
        return []
    return [
        json.loads(line)
        for line in cfg.notifications_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_notifications(notifications: list[dict[str, Any]], config: SuccessorConfig | None = None) -> None:
    cfg = config or SuccessorConfig.from_yaml()
    cfg.notifications_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.notifications_path.open("w", encoding="utf-8") as handle:
        for notification in notifications:
            handle.write(json.dumps(notification, sort_keys=True) + "\n")


def approve_successor(version: str, config: SuccessorConfig | None = None) -> dict[str, Any]:
    cfg = config or SuccessorConfig.from_yaml()
    version_dir = cfg.versions_dir / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Unknown model version: {version}")
    notifications = read_notifications(cfg)
    matched = False
    for notification in notifications:
        if notification.get("version") == version:
            matched = True
            notification["status"] = "approved"
            notification["approved_at"] = datetime.now(timezone.utc).isoformat()
    if not matched:
        raise FileNotFoundError(f"No pending notification for model version: {version}")
    _swap_active_symlink(cfg.active_model_symlink, version_dir)
    state = json.loads(cfg.state_path.read_text(encoding="utf-8")) if cfg.state_path.exists() else {}
    state["active_version"] = version
    state["active_model_path"] = str(version_dir)
    state["active_updated_at"] = datetime.now(timezone.utc).isoformat()
    for notification in notifications:
        if notification.get("version") == version and "trained_on_chunks" in notification:
            state["last_model_chunk_count"] = int(notification["trained_on_chunks"])
    cfg.state_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    write_notifications(notifications, cfg)
    return {"version": version, "status": "approved", "active_model": str(cfg.active_model_symlink)}


def reject_successor(version: str, config: SuccessorConfig | None = None) -> dict[str, Any]:
    cfg = config or SuccessorConfig.from_yaml()
    notifications = read_notifications(cfg)
    matched = False
    for notification in notifications:
        if notification.get("version") == version:
            matched = True
            notification["status"] = "rejected"
            notification["rejected_at"] = datetime.now(timezone.utc).isoformat()
    if not matched:
        raise FileNotFoundError(f"No notification for model version: {version}")
    write_notifications(notifications, cfg)
    return {"version": version, "status": "rejected"}


def list_model_versions(config: SuccessorConfig | None = None) -> list[dict[str, Any]]:
    cfg = config or SuccessorConfig.from_yaml()
    active = cfg.active_model_symlink.resolve() if cfg.active_model_symlink.exists() else None
    if not cfg.versions_dir.exists():
        return []
    versions: list[dict[str, Any]] = []
    for child in sorted(cfg.versions_dir.iterdir()):
        if not child.is_dir():
            continue
        metrics_path = child / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        versions.append(
            {
                "version": child.name,
                "path": str(child),
                "active": active == child.resolve() if active else False,
                "metrics": metrics,
            }
        )
    return versions


def _swap_active_symlink(active_path: Path, target_dir: Path) -> None:
    active_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_link = active_path.parent / f".{active_path.name}.tmp-{time.time_ns()}"
    os.symlink(target_dir.resolve(), temporary_link)
    if active_path.exists() or active_path.is_symlink():
        if active_path.is_dir() and not active_path.is_symlink():
            shutil.rmtree(active_path)
        else:
            active_path.unlink()
    os.replace(temporary_link, active_path)
