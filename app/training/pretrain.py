from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split

from app.llm.girivinity_architecture import GirivinityConfig, GirivinityModel
from app.llm.girivinity_tokenizer import GirivinityTokenizer


@dataclass(slots=True)
class PretrainConfig:
    model_config: GirivinityConfig
    batch_size: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    gradient_accumulation_steps: int
    checkpoint_every: int
    data_path: str
    output_dir: str
    use_amp: bool = True
    use_fsdp: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PretrainConfig":
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        pretrain = (raw.get("training", {}) or {}).get("pretrain", {}) or {}
        if not pretrain:
            raise ValueError("Missing training.pretrain section in config")

        model_config_raw = pretrain.get("model_config") or _model_config_from_root(raw)
        return cls(
            model_config=GirivinityConfig(**model_config_raw),
            batch_size=int(pretrain["batch_size"]),
            learning_rate=float(pretrain["learning_rate"]),
            warmup_steps=int(pretrain["warmup_steps"]),
            max_steps=int(pretrain["max_steps"]),
            gradient_accumulation_steps=int(pretrain["gradient_accumulation_steps"]),
            checkpoint_every=int(pretrain["checkpoint_every"]),
            data_path=str(pretrain["data_path"]),
            output_dir=str(pretrain["output_dir"]),
            use_amp=bool(pretrain.get("use_amp", True)),
            use_fsdp=bool(pretrain.get("use_fsdp", False)),
        )


class _PackedTokenDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError("model_config.max_seq_len must be at least 2")
        if len(token_ids) < seq_len + 1:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) to build one packed sequence of length {seq_len}"
            )

        usable_tokens = ((len(token_ids) - 1) // seq_len) * seq_len + 1
        self._tokens = torch.tensor(token_ids[:usable_tokens], dtype=torch.long)
        self._seq_len = seq_len
        self._num_sequences = (usable_tokens - 1) // seq_len

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        start = idx * self._seq_len
        end = start + self._seq_len
        return {
            "input_ids": self._tokens[start:end],
            "labels": self._tokens[start + 1 : end + 1],
        }


class GirivinityPretrainer:
    def __init__(self, config: PretrainConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GirivinityModel(config.model_config)
        self.model.gradient_checkpointing_enable()
        self.model.to(self.device)
        self._wrap_fsdp_if_requested()

        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self._base_model = self._unwrap_model(self.model)

    def load_data(self) -> DataLoader:
        tokenizer = self._load_tokenizer()
        texts = list(_iter_jsonl_texts(self.config.data_path))
        if not texts:
            raise ValueError(f"No training text found in {self.config.data_path}")

        eos_id = _token_id(tokenizer, "[EOS]")
        token_ids: list[int] = []
        for text in texts:
            ids = tokenizer.encode(text)
            if not ids:
                continue
            token_ids.extend(ids)
            if eos_id is not None:
                token_ids.append(eos_id)

        dataset = _PackedTokenDataset(token_ids, seq_len=self.config.model_config.max_seq_len)
        val_size = max(1, int(len(dataset) * 0.05)) if len(dataset) > 1 else 0
        train_size = len(dataset) - val_size
        if val_size:
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        else:
            train_dataset = dataset
            val_dataset = dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return self.train_loader

    def train(self) -> None:
        train_loader = self.train_loader or self.load_data()
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=self._lr_multiplier)
        amp_enabled = self.config.use_amp and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_tokens = 0
        window_started = time.perf_counter()
        steps = 0

        for batch in cycle(train_loader):
            loss, token_count = self._training_step(batch, amp_enabled)
            scaled_loss = loss / self.config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            running_loss += loss.detach().item()
            running_tokens += token_count

            if (steps + 1) % self.config.gradient_accumulation_steps != 0:
                steps += 1
                continue

            scaler.unscale_(optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step = (steps + 1) // self.config.gradient_accumulation_steps
            elapsed = max(time.perf_counter() - window_started, 1e-9)
            current_loss = running_loss / self.config.gradient_accumulation_steps
            tps = running_tokens / elapsed
            lr = scheduler.get_last_lr()[0]
            print(f"step={optimizer_step} loss={current_loss:.4f} lr={lr:.2e} tokens/sec={tps:.0f}")

            checkpoint_saved = False
            if optimizer_step % self.config.checkpoint_every == 0:
                self._save_checkpoint(optimizer_step)
                checkpoint_saved = True

            running_loss = 0.0
            running_tokens = 0
            window_started = time.perf_counter()
            steps += 1

            if optimizer_step >= self.config.max_steps:
                if not checkpoint_saved:
                    self._save_checkpoint(optimizer_step)
                return

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        if self.train_loader is None or self.val_loader is None:
            self.load_data()
        assert self.train_loader is not None
        assert self.val_loader is not None

        was_training = self.model.training
        self.model.eval()
        losses = {
            "train": self._loader_loss(self.train_loader),
            "val": self._loader_loss(self.val_loader),
        }
        if was_training:
            self.model.train()
        return losses

    def _training_step(self, batch: dict[str, Tensor], amp_enabled: bool) -> tuple[Tensor, int]:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits, _ = self.model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        return loss, labels.numel()

    def _loader_loss(self, loader: DataLoader) -> float:
        total_loss = 0.0
        total_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            logits, _ = self.model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item()
            total_batches += 1
        return total_loss / max(total_batches, 1)

    def _lr_multiplier(self, step: int) -> float:
        if self.config.warmup_steps > 0 and step < self.config.warmup_steps:
            return max((step + 1) / self.config.warmup_steps, 1e-8)
        decay_steps = max(self.config.max_steps - self.config.warmup_steps, 1)
        progress = min(max((step - self.config.warmup_steps) / decay_steps, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _save_checkpoint(self, step: int) -> None:
        checkpoint_dir = Path(self.config.output_dir) / f"step-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_model(checkpoint_dir)
        (checkpoint_dir / "pretrain_config.json").write_text(
            json.dumps(_dataclass_to_dict(self.config), indent=2),
            encoding="utf-8",
        )

    def _save_model(self, checkpoint_dir: Path) -> None:
        if _is_fsdp_model(self.model):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self.model.state_dict()
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                (checkpoint_dir / "config.json").write_text(
                    json.dumps(asdict(self.config.model_config), indent=2),
                    encoding="utf-8",
                )
                torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
            return

        self._base_model.save_pretrained(str(checkpoint_dir))

    def _load_tokenizer(self) -> GirivinityTokenizer:
        for path in _candidate_tokenizer_paths(self.config.data_path, self.config.output_dir):
            try:
                return GirivinityTokenizer.from_file(str(path))
            except FileNotFoundError:
                continue
        candidates = ", ".join(
            str(path)
            for path in _candidate_tokenizer_paths(self.config.data_path, self.config.output_dir)
        )
        raise FileNotFoundError(f"Girivinity tokenizer not found. Tried: {candidates}")

    def _wrap_fsdp_if_requested(self) -> None:
        if not self.config.use_fsdp:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            print(
                "use_fsdp=True requested but torch.distributed is not initialized; "
                "training without FSDP"
            )
            return
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        self.model = FSDP(self.model)

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> GirivinityModel:
        return model.module if hasattr(model, "module") else model  # type: ignore[return-value]


def _iter_jsonl_texts(data_path: str | Path) -> Iterable[str]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    with path.open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            yield from _extract_texts(record)


def _extract_texts(record: Any) -> list[str]:
    if isinstance(record, str):
        return [record.strip()] if record.strip() else []
    if isinstance(record, list):
        texts: list[str] = []
        for item in record:
            texts.extend(_extract_texts(item))
        return texts
    if not isinstance(record, dict):
        return []

    preferred_keys = (
        "instruction",
        "question",
        "prompt",
        "input",
        "context",
        "response",
        "answer",
        "completion",
        "output",
        "text",
        "code",
    )
    texts = [str(record[key]).strip() for key in preferred_keys if record.get(key)]

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("content"):
                texts.append(str(message["content"]).strip())

    sources = record.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict):
                for key in ("title", "text", "snippet", "content"):
                    if source.get(key):
                        texts.append(str(source[key]).strip())

    return [text for text in texts if text]


def _candidate_tokenizer_paths(data_path: str, output_dir: str) -> list[Path]:
    data_parent = Path(data_path).parent
    return [
        data_parent,
        data_parent / "tokenizer",
        Path(output_dir),
        Path(output_dir) / "tokenizer",
        Path("girivinity_tokenizer"),
        Path("models/tokenizer"),
        Path("models/base"),
    ]


def _token_id(tokenizer: GirivinityTokenizer, token: str) -> int | None:
    raw_tokenizer = getattr(tokenizer, "tokenizer", None)
    if raw_tokenizer is not None and hasattr(raw_tokenizer, "token_to_id"):
        return raw_tokenizer.token_to_id(token)
    return None


def _model_config_from_root(raw: dict[str, Any]) -> dict[str, Any]:
    model = raw.get("model", {}) or {}
    architecture = model.get("architecture", {}) or {}
    if architecture:
        return architecture
    key_map = {
        "vocab_size": "vocab_size",
        "hidden_dim": "hidden_dim",
        "dim": "hidden_dim",
        "num_heads": "num_heads",
        "n_heads": "num_heads",
        "num_layers": "num_layers",
        "n_layers": "num_layers",
        "ffn_dim": "ffn_dim",
        "max_seq_len": "max_seq_len",
        "num_kv_heads": "num_kv_heads",
        "n_kv_heads": "num_kv_heads",
    }
    return {target: model[source] for source, target in key_map.items() if source in model}


def _dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _dataclass_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dataclass_to_dict(item) for item in value]
    return value


def _is_fsdp_model(model: torch.nn.Module) -> bool:
    return model.__class__.__name__ == "FullyShardedDataParallel"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain GirivinityModel with pure PyTorch")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = PretrainConfig.from_yaml(args.config)
    GirivinityPretrainer(config).train()


if __name__ == "__main__":
    main()
