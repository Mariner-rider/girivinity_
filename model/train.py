from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.architecture import GirivinityConfig, GirivinityModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    def __init__(
        self, jsonl_path: str, tokeniser_path: str, max_len: int = 1024
    ) -> None:
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(tokeniser_path)
        self.max_len = max_len
        self.samples: list[list[int]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = (
                    f"### Instruction:\n{rec['instruction']}\n\n"
                    f"### Response:\n{rec['response']}"
                )
                enc = self.tok.encode(text)
                ids = enc.ids[: self.max_len]
                if len(ids) > 8:
                    self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    xs_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    ys_pad = torch.full((len(ys), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xs_pad[i, : x.size(0)] = x
        ys_pad[i, : y.size(0)] = y
    return xs_pad, ys_pad


def train(
    data_path: str,
    tokeniser_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 3e-4,
    grad_accum: int = 8,
    max_len: int = 1024,
    save_every: int = 1000,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    cfg = GirivinityConfig.from_yaml()
    model = GirivinityModel(cfg).to(device)
    logger.info("Model: %s", model.param_count())

    dataset = InstructionDataset(data_path, tokeniser_path, max_len)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=True,
    )
    logger.info("Dataset: %d samples", len(dataset))

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs * len(loader)
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            output = model(x)
            logits = output[0]
            loss = nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size),
                y.view(-1),
                ignore_index=-100,
            )
            (loss / grad_accum).backward()

            if step % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()

            total_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "epoch=%d step=%d loss=%.4f lr=%.2e elapsed=%.1fs",
                    epoch, global_step,
                    total_loss / step,
                    scheduler.get_last_lr()[0],
                    elapsed,
                )

            if global_step % save_every == 0:
                ckpt = out / f"checkpoint_{global_step}"
                ckpt.mkdir(exist_ok=True)
                torch.save(model.state_dict(), ckpt / "model.pt")
                logger.info("Checkpoint saved to %s", ckpt)

        avg_loss = total_loss / len(loader)
        logger.info("Epoch %d complete — avg loss: %.4f", epoch, avg_loss)

    final = out / "final"
    final.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final / "model.pt")
    cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    import yaml
    (final / "config.yaml").write_text(yaml.dump(cfg_dict))
    logger.info("Training complete. Model saved to %s", final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/training_queue")
    parser.add_argument("--tokeniser", default="models/tokeniser/tokeniser.json")
    parser.add_argument("--output",    default="models/base")
    parser.add_argument("--epochs",    type=int,   default=3)
    parser.add_argument("--batch",     type=int,   default=4)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--grad-accum",type=int,   default=8)
    parser.add_argument("--max-len",   type=int,   default=1024)
    parser.add_argument("--save-every",type=int,   default=1000)
    args = parser.parse_args()
    train(
        data_path=args.data,
        tokeniser_path=args.tokeniser,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        grad_accum=args.grad_accum,
        max_len=args.max_len,
        save_every=args.save_every,
    )
