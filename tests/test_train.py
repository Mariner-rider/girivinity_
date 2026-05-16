import json
import tempfile
from pathlib import Path
import pytest


def test_instruction_dataset_loads():
    pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    from model.architecture import GirivinityConfig
    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / "data.jsonl"
        data_file.write_text(
            json.dumps({
                "instruction": "What is Python?",
                "response": "Python is a programming language.",
            }) + "\n"
        )
        from model.tokeniser import train_tokeniser
        tok_path = Path(tmp) / "tok"
        corpus = Path(tmp) / "corpus.txt"
        corpus.write_text("What is Python? Python is a programming language.")
        train_tokeniser(str(corpus), vocab_size=200, save_path=str(tok_path))
        from model.train import InstructionDataset
        ds = InstructionDataset(
            str(data_file),
            str(tok_path / "tokeniser.json"),
            max_len=64,
        )
        assert len(ds) >= 1
        x, y = ds[0]
        assert x.shape == y.shape
        assert x.dtype.is_floating_point is False


def test_collate_pads_correctly():
    pytest.importorskip("torch")
    import torch
    from model.train import collate
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
        (torch.tensor([5, 6]),    torch.tensor([6, 7])),
    ]
    xs, ys = collate(batch)
    assert xs.shape == (2, 3)
    assert ys.shape == (2, 3)
    assert ys[1, 2].item() == -100
