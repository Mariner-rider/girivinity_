from __future__ import annotations
from pathlib import Path


def train_tokeniser(
    corpus_path: str,
    vocab_size: int = 32000,
    save_path: str = "models/tokeniser",
) -> None:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        min_frequency=2,
    )
    tokenizer.train([corpus_path], trainer)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    tokenizer.save(f"{save_path}/tokeniser.json")
    print(f"Tokeniser saved to {save_path}/tokeniser.json")


if __name__ == "__main__":
    train_tokeniser("data/seed_corpus.txt")
