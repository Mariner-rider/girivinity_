from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]
TOKENIZER_FILE = "tokenizer.json"


class GirivinityTokenizer:
    """Girivinity's standalone BPE tokenizer built with HuggingFace tokenizers.

    This class intentionally does not import or depend on transformers.AutoTokenizer.
    It trains a fresh BPE vocabulary suitable for multilingual text (including Hindi),
    English, and source code using Unicode normalization plus byte-level handling.
    """

    def __init__(self, tokenizer: Any | None = None) -> None:
        self.tokenizer = tokenizer

    def train(
        self,
        corpus_files: list[str],
        vocab_size: int = 100000,
        min_frequency: int = 2,
        output_path: str = "girivinity_tokenizer/",
    ) -> None:
        """Train a BPE tokenizer from raw text corpus files and save it."""
        if not corpus_files:
            raise ValueError("At least one corpus file is required to train the tokenizer")
        missing = [path for path in corpus_files if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Corpus files not found: {missing}")

        tokenizers = self._tokenizers()
        models = self._tokenizers_module("models")
        trainers = self._tokenizers_module("trainers")
        normalizers = self._tokenizers_module("normalizers")
        pre_tokenizers = self._tokenizers_module("pre_tokenizers")
        decoders = self._tokenizers_module("decoders")
        processors = self._tokenizers_module("processors")

        tokenizer = tokenizers.Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train(corpus_files, trainer=trainer)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [SEP] $B [EOS]",
            special_tokens=[
                ("[BOS]", tokenizer.token_to_id("[BOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

        self.tokenizer = tokenizer
        self.save(output_path)

    def encode(self, text: str) -> list[int]:
        self._require_tokenizer()
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        self._require_tokenizer()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        self._require_tokenizer()
        return [encoded.ids for encoded in self.tokenizer.encode_batch(texts)]

    def save(self, path: str) -> None:
        self._require_tokenizer()
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(output_dir / TOKENIZER_FILE))
        (output_dir / "special_tokens.json").write_text(
            json.dumps({"special_tokens": SPECIAL_TOKENS}, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str) -> None:
        tokenizers = self._tokenizers()
        tokenizer_path = Path(path)
        if tokenizer_path.is_dir():
            tokenizer_path = tokenizer_path / TOKENIZER_FILE
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

    @classmethod
    def from_file(cls, path: str) -> "GirivinityTokenizer":
        tokenizer = cls()
        tokenizer.load(path)
        return tokenizer

    def __len__(self) -> int:
        self._require_tokenizer()
        return self.tokenizer.get_vocab_size()

    def _require_tokenizer(self) -> None:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded yet")

    @staticmethod
    def _tokenizers() -> Any:
        try:
            import tokenizers
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'tokenizers' package is required. Install dependencies from requirements.txt."
            ) from exc
        return tokenizers

    @staticmethod
    def _tokenizers_module(name: str) -> Any:
        try:
            return __import__(f"tokenizers.{name}", fromlist=[name])
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'tokenizers' package is required. Install dependencies from requirements.txt."
            ) from exc


def build_training_corpus_from_jsonl(jsonl_path: str, output_txt: str) -> None:
    """Convert knowledge-distillation JSONL output into a plain text corpus.

    The distillation pipeline may write records with fields such as instruction,
    question, prompt, input, context, response, answer, completion, output, text,
    code, or chat-style messages. This utility extracts the useful text portions
    and writes one separated training block per JSONL record.
    """
    input_path = Path(jsonl_path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            texts = _extract_training_texts(record)
            if texts:
                dst.write("\n".join(texts))
                dst.write("\n\n")


def _extract_training_texts(record: Any) -> list[str]:
    if isinstance(record, str):
        return [record.strip()] if record.strip() else []
    if isinstance(record, list):
        texts: list[str] = []
        for item in record:
            texts.extend(_extract_training_texts(item))
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
