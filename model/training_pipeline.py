from __future__ import annotations
import hashlib
import json
import logging
import random
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    chunk_id: str
    text: str
    score: float
    signals: dict
    tier: str


class DataQualityScorer:
    def score(self, text: str, chunk_id: str = "") -> QualityScore:
        signals = {}

        length = len(text.split())
        if length < 20:
            signals["length"] = 0.0
        elif length < 50:
            signals["length"] = 0.5
        elif length < 500:
            signals["length"] = 1.0
        else:
            signals["length"] = 0.8

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        valid_sentences = [s for s in sentences if len(s.split()) >= 4]
        sentence_ratio = len(valid_sentences) / max(len(sentences), 1)
        signals["sentence_quality"] = round(sentence_ratio, 3)

        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        signals["diversity"] = round(unique_ratio, 3)

        boilerplate = [
            "click here", "subscribe", "cookie policy", "privacy policy",
            "terms of service", "all rights reserved", "follow us on",
            "sign up now", "buy now", "free trial",
        ]
        boilerplate_hits = sum(1 for b in boilerplate if b in text.lower())
        signals["boilerplate"] = max(0.0, 1.0 - boilerplate_hits * 0.3)

        if any(w in text for w in ["def ", "class ", "__init__", "return ", "__global__", "kernel", "void ", "#include"]):
            signals["code_bonus"] = 0.2
        else:
            signals["code_bonus"] = 0.0

        academic = ["research", "study", "analysis", "according to", "demonstrates", "evidence", "methodology"]
        academic_hits = sum(1 for a in academic if a in text.lower())
        signals["academic_signal"] = min(0.2, academic_hits * 0.05)

        score = (
            signals["length"] * 0.25
            + signals["sentence_quality"] * 0.25
            + signals["diversity"] * 0.20
            + signals["boilerplate"] * 0.15
            + signals["code_bonus"] * 0.10
            + signals["academic_signal"] * 0.05
        )
        score = round(min(1.0, score), 4)

        tier = (
            "gold" if score >= 0.8 else
            "silver" if score >= 0.6 else
            "bronze" if score >= 0.4 else
            "discard"
        )

        return QualityScore(chunk_id=chunk_id, text=text, score=score, signals=signals, tier=tier)


class Deduplicator:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def is_duplicate(self, text: str) -> bool:
        exact_hash = self._hash(text)
        if self._seen_exact(exact_hash):
            return True
        shingle_hash = self._shingle_hash(text)
        if self._seen_shingle(shingle_hash):
            return True
        self._store(exact_hash, shingle_hash)
        return False

    def filter_batch(self, chunks: list[dict]) -> tuple[list[dict], int]:
        unique = []
        dup_count = 0
        for chunk in chunks:
            text = chunk.get("text", "")
            if not self.is_duplicate(text):
                unique.append(chunk)
            else:
                dup_count += 1
        return unique, dup_count

    def _hash(self, text: str) -> str:
        normalised = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalised.encode()).hexdigest()

    def _shingle_hash(self, text: str, k: int = 5) -> str:
        words = text.lower().split()
        shingles = {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}
        combined = " ".join(sorted(shingles)[:20])
        return hashlib.md5(combined.encode()).hexdigest()

    def _seen_exact(self, hash_val: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT 1 FROM dedup_hashes WHERE exact_hash = ?", (hash_val,)).fetchone()
            return row is not None
        except Exception:
            return False

    def _seen_shingle(self, hash_val: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT 1 FROM dedup_hashes WHERE shingle_hash = ?", (hash_val,)).fetchone()
            return row is not None
        except Exception:
            return False

    def _store(self, exact: str, shingle: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO dedup_hashes (exact_hash, shingle_hash, seen_at) VALUES (?, ?, ?)",
                    (exact, shingle, datetime.now(timezone.utc).isoformat()),
                )
        except Exception as exc:
            logger.warning("Dedup store failed: %s", exc)

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dedup_hashes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    exact_hash   TEXT UNIQUE NOT NULL,
                    shingle_hash TEXT NOT NULL,
                    seen_at      TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shingle ON dedup_hashes (shingle_hash)")


class ReplayBuffer:
    def __init__(self, db_path: str, max_size: int = 10000) -> None:
        self.db_path = db_path
        self.max_size = max_size
        self._init_db()

    def add(self, instruction: str, response: str, quality_score: float, domain: str = "general") -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM replay_buffer").fetchone()[0]
                if count >= self.max_size:
                    conn.execute(
                        "DELETE FROM replay_buffer WHERE id IN (SELECT id FROM replay_buffer ORDER BY quality_score ASC LIMIT 100)"
                    )
                conn.execute(
                    "INSERT OR IGNORE INTO replay_buffer (instruction, response, quality_score, domain, added_at) VALUES (?, ?, ?, ?, ?)",
                    (instruction[:1000], response[:2000], quality_score, domain, datetime.now(timezone.utc).isoformat()),
                )
        except Exception as exc:
            logger.warning("ReplayBuffer add failed: %s", exc)

    def sample(self, n: int, min_quality: float = 0.6) -> list[dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT instruction, response, quality_score, domain FROM replay_buffer WHERE quality_score >= ? ORDER BY RANDOM() LIMIT ?",
                    (min_quality, n),
                ).fetchall()
            return [{"instruction": r[0], "response": r[1], "quality": r[2], "domain": r[3], "is_replay": True} for r in rows]
        except Exception as exc:
            logger.warning("ReplayBuffer sample failed: %s", exc)
            return []

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS replay_buffer (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    instruction   TEXT NOT NULL,
                    response      TEXT NOT NULL,
                    quality_score REAL DEFAULT 0.5,
                    domain        TEXT DEFAULT 'general',
                    added_at      TEXT NOT NULL,
                    UNIQUE(instruction)
                )
                """
            )


class CurriculumScheduler:
    def schedule(self, scored_chunks: list[QualityScore]) -> list[QualityScore]:
        gold = [c for c in scored_chunks if c.tier == "gold"]
        silver = [c for c in scored_chunks if c.tier == "silver"]
        bronze = [c for c in scored_chunks if c.tier == "bronze"]

        random.shuffle(gold)
        random.shuffle(silver)
        random.shuffle(bronze)

        oversampled_gold = gold * 3

        stage_1 = bronze + silver
        random.shuffle(stage_1)

        stage_2 = silver + gold
        random.shuffle(stage_2)

        stage_3 = oversampled_gold

        scheduled = stage_1 + stage_2 + stage_3

        logger.info(
            "Curriculum: bronze=%d silver=%d gold=%d total_scheduled=%d",
            len(bronze), len(silver), len(gold), len(scheduled),
        )
        return scheduled


class DiversityEnforcer:
    MAX_TOPIC_RATIO = 0.30

    def enforce(self, chunks: list[dict], topic_field: str = "query") -> list[dict]:
        if not chunks:
            return chunks

        topic_counts: dict[str, int] = defaultdict(int)
        topic_cap = max(1, int(len(chunks) * self.MAX_TOPIC_RATIO))

        selected = []
        for chunk in chunks:
            topic = self._extract_topic(chunk.get(topic_field, ""))
            if topic_counts[topic] < topic_cap:
                selected.append(chunk)
                topic_counts[topic] += 1

        removed = len(chunks) - len(selected)
        if removed > 0:
            logger.info("DiversityEnforcer: removed %d over-represented chunks", removed)
        return selected

    def _extract_topic(self, query: str) -> str:
        words = [w for w in query.lower().split() if len(w) > 4 and w.isalpha()]
        return words[0] if words else "general"


class ImprovedLoRATrainer:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        lora = cfg.get("lora", {})
        tr = cfg.get("training", {})

        self.r = int(lora.get("r", 16))
        self.alpha = int(lora.get("alpha", 32))
        self.dropout = float(lora.get("dropout", 0.05))
        self.target_modules = lora.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
        self.epochs = int(lora.get("epochs", 3))
        self.lr = float(lora.get("learning_rate", 2e-4))
        self.batch_size = int(lora.get("batch_size", 2))
        self.grad_accum = int(lora.get("grad_accum", 8))
        self.max_seq_len = int(lora.get("max_seq_len", 2048))
        self.loss_abort = float(lora.get("loss_abort_threshold", 2.0))
        self.adapters_dir = Path(lora.get("adapters_dir", "models/adapters"))

        db_base = Path(tr.get("queue_db", "data/training.db")).parent
        self.dedup = Deduplicator(str(db_base / "dedup.db"))
        self.replay = ReplayBuffer(str(db_base / "replay.db"))
        self.scorer = DataQualityScorer()
        self.curriculum = CurriculumScheduler()
        self.diversity = DiversityEnforcer()

    def train_from_jsonl(self, dataset_path: str, version: str) -> bool:
        raw_records = self._load_jsonl(dataset_path)
        if not raw_records:
            logger.warning("ImprovedLoRATrainer: empty dataset")
            return False

        logger.info("ImprovedLoRATrainer: loaded %d raw records", len(raw_records))

        scored = [
            self.scorer.score(
                r.get("response", ""),
                chunk_id=hashlib.md5(r.get("response", "").encode()).hexdigest()[:8],
            )
            for r in raw_records
        ]

        valid_pairs = [(raw_records[i], scored[i]) for i in range(len(raw_records)) if scored[i].tier != "discard"]
        discarded = len(raw_records) - len(valid_pairs)
        logger.info("Quality filter: kept %d, discarded %d", len(valid_pairs), discarded)

        if not valid_pairs:
            logger.warning("All chunks discarded by quality filter")
            return False

        texts_for_dedup = [{"text": p[0].get("response", ""), "idx": i} for i, p in enumerate(valid_pairs)]
        unique_texts, dups_removed = self.dedup.filter_batch(texts_for_dedup)
        unique_indices = {u["idx"] for u in unique_texts}
        valid_pairs = [p for i, p in enumerate(valid_pairs) if i in unique_indices]
        logger.info("Deduplication: kept %d, removed %d duplicates", len(valid_pairs), dups_removed)

        diversity_input = [{"text": p[0].get("response", ""), "query": p[0].get("instruction", ""), "pair_idx": i} for i, p in enumerate(valid_pairs)]
        diverse = self.diversity.enforce(diversity_input, "query")
        diverse_indices = {d["pair_idx"] for d in diverse}
        valid_pairs = [p for i, p in enumerate(valid_pairs) if i in diverse_indices]
        logger.info("Diversity filter: %d balanced chunks", len(valid_pairs))

        scored_only = [p[1] for p in valid_pairs]
        scheduled = self.curriculum.schedule(scored_only)

        ordered_pairs = []
        for s in scheduled:
            for record, score_obj in valid_pairs:
                if score_obj.chunk_id == s.chunk_id:
                    ordered_pairs.append((record, score_obj))
                    break

        if not ordered_pairs:
            ordered_pairs = valid_pairs

        replay_n = max(1, len(ordered_pairs) // 10)
        replay_examples = self.replay.sample(replay_n, min_quality=0.6)

        all_records = []
        for record, score_obj in ordered_pairs:
            all_records.append({
                "instruction": record.get("instruction", ""),
                "response": record.get("response", ""),
                "quality": score_obj.score,
                "is_replay": False,
            })
        all_records.extend(replay_examples)
        random.shuffle(all_records[-replay_n:])

        logger.info("Final training set: %d new + %d replay = %d total", len(ordered_pairs), len(replay_examples), len(all_records))

        refined_path = Path(dataset_path).with_suffix(".refined.jsonl")
        with open(refined_path, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps({"instruction": r["instruction"], "response": r["response"]}) + "\n")

        success = self._run_qlora(str(refined_path), version)

        if success:
            gold_count = 0
            for record, score_obj in ordered_pairs:
                if score_obj.tier == "gold":
                    self.replay.add(
                        instruction=record.get("instruction", ""),
                        response=record.get("response", ""),
                        quality_score=score_obj.score,
                    )
                    gold_count += 1
            logger.info("Stored %d gold examples in replay buffer", gold_count)

        return success

    def _run_qlora(self, dataset_path: str, version: str) -> bool:
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )

            base_model_path = yaml.safe_load(Path("config.yaml").read_text()).get("modules", {}).get("self_training", {}).get("base_model_path", "models/base")

            if not Path(base_model_path).exists():
                logger.warning("Base model not found at %s", base_model_path)
                return False

            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            model.config.use_cache = False
            model.enable_input_require_grads()

            latest = self.adapters_dir / "latest"
            if latest.exists() and latest.is_symlink():
                model = PeftModel.from_pretrained(model, str(latest), is_trainable=True)
                logger.info("Loaded existing LoRA adapter from %s", latest)
            else:
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=self.r,
                        lora_alpha=self.alpha,
                        target_modules=self.target_modules,
                        lora_dropout=self.dropout,
                        task_type=TaskType.CAUSAL_LM,
                        bias="none",
                    ),
                )
                model.print_trainable_parameters()

            records = []
            with open(dataset_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    records.append(
                        {
                            "text": (
                                "<|begin_of_text|>"
                                "<|start_header_id|>user<|end_header_id|>\n"
                                f"{r['instruction']}"
                                "<|eot_id|>"
                                "<|start_header_id|>assistant<|end_header_id|>\n"
                                f"{r['response']}"
                                "<|eot_id|>"
                            )
                        }
                    )

            def tokenize(ex: dict) -> dict:
                return tokenizer(ex["text"], truncation=True, max_length=self.max_seq_len, padding="max_length")

            tokenized = Dataset.from_list(records).map(tokenize, batched=True, remove_columns=["text"])

            adapter_out = self.adapters_dir / version
            adapter_out.mkdir(parents=True, exist_ok=True)

            result = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(adapter_out),
                    num_train_epochs=self.epochs,
                    per_device_train_batch_size=self.batch_size,
                    gradient_accumulation_steps=self.grad_accum,
                    learning_rate=self.lr,
                    bf16=torch.cuda.is_available(),
                    fp16=False,
                    logging_steps=10,
                    save_strategy="no",
                    report_to="none",
                    optim="paged_adamw_8bit",
                    warmup_ratio=0.05,
                    lr_scheduler_type="cosine",
                    group_by_length=True,
                    dataloader_drop_last=True,
                ),
                train_dataset=tokenized,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            ).train()

            loss = result.training_loss
            logger.info("QLoRA training complete: loss=%.4f version=%s", loss, version)

            if loss > self.loss_abort:
                logger.error("Loss %.4f exceeds abort threshold %.4f", loss, self.loss_abort)
                return False

            model.save_pretrained(str(adapter_out))
            tokenizer.save_pretrained(str(adapter_out))

            latest_link = self.adapters_dir / "latest"
            if latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(adapter_out.resolve())

            self._log_training_event(version, loss, len(records))
            return True

        except Exception as exc:
            logger.error("QLoRA training failed: %s", exc)
            return False

    def _load_jsonl(self, path: str) -> list[dict]:
        records = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as exc:
            logger.error("JSONL load failed: %s", exc)
        return records

    def _log_training_event(self, version: str, loss: float, sample_count: int) -> None:
        log_path = Path("logs/self_training.jsonl")
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "version": version,
                        "loss": round(loss, 4),
                        "sample_count": sample_count,
                        "method": "QLoRA+Replay+Curriculum+Dedup",
                        "status": "success",
                    }
                )
                + "\n"
            )
