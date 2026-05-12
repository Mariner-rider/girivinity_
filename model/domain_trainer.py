from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DOMAINS = {
    "cuda_kernels": {
        "description": "CUDA GPU kernel programming and optimisation",
        "search_queries": ["CUDA kernel optimization shared memory"],
        "target_skill_level": "expert",
        "weight": 2.0,
    },
    "indian_legal": {
        "description": "Indian law, IPC, Constitution, Supreme Court judgments",
        "search_queries": ["Indian Penal Code sections explanation"],
        "target_skill_level": "expert",
        "weight": 1.5,
    },
    "medical": {
        "description": "Medical knowledge",
        "search_queries": ["clinical diagnosis differential diagnosis approach"],
        "target_skill_level": "expert",
        "weight": 1.5,
    },
    "engineering": {
        "description": "Engineering",
        "search_queries": ["data structures algorithms implementation"],
        "target_skill_level": "expert",
        "weight": 1.5,
    },
    "mathematics": {
        "description": "Mathematics",
        "search_queries": ["linear algebra matrix operations proofs"],
        "target_skill_level": "expert",
        "weight": 1.5,
    },
    "hindi_language": {
        "description": "Hindi",
        "search_queries": ["हिंदी व्याकरण नियम"],
        "target_skill_level": "fluent",
        "weight": 1.0,
    },
    "general_reasoning": {
        "description": "Reasoning",
        "search_queries": ["logical reasoning puzzles solutions"],
        "target_skill_level": "expert",
        "weight": 1.0,
    },
}


@dataclass
class DomainDataset:
    domain: str
    samples: list[dict] = field(default_factory=list)
    token_count: int = 0

    def add(self, instruction: str, response: str, source: str = "") -> None:
        self.samples.append(
            {
                "instruction": instruction,
                "response": response,
                "source": source,
                "domain": self.domain,
            }
        )
        self.token_count += len(instruction.split()) + len(response.split())


class DomainCrawler:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        self.output_dir = Path(
            cfg.get("domain_training", {}).get("data_dir", "data/domain_training")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_per_domain = int(
            cfg.get("domain_training", {}).get("target_tokens_per_domain", 5_000_000)
        )

    def crawl_domain(self, domain_key: str) -> DomainDataset:
        domain = DOMAINS[domain_key]
        dataset = DomainDataset(domain=domain_key)
        try:
            from app.core.web_intelligence import WebIntelligence

            wi = WebIntelligence()
            for query in domain["search_queries"]:
                result = wi.search(query)
                for chunk in result.get("answer_chunks", []):
                    text = chunk.get("text", "").strip()
                    if len(text) < 100:
                        continue
                    dataset.add(
                        instruction=f"{query}",
                        response=text,
                        source=chunk.get("url", ""),
                    )
                if dataset.token_count >= self.target_per_domain:
                    break
        except Exception as exc:
            logger.error("Domain crawl failed for %s: %s", domain_key, exc)
        return dataset

    def crawl_all(self) -> dict[str, DomainDataset]:
        datasets = {}
        for key in DOMAINS:
            datasets[key] = self.crawl_domain(key)
            self._save_domain(datasets[key])
        return datasets

    def _save_domain(self, dataset: DomainDataset) -> None:
        out = self.output_dir / f"{dataset.domain}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for sample in dataset.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def build_mixed_dataset(self, output_path: str = "data/domain_training/mixed.jsonl") -> str:
        import random

        all_samples: list[dict] = []
        for key, domain in DOMAINS.items():
            domain_file = self.output_dir / f"{key}.jsonl"
            if not domain_file.exists():
                continue
            with open(domain_file, encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
            weight = domain.get("weight", 1.0)
            repeated = samples * int(weight)
            if weight % 1 > 0 and samples:
                repeated += random.sample(samples, int(len(samples) * (weight % 1)))
            all_samples.extend(repeated)
        random.shuffle(all_samples)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return str(out)


class DomainFineTuner:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        dt = cfg.get("domain_training", {})
        self.epochs = int(dt.get("finetune_epochs", 3))
        self.lr = float(dt.get("finetune_lr", 5e-5))
        self.batch_size = int(dt.get("finetune_batch_size", 2))
        self.grad_accum = int(dt.get("grad_accum", 16))
        self.max_len = int(dt.get("max_seq_len", 2048))
        self.output_dir = Path(dt.get("output_dir", "models/domain_finetuned"))

    def finetune(self, dataset_path: str) -> None:
        from model.train import train

        train(
            data_path=dataset_path,
            tokeniser_path="models/tokeniser/tokeniser.json",
            output_dir=str(self.output_dir),
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            grad_accum=self.grad_accum,
            max_len=self.max_len,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl", action="store_true")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--dataset", default="data/domain_training/mixed.jsonl")
    args = parser.parse_args()

    if args.crawl:
        crawler = DomainCrawler()
        if args.domain == "all":
            crawler.crawl_all()
        else:
            ds = crawler.crawl_domain(args.domain)
            crawler._save_domain(ds)

    if args.mix:
        DomainCrawler().build_mixed_dataset()

    if args.finetune:
        DomainFineTuner().finetune(args.dataset)
