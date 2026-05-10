import json
from pathlib import Path

from knowledge_distillation_engine import KnowledgeDistillationSystem


def test_knowledge_distillation_pipeline_and_storage(tmp_path: Path):
    system = KnowledgeDistillationSystem()
    crawler_output = [
        {
            "url": "https://example.org/a",
            "title": "Doc A",
            "text": "AI is transforming healthcare. AI is transforming healthcare. Hospitals have 20% faster triage.",
        },
        {
            "url": "https://example.org/b",
            "title": "Doc B",
            "text": "Short.",
        },
    ]

    distilled = system.distill(crawler_output, min_quality=0.2)
    assert len(distilled) >= 1
    assert distilled[0].source_url == "https://example.org/a"
    assert len(distilled[0].key_facts) >= 1

    out = tmp_path / "distilled.jsonl"
    path = system.store_structured(distilled, str(out))
    assert Path(path).exists()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    first = json.loads(lines[0])
    assert "summary" in first
    assert "key_facts" in first
    assert "quality_score" in first
