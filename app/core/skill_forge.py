from __future__ import annotations
import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    slug: str
    topic: str
    version: int
    instructions: str
    examples: list[dict]
    edge_cases: list[str]
    confidence: float
    usage_count: int
    avg_feedback: float
    created_at: str
    updated_at: str
    parent_skills: list[str]
    source_urls: list[str]

    def to_prompt_block(self) -> str:
        """Format skill for injection into LLM prompt."""
        lines = [
            f"## Skill: {self.topic}",
            f"Confidence: {self.confidence:.2f} | "
            f"Used: {self.usage_count}x | "
            f"Avg rating: {self.avg_feedback:.1f}/5.0",
            "",
            "### Instructions",
            self.instructions,
        ]
        if self.examples:
            lines.append("\n### Examples")
            for ex in self.examples[:3]:
                lines.append(f"Input: {ex.get('input', '')}")
                lines.append(f"Output: {ex.get('output', '')}")
                lines.append("")
        if self.edge_cases:
            lines.append("### Watch out for")
            for ec in self.edge_cases[:3]:
                lines.append(f"- {ec}")
        return "\n".join(lines)


@dataclass
class SkillEvalResult:
    skill_slug: str
    score: float
    passed: int
    total: int
    failures: list[str]
    improvement_vs_baseline: float


class SkillForge:
    """
    Girivinity's native skill generation engine.

    Generates living skills from web content and user interactions.
    Skills are semantically indexed, auto-versioned, and composed
    for complex queries. No external API required.
    """

    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        sf = cfg.get("skill_forge", {})
        self.skills_dir = Path(sf.get("skills_dir", "skills"))
        self.db_path = Path(sf.get("db_path", "data/skill_forge.db"))
        self.chroma_path = cfg["rag"]["chroma_path"]
        self.min_chunks = int(sf.get("min_chunks_to_generate", 3))
        self.max_skills_composed = int(sf.get("max_skills_composed", 3))

        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        client = chromadb.PersistentClient(path=self.chroma_path)
        self.skill_index = client.get_or_create_collection("skill_index")

    def generate_async(self, topic: str, chunks: list[dict], urls: list[str]) -> None:
        """Generate a skill in background. Never blocks response."""
        threading.Thread(
            target=self._generate,
            args=(topic, chunks, urls),
            daemon=True,
        ).start()

    def get_skill_for_query(self, query: str) -> Skill | None:
        from app.core.query_router import get_embedder

        embedder = get_embedder()
        vec = embedder.encode(query).tolist()

        try:
            results = self.skill_index.query(
                query_embeddings=[vec],
                n_results=self.max_skills_composed,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("Skill index query failed: %s", exc)
            return None

        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []

        scores = [max(0.0, 1.0 - d / 2.0) for d in distances]
        qualifying = [
            (docs[i], metas[i], scores[i])
            for i in range(len(docs))
            if scores[i] >= 0.60
        ]

        if not qualifying:
            return None

        if len(qualifying) == 1:
            return self._load_skill(qualifying[0][1]["slug"])

        return self._compose_skills(query, [self._load_skill(q[1]["slug"]) for q in qualifying])

    def update_skill_feedback(self, skill_slug: str, user_score: float) -> None:
        """Called after user rates a response. Updates skill quality."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO skill_feedback (slug, score, timestamp) "
                "VALUES (?, ?, ?)",
                (skill_slug, user_score, datetime.now(timezone.utc).isoformat()),
            )
        threading.Thread(
            target=self._maybe_improve_skill,
            args=(skill_slug,),
            daemon=True,
        ).start()

    def evaluate_skill(self, skill: Skill) -> SkillEvalResult:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT query, response, feedback_score FROM skill_interactions "
                "WHERE skill_slug = ? ORDER BY timestamp DESC LIMIT 20",
                (skill.slug,),
            ).fetchall()

        if not rows:
            return SkillEvalResult(
                skill_slug=skill.slug,
                score=skill.confidence,
                passed=0,
                total=0,
                failures=[],
                improvement_vs_baseline=0.0,
            )

        passed = 0
        failures = []
        for query, response, feedback in rows:
            if feedback and float(feedback) >= 3.5:
                passed += 1
            elif feedback and float(feedback) < 3.5:
                failures.append(f"Low score ({feedback}/5): Q={query[:60]}")

        score = passed / len(rows) if rows else 0.0
        baseline = 0.6
        return SkillEvalResult(
            skill_slug=skill.slug,
            score=score,
            passed=passed,
            total=len(rows),
            failures=failures,
            improvement_vs_baseline=round(score - baseline, 3),
        )

    def list_skills(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT slug, topic, version, confidence, usage_count, "
                "avg_feedback, updated_at FROM skills ORDER BY usage_count DESC"
            ).fetchall()
        return [
            {
                "slug": r[0],
                "topic": r[1],
                "version": r[2],
                "confidence": r[3],
                "usage_count": r[4],
                "avg_feedback": r[5],
                "updated_at": r[6],
            }
            for r in rows
        ]

    def record_interaction(
        self,
        skill_slug: str,
        query: str,
        response: str,
        feedback: float | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO skill_interactions "
                "(skill_slug, query, response, feedback_score, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    skill_slug,
                    query,
                    response[:2000],
                    feedback,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _generate(self, topic: str, chunks: list[dict], urls: list[str]) -> None:
        slug = self._slugify(topic)
        existing = self._load_skill(slug)

        if existing:
            self._update_skill(existing, chunks, urls)
            return

        if len(chunks) < self.min_chunks:
            logger.info(
                "SkillForge: not enough chunks (%d < %d) for: %s",
                len(chunks),
                self.min_chunks,
                slug,
            )
            return

        instructions = self._distil_instructions(topic, chunks)
        examples = self._extract_examples(topic, chunks)
        edge_cases = self._extract_edge_cases(topic, chunks)
        confidence = self._initial_confidence(chunks)

        skill = Skill(
            slug=slug,
            topic=topic,
            version=1,
            instructions=instructions,
            examples=examples,
            edge_cases=edge_cases,
            confidence=confidence,
            usage_count=0,
            avg_feedback=0.0,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            parent_skills=[],
            source_urls=urls[:5],
        )

        self._save_skill(skill)
        self._index_skill(skill)
        logger.info("SkillForge: generated skill '%s' v%d", slug, skill.version)

    def _update_skill(self, skill: Skill, new_chunks: list[dict], urls: list[str]) -> None:
        """Merge new knowledge into existing skill — living update."""
        new_instructions = self._distil_instructions(skill.topic, new_chunks)
        new_examples = self._extract_examples(skill.topic, new_chunks)
        new_edge_cases = self._extract_edge_cases(skill.topic, new_chunks)

        merged_instructions = self._merge_text(skill.instructions, new_instructions)
        merged_examples = self._merge_examples(skill.examples, new_examples)
        merged_edge_cases = list(dict.fromkeys(skill.edge_cases + new_edge_cases))[:10]
        merged_urls = list(dict.fromkeys(skill.source_urls + urls))[:10]

        skill.instructions = merged_instructions
        skill.examples = merged_examples
        skill.edge_cases = merged_edge_cases
        skill.source_urls = merged_urls
        skill.version += 1
        skill.updated_at = datetime.now(timezone.utc).isoformat()
        skill.confidence = min(1.0, skill.confidence + 0.05 * len(new_chunks))

        self._save_skill(skill)
        self._index_skill(skill)
        logger.info("SkillForge: updated skill '%s' to v%d", skill.slug, skill.version)

    def _distil_instructions(self, topic: str, chunks: list[dict]) -> str:
        """Extract structured instructions from raw web chunks."""
        all_text = "\n".join(c.get("text", "") for c in chunks[:5])

        sentences = re.split(r"(?<=[.!?])\s+", all_text)
        scored = []
        topic_words = set(topic.lower().split())

        for s in sentences:
            s = s.strip()
            if len(s) < 30 or len(s) > 400:
                continue
            s_lower = s.lower()
            score = 0
            if any(
                w in s_lower
                for w in [
                    "is",
                    "are",
                    "means",
                    "refers to",
                    "defined as",
                    "used to",
                    "allows",
                    "enables",
                    "should",
                    "must",
                    "important",
                    "key",
                    "note",
                    "remember",
                    "avoid",
                ]
            ):
                score += 2
            for word in topic_words:
                if word in s_lower:
                    score += 1
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in scored[:12]]

        instructions = (
            f"This skill covers: {topic}\n\n"
            f"Key knowledge:\n"
            + "\n".join(f"- {s}" for s in top_sentences)
        )
        return instructions

    def _extract_examples(self, topic: str, chunks: list[dict]) -> list[dict]:
        """Find input/output pattern pairs from retrieved content."""
        examples = []
        all_text = "\n".join(c.get("text", "") for c in chunks[:5])

        code_blocks = re.findall(r"```[\w]*\n(.*?)```", all_text, re.DOTALL)
        for i, block in enumerate(code_blocks[:3]):
            examples.append(
                {
                    "input": f"Show me an example of {topic}",
                    "output": block.strip()[:500],
                }
            )

        example_patterns = re.findall(r"[Ff]or example[,:]?\s+(.{20,200}?)(?:\.|$)", all_text)
        for pattern in example_patterns[:2]:
            examples.append(
                {
                    "input": f"Give me an example related to {topic}",
                    "output": pattern.strip(),
                }
            )

        return examples[:5]

    def _extract_edge_cases(self, topic: str, chunks: list[dict]) -> list[str]:
        """Find warnings, caveats, and edge cases from content."""
        edge_cases = []
        all_text = "\n".join(c.get("text", "") for c in chunks[:5])
        sentences = re.split(r"(?<=[.!?])\s+", all_text)

        warning_words = [
            "however",
            "but",
            "except",
            "careful",
            "warning",
            "note that",
            "important",
            "avoid",
            "never",
            "always",
            "common mistake",
            "pitfall",
            "gotcha",
            "be aware",
        ]
        for s in sentences:
            s = s.strip()
            if 20 < len(s) < 300:
                if any(w in s.lower() for w in warning_words):
                    edge_cases.append(s)

        return edge_cases[:5]

    def _initial_confidence(self, chunks: list[dict]) -> float:
        if not chunks:
            return 0.0
        avg_score = sum(c.get("score", 0.5) for c in chunks) / len(chunks)
        volume_bonus = min(0.2, len(chunks) * 0.02)
        return min(1.0, round(avg_score + volume_bonus, 3))

    def _compose_skills(self, query: str, skills: list[Skill | None]) -> Skill | None:
        """Combine multiple skills into a meta-skill for complex queries."""
        valid = [s for s in skills if s is not None]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]

        combined_instructions = "\n\n".join([f"=== {s.topic} ===\n{s.instructions}" for s in valid])
        combined_examples = []
        combined_edge_cases = []
        for s in valid:
            combined_examples.extend(s.examples)
            combined_edge_cases.extend(s.edge_cases)

        composite_confidence = sum(s.confidence for s in valid) / len(valid)

        return Skill(
            slug="__composed__",
            topic=f"Composed: {query[:60]}",
            version=0,
            instructions=combined_instructions,
            examples=combined_examples[:6],
            edge_cases=list(dict.fromkeys(combined_edge_cases))[:6],
            confidence=composite_confidence,
            usage_count=0,
            avg_feedback=0.0,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            parent_skills=[s.slug for s in valid],
            source_urls=[u for s in valid for u in s.source_urls][:5],
        )

    def _maybe_improve_skill(self, slug: str) -> None:
        """Re-distil skill if recent feedback is below threshold."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT AVG(score) FROM "
                "(SELECT score FROM skill_feedback WHERE slug = ? "
                " ORDER BY id DESC LIMIT 10)",
                (slug,),
            ).fetchone()

        if not row or row[0] is None:
            return

        avg = float(row[0])
        if avg < 3.5:
            logger.info(
                "SkillForge: avg feedback %.2f for '%s' "
                "below threshold — queuing improvement",
                avg,
                slug,
            )
            skill = self._load_skill(slug)
            if skill:
                skill.confidence = max(0.1, skill.confidence - 0.1)
                skill.updated_at = datetime.now(timezone.utc).isoformat()
                self._save_skill(skill)

    def _save_skill(self, skill: Skill) -> None:
        skill_path = self.skills_dir / skill.slug
        skill_path.mkdir(parents=True, exist_ok=True)
        (skill_path / "skill.json").write_text(json.dumps(asdict(skill), indent=2), encoding="utf-8")
        (skill_path / "SKILL.md").write_text(skill.to_prompt_block(), encoding="utf-8")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO skills
                  (slug, topic, version, confidence, usage_count,
                   avg_feedback, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                  version=excluded.version,
                  confidence=excluded.confidence,
                  usage_count=excluded.usage_count,
                  avg_feedback=excluded.avg_feedback,
                  updated_at=excluded.updated_at
            """,
                (
                    skill.slug,
                    skill.topic,
                    skill.version,
                    skill.confidence,
                    skill.usage_count,
                    skill.avg_feedback,
                    skill.updated_at,
                ),
            )

    def _load_skill(self, slug: str) -> Skill | None:
        skill_path = self.skills_dir / slug / "skill.json"
        if not skill_path.exists():
            return None
        try:
            data = json.loads(skill_path.read_text(encoding="utf-8"))
            return Skill(**data)
        except Exception as exc:
            logger.warning("Failed to load skill %s: %s", slug, exc)
            return None

    def _index_skill(self, skill: Skill) -> None:
        """Index skill in ChromaDB for semantic retrieval."""
        from app.core.query_router import get_embedder

        embedder = get_embedder()
        vec = embedder.encode(f"{skill.topic} {skill.instructions[:200]}").tolist()
        try:
            self.skill_index.upsert(
                ids=[skill.slug],
                embeddings=[vec],
                documents=[skill.to_prompt_block()],
                metadatas=[
                    {
                        "slug": skill.slug,
                        "topic": skill.topic,
                        "version": skill.version,
                        "confidence": skill.confidence,
                    }
                ],
            )
        except Exception as exc:
            logger.warning("Skill index upsert failed: %s", exc)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    slug        TEXT PRIMARY KEY,
                    topic       TEXT NOT NULL,
                    version     INTEGER DEFAULT 1,
                    confidence  REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    avg_feedback REAL DEFAULT 0.0,
                    updated_at  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS skill_feedback (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug      TEXT NOT NULL,
                    score     REAL NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS skill_interactions (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_slug     TEXT NOT NULL,
                    query          TEXT NOT NULL,
                    response       TEXT NOT NULL,
                    feedback_score REAL,
                    timestamp      TEXT NOT NULL
                );
            """
            )

    def _slugify(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_-]+", "-", text)
        return text[:60]

    def _merge_text(self, existing: str, new: str) -> str:
        existing_lines = set(existing.split("\n"))
        new_lines = [line for line in new.split("\n") if line not in existing_lines]
        return existing + "\n" + "\n".join(new_lines[:8])

    def _merge_examples(self, existing: list[dict], new: list[dict]) -> list[dict]:
        seen = {e["input"] for e in existing}
        unique_new = [e for e in new if e["input"] not in seen]
        return (existing + unique_new)[:8]
