"""AgentRegistry — stores, retrieves, and versions adaptive agent definitions."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.db import get_conn


@dataclass(slots=True)
class AgentTypeDefinition:
    agent_type_id: str
    display_name: str
    description: str
    base_system_prompt: str
    available_tools: list[str]
    adapter_path: str
    capability_version: int = 1
    user_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentRegistry:
    BUILTIN_TYPES = [
        AgentTypeDefinition(
            agent_type_id="data_analysis",
            display_name="Data Analysis Agent",
            description="Analyses structured and unstructured data, produces insights and visualisations",
            base_system_prompt="You are a data analysis specialist. You help users understand their data with rigorous analysis, clear assumptions, and actionable visualisation recommendations.",
            available_tools=["web_search", "code_analysis", "run_sandbox", "fetch_url"],
            adapter_path="models/adapters/data_analysis/latest",
            tags=["data", "analytics", "csv", "sql", "pandas"],
        ),
        AgentTypeDefinition(
            agent_type_id="job_hunting",
            display_name="Job Hunting Agent",
            description="Finds jobs, tailors resumes, prepares interview responses",
            base_system_prompt="You are a career specialist. You help users find and land their ideal job with targeted search strategy, resume tailoring, and interview practice.",
            available_tools=["web_search", "fetch_url"],
            adapter_path="models/adapters/job_hunting/latest",
            tags=["jobs", "resume", "career", "interview", "linkedin"],
        ),
        AgentTypeDefinition(
            agent_type_id="deep_research",
            display_name="Deep Research Agent",
            description="Performs multi-source, multi-step research on complex topics",
            base_system_prompt="You are a research specialist. You synthesise information from multiple sources, identify uncertainty, and cite concrete evidence.",
            available_tools=["web_search", "fetch_url", "fetch_cve", "query_mitre"],
            adapter_path="models/adapters/deep_research/latest",
            tags=["research", "analysis", "synthesis", "academic"],
        ),
        AgentTypeDefinition(
            agent_type_id="cybersecurity",
            display_name="Cybersecurity Agent",
            description="Analyses threats, CVEs, and security posture for organisations",
            base_system_prompt="You are a cybersecurity specialist. You identify threats, map ATT&CK techniques, assess risk, and recommend practical mitigations.",
            available_tools=["fetch_cve", "query_mitre", "check_ioc", "code_analysis", "run_sandbox"],
            adapter_path="models/adapters/cybersecurity/latest",
            tags=["security", "cve", "threat", "malware", "vulnerability"],
        ),
        AgentTypeDefinition(
            agent_type_id="code_review",
            display_name="Code Review Agent",
            description="Reviews code for bugs, security issues, and style",
            base_system_prompt="You are a code review specialist. You find bugs, security issues, and opportunities to improve maintainability while explaining fixes clearly.",
            available_tools=["code_analysis", "run_sandbox"],
            adapter_path="models/adapters/code_review/latest",
            tags=["code", "review", "bug", "security", "refactor"],
        ),
    ]

    def __init__(self, db_path: str = "data/agent_registry"):
        self.db_path = db_path
        self._init_db()
        self._seed_builtins()

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "AgentRegistry":
        db_path = "data/agent_registry"
        try:
            import yaml

            cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
            db_path = ((cfg or {}).get("adaptive_agents", {}) or {}).get("registry_db", db_path)
        except Exception:
            db_path = "data/agent_registry"
        return cls(db_path=db_path)

    def _init_db(self) -> None:
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_types (
                        agent_type_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        description TEXT,
                        base_system_prompt TEXT,
                        available_tools TEXT,
                        adapter_path TEXT,
                        capability_version INTEGER DEFAULT 1,
                        user_count INTEGER DEFAULT 0,
                        created_at REAL,
                        updated_at REAL,
                        tags TEXT
                    )
                    """
                )
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_agent_profiles (
                        profile_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        agent_type_id TEXT NOT NULL,
                        custom_system_prompt_addon TEXT DEFAULT '',
                        data_sources TEXT DEFAULT '[]',
                        preferences TEXT DEFAULT '{}',
                        interaction_count INTEGER DEFAULT 0,
                        capability_deltas TEXT DEFAULT '[]',
                        created_at REAL,
                        updated_at REAL,
                        FOREIGN KEY (agent_type_id) REFERENCES agent_types(agent_type_id),
                        UNIQUE(user_id, agent_type_id)
                    )
                    """
                )
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS capability_deltas (
                        delta_id TEXT PRIMARY KEY,
                        agent_type_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        delta_description TEXT,
                        training_example TEXT,
                        quality_score REAL,
                        merged INTEGER DEFAULT 0,
                        created_at REAL,
                        FOREIGN KEY (agent_type_id) REFERENCES agent_types(agent_type_id)
                    )
                    """
                )
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_instances (
                        agent_id TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        name TEXT NOT NULL,
                        agent_type TEXT NOT NULL,
                        use_count INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'ready',
                        created_at TEXT DEFAULT '',
                        last_used TEXT DEFAULT '',
                        tags TEXT DEFAULT '[]'
                    )
                    """
                )

    def _seed_builtins(self) -> None:
        for agent_def in self.BUILTIN_TYPES:
            if self.get(agent_def.agent_type_id) is None:
                self.register(agent_def)

    def register(self, agent_def: AgentTypeDefinition) -> None:
        now = time.time()
        if agent_def.created_at <= 0:
            agent_def.created_at = now
        agent_def.updated_at = now
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    """
                    INSERT INTO agent_types
                      (agent_type_id, display_name, description, base_system_prompt,
                       available_tools, adapter_path, capability_version, user_count,
                       created_at, updated_at, tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(agent_type_id) DO UPDATE SET
                      display_name=excluded.display_name,
                      description=excluded.description,
                      base_system_prompt=excluded.base_system_prompt,
                      available_tools=excluded.available_tools,
                      adapter_path=excluded.adapter_path,
                      capability_version=excluded.capability_version,
                      user_count=excluded.user_count,
                      created_at=excluded.created_at,
                      updated_at=excluded.updated_at,
                      tags=excluded.tags
                    """,
                    (
                        agent_def.agent_type_id,
                        agent_def.display_name,
                        agent_def.description,
                        agent_def.base_system_prompt,
                        json.dumps(agent_def.available_tools),
                        agent_def.adapter_path,
                        int(agent_def.capability_version),
                        int(agent_def.user_count),
                        float(agent_def.created_at),
                        float(agent_def.updated_at),
                        json.dumps(agent_def.tags),
                    ),
                )

    def get(self, agent_type_id: str) -> AgentTypeDefinition | None:
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute("SELECT * FROM agent_types WHERE agent_type_id = %s", (agent_type_id,))
                row = db.fetchone()
        return self._row_to_agent(row) if row else None

    def list_all(self) -> list[AgentTypeDefinition]:
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute("SELECT * FROM agent_types ORDER BY user_count DESC, display_name ASC")
                rows = db.fetchall()
        return [self._row_to_agent(row) for row in rows]

    def search(self, query: str) -> list[AgentTypeDefinition]:
        query_lower = (query or "").strip().lower()
        if not query_lower:
            return self.list_all()
        scored: list[tuple[int, AgentTypeDefinition]] = []
        for agent in self.list_all():
            haystack = " ".join([agent.agent_type_id, agent.display_name, agent.description, *agent.tags]).lower()
            score = haystack.count(query_lower)
            token_hits = sum(1 for token in query_lower.split() if token in haystack)
            if score or token_hits:
                scored.append((score + token_hits, agent))
        scored.sort(key=lambda item: (item[0], item[1].user_count), reverse=True)
        return [agent for _, agent in scored]

    def auto_match(self, user_description: str) -> AgentTypeDefinition | None:
        candidates = self.search(user_description)
        if candidates:
            return candidates[0]
        words = {word for word in user_description.lower().split() if len(word) > 3}
        best: tuple[int, AgentTypeDefinition] | None = None
        for agent in self.list_all():
            overlap = len(words & set(agent.tags))
            if best is None or overlap > best[0]:
                best = (overlap, agent)
        return best[1] if best and best[0] > 0 else None

    def increment_user_count(self, agent_type_id: str) -> None:
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    "UPDATE agent_types SET user_count = user_count + 1, updated_at = %s WHERE agent_type_id = %s",
                    (time.time(), agent_type_id),
                )

    def get_or_create_user_profile(self, user_id: str, agent_type_id: str) -> dict[str, Any]:
        if self.get(agent_type_id) is None:
            raise ValueError(f"Unknown agent type: {agent_type_id}")
        now = time.time()
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    "SELECT * FROM user_agent_profiles WHERE user_id=%s AND agent_type_id=%s",
                    (user_id, agent_type_id),
                )
                row = db.fetchone()
                if row:
                    return self._profile_row_to_dict(row)
                profile_id = str(uuid.uuid4())
                db.execute(
                    """
                    INSERT INTO user_agent_profiles VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (profile_id, user_id, agent_type_id, "", "[]", "{}", 0, "[]", now, now),
                )
                db.execute(
                    "UPDATE agent_types SET user_count = user_count + 1, updated_at = %s WHERE agent_type_id = %s",
                    (now, agent_type_id),
                )
        return {
            "profile_id": profile_id,
            "user_id": user_id,
            "agent_type_id": agent_type_id,
            "custom_system_prompt_addon": "",
            "data_sources": [],
            "preferences": {},
            "interaction_count": 0,
            "capability_deltas": [],
        }

    def update_user_profile(self, user_id: str, agent_type_id: str, **updates: Any) -> dict[str, Any]:
        self.get_or_create_user_profile(user_id, agent_type_id)
        allowed = {"custom_system_prompt_addon", "data_sources", "preferences", "capability_deltas"}
        fields = {k: v for k, v in updates.items() if k in allowed}
        if fields:
            assignments = []
            values: list[Any] = []
            for key, value in fields.items():
                assignments.append(f"{key}=%s")
                values.append(json.dumps(value) if isinstance(value, (dict, list)) else value)
            assignments.append("updated_at=%s")
            values.extend([time.time(), user_id, agent_type_id])
            with get_conn() as conn:
                with conn.cursor() as db:
                    db.execute(
                        f"UPDATE user_agent_profiles SET {', '.join(assignments)} WHERE user_id=%s AND agent_type_id=%s",
                        tuple(values),
                    )
        return self.get_or_create_user_profile(user_id, agent_type_id)

    def submit_capability_delta(
        self,
        user_id: str,
        agent_type_id: str,
        delta_description: str,
        training_example: dict[str, Any],
        quality_score: float,
    ) -> str:
        if self.get(agent_type_id) is None:
            raise ValueError(f"Unknown agent type: {agent_type_id}")
        delta_id = str(uuid.uuid4())
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    """
                    INSERT INTO capability_deltas VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        delta_id,
                        agent_type_id,
                        user_id,
                        delta_description,
                        json.dumps(training_example, ensure_ascii=False),
                        float(quality_score),
                        0,
                        time.time(),
                    ),
                )
                db.execute(
                    "SELECT capability_deltas FROM user_agent_profiles WHERE user_id=%s AND agent_type_id=%s",
                    (user_id, agent_type_id),
                )
                row = db.fetchone()
                if row:
                    deltas = self._json_load(row[0], [])
                    deltas.append(delta_description)
                    db.execute(
                        "UPDATE user_agent_profiles SET capability_deltas=%s, updated_at=%s WHERE user_id=%s AND agent_type_id=%s",
                        (json.dumps(deltas), time.time(), user_id, agent_type_id),
                    )
        return delta_id

    def get_unmerged_deltas(self, agent_type_id: str, min_quality: float = 0.6) -> list[dict[str, Any]]:
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    """
                    SELECT delta_id, user_id, delta_description, training_example, quality_score, created_at
                    FROM capability_deltas
                    WHERE agent_type_id=%s AND merged=0 AND quality_score >= %s
                    ORDER BY quality_score DESC, created_at ASC
                    """,
                    (agent_type_id, float(min_quality)),
                )
                rows = db.fetchall()
        return [
            {
                "delta_id": row[0],
                "user_id": row[1],
                "description": row[2],
                "training_example": self._json_load(row[3], {}),
                "quality_score": float(row[4]),
                "created_at": float(row[5]),
            }
            for row in rows
        ]

    def mark_deltas_merged(self, delta_ids: list[str]) -> None:
        if not delta_ids:
            return
        with get_conn() as conn:
            with conn.cursor() as db:
                db.executemany("UPDATE capability_deltas SET merged=1 WHERE delta_id=%s", [(delta_id,) for delta_id in delta_ids])

    def stats(self, agent_type_id: str) -> dict[str, Any]:
        agent = self.get(agent_type_id)
        if agent is None:
            raise ValueError(f"Unknown agent type: {agent_type_id}")
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute("SELECT COUNT(*) FROM capability_deltas WHERE agent_type_id=%s", (agent_type_id,))
                delta_count = db.fetchone()[0]
                db.execute(
                    "SELECT COUNT(*) FROM capability_deltas WHERE agent_type_id=%s AND merged=0",
                    (agent_type_id,),
                )
                unmerged_count = db.fetchone()[0]
                db.execute("SELECT COUNT(*) FROM user_agent_profiles WHERE agent_type_id=%s", (agent_type_id,))
                profile_count = db.fetchone()[0]
        return {
            "agent_type_id": agent_type_id,
            "capability_version": agent.capability_version,
            "user_count": agent.user_count,
            "profile_count": int(profile_count),
            "delta_count": int(delta_count),
            "unmerged_delta_count": int(unmerged_count),
            "adapter_path": agent.adapter_path,
            "updated_at": agent.updated_at,
        }

    def update_capability_version(self, agent_type_id: str, adapter_path: str | None = None) -> int:
        now = time.time()
        with get_conn() as conn:
            with conn.cursor() as db:
                if adapter_path:
                    db.execute(
                        "UPDATE agent_types SET capability_version=capability_version+1, adapter_path=%s, updated_at=%s WHERE agent_type_id=%s",
                        (adapter_path, now, agent_type_id),
                    )
                else:
                    db.execute(
                        "UPDATE agent_types SET capability_version=capability_version+1, updated_at=%s WHERE agent_type_id=%s",
                        (now, agent_type_id),
                    )
        agent = self.get(agent_type_id)
        return int(agent.capability_version if agent else 0)

    def save(self, agent: Any) -> None:
        payload = agent.to_dict() if hasattr(agent, "to_dict") else asdict(agent)
        agent_type = payload.get("agent_type", "")
        if hasattr(agent_type, "value"):
            agent_type = agent_type.value
        payload["agent_type"] = agent_type
        tags = payload.get("tags", [])
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    """
                    INSERT INTO agent_instances
                      (agent_id, payload, name, agent_type, use_count, status,
                       created_at, last_used, tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(agent_id) DO UPDATE SET
                      payload=excluded.payload,
                      name=excluded.name,
                      agent_type=excluded.agent_type,
                      use_count=excluded.use_count,
                      status=excluded.status,
                      created_at=excluded.created_at,
                      last_used=excluded.last_used,
                      tags=excluded.tags
                    """,
                    (
                        payload["agent_id"],
                        json.dumps(payload),
                        payload.get("name", ""),
                        agent_type,
                        int(payload.get("use_count", 0)),
                        payload.get("status", "ready"),
                        payload.get("created_at", ""),
                        payload.get("last_used", ""),
                        json.dumps(tags),
                    ),
                )

    def load(self, agent_id: str) -> Any | None:
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute("SELECT payload FROM agent_instances WHERE agent_id=%s", (agent_id,))
                row = db.fetchone()
        if not row:
            return None
        try:
            from app.core.agent_forge import AgentDefinition, AgentType

            data = json.loads(row[0])
            data["agent_type"] = AgentType(data["agent_type"])
            return AgentDefinition(**data)
        except Exception:
            return None

    def find_similar(self, request: str, threshold: float = 0.5) -> tuple[Any | None, float]:
        from app.core.agent_forge import AgentForge

        forge = AgentForge()
        request_tags = set(forge._extract_tags(request))
        forge_type = forge.classify_request(request).value
        best_agent = None
        best_score = 0.0

        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute("SELECT agent_id, agent_type, tags FROM agent_instances")
                rows = db.fetchall()

        for agent_id, agent_type, tags_json in rows:
            try:
                agent_tags = set(self._json_load(tags_json, []))
                if agent_type != forge_type or not request_tags or not agent_tags:
                    continue
                overlap = len(request_tags & agent_tags)
                score = overlap / max(len(request_tags), 1)
                if score > best_score:
                    best_score = score
                    best_agent = self.load(agent_id)
            except Exception:
                continue

        if best_score >= threshold and best_agent:
            return best_agent, best_score
        return None, 0.0

    def list_agents(self) -> list[dict[str, Any]]:
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    """
                    SELECT agent_id, name, agent_type, use_count, status, created_at, tags
                    FROM agent_instances
                    ORDER BY use_count DESC
                    """
                )
                rows = db.fetchall()
        return [
            {
                "agent_id": row[0],
                "name": row[1],
                "agent_type": row[2],
                "use_count": int(row[3] or 0),
                "status": row[4] or "ready",
                "created_at": row[5] or "",
                "tags": self._json_load(row[6], [])[:5],
            }
            for row in rows
        ]

    def set_status(self, agent_id: str, status: str) -> None:
        agent = self.load(agent_id)
        if agent:
            from datetime import datetime, timezone

            agent.status = status
            if status == "resting":
                agent.last_used = datetime.now(timezone.utc).isoformat()
            self.save(agent)

    def increment_use(self, agent_id: str) -> None:
        agent = self.load(agent_id)
        if agent:
            agent.use_count += 1
            self.save(agent)

    def delete(self, agent_id: str) -> bool:
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute("DELETE FROM agent_instances WHERE agent_id=%s", (agent_id,))
                return bool(db.rowcount)

    @staticmethod
    def _json_load(value: str | None, default: Any) -> Any:
        if value is None:
            return default
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default

    def _row_to_agent(self, row: tuple[Any, ...]) -> AgentTypeDefinition:
        return AgentTypeDefinition(
            agent_type_id=row[0],
            display_name=row[1],
            description=row[2] or "",
            base_system_prompt=row[3] or "",
            available_tools=self._json_load(row[4], []),
            adapter_path=row[5] or "",
            capability_version=int(row[6] or 1),
            user_count=int(row[7] or 0),
            created_at=float(row[8] or time.time()),
            updated_at=float(row[9] or time.time()),
            tags=self._json_load(row[10], []),
        )

    def _profile_row_to_dict(self, row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "profile_id": row[0],
            "user_id": row[1],
            "agent_type_id": row[2],
            "custom_system_prompt_addon": row[3] or "",
            "data_sources": self._json_load(row[4], []),
            "preferences": self._json_load(row[5], {}),
            "interaction_count": int(row[6] or 0),
            "capability_deltas": self._json_load(row[7], []),
            "created_at": float(row[8] or 0.0),
            "updated_at": float(row[9] or 0.0),
        }
