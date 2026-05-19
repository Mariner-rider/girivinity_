from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentRegistry:
    def __init__(self) -> None:
        try:
            import yaml

            cfg = yaml.safe_load(Path("config.yaml").read_text())
            self.agents_dir = Path(cfg.get("agent_mode", {}).get("agents_dir", "data/agents"))
        except Exception:
            self.agents_dir = Path("data/agents")
        self.agents_dir.mkdir(parents=True, exist_ok=True)

    def save(self, agent) -> None:
        path = self.agents_dir / f"{agent.agent_id}.json"
        path.write_text(json.dumps(asdict(agent), indent=2), encoding="utf-8")
        logger.info("Agent saved: %s (v%d)", agent.name, agent.version)

    def load(self, agent_id: str):
        path = self.agents_dir / f"{agent_id}.json"
        if not path.exists():
            return None
        try:
            from app.core.agent_forge import AgentDefinition, AgentType

            data = json.loads(path.read_text(encoding="utf-8"))
            data["agent_type"] = AgentType(data["agent_type"])
            return AgentDefinition(**data)
        except Exception as exc:
            logger.warning("Agent load failed %s: %s", agent_id, exc)
            return None

    def find_similar(self, request: str, threshold: float = 0.5):
        from app.core.agent_forge import AgentForge

        request_tags = set(AgentForge()._extract_tags(request))
        best_agent = None
        best_score = 0.0

        for path in self.agents_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                agent_tags = set(data.get("tags", []))
                agent_type = data.get("agent_type", "")
                forge_type = AgentForge().classify_request(request).value
                if agent_type != forge_type or not request_tags or not agent_tags:
                    continue
                overlap = len(request_tags & agent_tags)
                score = overlap / max(len(request_tags), 1)
                if score > best_score:
                    best_score = score
                    best_agent = self.load(data["agent_id"])
            except Exception:
                continue

        if best_score >= threshold and best_agent:
            logger.info("Found similar agent: '%s' (score=%.2f)", best_agent.name, best_score)
            return best_agent, best_score
        return None, 0.0

    def list_agents(self) -> list[dict]:
        agents = []
        for path in self.agents_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                agents.append({"agent_id": data["agent_id"], "name": data["name"], "agent_type": data["agent_type"], "use_count": data.get("use_count", 0), "status": data.get("status", "ready"), "created_at": data.get("created_at", ""), "tags": data.get("tags", [])[:5]})
            except Exception:
                continue
        return sorted(agents, key=lambda x: x["use_count"], reverse=True)

    def set_status(self, agent_id: str, status: str) -> None:
        agent = self.load(agent_id)
        if agent:
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
        path = self.agents_dir / f"{agent_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
