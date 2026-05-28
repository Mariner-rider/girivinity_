from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    RESEARCH = "research"
    CODE = "code"
    DATA = "data_analysis"
    MONITORING = "monitoring"
    WRITING = "writing"
    LEGAL = "legal_research"
    FINANCIAL = "financial"
    TEACHING = "teaching"
    SECURITY = "security"
    CUSTOM = "custom"


@dataclass
class AgentDefinition:
    agent_id: str
    name: str
    description: str
    agent_type: AgentType
    created_for: str
    tools: list[str]
    steps: list[dict]
    output_format: str
    version: int = 1
    use_count: int = 0
    created_at: str = ""
    last_used: str = ""
    status: str = "ready"
    tags: list[str] = field(default_factory=list)


AGENT_TYPE_PATTERNS = {
    AgentType.RESEARCH: ["research", "find information", "gather data", "investigate", "explore", "study", "analyse topic", "deep dive", "comprehensive report"],
    AgentType.CODE: ["write code", "build", "create function", "implement", "develop", "program", "script", "fix bug", "debug", "create app", "build tool"],
    AgentType.DATA: ["analyse data", "data analysis", "statistics", "visualise", "chart", "csv", "dataset", "trends", "correlation", "regression"],
    AgentType.MONITORING: ["monitor", "track", "watch", "alert", "notify", "check regularly", "daily update", "price tracking", "news monitoring"],
    AgentType.WRITING: ["write article", "create content", "draft", "compose", "write report", "documentation", "blog post", "newsletter", "pitch deck"],
    AgentType.LEGAL: ["legal research", "find case law", "section", "judgment", "legal opinion", "contract review", "legal analysis", "law research"],
    AgentType.FINANCIAL: ["financial analysis", "stock", "investment", "valuation", "balance sheet", "profit loss", "audit", "tax", "financial model"],
    AgentType.TEACHING: ["teach", "create curriculum", "make lesson", "design course", "explain step by step", "create exercises", "training material"],
    AgentType.SECURITY: ["security audit", "vulnerability", "penetration", "threat analysis", "security review", "scan"],
}

AGENT_TOOL_MAP = {
    AgentType.RESEARCH: ["WebIntelligence", "QueryRouter", "CitationEngine", "SkillForge"],
    AgentType.CODE: ["CUDAEngine", "QueryRouter", "SkillForge"],
    AgentType.DATA: ["WebIntelligence", "QueryRouter", "CitationEngine"],
    AgentType.MONITORING: ["WebIntelligence", "QueryRouter"],
    AgentType.WRITING: ["WebIntelligence", "QueryRouter", "CitationEngine", "SkillForge", "TeachingEngine"],
    AgentType.LEGAL: ["QueryRouter", "WebIntelligence", "CitationEngine", "DomainRouter"],
    AgentType.FINANCIAL: ["WebIntelligence", "QueryRouter", "CitationEngine"],
    AgentType.TEACHING: ["QueryRouter", "TeachingEngine", "SkillForge", "MemoryEngine"],
    AgentType.SECURITY: ["ThreatDetector", "QueryRouter", "WebIntelligence"],
    AgentType.CUSTOM: ["QueryRouter", "WebIntelligence"],
}


class AgentForge:
    def classify_request(self, request: str) -> AgentType:
        req_lower = request.lower()
        scores: dict[AgentType, int] = {}
        for atype, patterns in AGENT_TYPE_PATTERNS.items():
            scores[atype] = sum(1 for p in patterns if p in req_lower)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else AgentType.CUSTOM

    def forge(self, request: str, user_id: str) -> AgentDefinition:
        agent_type = self.classify_request(request)
        agent_id = self._generate_id(request, user_id)
        return AgentDefinition(
            agent_id=agent_id,
            name=self._generate_name(request, agent_type),
            description=request[:200],
            agent_type=agent_type,
            created_for=request,
            tools=AGENT_TOOL_MAP.get(agent_type, ["QueryRouter"]),
            steps=self._build_steps(request, agent_type),
            output_format=self._detect_output_format(request),
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=self._extract_tags(request),
            status="ready",
        )

    def adapt(self, existing: AgentDefinition, new_request: str) -> AgentDefinition:
        adapted = AgentDefinition(**existing.__dict__.copy())
        adapted.version += 1
        adapted.description = new_request[:200]
        adapted.created_for = new_request
        adapted.steps = self._build_steps(new_request, existing.agent_type)
        adapted.last_used = datetime.now(timezone.utc).isoformat()
        adapted.tags = self._extract_tags(new_request)
        logger.info("AgentForge: adapted '%s' to v%d", adapted.name, adapted.version)
        return adapted

    def _build_steps(self, request: str, agent_type: AgentType) -> list[dict]:
        base_steps: list[dict] = [{"step": 1, "action": "understand_request", "description": f"Analyse and clarify: {request[:100]}", "tool": "QueryRouter"}]
        type_steps = {
            AgentType.RESEARCH: [{"step": 2, "action": "search_knowledge_base", "description": "Check existing knowledge", "tool": "QueryRouter"}, {"step": 3, "action": "web_search", "description": "Search web for latest information", "tool": "WebIntelligence"}, {"step": 4, "action": "generate_citations", "description": "Generate academic citations", "tool": "CitationEngine"}, {"step": 5, "action": "compile_report", "description": "Compile structured research report", "tool": "LLMSynthesiser"}],
            AgentType.CODE: [{"step": 2, "action": "search_patterns", "description": "Find relevant code patterns", "tool": "QueryRouter"}, {"step": 3, "action": "generate_code", "description": "Generate implementation", "tool": "LLMSynthesiser"}, {"step": 4, "action": "review_code", "description": "Review for correctness", "tool": "LLMSynthesiser"}],
            AgentType.LEGAL: [{"step": 2, "action": "search_provisions", "description": "Search BNS/IPC/relevant acts", "tool": "QueryRouter"}, {"step": 3, "action": "find_case_law", "description": "Find relevant judgments", "tool": "WebIntelligence"}, {"step": 4, "action": "legal_analysis", "description": "Provide legal analysis", "tool": "LLMSynthesiser"}, {"step": 5, "action": "cite_sources", "description": "Add legal citations", "tool": "CitationEngine"}],
            AgentType.FINANCIAL: [{"step": 2, "action": "gather_financials", "description": "Gather financial data", "tool": "WebIntelligence"}, {"step": 3, "action": "analyse", "description": "Financial analysis", "tool": "LLMSynthesiser"}],
        }
        return base_steps + type_steps.get(agent_type, [{"step": 2, "action": "process", "description": "Process request", "tool": "LLMSynthesiser"}])

    def _generate_name(self, request: str, agent_type: AgentType) -> str:
        words = [w.title() for w in request.split()[:3] if len(w) > 3 and w.isalpha()]
        prefix = agent_type.value.replace("_", " ").title()
        suffix = " ".join(words) if words else "Agent"
        return f"{prefix}: {suffix}"

    def _generate_id(self, request: str, user_id: str) -> str:
        import hashlib

        raw = f"{user_id}:{request[:100]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _detect_output_format(self, request: str) -> str:
        req = request.lower()
        if any(w in req for w in ["code", "script", "function"]):
            return "code"
        if any(w in req for w in ["json", "data", "structured"]):
            return "json"
        if any(w in req for w in ["report", "analysis", "research", "comprehensive"]):
            return "report"
        return "text"

    def _extract_tags(self, request: str) -> list[str]:
        stop_words = {"a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "by", "with", "from"}
        words = [w.lower() for w in re.findall(r"\b\w{4,}\b", request) if w.lower() not in stop_words]
        return list(dict.fromkeys(words))[:10]
