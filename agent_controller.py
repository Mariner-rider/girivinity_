"""Multi-agent orchestration with LLM-backed agents, cognition, and shared memory."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.agents.base_agent import BaseAgent
from app.cognition.causal_reasoner import CausalReasoningEngine
from app.cognition.sentiment_engine import SentimentEngine
from app.cognition.theory_of_mind import TheoryOfMindEngine
from app.reasoning_planner import ReasoningPlanner
from app.security.policy import SecurityGuard, secure_operation


@dataclass(slots=True)
class AgentResult:
    agent_name: str
    output: str
    confidence: float
    citations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SharedMemory:
    facts: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def add_fact(self, fact: str) -> None:
        if fact and fact not in self.facts:
            self.facts.append(fact)


@dataclass(slots=True)
class AgentMessage:
    sender: str
    recipient: str
    content: str


class Agent(Protocol):
    name: str

    def run(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        hidden_scratchpad: list[str],
        user_id: str | None = None,
        sentiment: dict[str, Any] | None = None,
    ) -> AgentResult:
        ...


class LocalLLMEngine:
    """Small deterministic fallback LLM for tests and offline controller use."""

    def generate(self, prompt: str, max_tokens: int = 512, stream: bool = False) -> str:
        lower = prompt.lower()
        if "cybersecurity intelligence agent" in lower:
            return json.dumps(
                {
                    "threat_assessment": "No confirmed active threat identified from local context.",
                    "severity": "informational",
                    "affected_systems": [],
                    "attack_vector": "unknown",
                    "cve_ids": [],
                    "attack_techniques": [],
                    "mitigations": ["Validate findings against authoritative sources."],
                    "ioc_matches": [],
                    "confidence": 0.55,
                }
            )
        if "critical review agent" in lower:
            return json.dumps(
                {
                    "critique": "Validate assumptions and cite stronger evidence before finalising.",
                    "severity": "medium",
                    "specific_flaws": ["Evidence may be incomplete."],
                    "corrections": ["Cross-check retrieved sources."],
                    "confidence": 0.72,
                }
            )
        if "knowledge consolidation agent" in lower:
            return json.dumps(
                {
                    "summary": "Consolidated findings and next actions from agent outputs.",
                    "key_facts": ["Shared memory was updated."],
                    "action_items": ["Review citations."],
                    "knowledge_type": "conceptual",
                    "confidence": 0.8,
                }
            )
        if "logical reasoning agent" in lower:
            return json.dumps(
                {
                    "analysis": "Structured analysis produced from research findings and context.",
                    "causal_chains": [],
                    "assumptions": ["Retrieved context is relevant."],
                    "plan": ["Review facts", "Apply reasoning", "Produce answer"],
                    "confidence": 0.76,
                }
            )
        return json.dumps(
            {
                "findings": ["Relevant context gathered for the task."],
                "key_facts": ["Context is available for downstream reasoning."],
                "confidence": 0.74,
                "uncertainty": "Source coverage may be incomplete.",
                "sources": [],
            }
        )

    def get_token_entropy(self, prompt: str) -> float:
        return min(2.0, max(0.1, len(prompt.split()) / 500))


class QueryRouterRAGEngine:
    def query(self, task: str, top_k: int = 6) -> list[dict[str, Any]]:
        try:
            from app.core.query_router import QueryRouter

            result = QueryRouter().route(task)
            text = result.get("context_string") or f"No data found for: {task}"
            urls = result.get("urls", []) or []
            confidence = float(result.get("confidence", 0.5))
            if urls:
                return [
                    {"text": text, "source": url, "score": confidence}
                    for url in urls[:top_k]
                ]
            return [{"text": text, "source": "internal:query_router", "score": confidence}]
        except (RuntimeError, ValueError, ImportError, KeyError, TypeError, AttributeError) as exc:
            return [{"text": f"Research error for '{task}': {exc}", "source": "internal:error", "score": 0.1}]

    def add(self, text: str, source: str, metadata: dict[str, Any] | None = None) -> None:
        return None


class SecurityIntelligenceEngine:
    def search_cve(self, task: str) -> list[dict[str, Any]]:
        return []

    def search_attack_techniques(self, task: str) -> list[dict[str, Any]]:
        return []


class ResearchAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "research_agent"

    def system_prompt(self, user_model: Any | None = None) -> str:
        return (
            "You are a research agent specialising in fact collection. Given a task and context, "
            "extract the most relevant facts. OUTPUT JSON ONLY: {\"findings\": [\"...\", ...], "
            "\"key_facts\": [\"...\"], \"confidence\": 0.0–1.0, \"uncertainty\": \"...\", "
            "\"sources\": [\"...\"]}"
        )

    def build_prompt(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        rag_results: list[dict[str, Any]],
        user_model: Any | None,
        episodes: list[Any],
    ) -> str:
        return "\n\n".join(
            [
                self.system_prompt(user_model),
                self._format_user_context(user_model),
                self._format_episodes(episodes),
                self._format_rag_context(rag_results),
                f"## Task\n{task}",
            ]
        )

    def parse_response(self, raw: str) -> dict[str, Any]:
        parsed = self._safe_json_parse(raw)
        findings = self._as_list(parsed.get("findings", parsed.get("key_facts", [])))
        parsed["findings"] = findings
        parsed.setdefault("answer", "\n".join(findings) if findings else raw[:500])
        return parsed

    def run(self, *args: Any, **kwargs: Any) -> AgentResult:
        result = super().run(*args, **kwargs)
        memory = args[1]
        if isinstance(memory, SharedMemory):
            memory.add_fact(result.output)
            memory.notes["research_findings"] = result.output
        return result


class ReasoningAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "reasoning_agent"

    def __init__(self, *args: Any, causal_reasoner: CausalReasoningEngine | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.causal = causal_reasoner or CausalReasoningEngine({"chain_of_thought_steps": 5}, self.llm)

    def system_prompt(self, user_model: Any | None = None) -> str:
        return (
            "You are a logical reasoning agent. Given research findings, produce a structured analysis. "
            "Consider causal relationships, not just correlations. OUTPUT JSON ONLY: {\"analysis\": \"...\", "
            "\"causal_chains\": [{\"cause\": \"...\", \"effect\": \"...\", \"confidence\": 0.0}], "
            "\"assumptions\": [\"...\"], \"plan\": [\"step 1\", ...], \"confidence\": 0.0}"
        )

    def build_prompt(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        rag_results: list[dict[str, Any]],
        user_model: Any | None,
        episodes: list[Any],
    ) -> str:
        context = "\n".join(memory.facts + [str(item) for item in memory.notes.values()])
        user_addon = self._format_user_context(user_model)
        if self.causal.complexity_score(task) > 0.5:
            return self.system_prompt(user_model) + "\n\n" + self.causal.build_reasoning_prompt(task, context, user_addon)
        return "\n\n".join([self.system_prompt(user_model), user_addon, self._format_rag_context(rag_results), context, task])

    def parse_response(self, raw: str) -> dict[str, Any]:
        parsed = self._safe_json_parse(raw)
        parsed.setdefault("analysis", parsed.get("raw", raw[:500]))
        return parsed

    def run(self, *args: Any, **kwargs: Any) -> AgentResult:
        result = super().run(*args, **kwargs)
        memory = args[1]
        mailbox = args[2]
        if isinstance(memory, SharedMemory):
            memory.notes["draft_plan"] = result.output
        mailbox.append(AgentMessage(self.name, "critic_agent", result.output))
        return result


class CriticAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "critic_agent"

    def __init__(self, *args: Any, causal_reasoner: CausalReasoningEngine | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.causal = causal_reasoner or CausalReasoningEngine({"chain_of_thought_steps": 5}, self.llm)

    def system_prompt(self, user_model: Any | None = None) -> str:
        return (
            "You are a critical review agent. Identify flaws, unsupported claims, and missing evidence. "
            "Be specific, not vague. OUTPUT JSON ONLY: {\"critique\": \"...\", \"severity\": "
            "\"low|medium|high\", \"specific_flaws\": [\"...\"], \"corrections\": [\"...\"], "
            "\"confidence\": 0.0}"
        )

    def build_prompt(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        rag_results: list[dict[str, Any]],
        user_model: Any | None,
        episodes: list[Any],
    ) -> str:
        return "\n\n".join(
            [
                self.system_prompt(user_model),
                self._format_user_context(user_model),
                f"## Research Findings\n{memory.notes.get('research_findings', '')}",
                f"## Draft Plan\n{memory.notes.get('draft_plan', '')}",
                f"## Task\n{task}",
            ]
        )

    def parse_response(self, raw: str) -> dict[str, Any]:
        parsed = self._safe_json_parse(raw)
        parsed.setdefault("analysis", parsed.get("critique", parsed.get("raw", raw[:500])))
        return parsed

    def run(self, *args: Any, **kwargs: Any) -> AgentResult:
        result = super().run(*args, **kwargs)
        memory = args[1]
        check = self.causal.counterfactual_check(memory.notes.get("draft_plan", ""), result.output)
        if isinstance(memory, SharedMemory):
            memory.notes["critique"] = result.output
            memory.notes["counterfactual_check"] = check
        return result


class MemoryAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "memory_agent"

    def system_prompt(self, user_model: Any | None = None) -> str:
        return (
            "You are a knowledge consolidation agent. Compress all findings into a permanent knowledge update. "
            "OUTPUT JSON ONLY: {\"summary\": \"...\", \"key_facts\": [\"...\"], \"action_items\": [\"...\"], "
            "\"knowledge_type\": \"factual|procedural|conceptual\"}"
        )

    def build_prompt(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        rag_results: list[dict[str, Any]],
        user_model: Any | None,
        episodes: list[Any],
    ) -> str:
        inbound = [m.content for m in mailbox if m.recipient in {self.name, "next"}]
        return "\n\n".join(
            [
                self.system_prompt(user_model),
                self._format_user_context(user_model),
                f"## Shared Facts\n{memory.facts}",
                f"## Notes\n{memory.notes}",
                f"## Mailbox\n{inbound}",
                f"## Task\n{task}",
            ]
        )

    def parse_response(self, raw: str) -> dict[str, Any]:
        parsed = self._safe_json_parse(raw)
        parsed.setdefault("summary", parsed.get("raw", raw[:500]))
        return parsed

    def run(self, *args: Any, **kwargs: Any) -> AgentResult:
        result = super().run(*args, **kwargs)
        memory = args[1]
        parsed = self.parse_response(result.output)
        summary = parsed.get("summary", result.output)
        knowledge_type = parsed.get("knowledge_type", "conceptual")
        if self.rag and hasattr(self.rag, "add"):
            self.rag.add(summary, source="agent_memory", metadata={"type": knowledge_type})
        if isinstance(memory, SharedMemory):
            memory.notes["memory_summary"] = summary
        return result


class CybersecurityAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "cybersecurity_agent"

    def __init__(self, *args: Any, security_engine: SecurityIntelligenceEngine | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.security_engine = security_engine or SecurityIntelligenceEngine()

    def system_prompt(self, user_model: Any | None = None) -> str:
        return (
            "You are a cybersecurity intelligence agent. Analyse threats, CVEs, IOCs, and attack techniques. "
            "Always cite specific CVE IDs and ATT&CK technique IDs when known. OUTPUT JSON ONLY: "
            "{\"threat_assessment\": \"...\", \"severity\": \"critical|high|medium|low|informational\", "
            "\"affected_systems\": [\"...\"], \"attack_vector\": \"...\", \"cve_ids\": [\"CVE-...\"], "
            "\"attack_techniques\": [\"T...\"], \"mitigations\": [\"...\"], \"ioc_matches\": [], "
            "\"confidence\": 0.0}"
        )

    def build_prompt(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        rag_results: list[dict[str, Any]],
        user_model: Any | None,
        episodes: list[Any],
    ) -> str:
        cves = self.security_engine.search_cve(task)
        techniques = self.security_engine.search_attack_techniques(task)
        return "\n\n".join(
            [
                self.system_prompt(user_model),
                self._format_user_context(user_model),
                self._format_rag_context(rag_results),
                f"## CVE Results\n{json.dumps(cves)}",
                f"## ATT&CK Techniques\n{json.dumps(techniques)}",
                f"## Task\n{task}",
            ]
        )

    def parse_response(self, raw: str) -> dict[str, Any]:
        parsed = self._safe_json_parse(raw)
        parsed.setdefault("analysis", parsed.get("threat_assessment", parsed.get("raw", raw[:500])))
        return parsed


class SentimentAwareAgent:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.name = agent.name

    def run(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        hidden_scratchpad: list[str],
        user_id: str | None = None,
        sentiment: dict[str, Any] | None = None,
    ) -> AgentResult:
        result = self.agent.run(task, memory, mailbox, hidden_scratchpad, user_id=user_id, sentiment=sentiment)
        sentiment = sentiment or {}
        if sentiment.get("crisis_signal"):
            result.output = (
                "I’m sorry you’re carrying this. You are not alone, and it may help to pause, breathe, "
                "and reach out to someone you trust right now.\n\n"
                f"{result.output}\n\nIf you might hurt yourself or feel in immediate danger, contact local emergency services or a crisis hotline now."
            )
        elif sentiment.get("requires_empathy"):
            result.output = f"I understand this may feel difficult. I’ll keep this clear and supportive.\n\n{result.output}"
        elif sentiment.get("dominant_emotion") == "urgency":
            result.output = result.output.strip()
        return result


class AgentController:
    SECURITY_KEYWORDS = ("cve", "exploit", "malware", "ioc", "threat", "attack", "vulnerability", "ransomware")
    EMOTIONAL_KEYWORDS = ("sad", "angry", "frustrated", "worried", "scared", "urgent", "confused", "suicide")

    def __init__(
        self,
        security_guard: SecurityGuard | None = None,
        llm_engine: Any | None = None,
        rag_engine: Any | None = None,
        sentiment_engine: SentimentEngine | None = None,
        theory_of_mind: TheoryOfMindEngine | None = None,
        episodic_memory: Any | None = None,
        security_engine: SecurityIntelligenceEngine | None = None,
    ) -> None:
        self.security_guard = security_guard or SecurityGuard()
        self.memory = SharedMemory()
        self.reasoning_planner = ReasoningPlanner()
        self.llm = llm_engine or LocalLLMEngine()
        self.rag = rag_engine or QueryRouterRAGEngine()
        self.sentiment_engine = sentiment_engine or SentimentEngine({})
        self.tom = theory_of_mind or TheoryOfMindEngine({"user_model_window": 20})
        self.episodic = episodic_memory
        self.security_engine = security_engine or SecurityIntelligenceEngine()
        self.causal_reasoner = CausalReasoningEngine({"chain_of_thought_steps": 5}, self.llm)
        self.research_agent = ResearchAgent(self.llm, self.rag, self.tom, self.episodic)
        self.reasoning_agent = ReasoningAgent(self.llm, self.rag, self.tom, self.episodic, causal_reasoner=self.causal_reasoner)
        self.critic_agent = CriticAgent(self.llm, self.rag, self.tom, self.episodic, causal_reasoner=self.causal_reasoner)
        self.memory_agent = MemoryAgent(self.llm, self.rag, self.tom, self.episodic)
        self.cybersecurity_agent = CybersecurityAgent(
            self.llm,
            self.rag,
            self.tom,
            self.episodic,
            security_engine=self.security_engine,
        )

    @secure_operation("agents.route_task")
    def route_task(self, task: str) -> list[Agent]:
        self.security_guard.validate_prompt(task)
        lowered = task.lower()
        route: list[Agent] = [self.research_agent, self.reasoning_agent, self.critic_agent, self.memory_agent]

        if "quick" in lowered:
            route = [self.reasoning_agent, self.critic_agent, self.memory_agent]
        elif "research" in lowered:
            route = [self.research_agent, self.reasoning_agent, self.critic_agent, self.memory_agent]

        if self._is_security_related(task):
            route.insert(1, self.cybersecurity_agent)
        if self._is_multi_step(task) and self.reasoning_agent not in route:
            route.insert(1, self.reasoning_agent)
        if self._has_emotional_keywords(task):
            route = [SentimentAwareAgent(agent) for agent in route]
        return route

    @secure_operation("agents.execute")
    def execute(
        self,
        task: str,
        user_id: str | None = None,
        multimodal_payload: Any | None = None,
    ) -> dict[str, Any]:
        self.security_guard.validate_prompt(task)
        plan = self.reasoning_planner.build_plan(task)
        agents = self.route_task(task)
        hidden_scratchpad: list[str] = []
        mailbox: list[AgentMessage] = []
        results: list[AgentResult] = []
        sentiment = self.sentiment_engine.analyze(task)

        hidden_scratchpad.append(f"internal_reasoning_plan={plan.to_json()}")
        if self._is_security_related(task):
            first_results = self._run_security_parallel(task, mailbox, hidden_scratchpad, user_id, sentiment)
            results.extend(first_results)
            completed = {result.agent_name for result in first_results}
            agents = [agent for agent in agents if agent.name not in completed]

        for agent in agents:
            results.append(agent.run(task, self.memory, mailbox, hidden_scratchpad, user_id=user_id, sentiment=sentiment))

        avg_conf = round(sum(item.confidence for item in results) / max(len(results), 1), 3)
        sources = sorted({citation for item in results for citation in item.citations})
        context = "\n".join(self.memory.facts + [str(value) for value in self.memory.notes.values()])
        self.security_guard.require_grounding(sources=sources, context=context)
        final = self._final_synthesis(task, results, multimodal_payload)

        return {
            "task": task,
            "results": [
                {"agent": r.agent_name, "output": r.output, "confidence": r.confidence, "citations": r.citations}
                for r in results
            ],
            "final": final,
            "confidence": avg_conf,
            "sources": sources,
            "sentiment": sentiment,
            "shared_memory": {"facts": list(self.memory.facts), "notes": dict(self.memory.notes)},
            "inter_agent_messages": [
                {"sender": m.sender, "recipient": m.recipient, "content": m.content} for m in mailbox
            ],
            # chain-of-thought stays hidden intentionally
        }

    def _run_security_parallel(
        self,
        task: str,
        mailbox: list[AgentMessage],
        hidden_scratchpad: list[str],
        user_id: str | None,
        sentiment: dict[str, Any],
    ) -> list[AgentResult]:
        async def run_pair() -> list[AgentResult]:
            return await asyncio.gather(
                asyncio.to_thread(
                    self.research_agent.run,
                    task,
                    self.memory,
                    mailbox,
                    hidden_scratchpad,
                    user_id,
                    sentiment,
                ),
                asyncio.to_thread(
                    self.cybersecurity_agent.run,
                    task,
                    self.memory,
                    mailbox,
                    hidden_scratchpad,
                    user_id,
                    sentiment,
                ),
            )

        return asyncio.run(run_pair())

    def _final_synthesis(self, task: str, results: list[AgentResult], multimodal_payload: Any | None) -> str:
        if multimodal_payload is not None and hasattr(self.llm, "generate_from_embeds"):
            generated = self.llm.generate_from_embeds(
                multimodal_payload.fused_embeds,
                getattr(multimodal_payload, "attention_mask", None),
            )
            return str(getattr(generated, "text", generated))
        return results[-1].output if results else ""

    def _is_security_related(self, task: str) -> bool:
        lowered = task.lower()
        return any(keyword in lowered for keyword in self.SECURITY_KEYWORDS)

    def _has_emotional_keywords(self, task: str) -> bool:
        lowered = task.lower()
        return any(keyword in lowered for keyword in self.EMOTIONAL_KEYWORDS)

    def _is_multi_step(self, task: str) -> bool:
        lowered = task.lower()
        return any(keyword in lowered for keyword in ["then", "after", "step", "multi", "roadmap", "plan"])
