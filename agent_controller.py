"""Multi-agent orchestration with task routing, inter-agent communication, and shared memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

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
    notes: dict[str, str] = field(default_factory=dict)

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
    ) -> AgentResult:
        ...


class ResearchAgent:
    name = "research_agent"

    def run(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        hidden_scratchpad: list[str],
    ) -> AgentResult:
        hidden_scratchpad.append(f"Researching task: {task}")
        findings = f"Collected background facts for: {task}"
        memory.add_fact(findings)
        mailbox.append(AgentMessage(self.name, "reasoning_agent", findings))
        return AgentResult(self.name, findings, confidence=0.72, citations=["internal:memory"])


class ReasoningAgent:
    name = "reasoning_agent"

    def run(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        hidden_scratchpad: list[str],
    ) -> AgentResult:
        inbound = [m.content for m in mailbox if m.recipient == self.name]
        hidden_scratchpad.append(f"Reasoning over {len(memory.facts)} facts and {len(inbound)} messages")
        conclusion = f"Reasoned plan for '{task}' using {len(memory.facts)} facts."
        memory.notes["draft_plan"] = conclusion
        mailbox.append(AgentMessage(self.name, "critic_agent", conclusion))
        return AgentResult(self.name, conclusion, confidence=0.79, citations=["internal:memory"])


class CriticAgent:
    name = "critic_agent"

    def run(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        hidden_scratchpad: list[str],
    ) -> AgentResult:
        draft = memory.notes.get("draft_plan", "")
        inbound = [m.content for m in mailbox if m.recipient == self.name]
        hidden_scratchpad.append(f"Critiquing draft plan with {len(inbound)} upstream messages")
        critique = f"Critique for '{task}': validate assumptions in -> {draft}"
        memory.notes["critique"] = critique
        mailbox.append(AgentMessage(self.name, "memory_agent", critique))
        return AgentResult(
            self.name,
            critique,
            confidence=0.75,
            citations=["internal:memory", "internal:draft_plan"],
        )


class MemoryAgent:
    name = "memory_agent"

    def run(
        self,
        task: str,
        memory: SharedMemory,
        mailbox: list[AgentMessage],
        hidden_scratchpad: list[str],
    ) -> AgentResult:
        inbound = [m.content for m in mailbox if m.recipient == self.name]
        summary = f"Memory summary for '{task}': {len(memory.facts)} facts, {len(memory.notes)} notes, {len(inbound)} messages."
        memory.notes["memory_summary"] = summary
        hidden_scratchpad.append("Persisted shared memory summary")
        return AgentResult(self.name, summary, confidence=0.83, citations=["internal:memory"])


class AgentController:
    def __init__(self, security_guard: SecurityGuard | None = None) -> None:
        self.security_guard = security_guard or SecurityGuard()
        self.memory = SharedMemory()
        self.reasoning_planner = ReasoningPlanner()
        self.research_agent = ResearchAgent()
        self.reasoning_agent = ReasoningAgent()
        self.critic_agent = CriticAgent()
        self.memory_agent = MemoryAgent()

    @secure_operation("agents.route_task")
    def route_task(self, task: str) -> list[Agent]:
        self.security_guard.validate_prompt(task)
        lowered = task.lower()
        route: list[Agent] = [
            self.research_agent,
            self.reasoning_agent,
            self.critic_agent,
            self.memory_agent,
        ]

        if "quick" in lowered:
            route = [self.reasoning_agent, self.critic_agent, self.memory_agent]
        elif "research" in lowered:
            route = [self.research_agent, self.reasoning_agent, self.critic_agent, self.memory_agent]
        return route

    @secure_operation("agents.execute")
    def execute(self, task: str) -> dict:
        self.security_guard.validate_prompt(task)
        plan = self.reasoning_planner.build_plan(task)
        agents = self.route_task(task)
        hidden_scratchpad: list[str] = []
        mailbox: list[AgentMessage] = []
        results: list[AgentResult] = []

        hidden_scratchpad.append(f"internal_reasoning_plan={plan.to_json()}")
        for agent in agents:
            results.append(agent.run(task, self.memory, mailbox, hidden_scratchpad))

        avg_conf = round(sum(item.confidence for item in results) / max(len(results), 1), 3)
        sources = sorted({citation for item in results for citation in item.citations})
        context = "\n".join(self.memory.facts + list(self.memory.notes.values()))
        self.security_guard.require_grounding(sources=sources, context=context)

        return {
            "task": task,
            "results": [
                {
                    "agent": r.agent_name,
                    "output": r.output,
                    "confidence": r.confidence,
                    "citations": r.citations,
                }
                for r in results
            ],
            "final": results[-1].output if results else "",
            "confidence": avg_conf,
            "sources": sources,
            "shared_memory": {
                "facts": list(self.memory.facts),
                "notes": dict(self.memory.notes),
            },
            "inter_agent_messages": [
                {"sender": m.sender, "recipient": m.recipient, "content": m.content} for m in mailbox
            ],
            # chain-of-thought stays hidden intentionally
        }
