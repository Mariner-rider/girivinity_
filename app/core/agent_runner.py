from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    agent_id: str
    agent_name: str
    success: bool
    output: str
    steps_completed: int
    total_steps: int
    execution_time_s: float
    sources: list[dict] = field(default_factory=list)
    learned_chunks: int = 0
    error: str = ""


class AgentRunner:
    def run(self, agent, user_id: str = "anonymous") -> AgentResult:
        start = time.time()
        registry = self._get_registry()
        registry.set_status(agent.agent_id, "running")
        registry.increment_use(agent.agent_id)

        results: list[str] = []
        all_sources: list[dict] = []
        steps_done = 0
        try:
            for step in agent.steps:
                step_result, sources = self._execute_step(step, agent, user_id)
                if step_result:
                    results.append(f"**Step {step['step']} — {step['description']}:**\n{step_result}")
                all_sources.extend(sources)
                steps_done += 1

            final_output = "\n\n".join(results)
            learned = self._learn_from_results(agent, final_output, all_sources)
            registry.set_status(agent.agent_id, "resting")
            elapsed = round(time.time() - start, 2)
            return AgentResult(agent_id=agent.agent_id, agent_name=agent.name, success=True, output=final_output, steps_completed=steps_done, total_steps=len(agent.steps), execution_time_s=elapsed, sources=all_sources, learned_chunks=learned)
        except Exception as exc:
            registry.set_status(agent.agent_id, "resting")
            logger.error("AgentRunner error: %s", exc)
            return AgentResult(agent_id=agent.agent_id, agent_name=agent.name, success=False, output="", steps_completed=steps_done, total_steps=len(agent.steps), execution_time_s=round(time.time() - start, 2), error=str(exc))

    def _execute_step(self, step: dict, agent, user_id: str) -> tuple[str, list[dict]]:
        action = step.get("action", "")
        desc = step.get("description", "")
        sources: list[dict] = []
        try:
            if action == "understand_request":
                return f"Understood: {agent.created_for[:150]}", []
            if action in ("search_knowledge_base", "search_provisions"):
                from app.core.query_router import QueryRouter

                result = QueryRouter().route(agent.created_for)
                context = result.get("context_string", "")
                sources = [{"url": u, "title": u} for u in result.get("urls", [])]
                return context or f"Searched: {desc}", sources
            if action in ("web_search", "gather_financials", "find_case_law"):
                from app.core.web_intelligence import WebIntelligence

                web = WebIntelligence().search(agent.created_for)
                chunks = web.get("answer_chunks", [])
                sources = web.get("sources", [])
                return "\n".join(c.get("text", "") for c in chunks[:3]) or f"Searched web for: {desc}", sources
            if action == "generate_citations":
                from app.core.citation_engine import CitationEngine

                if sources:
                    cits = CitationEngine().generate_citations(sources)
                    return CitationEngine().format_citations_block(cits, "apa"), sources
                return "", sources
            if action in ("compile_report", "generate_code", "review_code", "legal_analysis", "analyse", "process"):
                from app.core.llm_synthesiser import LLMSynthesiser

                context = "\n".join(s.get("text", s.get("url", "")) for s in sources[:3])
                result = LLMSynthesiser().synthesise(query=agent.created_for, context=context, urls=[s.get("url", "") for s in sources], stream=False, web_sources=sources, user_id=user_id)
                if not isinstance(result, str):
                    result = "".join(result)
                return result, sources
            return f"Completed: {desc}", []
        except Exception as exc:
            logger.warning("Step '%s' failed: %s", action, exc)
            return f"Step '{desc}' encountered an issue: {exc}", []

    def _learn_from_results(self, agent, output: str, sources: list[dict]) -> int:
        learned = 0
        try:
            from app.core.self_trainer import SelfTrainer

            chunks = []
            if output:
                chunks = [{"text": output[:2000], "url": "", "score": 0.8}]
                for src in sources[:5]:
                    if src.get("text") or src.get("url"):
                        chunks.append({"text": src.get("text", src.get("url", "")), "url": src.get("url", ""), "score": 0.7})
            if chunks:
                SelfTrainer().queue(query=agent.created_for, chunks=chunks)
                learned += len(chunks)

            from app.core.skill_forge import SkillForge

            SkillForge().generate_async(topic=f"agent:{agent.agent_type.value}:{agent.name[:40]}", chunks=chunks[:3], urls=[s.get("url", "") for s in sources[:3]])

            from app.core.memory_engine import MemoryEngine

            MemoryEngine().remember_async(user_id="agent_system", query=agent.created_for, response=output[:500])
        except Exception as exc:
            logger.warning("Agent learning failed: %s", exc)
        return learned

    def _get_registry(self):
        from app.core.agent_registry import AgentRegistry

        return AgentRegistry()
