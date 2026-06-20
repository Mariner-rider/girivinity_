"""Runtime executor for adaptive, user-personalised agent types."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from app.agents.agent_registry import AgentRegistry, AgentTypeDefinition
from app.agents.capability_merger import CapabilityMerger


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str


class BasicToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, Any]] = {}
        self.register("web_search", "Return web-search guidance for the query", self._web_search)
        self.register("fetch_url", "Fetch and summarise a URL", self._fetch_url)
        self.register("fetch_cve", "Return a CVE detail link and local summary", self._fetch_cve)
        self.register("query_mitre", "Return a MITRE ATT&CK technique link", self._query_mitre)
        self.register("check_ioc", "Classify an indicator of compromise", self._check_ioc)
        self.register("code_analysis", "Analyse code for common risky patterns", self._code_analysis)
        self.register("run_sandbox", "Describe a sandboxed execution plan", self._run_sandbox)

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "BasicToolRegistry":
        return cls()

    def register(self, name: str, description: str, fn: Any) -> None:
        self._tools[name] = (ToolDefinition(name=name, description=description), fn)

    def get(self, name: str) -> ToolDefinition | None:
        item = self._tools.get(name)
        return item[0] if item else None

    def execute(self, name: str, params: dict[str, Any]) -> str:
        item = self._tools.get(name)
        if item is None:
            raise ValueError(f"Unknown tool: {name}")
        result = item[1](params or {})
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def _web_search(self, params: dict[str, Any]) -> str:
        query = params.get("query") or params.get("q") or ""
        return f"Search requested for: {query}. Live web execution is delegated to the production search connector when configured."

    def _fetch_url(self, params: dict[str, Any]) -> str:
        url = params.get("url", "")
        if not url:
            return "No URL provided."
        try:
            from crawler_engine.engine import CrawlerEngine

            item = CrawlerEngine([]).crawl_url(url)
            return json.dumps(item or {"url": url, "status": "no_content"})[:2000]
        except Exception as exc:
            return f"Fetch failed for {url}: {exc}"

    def _fetch_cve(self, params: dict[str, Any]) -> str:
        cve_id = str(params.get("cve_id") or params.get("id") or "").upper()
        return json.dumps({"id": cve_id, "url": f"https://nvd.nist.gov/vuln/detail/{cve_id}", "source": "NVD"})

    def _query_mitre(self, params: dict[str, Any]) -> str:
        technique_id = str(params.get("technique_id") or params.get("id") or "").upper()
        return json.dumps({"technique_id": technique_id, "url": f"https://attack.mitre.org/techniques/{technique_id.replace('.', '/')}/"})

    def _check_ioc(self, params: dict[str, Any]) -> str:
        value = str(params.get("value") or "")
        value_type = "hash" if re.fullmatch(r"[A-Fa-f0-9]{32,64}", value) else "ip" if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value) else "url" if value.startswith(("http://", "https://")) else "domain"
        return json.dumps({"value": value, "value_type": value_type, "is_ioc": bool(value), "feeds": ["local_regex"]})

    def _code_analysis(self, params: dict[str, Any]) -> str:
        code = str(params.get("code") or "")
        findings = []
        if "eval(" in code or "exec(" in code:
            findings.append({"severity": "high", "finding": "Dynamic execution detected"})
        if "shell=True" in code:
            findings.append({"severity": "critical", "finding": "Shell execution with shell=True"})
        return json.dumps({"findings": findings, "severity": "critical" if any(f["severity"] == "critical" for f in findings) else "high" if findings else "low"})

    def _run_sandbox(self, params: dict[str, Any]) -> str:
        return json.dumps({"status": "planned", "isolation": "docker/no-network", "command": params.get("command", "")})


class InMemoryRAG:
    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []

    def query(self, query: str, top_k: int = 5, filter_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        candidates = []
        for doc in self.docs:
            metadata = doc.get("metadata") or {}
            if filter_metadata and any(metadata.get(k) != v for k, v in filter_metadata.items()):
                continue
            text = doc.get("text", "")
            score = len(tokens & set(re.findall(r"[a-zA-Z0-9_]+", text.lower())))
            if score > 0:
                candidates.append((score, doc))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [{**doc, "score": float(score)} for score, doc in candidates[:top_k]]

    def add(self, text: str, source: str = "", metadata: dict[str, Any] | None = None) -> str:
        doc_id = hashlib.sha256(f"{source}:{text}:{time.time()}".encode()).hexdigest()[:16]
        self.docs.append({"doc_id": doc_id, "text": text, "source": source, "metadata": dict(metadata or {})})
        return doc_id


class AdaptiveAgentExecutor:
    def __init__(self, llm_engine: Any, agent_registry: AgentRegistry, capability_merger: CapabilityMerger, tool_registry: Any, rag_engine: Any):
        self.llm = llm_engine
        self.registry = agent_registry
        self.merger = capability_merger
        self.tools = tool_registry
        self.rag = rag_engine

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "AdaptiveAgentExecutor":
        from agent_controller import LocalLLMEngine

        registry = AgentRegistry.from_config(path)
        merger = CapabilityMerger.from_config(path)
        return cls(LocalLLMEngine(), registry, merger, BasicToolRegistry.from_config(path), InMemoryRAG())

    async def execute(self, agent_type_id: str, user_id: str, task: str, user_context: dict[str, Any] | None = None) -> dict[str, Any]:
        agent_def = self.registry.get(agent_type_id)
        if agent_def is None:
            raise ValueError(f"Unknown agent type: {agent_type_id}")
        self._load_adapter(agent_def.adapter_path)
        user_profile = self.registry.get_or_create_user_profile(user_id, agent_type_id)
        system_prompt = self._build_personalised_prompt(agent_def, user_profile, user_context)
        rag_results = self._rag_query(task, agent_type_id)
        rag_context = self._format_rag_context(rag_results)
        tool_descriptions = self._get_tool_descriptions(agent_def.available_tools)
        full_prompt = f"""{system_prompt}

## Available Tools
{tool_descriptions}

## Retrieved Knowledge
{rag_context}

## User's Personal Context
Data sources: {user_profile.get('data_sources', [])}
Preferences: {user_profile.get('preferences', {})}

## Task
{task}

## Instructions
Think step by step. If you need to use a tool, output:
TOOL_CALL: {{"tool": "tool_name", "params": {{...}}}}
Then wait for the tool result before continuing.
Output your final answer as plain text after your reasoning.
"""
        output, tool_calls = await self._agentic_loop(full_prompt, agent_def)
        confidence = self._confidence(full_prompt[:500])
        self._rag_add(
            text=f"Task: {task}\nAnswer: {output[:500]}",
            source=f"agent_interaction:{agent_type_id}:{self._hash(user_id)}",
            metadata={"agent_type": agent_type_id, "user_id_hash": self._hash(user_id)},
        )
        delta_submitted = False
        example = self.merger.extract_training_example(task, output, user_rating=None)
        if example:
            self.registry.submit_capability_delta(user_id, agent_type_id, f"New {agent_type_id} capability", example, example["quality_score"])
            delta_submitted = True
        try:
            asyncio.create_task(self.merger.check_and_merge(agent_type_id))
        except RuntimeError:
            await self.merger.check_and_merge(agent_type_id)
        self._increment_interactions(user_id, agent_type_id)
        latest_def = self.registry.get(agent_type_id) or agent_def
        return {
            "output": output,
            "confidence": confidence,
            "sources": [result.get("source", "") for result in rag_results],
            "tool_calls": tool_calls,
            "capability_delta_submitted": delta_submitted,
            "agent_type": agent_type_id,
            "capability_version": latest_def.capability_version,
        }

    async def _agentic_loop(self, prompt: str, agent_def: AgentTypeDefinition, max_iterations: int = 5) -> tuple[str, list[dict[str, Any]]]:
        current_prompt = prompt
        tool_calls: list[dict[str, Any]] = []
        text = ""
        for _ in range(max_iterations):
            result = self.llm.generate(current_prompt, max_new_tokens=512, max_tokens=512, temperature=0.2)
            text = str(getattr(result, "text", result))
            match = re.search(r"TOOL_CALL:\s*(\{.*?\})", text, re.DOTALL)
            if match:
                tool_record = self._execute_tool_call(match.group(1), agent_def)
                if tool_record is not None:
                    tool_calls.append(tool_record)
                    current_prompt = f"{current_prompt}\n\nTOOL_CALL: {match.group(1)}\nTOOL_RESULT: {tool_record['result']}\n\nContinue:"
                    continue
            final = re.sub(r"TOOL_CALL:.*?(\n|$)", "", text, flags=re.DOTALL).strip()
            return final or text.strip(), tool_calls
        return text.strip(), tool_calls

    def _execute_tool_call(self, raw_json: str, agent_def: AgentTypeDefinition) -> dict[str, Any] | None:
        try:
            tool_spec = json.loads(raw_json)
            tool_name = str(tool_spec.get("tool", ""))
            params = tool_spec.get("params", {}) if isinstance(tool_spec.get("params", {}), dict) else {}
            if tool_name not in agent_def.available_tools:
                return {"tool": tool_name, "params": params, "result": "Tool is not available for this agent type."}
            tool_result = self.tools.execute(tool_name, params)
            return {"tool": tool_name, "params": params, "result": str(tool_result)[:2000]}
        except Exception as exc:
            return {"tool": "invalid", "params": {}, "result": f"Tool call failed: {exc}"}

    def _build_personalised_prompt(self, agent_def: AgentTypeDefinition, user_profile: dict[str, Any], user_context: dict[str, Any] | None) -> str:
        addon = user_profile.get("custom_system_prompt_addon", "") or ""
        preferences = user_profile.get("preferences", {}) or {}
        if preferences:
            addon += f"\nUser preferences: {preferences}"
        if user_context:
            addon += f"\nUser context: {user_context}"
        return f"{agent_def.base_system_prompt}\n\n{addon}".strip()

    def _format_rag_context(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "No specific context retrieved."
        return "\n".join(f"[{index + 1}] {str(result.get('text', ''))[:300]}" for index, result in enumerate(results))

    def _get_tool_descriptions(self, tool_names: list[str]) -> str:
        descriptions = []
        for name in tool_names:
            tool = self.tools.get(name) if hasattr(self.tools, "get") else None
            if tool:
                descriptions.append(f"- {name}: {getattr(tool, 'description', '')}")
        return "\n".join(descriptions) if descriptions else "No tools available."

    def _hash(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def _confidence(self, prompt: str) -> float:
        try:
            entropy = float(self.llm.get_token_entropy(prompt))
            return round(1.0 / (1.0 + entropy), 3)
        except Exception:
            return 0.5

    def _rag_query(self, task: str, agent_type_id: str) -> list[dict[str, Any]]:
        try:
            return list(self.rag.query(task, top_k=5, filter_metadata={"agent_type": agent_type_id}) or [])
        except TypeError:
            return list(self.rag.query(task, top_k=5) or [])
        except Exception:
            return []

    def _rag_add(self, text: str, source: str, metadata: dict[str, Any]) -> None:
        if hasattr(self.rag, "add"):
            self.rag.add(text=text, source=source, metadata=metadata)
        elif hasattr(self.rag, "add_batch"):
            self.rag.add_batch([{"text": text, "source": source, "metadata": metadata}])

    def _load_adapter(self, adapter_path: str) -> None:
        model = getattr(self.llm, "model", None)
        if model is not None and adapter_path and hasattr(model, "load_adapter") and hasattr(model, "set_adapter"):
            try:
                model.load_adapter(adapter_path, adapter_name="adaptive_current")
                model.set_adapter("adaptive_current")
            except Exception:
                return

    def _increment_interactions(self, user_id: str, agent_type_id: str) -> None:
        with self.registry._conn() as conn:
            conn.execute(
                """
                UPDATE user_agent_profiles
                SET interaction_count = interaction_count + 1, updated_at = ?
                WHERE user_id = ? AND agent_type_id = ?
                """,
                (time.time(), user_id, agent_type_id),
            )
            conn.commit()
