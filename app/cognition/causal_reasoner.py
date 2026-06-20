"""
CausalReasoningEngine — forces structured step-by-step reasoning before answering.

For complex queries, injects a chain-of-thought scaffold into the prompt:
  Step 1: Identify the core question / goal
  Step 2: List relevant facts from memory and retrieved context
  Step 3: Identify causal relationships (A causes B because...)
  Step 4: Identify what is NOT known / uncertain
  Step 5: Apply counterfactual check — "If X were false, would this still hold?"
  Step 6: Synthesize answer

For simpler queries (detected by low complexity score), skips chain-of-thought.
"""

from __future__ import annotations

import json
import re
from typing import Any


class CausalReasoningEngine:
    def __init__(self, config: Any, llm_engine: Any) -> None:
        self.config = config
        self.llm = llm_engine
        self.steps = int(self._cfg("chain_of_thought_steps", 5))

    def complexity_score(self, query: str) -> float:
        """0.0–1.0. High score = complex query needing chain-of-thought."""
        words = query.split()
        has_why = any(w.lower().startswith("why") for w in words)
        has_how = any(w.lower().startswith("how") for w in words)
        is_long = len(words) > 15
        has_comparison = any(w in query.lower() for w in ["vs", "compare", "difference", "better"])
        has_causal = any(w in query.lower() for w in ["cause", "effect", "lead to", "result in", "because"])
        return min(
            1.0,
            sum(
                [
                    has_why * 0.3,
                    has_how * 0.2,
                    is_long * 0.2,
                    has_comparison * 0.15,
                    has_causal * 0.15,
                ]
            ),
        )

    def build_reasoning_prompt(self, query: str, context: str, user_model_addon: str) -> str:
        """Returns a structured prompt that forces step-by-step reasoning."""
        return f"""{user_model_addon}

## Task
{query}

## Context
{context}

## Reasoning Protocol
You MUST follow these steps before writing your final answer.
Do not skip steps. Think explicitly at each stage.

Step 1 — IDENTIFY: What exactly is being asked? What is the core goal?
Step 2 — FACTS: What relevant facts do I know from context and memory?
Step 3 — CAUSALITY: What causes what here? Identify at least one A → B causal chain.
Step 4 — UNCERTAINTY: What am I not sure about? What might I be wrong about?
Step 5 — COUNTERFACTUAL: If one of my key assumptions were false, does the answer change?
Step 6 — SYNTHESIS: Given steps 1–5, what is my best answer?

[Begin reasoning]
"""

    def counterfactual_check(self, reasoning: str, answer: str) -> dict[str, Any]:
        """Ask the LLM to challenge its own answer. Returns {robust: bool, concern: str}."""
        prompt = f"""You just reasoned:
{reasoning}

And concluded:
{answer}

Now challenge this. Identify the single most important assumption in your reasoning.
If that assumption were FALSE, would your conclusion change?
Answer with JSON: {{"assumption": "...", "if_false_conclusion_changes": true/false, "concern": "..."}}"""
        result = self._generate(prompt)
        match = re.search(r"\{.*?\}", result, re.DOTALL)
        if not match:
            return {"robust": True, "concern": ""}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {"robust": True, "concern": ""}

    def _generate(self, prompt: str) -> str:
        if hasattr(self.llm, "generate"):
            generated = self.llm.generate(prompt, max_tokens=200, stream=False)
            return getattr(generated, "text", generated)
        raise RuntimeError("LLM engine does not expose generate()")

    def _cfg(self, name: str, default: Any) -> Any:
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return getattr(self.config, name, default)
