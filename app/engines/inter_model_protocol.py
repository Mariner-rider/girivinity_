"""Inter-model communication protocol with API interaction, task delegation, result aggregation, and trust scoring."""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelEndpoint:
    model_id: str
    base_url: str
    api_key: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class DelegatedTask:
    task_id: str
    instruction: str
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class ModelResponse:
    model_id: str
    task_id: str
    output: str
    latency_ms: float
    success: bool
    confidence: float = 0.0


class TrustScorer:
    def score(self, endpoint: ModelEndpoint, response: ModelResponse) -> float:
        reliability = float(endpoint.metadata.get("reliability", 0.7))
        confidence = max(0.0, min(1.0, response.confidence))
        latency_penalty = 0.0 if response.latency_ms < 1500 else 0.2
        success_factor = 1.0 if response.success else 0.2
        score = (0.45 * reliability) + (0.35 * confidence) + (0.2 * success_factor) - latency_penalty
        return round(max(0.0, min(1.0, score)), 3)


class InterModelCommunicationProtocol:
    def __init__(self, trust_scorer: TrustScorer | None = None) -> None:
        self.trust_scorer = trust_scorer or TrustScorer()

    def _post_json(self, endpoint: ModelEndpoint, path: str, body: dict, timeout_s: int = 15) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url=f"{endpoint.base_url.rstrip('/')}/{path.lstrip('/')}",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}),
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8", errors="ignore")
        return json.loads(payload) if payload else {}

    def delegate_task(self, endpoint: ModelEndpoint, task: DelegatedTask) -> ModelResponse:
        start = time.perf_counter()
        try:
            payload = self._post_json(
                endpoint,
                "/infer",
                {"task_id": task.task_id, "instruction": task.instruction, "payload": task.payload},
            )
            success = bool(payload.get("success", True))
            output = str(payload.get("output", ""))
            confidence = float(payload.get("confidence", 0.0))
        except Exception as exc:  # network/runtime protection boundary
            success = False
            output = f"delegation_error: {exc}"
            confidence = 0.0
        latency_ms = (time.perf_counter() - start) * 1000
        return ModelResponse(
            model_id=endpoint.model_id,
            task_id=task.task_id,
            output=output,
            latency_ms=round(latency_ms, 3),
            success=success,
            confidence=confidence,
        )

    def aggregate_results(self, endpoint_map: dict[str, ModelEndpoint], responses: list[ModelResponse]) -> dict:
        scored = []
        for res in responses:
            endpoint = endpoint_map[res.model_id]
            trust = self.trust_scorer.score(endpoint, res)
            scored.append({"response": res, "trust": trust})

        scored.sort(key=lambda x: (x["trust"], x["response"].confidence), reverse=True)
        best = scored[0] if scored else None

        return {
            "selected_model": best["response"].model_id if best else "",
            "selected_output": best["response"].output if best else "",
            "consensus": [item["response"].output for item in scored],
            "trust_scores": {item["response"].model_id: item["trust"] for item in scored},
            "all_success": all(item["response"].success for item in scored) if scored else False,
        }
