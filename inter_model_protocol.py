"""InterModelProtocol — standardised in-process communication for AI models."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class ModelMessage:
    task: str
    sender_model: str
    recipient_model: str
    context: dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"
    timeout_seconds: int = 30
    reply_to: str | None = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "ModelMessage":
        payload = json.loads(data)
        return cls(**payload)


class InterModelRouter:
    """Routes tasks between registered model handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[ModelMessage], Any]] = {}
        self._endpoints: dict[str, asyncio.Queue[ModelMessage]] = {}

    def register(self, model_id: str, handler: Callable[[ModelMessage], Any]) -> None:
        if not model_id:
            raise ValueError("model_id is required")
        self._handlers[model_id] = handler
        self._endpoints.setdefault(model_id, asyncio.Queue())

    async def send(self, message: ModelMessage) -> dict[str, Any]:
        if message.recipient_model not in self._handlers:
            raise ValueError(f"Unknown model: {message.recipient_model}")

        handler = self._handlers[message.recipient_model]
        try:
            if inspect.iscoroutinefunction(handler):
                result = await asyncio.wait_for(handler(message), timeout=message.timeout_seconds)
            else:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, handler, message), timeout=message.timeout_seconds
                )
            return {"success": True, "result": result, "message_id": message.message_id}
        except asyncio.TimeoutError:
            return {"success": False, "error": "timeout", "message_id": message.message_id}
        except Exception as exc:  # handlers are external plugins; surface failures as protocol errors
            return {"success": False, "error": str(exc), "message_id": message.message_id}

    def broadcast(self, task: str, sender: str, recipients: list[str]) -> list[asyncio.Task[dict[str, Any]]]:
        tasks = []
        for recipient in recipients:
            if recipient in self._handlers:
                message = ModelMessage(task=task, sender_model=sender, recipient_model=recipient)
                tasks.append(asyncio.create_task(self.send(message)))
        return tasks


class CapabilityNegotiator:
    """Allows models to advertise and discover each other's capabilities."""

    def __init__(self) -> None:
        self._registry: dict[str, list[str]] = {}

    def advertise(self, model_id: str, capabilities: list[str]) -> None:
        self._registry[model_id] = [cap.lower() for cap in capabilities]

    def find_capable(self, capability: str) -> list[str]:
        needle = capability.lower()
        return [model_id for model_id, caps in self._registry.items() if needle in caps]

    def best_for(self, task: str) -> str | None:
        words = set(re for re in task.lower().replace("-", " ").split() if re)
        best_model: str | None = None
        best_score = 0
        for model_id, capabilities in self._registry.items():
            score = sum(1 for capability in capabilities if capability in task.lower() or capability in words)
            if score > best_score:
                best_model, best_score = model_id, score
        return best_model
