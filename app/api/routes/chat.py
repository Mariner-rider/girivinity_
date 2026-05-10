from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.query_router import QueryRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    source: str
    confidence: float
    urls: list[str]


@router.post("/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = QueryRouter().route(req.query)
    except Exception as exc:
        logger.error("QueryRouter failed: %s", exc)
        raise HTTPException(status_code=500, detail="Retrieval failed")

    context = result.get("context_string", "")
    source = result.get("source", "none")
    urls = result.get("urls", [])
    confidence = float(result.get("confidence", 0.0))

    if context:
        answer = _format_answer(req.query, context, urls)
    else:
        answer = (
            "I could not find verified information on this topic right now. "
            "The system will search and learn more about it in the background."
        )

    try:
        from app.core.analytics_engine import AnalyticsEngine

        AnalyticsEngine().log_query(
            user_id=req.user_id,
            query=req.query,
            source=source,
            confidence=confidence,
        )
    except Exception:
        pass

    return ChatResponse(
        answer=answer,
        source=source,
        confidence=confidence,
        urls=urls,
    )


@router.post("/message/stream")
async def chat_message_stream(req: ChatRequest):
    """Streaming version — yields answer tokens as they are built."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = QueryRouter().route(req.query)
    except Exception as exc:
        logger.error("QueryRouter failed: %s", exc)
        raise HTTPException(status_code=500, detail="Retrieval failed")

    context = result.get("context_string", "")
    urls = result.get("urls", [])

    def generate():
        answer = _format_answer(req.query, context, urls) if context else (
            "I could not find verified information on this topic right now. "
            "The system will search and learn more about it in the background."
        )
        for word in answer.split(" "):
            yield word + " "

    return StreamingResponse(generate(), media_type="text/plain")


def _format_answer(query: str, context: str, urls: list[str]) -> str:
    """Structure the retrieved context into a clean answer."""
    lines = ["Here is what I found:\n", context]
    if urls:
        lines.append("\nSources:")
        for i, url in enumerate(urls[:3], 1):
            lines.append(f"  [{i}] {url}")
    return "\n".join(lines)
