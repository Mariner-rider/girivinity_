from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.query_router import QueryRouter
from app.core.llm_synthesiser import LLMSynthesiser

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

    answer = LLMSynthesiser().synthesise(
        query=req.query,
        context=context,
        urls=urls,
        stream=False,
    )
    if not isinstance(answer, str):
        answer = "".join(answer)

    try:
        from app.engines.analytics_engine import AnalyticsEngine

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
        result = LLMSynthesiser().synthesise(
            query=req.query,
            context=context,
            urls=urls,
            stream=True,
        )
        if isinstance(result, str):
            for word in result.split(" "):
                yield word + " "
        else:
            yield from result

    return StreamingResponse(generate(), media_type="text/plain")

