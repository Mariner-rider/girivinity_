from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.query_router import QueryRouter
from app.core.llm_synthesiser import LLMSynthesiser
from app.core.cognitive_engine import CognitiveEngine
from app.core.sentiment_engine import SentimentEngine
from app.core.social_engine import SocialEngine
from app.core.memory_engine import MemoryEngine
from app.security.ai_threat_reasoner import AIThreatReasoner
from app.security.jailbreak_classifier import JailbreakClassifier

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

    # AI-powered threat check on query
    reasoner  = AIThreatReasoner()
    jailbreak = JailbreakClassifier()

    ai_threat = reasoner.assess(
        query=req.query,
        user_id=req.user_id,
    )
    jb_result = jailbreak.classify(req.query)

    if ai_threat.recommended_action == "block" or jb_result.is_jailbreak:
        raise HTTPException(
            status_code=400,
            detail="Request could not be processed",
        )

    # 1 — Analyse sentiment (silent)
    sentiment = SentimentEngine().analyse(req.query, req.user_id)

    # 2 — Update social/user model (silent)
    user_model = SocialEngine().update(req.user_id, req.query, sentiment)

    # 3 — Recall long-term memories for this user
    memory = MemoryEngine()
    memories = memory.recall(req.user_id, req.query)

    # 4 — Route query (KB or web)
    try:
        result = QueryRouter().route(req.query)
    except Exception as exc:
        logger.error("QueryRouter failed: %s", exc)
        raise HTTPException(status_code=500, detail="Retrieval failed")

    context  = result.get("context_string", "")
    source   = result.get("source", "none")
    urls     = result.get("urls", [])
    confidence = float(result.get("confidence", 0.0))

    # 5 — Cognitive thinking (enrich reasoning)
    cognitive   = CognitiveEngine()
    thought     = cognitive.think(req.query, context)

    # 6 — Build enriched context with memory + social + cognitive
    memory_ctx  = memory.build_memory_context(memories)
    social_ctx  = SocialEngine().get_context_injection(user_model)
    style_instr = SentimentEngine().analyse(
        req.query, req.user_id
    )

    enriched_context = "\n\n".join(filter(None, [
        memory_ctx,
        social_ctx,
        context,
    ]))


    # 7 — Synthesise answer with all context
    try:
        answer = LLMSynthesiser().synthesise(
            query=req.query,
            context=enriched_context,
            urls=urls,
            stream=False,
            web_sources=result.get("chunks", []),
        )
        if not isinstance(answer, str):
            answer = "".join(answer)
    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        answer = enriched_context or "I could not generate an answer."

    # 8 — Store memory async (non-blocking)
    MemoryEngine().remember_async(req.user_id, req.query, answer)

    # 9 — Log analytics (non-blocking)
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

    # AI-powered threat check on query
    reasoner  = AIThreatReasoner()
    jailbreak = JailbreakClassifier()

    ai_threat = reasoner.assess(
        query=req.query,
        user_id=req.user_id,
    )
    jb_result = jailbreak.classify(req.query)

    if ai_threat.recommended_action == "block" or jb_result.is_jailbreak:
        raise HTTPException(
            status_code=400,
            detail="Request could not be processed",
        )

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

