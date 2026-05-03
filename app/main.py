import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.config import get_settings
from app.llm.loader import QuantizedLLMLoader
from app.monitoring.logging import configure_logging
from app.monitoring.metrics import REQUEST_COUNTER

logger = logging.getLogger(__name__)

_llm_runtime: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    loader = QuantizedLLMLoader(settings)
    _llm_runtime["runtime"] = loader.load()
    yield
    _llm_runtime.clear()


app = FastAPI(title="Girivinity", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    REQUEST_COUNTER.labels(path=request.url.path, method=request.method).inc()
    return await call_next(request)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
