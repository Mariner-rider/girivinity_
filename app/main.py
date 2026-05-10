import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.config import get_settings
from app.core.system_config import REQUIRED_MODULES, get_system_config
from llm_loader import GirivinityLoader
from app.monitoring.logging import configure_logging
from app.monitoring.metrics import REQUEST_COUNTER
from app.core.self_trainer import SelfTrainer
from app.api.routes.admin import router as admin_router

logger = logging.getLogger(__name__)

_llm_runtime: dict[str, Any] = {}
_self_trainer_process: Any | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, structured=settings.structured_logging)
    _llm_runtime["config"] = get_system_config()
    _start_self_trainer_once()
    if settings.auto_load_model:
        _llm_runtime["runtime"] = GirivinityLoader().get_model()
    yield
    _llm_runtime.clear()


app = FastAPI(title="Girivinity", lifespan=lifespan)
app.include_router(admin_router)


def _start_self_trainer_once() -> None:
    global _self_trainer_process
    if _self_trainer_process is None or not _self_trainer_process.is_alive():
        _self_trainer_process = SelfTrainer.start()


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    REQUEST_COUNTER.labels(path=request.url.path, method=request.method).inc()
    return await call_next(request)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": "runtime" in _llm_runtime,
            "modules": list(REQUIRED_MODULES),
        }
    )


@app.get("/modules")
def modules() -> JSONResponse:
    config = _llm_runtime.get("config") or get_system_config()
    return JSONResponse({"modules": config.get("modules", {})})


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def start_self_trainer():
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    SelfTrainer.start()
