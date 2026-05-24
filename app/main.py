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
from app.core.successor_engine import SuccessorEngine
from app.core.migrations import run_migrations
from app.core.db import close_pool
from app.api.routes.admin import router as admin_router
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.skills import router as skills_router
from app.api.routes.cuda import router as cuda_router
from app.security.cyber_shield import CyberShieldMiddleware
from app.api.routes.security import router as security_router
from app.api.routes.tenant_security import router as tenant_security_router
from app.api.routes.agents import router as agents_router
from app.api.routes.rasp import router as rasp_router
from app.security.rasp.rasp_engine import RASPEngine

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
app.add_middleware(CyberShieldMiddleware)
app.include_router(chat_router)
app.include_router(health_router)
app.include_router(admin_router)
app.include_router(skills_router)
app.include_router(cuda_router)
app.include_router(security_router)
app.include_router(tenant_security_router)
app.include_router(agents_router)
app.include_router(rasp_router)


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


@app.on_event("startup")
async def start_successor_engine():
    SuccessorEngine.start()


@app.on_event("startup")
async def bootstrap_cuda():
    from app.core.cuda_crawler import CUDACrawler
    CUDACrawler().bootstrap_async()


@app.on_event("startup")
async def startup():
    run_migrations()

@app.on_event("shutdown")
async def shutdown():
    close_pool()


@app.on_event("startup")
async def start_rasp():
    RASPEngine.start()
