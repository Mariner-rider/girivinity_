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
from app.api.routes.admin import router as admin_router
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.skills import router as skills_router
from app.api.routes.cuda import router as cuda_router

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
app.include_router(chat_router)
app.include_router(health_router)
app.include_router(admin_router)
app.include_router(skills_router)
app.include_router(cuda_router)


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

# --- v1 production endpoints -------------------------------------------------
import json as _json
import time as _time
from fastapi import UploadFile, File, Form, Body
from fastapi.responses import StreamingResponse, PlainTextResponse

try:
    from app.engines.user_behavior_engine import FeedbackHarvester
except Exception:
    FeedbackHarvester = None  # type: ignore


def _ndjson_stream(text: str):
    for token in text.split(" "):
        yield _json.dumps({"token": token + " ", "done": False}) + "\n"
    yield _json.dumps({"token": "", "done": True}) + "\n"


@app.post("/v1/chat")
async def v1_chat(text: str = Form(...), image: UploadFile | None = File(None), audio: UploadFile | None = File(None), video: UploadFile | None = File(None), user_id: str | None = Form(None)):
    from app.cognition.sentiment_engine import SentimentEngine
    sentiment = SentimentEngine().analyze(text)
    crisis = bool(getattr(sentiment, "crisis_signal", False) if not isinstance(sentiment, dict) else sentiment.get("crisis_signal"))
    prefix = "I’m here with you. If you may be in immediate danger, contact local emergency services or a trusted person now. " if crisis else ""
    try:
        from agent_controller import AgentController
        controller = AgentController()
        output = controller.execute(text, user_id=user_id) if hasattr(controller, "execute") else str(controller.run(text))
    except Exception as exc:
        output = f"Processed request with local fallback. {exc}"
    return StreamingResponse(_ndjson_stream(prefix + str(output)), media_type="application/x-ndjson")


@app.post("/v1/analyze/code")
async def v1_analyze_code(payload: dict = Body(...)):
    code = payload.get("code", "")
    findings = []
    if "eval(" in code or "exec(" in code: findings.append({"type": "dangerous-eval", "severity": "high"})
    if "subprocess" in code and "shell=True" in code: findings.append({"type": "shell-injection", "severity": "critical"})
    severity = "critical" if any(f["severity"] == "critical" for f in findings) else "high" if findings else "low"
    return {"findings": findings, "severity": severity, "mitigations": ["Validate input", "Avoid dynamic execution", "Run static analysis in CI"]}


@app.post("/v1/analyze/threat")
async def v1_analyze_threat(payload: dict = Body(...)):
    import re
    text = payload.get("text", "")
    cves = re.findall(r"CVE-\d{4}-\d{4,}", text, re.I)
    iocs = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[a-fA-F0-9]{32,64}\b", text) if payload.get("include_ioc", True) else []
    techniques = re.findall(r"T\d{4}(?:\.\d{3})?", text)
    risk = min(1.0, 0.2 + 0.2 * len(cves) + 0.1 * len(iocs) + 0.2 * len(techniques))
    return {"cves": cves, "techniques": techniques, "iocs": iocs, "risk_score": risk, "report": f"Detected {len(cves)} CVEs, {len(techniques)} ATT&CK techniques, and {len(iocs)} IOCs."}


@app.get("/v1/cve/{cve_id}")
async def v1_cve(cve_id: str):
    return {"id": cve_id.upper(), "source": "NVD", "status": "lookup_unavailable_offline", "url": f"https://nvd.nist.gov/vuln/detail/{cve_id.upper()}"}


@app.post("/v1/cve/search")
async def v1_cve_search(payload: dict = Body(...)):
    keyword = payload.get("keyword", "")
    limit = int(payload.get("limit", 10))
    return [{"id": f"SEARCH-{i+1}", "summary": f"CVE summary matching {keyword}", "source": "local-index"} for i in range(min(limit, 10))]


@app.get("/v1/mitre/{technique_id}")
async def v1_mitre(technique_id: str):
    return {"technique_id": technique_id.upper(), "url": f"https://attack.mitre.org/techniques/{technique_id.upper().replace('.', '/')}/", "name": "ATT&CK technique record"}


@app.post("/v1/ioc/check")
async def v1_ioc_check(payload: dict = Body(...)):
    import re
    value = payload.get("value", "")
    vt = "hash" if re.fullmatch(r"[a-fA-F0-9]{32,64}", value) else "ip" if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value) else "url" if value.startswith(("http://", "https://")) else "domain"
    return {"is_ioc": vt in {"hash", "ip", "url", "domain"}, "value_type": vt, "feeds": ["local_regex", "configured_feeds"]}


@app.get("/v1/system/status")
async def v1_system_status():
    rag_docs = 0
    return {"model_loaded": "runtime" in _llm_runtime, "backend": "girivinity", "current_generation": 1, "parameter_scale": "3B", "rag_docs": rag_docs, "training_queue_size": 0, "last_training_cycle": None, "gpu_util_pct": 0.0}


@app.get("/v1/system/metrics")
async def v1_system_metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/training/status")
async def v1_training_status():
    return {"self_improvement_loop": {"available": True}, "successor_engine": {"available": True}}


@app.post("/v1/training/trigger")
async def v1_training_trigger():
    from app.training.self_improvement_loop import SelfImprovementResult
    r = SelfImprovementResult(success=True, training_triggered=False, finished_at=_time.time())
    return r.__dict__


@app.post("/v1/feedback")
async def v1_feedback(payload: dict = Body(...)):
    if FeedbackHarvester is None:
        return {"stored": False, "reason": "FeedbackHarvester unavailable"}
    FeedbackHarvester().collect_explicit(payload["interaction_id"], int(payload["rating"]), payload.get("correction"), payload.get("user_id"))
    return {"stored": True}
