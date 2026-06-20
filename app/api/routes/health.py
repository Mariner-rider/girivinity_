from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "girivinity"}


@router.get("/health/deep")
async def health_deep():
    """Checks that ChromaDB and embedder are reachable."""
    issues = []
    try:
        from app.core.query_router import get_embedder

        get_embedder()
    except Exception as exc:
        issues.append(f"embedder: {exc}")
    try:
        import chromadb
        from pathlib import Path

        import yaml

        cfg = yaml.safe_load(Path("config.yaml").read_text())
        chromadb.PersistentClient(path=cfg["rag"]["chroma_path"])
    except Exception as exc:
        issues.append(f"chromadb: {exc}")

    try:
        from llm_engine import LLMEngine

        compute = LLMEngine().get_device_info()
    except Exception as exc:
        compute = {
            "device": "cpu",
            "gpu_name": None,
            "gpu_vram_total_gb": None,
            "gpu_vram_free_gb": None,
            "inference_mode": "float32_cpu",
            "error": str(exc),
        }

    return {
        "status": "ok" if not issues else "degraded",
        "issues": issues,
        "compute": compute,
    }
