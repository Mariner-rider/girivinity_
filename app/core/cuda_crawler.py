from __future__ import annotations
import logging
import threading
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CUDA_KNOWLEDGE_SOURCES = [
    "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
    "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/",
    "https://docs.nvidia.com/cuda/cuda-math-api/",
    "https://raw.githubusercontent.com/tinygrad/tinygrad/master/README.md",
    "https://raw.githubusercontent.com/openai/triton/main/README.md",
    "https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md",
    "https://raw.githubusercontent.com/NVIDIA/cutlass/main/README.md",
    "https://raw.githubusercontent.com/NVIDIA/cub/main/README.md",
]

CUDA_TOPICS = [
    "CUDA memory coalescing optimization",
    "CUDA warp shuffle reduction kernel",
    "CUDA shared memory bank conflicts",
    "CUDA tensor core WMMA API",
    "CUDA occupancy optimization registers",
    "Flash attention CUDA kernel implementation",
    "CUDA cp.async double buffering A100",
    "CUDA kernel tiling matrix multiplication",
    "CUDA atomics warp level operations",
    "CUDA profiling nsight metrics",
    "CUDA thread block cluster H100",
    "CUDA cooperative groups synchronization",
]


class CUDACrawler:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        self.chroma_path = cfg["rag"]["chroma_path"]

    def bootstrap_async(self) -> None:
        threading.Thread(target=self._bootstrap, daemon=True).start()
        logger.info("CUDACrawler bootstrap started in background")

    def _bootstrap(self) -> None:
        logger.info("Phase 1: Crawling CUDA documentation sources")
        self._crawl_sources()
        logger.info("Phase 2: Searching CUDA topics")
        self._search_topics()
        logger.info("CUDACrawler bootstrap complete")

    def _crawl_sources(self) -> None:
        try:
            from app.core.web_intelligence import WebIntelligence

            wi = WebIntelligence()
            for url in CUDA_KNOWLEDGE_SOURCES:
                try:
                    import httpx
                    import trafilatura

                    r = httpx.get(url, timeout=10, follow_redirects=True, headers={"User-Agent": "GirivinityBot/1.0"})
                    if r.status_code != 200:
                        continue
                    text = trafilatura.extract(r.text)
                    if not text:
                        continue
                    chunks = wi._chunk(text, url, "CUDA Documentation")
                    scored = wi._score("CUDA kernel optimization", chunks)
                    above = [c for c in scored if c["score"] > 0.3]
                    if above:
                        wi._store_pending(above, "CUDA documentation")
                        from app.core.skill_forge import SkillForge

                        SkillForge().generate_async(topic="CUDA kernel optimization", chunks=above[:5], urls=[url])
                    logger.info("Crawled: %s", url)
                except Exception as exc:
                    logger.warning("Failed to crawl %s: %s", url, exc)
        except Exception as exc:
            logger.error("Source crawl failed: %s", exc)

    def _search_topics(self) -> None:
        try:
            from app.core.query_router import QueryRouter
            for topic in CUDA_TOPICS:
                try:
                    QueryRouter().route(topic)
                    logger.info("Searched topic: %s", topic)
                except Exception as exc:
                    logger.warning("Topic search failed '%s': %s", topic, exc)
        except Exception as exc:
            logger.error("Topic search phase failed: %s", exc)
