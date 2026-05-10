# Girivinity Autonomous LLM System

Production-grade modular AI system scaffold using Python, FastAPI, and PyTorch.

## Folder structure

```text
app/
  core/          # config loading and runtime settings
  llm/           # quantized HuggingFace/PyTorch model loading
  memory/        # short-term memory + FAISS-backed long-term memory
  agents/        # multi-agent orchestration entrypoints
  crawler/       # Scrapy crawler, URL queue, trust scoring, vector pipeline
  rag/           # retrieval augmented generation pipeline
  security/      # cross-module guardrails and policy checks
  analytics/     # analytics event interfaces/sinks
  multimodal/    # text/image/audio payload normalization
  training/      # training-facing exports for LoRA workflows
  finetune/      # LoRA dataset building, training, eval, update gate
  monitoring/    # structured logging and Prometheus metrics
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Centralized configuration

`config.yaml` is the base YAML configuration. Runtime overrides use double-underscore
environment paths with the `GIRIVINITY__` prefix, for example:

```bash
GIRIVINITY__MODEL__MODEL_ID=meta-llama/Meta-Llama-3-8B
GIRIVINITY__CRAWLER__MAX_DEPTH=3
GIRIVINITY__FEATURE_FLAGS__ENABLE_TRAINING=false
```

`app.core.config_loader.ConfigLoader` supports explicit `reload()` and mtime-based
`reload_if_changed()` for long-running services.

## Docker

```bash
docker build -t girivinity .
docker run --rm -p 8000:8000 girivinity
```

The base `config.yaml` is module-oriented and environment overrides live in `.env` / `.env.example`.
Set `AUTO_LOAD_MODEL=true` only when model weights and hardware are available.
