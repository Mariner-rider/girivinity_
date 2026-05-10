# Girivinity Internal Documentation

Developer-friendly internal guide for architecture, modules, deployment, training, and security.

---

## 1) System Architecture

Girivinity is organized as a modular platform with composable services and engines.

### High-level architecture

```text
Clients/API
  -> app.main (FastAPI entrypoint)
  -> app.agents / agent_controller (task orchestration)
  -> app.rag (retrieval + generation)
  -> app.memory (short-term + vector + graph memory)
  -> app.security (policy and runtime inspection)
  -> app.profiling (user capability detection)

Data/Processing engines
  -> analytics_engine.py
  -> crawler_engine/
  -> multimodal_engine.py
  -> code_intelligence_engine.py
  -> inter_model_protocol.py
  -> user_behavior_engine.py

Model lifecycle
  -> app.finetune (dataset build, LoRA training/eval, continual learning)
  -> app.training (model evolution and deployment gating)
```

### Design principles

- **Modular boundaries**: each domain lives in an isolated module.
- **Policy-gated execution**: sensitive operations are routed through security checks.
- **Adapter-first model updates**: continual improvements use LoRA-style updates (no full retraining pathways).
- **Testable defaults**: in-memory and placeholder implementations keep local/CI deterministic.

---

## 2) Module Responsibilities

## Core runtime (`app/`)

- `app/main.py`
  - API entrypoint and integration wiring.
- `app/agents/` + `agent_controller.py`
  - Multi-agent orchestration, task routing, inter-agent messaging, shared memory summary.
- `app/rag/`
  - Retrieval-augmented generation flow, source grounding, citation-aware responses.
- `app/memory/`
  - Hybrid memory system (vector retrieval + short-term recency + graph relations).
- `app/security/`
  - Guardrails (`policy.py`) and advanced security inspection (`layer.py`).
- `app/profiling/`
  - User intelligence profiling and response-adaptation planning.
- `app/finetune/`
  - Dataset construction, LoRA trainer/eval, update gate, continual learning.
- `app/training/`
  - Model evolution lifecycle and deploy/notification rules.

## Engine modules (repo root)

- `analytics_engine.py`
  - CSV/DB ingestion, SQL + pandas transforms, anomaly detection, forecasting, reports.
- `crawler_engine/`
  - Content extraction, trust scoring, dedupe, quality filtering, language/topic enrichment.
- `multimodal_engine.py`
  - Image/video/speech/youtube processing, summarization, insight extraction, TTS.
- `code_intelligence_engine.py`
  - Repository analysis, bug/static checks, API tests, fix suggestions, report export.
- `inter_model_protocol.py`
  - Cross-model API delegation, trust scoring, and aggregated result selection.
- `user_behavior_engine.py`
  - Privacy-aware behavior tracking, embeddings, recommendations, ad-segment targeting.

---

## 3) Deployment Guide

## Local development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Container deployment

```bash
docker build -t girivinity .
docker run --rm -p 8000:8000 girivinity
```

## Configuration

- Base configuration: `config.yaml`
- Environment override prefix: `GIRIVINITY__`

Examples:

```bash
GIRIVINITY__MODEL__MODEL_ID=meta-llama/Meta-Llama-3-8B
GIRIVINITY__CRAWLER__MAX_DEPTH=3
GIRIVINITY__FEATURE_FLAGS__ENABLE_TRAINING=false
```

## Recommended deployment stages

1. **Dev**: local/in-memory adapters, smoke tests.
2. **Staging**: real backing services (vector DB/redis/graph), integration tests.
3. **Prod**: monitoring + policy gates + rollback-ready model artifacts.

---

## 4) Training Workflow

Training stack is adapter-centric and split across `app/finetune` and `app/training`.

## Standard LoRA workflow

1. Build dataset from logs (`dataset_builder.py`)
2. Train adapter (`lora_trainer.py`)
3. Evaluate adapter (`evaluate.py`)
4. Gate promotion (`update_gate.py`)

## Continual learning workflow

`app/finetune/continual_learning.py` orchestrates:

1. collect logs,
2. filter high-quality samples,
3. build dataset,
4. run LoRA fine-tune hook,
5. benchmark candidate vs production,
6. rollback on policy rejection.

## Model evolution workflow

`app/training/model_evolution.py` orchestrates:

1. evaluate baseline,
2. identify weaknesses,
3. generate targeted training tasks,
4. fine-tune candidate,
5. benchmark and deploy only if:
   - accuracy improves,
   - hallucination decreases.

---

## 5) Security Policies

Security controls are split into two layers:

## Policy guard (`app/security/policy.py`)

- Prompt must be non-empty.
- Grounded generation requires sources + context.
- URL trust scoring for crawler inputs.
- Validation dataset requirements for finetuning.
- Benchmark improvement checks for model update approval.

## Runtime security layer (`app/security/layer.py`)

- Input sanitization (script/control-character cleanup).
- Prompt injection detection patterns.
- Anomaly detection (oversized/obfuscated payload heuristics).
- Threat intelligence pattern matching.
- Attack-event logging (JSONL).
- Self-improving rule engine for recurring attack signatures.

## Security best practices for contributors

- Prefer `@secure_operation(...)` on sensitive entrypoints.
- Never bypass grounding checks in generation paths.
- Route new model-promotion logic through benchmark gates.
- Add tests for malicious payloads and regression cases.

---

## Appendix: Suggested Developer Onboarding Path

1. Read this doc + `README.md`.
2. Run focused tests for your area.
3. Start with module-level changes and add unit tests first.
4. Validate policy/security implications before merging.
