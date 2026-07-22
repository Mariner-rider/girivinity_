# Girivinity — Self-Learning AI System

## Why We Built This

Girivinity was born in India with a single belief: the future of artificial
intelligence should not be owned by a handful of companies in Silicon Valley.
Every major AI model today — GPT, Claude, Gemini, Llama — was built abroad,
trained on data that underrepresents Indian languages, Indian culture, and
Indian knowledge. When you ask these models about local governance, regional
history, or technical questions in Hindi, the answers are shallow. The models
were not built for us.

We decided to change that.

Girivinity is our answer — a fully independent AI system, built from scratch
in Python and PyTorch, that:

- Owns its own transformer architecture (no AutoModelForCausalLM)
- Trains its own tokeniser natively on Hindi, English, and code
- Learns continuously from the web without human intervention
- Runs on minimal hardware so it can be deployed anywhere in India
- Never fabricates sources — every claim is verified before it reaches the user

This is not a product built for profit. It is infrastructure built for
independence. We are building the model that belongs to everyone.

Built with pride in India. Open for the world.

---

Girivinity is a self-improving, self-learning AI system built from scratch.
It retrieves live information from the web, synthesises answers using its
own language model, verifies every claim before responding, and continuously
trains itself on everything it learns — with no human intervention required
after deployment.

It is not a wrapper around ChatGPT, Claude, or any other external model.
Every component — the transformer architecture, the training pipeline, the
retrieval system, the self-improvement loop — is built and owned entirely
within this codebase.

---

## Table of Contents

1. [Why We Built This](#why-we-built-this)
2. [What Makes This Different](#what-makes-this-different)
3. [System Architecture](#system-architecture)
4. [How Self-Learning Works](#how-self-learning-works)
5. [How Successor Models Work](#how-successor-models-work)
6. [Model Generation Roadmap](#model-generation-roadmap)
7. [Repository Structure](#repository-structure)
8. [Component Reference](#component-reference)
9. [API Reference](#api-reference)
10. [Deployment Guide — Step by Step](#deployment-guide)
11. [Development Guide](#development-guide)
12. [Configuration Reference](#configuration-reference)
13. [Serving the API to Others](#serving-the-api-to-others)
14. [Roadmap](#roadmap)

---

## What Makes This Different

Most AI products are wrappers. They call OpenAI or Anthropic and format
the response. Girivinity is not that.

**It owns its own model.** The GirivinityModel is a decoder-only transformer
built from scratch in PyTorch — RMSNorm, Rotary Position Embeddings, Grouped
Query Attention, SwiGLU feed-forward, weight-tied LM head. The model family starts at 3B parameters (Girivinity-3B), scaling to 70B and beyond through the structured generation roadmap.
Quantised to Q4_K_M GGUF for CPU inference on a standard VPS.

**It learns while it runs.** Every query that misses the knowledge base
triggers a live web search. The retrieved content is queued for training.
Every 30 minutes the self-trainer fires LoRA fine-tuning automatically
in a background process. The model gets smarter every day it runs.

**It creates its own successors.** When the knowledge base crosses 100,000
trained chunks, the system automatically retrains a new model version from
scratch, evaluates it against the current model, and notifies the admin.
One API call approves it. The new model goes live.

**It never fabricates sources.** Every response passes through the
TruthEngine, which verifies each factual claim against the knowledge base
and attached web sources. Unverified claims are flagged. Confidence is
scored 0.0–1.0. Citations only appear for URLs that actually exist in
the current session's retrieved data.

**It runs on low compute.** Runs on both GPU and CPU — automatically. When a GPU is present, inference uses 4-bit NF4 quantization (BitsAndBytes) for maximum speed at minimum VRAM cost. When no GPU is available, it falls back to CPU float32 inference on as little as 4GB RAM. Same codebase, same config, zero changes needed.
LoRA fine-tuning triggers on demand on the cheapest GPU instance available
(Lambda Labs T4, ~$0.35/hour). Full successor retraining runs once every
few months automatically.

---

## System Architecture
User Query
│
▼
┌─────────────────────────────────────┐
│           QueryRouter               │
│  Embeds query → searches ChromaDB   │
│  Score ≥ 0.72 → Knowledge Base hit  │
│  Score < 0.72 → Web search          │
└──────────┬──────────────┬───────────┘
│              │
KB Hit │              │ KB Miss
│              ▼
│   ┌─────────────────────┐
│   │   WebIntelligence   │
│   │  DuckDuckGo search  │
│   │  httpx fetch        │
│   │  trafilatura clean  │
│   │  chunk + score      │
│   └────────┬────────────┘
│            │
│            ├──────────────────────────────────┐
│            │                                  │
│            ▼                                  ▼
│   Returns top 3 chunks            Background daemon thread
│                                   SelfTrainer.queue()
│                                   → SQLite pending_training
│                                   → Every 30min: LoRA update
▼
┌─────────────────────────────────────┐
│          LLMSynthesiser             │
│  Builds prompt: system + context    │
│  Calls GirivinityEngine.generate()  │
│  Fallback to extraction if no model │
└──────────┬──────────────────────────┘
│
▼
┌─────────────────────────────────────┐
│           TruthEngine               │
│  Extracts factual claims            │
│  Verifies each claim vs ChromaDB    │
│  Labels: KB_SOURCED / WEB_SOURCED   │
│          / UNVERIFIED               │
│  Adds disclaimer if >30% unverified │
│  Appends real citations only        │
│  Scores confidence 0.0–1.0          │
└──────────┬──────────────────────────┘
│
▼
Response to User
{answer, source, confidence, urls}

---

## How Self-Learning Works

Every time a user asks something the knowledge base does not know:

1. **WebIntelligence** fires — searches DuckDuckGo, fetches top 5 URLs,
   extracts clean text via trafilatura, chunks into 400-token segments,
   scores relevance via cosine similarity.

2. Top 3 chunks become the **context** for the current answer.

3. All chunks above the relevance threshold are written to the
   **SQLite training queue** in a background daemon thread.
   The user receives their answer immediately — training never blocks.

4. Every 30 minutes the **SelfTrainer** daemon wakes up and checks
   how many chunks are pending.

5. When the count reaches 50 (configurable), it:
   - Formats chunks as instruction pairs
   - Saves as JSONL dataset
   - Loads the current base model + latest LoRA adapter
   - Runs LoRA fine-tuning (3 epochs, lr=2e-4)
   - Saves the new adapter
   - Updates the `models/adapters/latest` symlink
   - Logs the training event

6. If training loss exceeds the abort threshold, the update is
   discarded and an alert is written to `logs/alerts.jsonl`.

The model gets a LoRA update multiple times per day under normal usage.
Each update makes it more knowledgeable about the topics users actually
ask about.

---

## How Successor Models Work

LoRA adapters improve the existing model. Successor models are full
retrains from scratch — a completely new generation.

The **SuccessorEngine** daemon checks two triggers every 24 hours:

| Trigger | Condition | Meaning |
|---------|-----------|---------|
| Knowledge growth | trained chunks ≥ 100,000 | Enough new knowledge for a fundamentally better model |
| Quality degradation | rolling avg feedback < 3.5/5.0 | Users rating answers poorly |

When either triggers:

1. Full corpus exported from trained SQLite rows to JSONL
2. GirivinityModel retrained from scratch on the full corpus
3. Perplexity evaluated on held-out sample
4. If new model perplexity is lower (better) than current:
   - Saved to `models/versions/{timestamp}/`
   - Automatically quantized to Q4_K_M GGUF at `models/versions/{timestamp}/model.gguf` for CPU inference
   - Notification written to `logs/admin_notifications.jsonl`
   - Status: `awaiting_admin_approval`
5. Admin calls `POST /admin/approve-successor/{version}`
6. `models/active` symlink updated → new model goes live

**The system never deploys itself without admin approval.**
Every generation upgrade requires one human confirmation.

Each successor model has measurably lower perplexity than its predecessor.
The improvement percentage is included in the notification.

---


## Model Generation Roadmap

| Generation | Model | Target Parameters | Architecture Direction |
|---|---|---:|---|
| 1 | Girivinity-3B | 3B | First production-scale native Girivinity generation |
| 2 | Girivinity-7B | 7B | Larger general-purpose model with stronger reasoning |
| 3 | Girivinity-13B | 13B | Higher-capacity reasoning and multilingual depth |
| 4 | Girivinity-34B | 34B | Large-scale expert generation for complex domains |
| 5 | Girivinity-70B | 70B | Last explicitly defined, hand-tuned generation |

### The Roadmap Never Ends

Generations 1–5 are explicitly defined with hand-tuned architectures.
Beyond Generation 5 (Girivinity-70B), the system automatically extrapolates
the next generation by doubling parameters and scaling the architecture proportionally.

There is no ceiling. Girivinity is designed to keep growing as long as
data and compute are available. Each generation is twice as capable as the last.

The question is never "when do we stop?" — it is always "what does the next generation need?"

---

## Repository Structure

The Python source tree below is generated from `find . -type f -name "*.py" | sort` after the root-level engine cleanup.

```text
girivinity/
├── agent_controller.py
├── app/__init__.py
├── app/agents/__init__.py
├── app/agents/controller.py
├── app/agents/self_critic.py
├── app/analytics/__init__.py
├── app/analytics/events.py
├── app/api/__init__.py
├── app/api/routes/admin.py
├── app/api/routes/agents.py
├── app/api/routes/chat.py
├── app/api/routes/cuda.py
├── app/api/routes/health.py
├── app/api/routes/rasp.py
├── app/api/routes/security.py
├── app/api/routes/skills.py
├── app/api/routes/tenant_security.py
├── app/core/__init__.py
├── app/core/agent_forge.py
├── app/core/agent_orchestrator.py
├── app/core/agent_registry.py
├── app/core/agent_runner.py
├── app/core/citation_engine.py
├── app/core/cognitive_engine.py
├── app/core/config.py
├── app/core/config_loader.py
├── app/core/config_schema.py
├── app/core/cuda_crawler.py
├── app/core/cuda_engine.py
├── app/core/db.py
├── app/core/domain_router.py
├── app/core/llm_synthesiser.py
├── app/core/memory_engine.py
├── app/core/migrations.py
├── app/core/query_router.py
├── app/core/self_trainer.py
├── app/core/sentiment_engine.py
├── app/core/skill_forge.py
├── app/core/social_engine.py
├── app/core/successor_engine.py
├── app/core/system_config.py
├── app/core/teaching_engine.py
├── app/core/truth_engine.py
├── app/core/web_intelligence.py
├── app/crawler/__init__.py
├── app/crawler/items.py
├── app/crawler/pipeline.py
├── app/crawler/pipelines.py
├── app/crawler/queue.py
├── app/crawler/runner.py
├── app/crawler/settings.py
├── app/crawler/spiders/__init__.py
├── app/crawler/spiders/scalable_spider.py
├── app/crawler/vector_db.py
├── app/engines/__init__.py
├── app/engines/analytics_engine.py
├── app/engines/code_intelligence_engine.py
├── app/engines/context_optimization_engine.py
├── app/engines/instruction_following_engine.py
├── app/engines/inter_model_protocol.py
├── app/engines/language_router.py
├── app/engines/response_planning_engine.py
├── app/engines/tool_selection_engine.py
├── app/engines/user_behavior_engine.py
├── app/finetune/__init__.py
├── app/finetune/continual_learning.py
├── app/finetune/dataset_builder.py
├── app/finetune/evaluate.py
├── app/finetune/lora_trainer.py
├── app/finetune/update_gate.py
├── app/llm/__init__.py
├── app/llm/girivinity_architecture.py
├── app/llm/girivinity_tokenizer.py
├── app/llm/loader.py
├── app/main.py
├── app/memory/__init__.py
├── app/memory/system.py
├── app/monitoring/__init__.py
├── app/monitoring/logging.py
├── app/monitoring/metrics.py
├── app/multimodal/__init__.py
├── app/multimodal/processor.py
├── app/profiling/__init__.py
├── app/profiling/user_profiler.py
├── app/rag/__init__.py
├── app/rag/system.py
├── app/rag/truth_verifier.py
├── app/reasoning_planner.py
├── app/security/__init__.py
├── app/security/ai_threat_reasoner.py
├── app/security/alignment.py
├── app/security/anomaly_scorer.py
├── app/security/cyber_shield.py
├── app/security/emergency_shutdown.py
├── app/security/jailbreak_classifier.py
├── app/security/layer.py
├── app/security/model_steal_detector.py
├── app/security/policy.py
├── app/security/policy_engine.py
├── app/security/rasp/__init__.py
├── app/security/rasp/hardware_monitor.py
├── app/security/rasp/process_guard.py
├── app/security/rasp/rasp_engine.py
├── app/security/rasp/runtime_interceptor.py
├── app/security/rasp/self_healer.py
├── app/security/rate_limiter.py
├── app/security/session_manager.py
├── app/security/tenant_security.py
├── app/security/threat_detector.py
├── app/security/training_poison_guard.py
├── app/training/__init__.py
├── app/training/benchmarking.py
├── app/training/model_evolution.py
├── app/training/pretrain.py
├── config_loader.py
├── core/__init__.py
├── core/query_router.py
├── core/self_trainer.py
├── core/successor_engine.py
├── core/truth_engine.py
├── core/web_intelligence.py
├── crawler_engine/__init__.py
├── crawler_engine/engine.py
├── knowledge_distillation_engine.py
├── llm_engine.py
├── llm_loader.py
├── model/                       # Legacy compatibility/export utilities; native LLM code lives in app/llm/
├── multimodal_engine.py
├── tests/test_agent_controller.py
├── tests/test_agent_mode.py
├── tests/test_ai_threat_reasoner.py
├── tests/test_alignment_layer.py
├── tests/test_analytics_engine.py
├── tests/test_architecture.py
├── tests/test_architecture_3b.py
├── tests/test_architecture_v2.py
├── tests/test_base_model.py
├── tests/test_benchmarking_system.py
├── tests/test_chat_endpoint.py
├── tests/test_citation_engine.py
├── tests/test_code_intelligence_engine.py
├── tests/test_cognitive_engine.py
├── tests/test_config.py
├── tests/test_config_loader.py
├── tests/test_context_optimization_engine.py
├── tests/test_continual_learning.py
├── tests/test_crawler_engine.py
├── tests/test_crawler_pipeline_integration.py
├── tests/test_crawler_queue.py
├── tests/test_cuda_engine.py
├── tests/test_dataset_builder.py
├── tests/test_db.py
├── tests/test_domain_router.py
├── tests/test_emergency_shutdown.py
├── tests/test_girivinity_model.py
├── tests/test_instruction_following_engine.py
├── tests/test_inter_model_protocol.py
├── tests/test_knowledge_distillation_engine.py
├── tests/test_language_router.py
├── tests/test_llm_engine.py
├── tests/test_llm_synthesiser.py
├── tests/test_memory_engine.py
├── tests/test_memory_system.py
├── tests/test_model_evolution.py
├── tests/test_model_inference.py
├── tests/test_multimodal_engine.py
├── tests/test_policy_engine.py
├── tests/test_project_structure.py
├── tests/test_query_router.py
├── tests/test_rag_system.py
├── tests/test_rasp.py
├── tests/test_reasoning_planner.py
├── tests/test_response_planning_engine.py
├── tests/test_security_layer.py
├── tests/test_security_policy.py
├── tests/test_self_critic.py
├── tests/test_self_trainer.py
├── tests/test_sentiment_engine.py
├── tests/test_skill_forge.py
├── tests/test_successor_engine.py
├── tests/test_teaching_engine.py
├── tests/test_threat_detector.py
├── tests/test_tool_selection_engine.py
├── tests/test_train.py
├── tests/test_training_pipeline.py
├── tests/test_training_poison_guard.py
├── tests/test_truth_engine.py
├── tests/test_truth_verifier.py
├── tests/test_user_behavior_engine.py
├── tests/test_user_profiler.py
├── tests/test_vector_pipeline.py
├── tests/test_web_intelligence.py
├── config.yaml
├── requirements.txt
├── pyproject.toml
├── Makefile
├── .github/DESCRIPTION.md
```

## Component Reference

### QueryRouter — `app/core/query_router.py`
The brain of every request. Embeds the query using
`all-MiniLM-L6-v2`, searches ChromaDB with cosine similarity.
Score ≥ 0.72 returns knowledge base chunks immediately.
Score < 0.72 delegates to WebIntelligence and fires self-training
in a background thread. Never blocks the response.

### WebIntelligence — `app/core/web_intelligence.py`
Searches DuckDuckGo (no API key needed), fetches top 5 URLs
via httpx, extracts clean article text via trafilatura,
chunks to 400 tokens with 50-token overlap, scores each chunk
for relevance to the original query, stores qualifying chunks
in the ChromaDB `pending_training` collection.

### SelfTrainer — `app/core/self_trainer.py`
Runs as a `multiprocessing.Process` daemon started at FastAPI
startup. Checks the SQLite queue every 30 minutes. When ≥50
pending chunks exist, formats them as instruction pairs,
runs LoRA fine-tuning via PEFT on the base model, saves the
new adapter, updates the `latest` symlink.
Aborts training if loss exceeds threshold and alerts admin.

### LLMSynthesiser — `app/core/llm_synthesiser.py`
Wraps the GirivinityEngine to produce natural language answers
from retrieved context. Uses a system prompt that constrains
the model to only use provided context — reducing hallucination
at the prompt level before TruthEngine verification.
Falls back to structured extraction if the model is not yet built.
Streaming responses append sources inline as they yield.

### TruthEngine — `app/core/truth_engine.py`
Post-generation verification layer. Splits every response into
individual factual claims, verifies each against ChromaDB
(similarity ≥ 0.70 = KB_SOURCED), checks against web sources
(WEB_SOURCED), or marks as UNVERIFIED. If >30% of claims are
unverified, prepends a disclaimer. Scores confidence 0.0–1.0.
Only appends citations for URLs that actually exist in the
current session's retrieved data — never fabricates sources.

### SuccessorEngine — `app/core/successor_engine.py`
24-hour daemon. Monitors two triggers: knowledge base chunk
count and rolling average feedback score. When triggered,
exports full training corpus, retrains GirivinityModel from
scratch, evaluates perplexity, and writes an admin notification
if the new model is better. Admin approves via API — the engine
never auto-deploys.

### GirivinityModel — `app/llm/girivinity_architecture.py`
Custom decoder-only transformer. PyTorch only — no HuggingFace.
Components: RMSNorm, RoPE, Grouped Query Attention (16 heads,
4 KV heads), SwiGLU FFN, weight-tied LM head. The model family starts at 3B parameters (Girivinity-3B), scaling to 70B and beyond through the structured generation roadmap. KV-cache supported for fast autoregressive
inference. Quantised to Q4_K_M GGUF via llama.cpp for CPU
inference on 4GB RAM.

---

## API Reference

### Chat

**POST /chat/message**
```json
Request:  { "query": "string", "user_id": "string", "stream": false }
Response: { "answer": "string", "source": "knowledge_base|web|none",
            "confidence": 0.0-1.0, "urls": ["string"] }
```

**POST /chat/message/stream**
Request:  { "query": "string", "user_id": "string" }
Response: text/plain streaming — tokens arrive as they are generated

### Admin

**GET /admin/notifications**
```json
Response: { "notifications": [
  { "type": "successor_ready", "version": "20240101_120000",
    "previous_version": "none", "improvement_percent": 12.5,
    "trained_on_chunks": 105000, "perplexity": 38.4,
    "quantization_status": "quantized",
    "timestamp": "ISO8601", "status": "awaiting_admin_approval" }
]}
```

**POST /admin/approve-successor/{version}**
```json
Response: { "status": "approved", "version": "20240101_120000" }
```

**POST /admin/reject-successor/{version}**
```json
Response: { "status": "rejected", "version": "20240101_120000" }
```

**GET /admin/model-versions**
```json
Response: { "versions": ["20240201_080000", "20240101_120000"] }
```

**POST /admin/feedback**
```json
Request:  { "user_id": "string", "score": 1.0-5.0 }
Response: { "status": "recorded" }
```

### Health

**GET /health** → `{ "status": "ok", "service": "girivinity" }`

**GET /health/deep** → `{ "status": "ok|degraded", "issues": [], "compute": { ... } }`

---

## Deployment Guide

### Requirements

| Resource | Minimum (CPU mode) | Recommended (GPU mode) |
|---|---|---|
| RAM | 4 GB | 8 GB |
| CPU | 2 cores | 4 cores |
| Storage | 20 GB | 40 GB |
| GPU | Not required | 6GB+ VRAM (RTX 3060 or better) |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

### GPU vs CPU Mode

Girivinity detects your hardware automatically at startup. You never need to change config.

| Operation | CPU Mode | GPU Mode |
|---|---|---|
| Inference | float32, slower | 4-bit NF4 quantized, 5-10x faster |
| LoRA fine-tuning | Runs, takes longer | AMP float16, much faster |
| Pretraining | Possible but slow | Recommended, use gradient checkpointing |
| Successor retraining | Hours to days | Minutes to hours |
| Min RAM / VRAM | 4GB RAM | 6GB VRAM |
| Cost | Free (any VPS) | ~$0.35/hr (Lambda Labs T4) |

**To force CPU mode** (e.g. for testing):
```yaml
# config.yaml
compute:
  device: "cpu"
```

**To verify which mode is active:**
```bash
curl http://localhost:8000/health/deep
# Returns: "compute": {"device": "cuda", "gpu_name": "...", "inference_mode": "4bit_nf4_gpu"}
# Or:      "compute": {"device": "cpu", "inference_mode": "float32_cpu"}
```

---

### Step 1 — Clone and install

```bash
git clone https://github.com/Mariner-rider/girivinity_
cd girivinity_
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import torch, chromadb, sentence_transformers; print('OK')"
```

---

### Step 2 — Create required directories

```bash
mkdir -p data/chroma data/training_queue logs \
         models/tokeniser models/base models/adapters \
         models/girivinity_quantised models/versions
```

---

### Step 3 — Create seed corpus

The tokeniser needs text to learn from. This seeds it:

```bash
cat > data/seed_corpus.txt << 'EOF'
Girivinity is an intelligent self-learning AI system built in India.
It retrieves information from the web and continuously improves itself.
Artificial intelligence is the simulation of human intelligence by machines.
Machine learning enables computers to learn from data without being programmed.
Natural language processing allows computers to understand human language.
Deep learning uses neural networks with many layers to learn complex patterns.
India is a country in South Asia with a population of over 1.4 billion people.
Technology is transforming every industry from healthcare to agriculture.
EOF
```

---

### Step 4 — Train the tokeniser

```bash
python app/llm/girivinity_tokenizer.py
```

Expected output:
Tokeniser saved to models/tokeniser/tokeniser.json

---

### Step 5 — Seed the training database

Before the model can be trained, the SQLite queue needs initial data:

```bash
python - << 'EOF'
import sqlite3, json, datetime
from pathlib import Path

db = Path("data/training.db")
db.parent.mkdir(exist_ok=True)
conn = sqlite3.connect(db)
conn.execute("""
    CREATE TABLE IF NOT EXISTS training_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL, chunk_text TEXT NOT NULL,
        url TEXT DEFAULT '', score REAL DEFAULT 0.0,
        timestamp TEXT NOT NULL, status TEXT DEFAULT 'pending'
    )
""")
seeds = [
    ("What is artificial intelligence",
     "Artificial intelligence is the simulation of human intelligence "
     "processes by machines, especially computer systems."),
    ("How does machine learning work",
     "Machine learning is a subset of AI that enables systems to learn "
     "and improve from experience without being explicitly programmed."),
    ("What is deep learning",
     "Deep learning is part of machine learning based on artificial "
     "neural networks with representation learning."),
]
ts = datetime.datetime.utcnow().isoformat()
for q, c in seeds:
    conn.execute(
        "INSERT INTO training_queue (query,chunk_text,url,score,timestamp,status)"
        " VALUES (?,?,?,?,?,?)", (q, c, "", 0.9, ts, "pending")
    )
conn.commit()
conn.close()
print(f"Seeded {len(seeds)} training records")
EOF
```

---

### Step 6 — Train the initial base model

This trains GirivinityModel from scratch. On CPU this takes time —
start it in a screen or tmux session:

```bash
screen -S girivinity-train
python -m app.training.pretrain --config config.yaml
```

Training logs appear every 100 steps. When complete:
Training complete. Model saved to models/base/final

Press `Ctrl+A, D` to detach from screen.

---

### Step 7 — Quantise to GGUF

Required for CPU inference. First clone llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make && cd ..
```

Then quantise:

```bash
python model/quantise.py \
  --weights models/base/final \
  --output models/girivinity_quantised \
  --quant Q4_K_M
```

Verify the output:
```bash
ls -lh models/girivinity_quantised/model.gguf
```

---

### Step 8 — Update config.yaml

Make sure these paths are set correctly:

```yaml
model:
  quantised_path: models/girivinity_quantised/model.gguf
  n_ctx: 4096
  n_threads: 3        # cpu_count - 1
  n_gpu_layers: 0     # 0 = CPU fallback for GGUF inference

training:
  queue_db: data/training.db

rag:
  chroma_path: data/chroma
```

---

### Step 9 — Start the server

```bash
# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Or with Makefile
make run
```

---

### Step 10 — Verify everything works

```bash
# Basic health
curl http://localhost:8000/health

# Deep health (checks ChromaDB + embedder)
curl http://localhost:8000/health/deep

# Send your first query
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?", "user_id": "test"}'
```

Expected response shape:
```json
{
  "answer": "...",
  "source": "knowledge_base",
  "confidence": 0.85,
  "urls": []
}
```

---

### Step 11 — Keep it running with systemd

```bash
sudo nano /etc/systemd/system/girivinity.service
```

Paste:
```ini
[Unit]
Description=Girivinity AI Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/girivinity_
ExecStart=/usr/bin/python3 -m uvicorn app.main:app \
          --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable girivinity
sudo systemctl start girivinity
sudo systemctl status girivinity
```

---

## Development Guide

### Run tests
```bash
pytest -q                          # all tests
pytest tests/test_architecture.py  # specific file
pytest -k "test_kb_hit"            # specific test
```

### Lint
```bash
ruff check .
```

### Watch self-training logs
```bash
tail -f logs/self_training.jsonl | python -m json.tool
```

### Watch for successor model notifications
```bash
tail -f logs/admin_notifications.jsonl | python -m json.tool
```

### Submit user feedback (to influence successor triggers)
```bash
curl -X POST http://localhost:8000/admin/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "score": 4.5}'
```

### Manually approve a successor model
```bash
# List available versions
curl http://localhost:8000/admin/model-versions

# Approve a specific version
curl -X POST http://localhost:8000/admin/approve-successor/20240101_120000
```

---

## Configuration Reference

All configuration lives in `config.yaml`. Key sections:

```yaml
modules:
  self_training:
    interval_seconds: 1800        # Check queue every 30 minutes
    chunk_threshold: 50           # Min chunks before LoRA fires
    base_model_path: models/base  # Where trained base model lives
    adapters_dir: models/adapters # Where LoRA adapters are saved
    epochs: 3
    learning_rate: 0.0002
    loss_abort_threshold: 2.0     # Abort LoRA if loss exceeds this

successor_engine:
  check_interval_seconds: 86400  # Check every 24 hours
  knowledge_base_threshold: 100000  # Chunks before full retrain
  quality_score_threshold: 3.5   # Avg feedback below this triggers
  versions_dir: models/versions
  notifications_path: logs/admin_notifications.jsonl

rag:
  chroma_path: data/chroma

model:
  # Set true after running app/training/pretrain.py to use Girivinity's own weights
  use_native_model: false
  # Set to 'girivinity-native' to use own trained model, or a HuggingFace model_id for the HF path
  model_id: girivinity-native
  architecture:
    vocab_size: 65536
    max_seq_len: 32768
    dim: 2048
    n_layers: 24
    n_heads: 16
    n_kv_heads: 4
```

---

## Serving the API to Others

Girivinity exposes a standard REST API. Any developer, app, or service
can connect to it once it is running.

### Public API (direct server)

If your server has a public IP, the API is immediately accessible:

```bash
# From any machine
curl -X POST http://YOUR_SERVER_IP:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "user_id": "user1"}'
```

### Add API Key Authentication

To protect your API before making it public, add a key check. In `app/main.py`,
add this middleware:

```python
from fastapi import Request, HTTPException

API_KEY = "your-secret-key-here"  # Move to .env in production

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path.startswith("/chat") or request.url.path.startswith("/admin"):
        key = request.headers.get("X-API-Key")
        if key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return await call_next(request)
```

Callers then pass the key:

```bash
curl -X POST http://YOUR_IP:8000/chat/message \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "user_id": "user1"}'
```

### Expose via Nginx (recommended for production)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Python SDK usage example

```python
import requests

class GirivinityClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    def ask(self, query: str, user_id: str = "default") -> dict:
        response = requests.post(
            f"{self.base_url}/chat/message",
            headers=self.headers,
            json={"query": query, "user_id": user_id}
        )
        return response.json()

# Usage
client = GirivinityClient("http://your-server:8000", "your-api-key")
result = client.ask("What is deep learning?")
print(result["answer"])
```

### Rate Limiting (optional)

Install slowapi:

```bash
pip install slowapi
```

Add to `app/main.py`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/chat/message")
@limiter.limit("10/minute")
async def chat(request: Request, ...):
    ...
```

---

## Roadmap

- [ ] Admin web dashboard (approve successors via UI)
- [ ] Hindi and regional Indian language support
- [ ] Girivinity-7B training run (Generation 2)
- [ ] Girivinity-13B training run (Generation 3)
- [ ] Multi-GPU training for successor models
- [ ] Streaming successor training progress via WebSocket
- [ ] Domain-specific knowledge packs (medical, legal, engineering)
- [ ] Mobile API client
- [ ] Rate limiting and user authentication layer
- [ ] Distributed ChromaDB for horizontal scaling

---

## License

Built in India. Open for the world.
