# Girivinity — Self-Learning AI System

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

1. [What Makes This Different](#what-makes-this-different)
2. [System Architecture](#system-architecture)
3. [How Self-Learning Works](#how-self-learning-works)
4. [How Successor Models Work](#how-successor-models-work)
5. [Repository Structure](#repository-structure)
6. [Component Reference](#component-reference)
7. [API Reference](#api-reference)
8. [Deployment Guide — Step by Step](#deployment-guide)
9. [Development Guide](#development-guide)
10. [Configuration Reference](#configuration-reference)
11. [Roadmap](#roadmap)

---

## What Makes This Different

Most AI products are wrappers. They call OpenAI or Anthropic and format
the response. Girivinity is not that.

**It owns its own model.** The GirivinityModel is a decoder-only transformer
built from scratch in PyTorch — RMSNorm, Rotary Position Embeddings, Grouped
Query Attention, SwiGLU feed-forward, weight-tied LM head. ~360M parameters.
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

**It runs on low compute.** Inference runs CPU-only on a 4GB VPS.
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
   - Notification written to `logs/admin_notifications.jsonl`
   - Status: `awaiting_admin_approval`
5. Admin calls `POST /admin/approve-successor/{version}`
6. `models/active` symlink updated → new model goes live

**The system never deploys itself without admin approval.**
Every generation upgrade requires one human confirmation.

Each successor model has measurably lower perplexity than its predecessor.
The improvement percentage is included in the notification.

---

## Repository Structure
girivinity/
│
├── app/                          # FastAPI application
│   ├── main.py                   # App entry point, startup wiring
│   ├── core/                     # Intelligence layer
│   │   ├── query_router.py       # Routes queries: KB or web
│   │   ├── web_intelligence.py   # Live web search + chunking
│   │   ├── self_trainer.py       # Continuous LoRA training daemon
│   │   ├── llm_synthesiser.py    # Context → answer via LLM
│   │   ├── truth_engine.py       # Claim verification + citations
│   │   ├── successor_engine.py   # Successor model creation daemon
│   │   ├── config_loader.py      # YAML config helpers
│   │   └── config_schema.py      # Config validation
│   └── api/
│       └── routes/
│           ├── chat.py           # POST /chat/message
│           ├── admin.py          # Admin successor management
│           └── health.py         # GET /health, /health/deep
│
├── model/                        # Custom transformer model
│   ├── architecture.py           # GirivinityModel (PyTorch from scratch)
│   ├── tokeniser.py              # BPE tokeniser trainer
│   ├── train.py                  # Full training script
│   └── quantise.py               # GGUF export for CPU inference
│
├── agent_controller.py           # Multi-agent orchestration
├── knowledge_distillation_engine.py
├── user_behavior_engine.py
├── analytics_engine.py
├── llm_loader.py                 # GGUF model loader (llama-cpp-python)
├── llm_engine.py                 # GirivinityEngine inference wrapper
├── language_router.py
├── context_optimization_engine.py
├── instruction_following_engine.py
├── multimodal_engine.py
├── response_planning_engine.py
├── tool_selection_engine.py
├── inter_model_protocol.py
│
├── tests/                        # Pytest test suite
│   ├── test_query_router.py
│   ├── test_web_intelligence.py
│   ├── test_self_trainer.py
│   ├── test_chat_endpoint.py
│   ├── test_llm_synthesiser.py
│   ├── test_truth_engine.py
│   ├── test_successor_engine.py
│   ├── test_architecture.py
│   └── test_train.py
│
├── data/                         # Runtime data (git-ignored)
│   ├── chroma/                   # ChromaDB vector store
│   ├── training_queue/           # JSONL training batches
│   └── seed_corpus.txt           # Bootstrap text
│
├── models/                       # Model files (git-ignored)
│   ├── tokeniser/                # Saved BPE tokeniser
│   ├── base/                     # Trained base model weights
│   ├── adapters/                 # LoRA adapters
│   │   └── latest -> {version}/  # Symlink to current adapter
│   ├── girivinity_quantised/     # GGUF model for inference
│   │   └── model.gguf
│   ├── versions/                 # Successor model versions
│   └── active -> {version}/      # Symlink to active model
│
├── logs/                         # Runtime logs (git-ignored)
│   ├── self_training.jsonl
│   ├── alerts.jsonl
│   └── admin_notifications.jsonl
│
├── config.yaml                   # All system configuration
├── requirements.txt
├── Makefile
└── docs/
└── DEPLOYMENT.md

---

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

### GirivinityModel — `model/architecture.py`
Custom decoder-only transformer. PyTorch only — no HuggingFace.
Components: RMSNorm, RoPE, Grouped Query Attention (16 heads,
4 KV heads), SwiGLU FFN, weight-tied LM head. ~360M parameters
at full config. KV-cache supported for fast autoregressive
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

**GET /health/deep** → `{ "status": "ok|degraded", "issues": [] }`

---

## Deployment Guide

### Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| CPU | 2 cores | 4 cores |
| Storage | 20 GB | 40 GB |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| GPU | Not required | Optional (speeds training) |

---

### Step 1 — Clone and install

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
python model/tokeniser.py
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
python model/train.py \
  --data data/training_queue \
  --tokeniser models/tokeniser/tokeniser.json \
  --output models/base \
  --epochs 3 \
  --batch 2 \
  --grad-accum 8
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
  n_gpu_layers: 0     # 0 = CPU only

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

### Watch for successor model notifications
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

architecture:
  dim: 1024
  n_layers: 16
  n_heads: 16
  n_kv_heads: 4
  vocab_size: 32000
  max_seq_len: 4096
```

---

## Roadmap

- [ ] Admin web dashboard (approve successors via UI)
- [ ] Hindi and regional Indian language support
- [ ] Multi-GPU training for successor models
- [ ] Streaming successor training progress via WebSocket
- [ ] Domain-specific knowledge packs (medical, legal, engineering)
- [ ] Mobile API client
- [ ] Rate limiting and user authentication layer
- [ ] Distributed ChromaDB for horizontal scaling

---

## License

Built in India. Open for the world.
