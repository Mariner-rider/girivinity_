# Girivinity — Autonomous Self-Learning AI Platform

## What Girivinity Is
Girivinity is a FastAPI-based AI platform that combines retrieval, reasoning, synthesis, security enforcement, memory, and continuous model improvement in one integrated runtime. A single user request can trigger security analysis, profile-aware response shaping, retrieval from local/web sources, grounded answer generation, and asynchronous learning updates.

What makes Girivinity different is that it is architected as a *system of collaborating engines* rather than a single chat wrapper. The platform includes a domain router, truth/citation stack, cyber-defense middleware, skill generation, CUDA code generation/benchmarking, successor-model governance, and autonomous agent orchestration that can create/reuse/adapt task agents on demand.

Girivinity is built for India-first and global usage: strong support for Indian legal/education/finance contexts, plus globally relevant research, engineering, and enterprise workflows. The goal is practical intelligence that is safer, auditable, and continuously improving for developers, researchers, students, founders, security teams, and institutions.

## The Complete System Architecture
```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                                CLIENTS                                      │
│  Web UI / API clients / internal services                                   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ HTTP
┌───────────────────────────────▼──────────────────────────────────────────────┐
│                           FASTAPI APPLICATION                               │
│  app/main.py                                                                 │
│  Routers: /chat /agents /security /skills /cuda /admin /health /tenant/*   │
│  Middleware: CyberShield                                                     │
└───────────────┬───────────────────────────────────────────────┬──────────────┘
                │                                               │
      ┌─────────▼─────────┐                           ┌────────▼────────┐
      │  CHAT PIPELINE     │                           │  AGENT MODE      │
      │  /chat/message     │                           │  /agents/*        │
      └─────────┬─────────┘                           └────────┬────────┘
                │                                              │
   Threat + Jailbreak checks                                   │
                │                                              │
   Agent request detection ────────────────yes─────────────────┘
                │ no
                ▼
      SentimentEngine + SocialEngine + MemoryEngine
                │
      QueryRouter ──► KB hit OR WebIntelligence fallback
                │
      CognitiveEngine context enrichment
                │
      LLMSynthesiser + DomainRouter prompt injection
                │
      Truth/Citation/Grounding controls
                │
             ChatResponse

Background / Async Planes:
- SelfTrainer queue + LoRA training
- SkillForge generation + evaluation
- Memory writes
- SuccessorEngine governance + model-version approvals
- Security telemetry and incident state
```

## Core Intelligence Pipeline
1. **Query ingress**: `/chat/message` receives query + user_id.
2. **Security first**: AI threat reasoner + jailbreak classifier evaluate request.
3. **Agent mode switch**: if query matches agent triggers, route to `AgentOrchestrator`.
4. **Sentiment analysis**: `SentimentEngine` infers tone/style needs.
5. **Social model update**: `SocialEngine` updates user-context profile.
6. **Memory recall**: `MemoryEngine` retrieves relevant long-term context.
7. **Cognitive framing**: `CognitiveEngine` adds reasoning guidance.
8. **Retrieval**: `QueryRouter` chooses KB/web path and gathers context/URLs.
9. **Synthesis**: `LLMSynthesiser` generates answer, domain-conditioned by `DomainRouter`.
10. **Truth-grounded response**: citation/trust controls shape final confidence and source set.
11. **Response + async learning hooks**: return answer; persist memory/analytics/training queue.

## Self-Learning Pipeline
Girivinity’s learning loop is orchestrated around retrieval outputs and post-response signals:

1. **Web acquisition**: `WebIntelligence` fetches, cleans, chunks, and scores web content.
2. **Poison guard**: `TrainingPoisonGuard` screens suspicious or adversarial training inputs.
3. **Quality scoring**: retrieval/chunk relevance scores rank candidate learning data.
4. **Deduplication**: chunk IDs/hash-based filtering prevent replaying duplicates.
5. **Curriculum queueing**: high-quality chunks are queued through `SelfTrainer`.
6. **Replay batches**: queued chunks are converted to trainable examples.
7. **QLoRA/LoRA training**: adapter fine-tuning runs with configured cadence/thresholds.
8. **Adapter promotion**: latest successful adapter becomes active (`models/adapters/latest`).

## Autonomous Agent Mode
Autonomous Agent Mode allows users to ask for specialized agents in natural language (e.g., “create an agent to monitor…”, “build an agent to research…”).

### Agent types
- research
- code
- data_analysis
- monitoring
- writing
- legal_research
- financial
- teaching
- security
- custom

### Lifecycle
1. **Forge / reuse / adapt** (`AgentOrchestrator` + `AgentForge` + `AgentRegistry`).
2. **Run** stepwise via `AgentRunner` and core tools.
3. **Learn** by pushing outputs/sources into `SelfTrainer`, `SkillForge`, `MemoryEngine`.
4. **Rest** with persisted status and usage metadata in registry storage.

## CyberShield — 7 Security Layers
1. **CyberShield middleware** — centralized request-time defense and policy enforcement.
2. **AI threat reasoning** — blocks high-risk intent and abuse patterns.
3. **Jailbreak classification** — blocks prompt-injection / override attempts.
4. **Threat detector** — flags exploit signatures (path traversal, SSRF patterns, etc.).
5. **Model-steal detector** — identifies extraction/probing behavior.
6. **Training poison guard** — blocks malicious/low-trust data from training pipeline.
7. **Security mode controls** — `/security` operations for incident mode and emergency response.

## Intelligence Engines (4 layers)
1. **Cognitive Engine**: adds structured reasoning scaffolding before synthesis.
2. **Sentiment Engine**: adapts tone, empathy, and communication style.
3. **Social Engine**: tracks user interaction profile for personalized context injection.
4. **Memory Engine**: recalls and persists user-specific conversational knowledge.

## Knowledge Domains (25 domains)
The current platform has **15 active router domains** and a **25-domain catalog target** for coverage scaling.

| # | Domain | Description | Status |
|---|---|---|---|
| 1 | cuda_kernels | CUDA kernels, NVCC, GPU optimization | Active |
| 2 | space_astronomy | ISRO/NASA/astrophysics/orbits | Active |
| 3 | computer_science | CS fundamentals, systems, networking | Active |
| 4 | three_d_design | 3D modeling/rendering/game engines | Active |
| 5 | artificial_intelligence | ML/DL/LLMs/CV/NLP | Active |
| 6 | medical_clinical | Clinical/diagnostic/medical concepts | Active |
| 7 | indian_legal | BNS/BNSS/IPC/case-law context | Active |
| 8 | international_law | Treaties, GDPR, comparative law | Active |
| 9 | business_strategy | Startups, GTM, funding, strategy | Active |
|10 | accounting_finance | Tax, audit, finance/compliance | Active |
|11 | mathematics | Core pure/applied mathematics | Active |
|12 | education_pedagogy | Teaching and learning workflows | Active |
|13 | research_academia | Papers, citations, methodology | Active |
|14 | history_geopolitics | Historical/geopolitical analysis | Active |
|15 | economics | Macro/micro/policy/economic data | Active |
|16 | cybersecurity_ops | SOC/DFIR/defensive operations | Planned |
|17 | devops_sre | CI/CD, reliability, observability | Planned |
|18 | cloud_architecture | Multi-cloud design and governance | Planned |
|19 | data_engineering | Pipelines, warehousing, ETL/ELT | Planned |
|20 | product_management | Discovery, roadmaps, prioritization | Planned |
|21 | ux_hci | UX research, IA, interaction design | Planned |
|22 | public_policy | Policy analysis and governance | Planned |
|23 | agriculture_food | Agri systems, crop/food value chains | Planned |
|24 | energy_climate | Energy systems, climate adaptation | Planned |
|25 | manufacturing_industry4 | Automation, QA, industrial analytics | Planned |

## Model Architecture
Girivinity uses a custom `GirivinityModel` family with roadmap scaling (current config is transformer-based with LoRA adapter workflows). The architecture notes include three practical improvements:

1. **KV sharing**: shares key/value structures efficiently to reduce memory pressure and improve inference throughput.
2. **PLE (Progressive Layer Efficiency)**: emphasizes compute where it yields the largest quality gain, improving cost-quality balance.
3. **mHC (multi-head calibration)**: calibrates attention heads for more stable reasoning and better cross-domain retention.

In plain language: these optimizations aim to make the model *faster, cheaper, and more consistent* as it grows.

## Generation Roadmap
| Generation | Target Size | Chunk Threshold Trigger | Goal |
|---|---:|---:|---|
| G1 | 3B | 100k trained chunks | Stable base + autonomous learning |
| G2 | 7B | 300k trained chunks | Better reasoning depth |
| G3 | 13B | 700k trained chunks | Strong multi-domain expert quality |
| G4 | 30B | 2M trained chunks | Enterprise-grade complex planning |
| G5 | 70B | 5M trained chunks | Frontier-scale autonomous platform |

## API Reference
### Chat

### Agents
- `POST /agents/run`
- `GET /agents/list`
- `GET /agents/{agent_id}`
- `DELETE /agents/{agent_id}`
- `POST /agents/{agent_id}/run`

### Security
- `GET /security/status`
- `GET /security/events`
- `POST /security/mode/{mode}`
- `POST /security/emergency/resolve`
- `GET /security/threat-summary`

### Skills
- `GET /skills/`
- `GET /skills/{slug}`
- `POST /skills/feedback`
- `POST /skills/{slug}/evaluate`
- `DELETE /skills/{slug}`

### CUDA
- `POST /cuda/generate`
- `POST /cuda/benchmark`
- `GET /cuda/types`
- `POST /cuda/bootstrap`

### Admin
- `GET /admin/notifications`
- `POST /admin/approve-successor/{version}`
- `POST /admin/reject-successor/{version}`
- `GET /admin/model-versions`
- `POST /admin/feedback`
- `GET /admin/generation-roadmap`

### Health
- `GET /health`
- `GET /health/deep`
- `GET /metrics`
- `GET /modules`

### Tenant security
- `GET /tenant/security/config`
- `PUT /tenant/security/config`

## Deployment Guide
1. **Clone**
   ```bash
   git clone <your-repo-url>
   cd girivinity_
   ```
2. **Create environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -U pip
   pip install -e .
   ```
4. **Configure**
   - Edit `config.yaml` (model, training, security, agent_mode, database).
5. **Run migrations / setup services**
   - Ensure Postgres and configured stores are available.
6. **Start app**
   ```bash
   make run
   ```
7. **Verify endpoints**
   - `GET /health`
   - `GET /health/deep`
   - `GET /modules`

## Repository Structure
```text
girivinity_/
├── app/
│   ├── main.py
│   ├── api/routes/
│   │   ├── chat.py
│   │   ├── agents.py
│   │   ├── security.py
│   │   ├── tenant_security.py
│   │   ├── skills.py
│   │   ├── cuda.py
│   │   ├── admin.py
│   │   └── health.py
│   ├── core/
│   │   ├── agent_forge.py
│   │   ├── agent_registry.py
│   │   ├── agent_runner.py
│   │   ├── agent_orchestrator.py
│   │   ├── query_router.py
│   │   ├── web_intelligence.py
│   │   ├── llm_synthesiser.py
│   │   ├── truth_engine.py
│   │   ├── citation_engine.py
│   │   ├── domain_router.py
│   │   ├── self_trainer.py
│   │   ├── successor_engine.py
│   │   ├── skill_forge.py
│   │   ├── memory_engine.py
│   │   ├── sentiment_engine.py
│   │   ├── social_engine.py
│   │   └── cognitive_engine.py
│   ├── security/
│   │   ├── cyber_shield.py
│   │   ├── ai_threat_reasoner.py
│   │   ├── jailbreak_classifier.py
│   │   ├── threat_detector.py
│   │   ├── model_steal_detector.py
│   │   └── training_poison_guard.py
│   └── monitoring/
├── model/
│   ├── architecture.py
│   ├── inference.py
│   ├── tokeniser.py
│   ├── training_pipeline.py
│   ├── train.py
│   ├── domain_trainer.py
│   └── quantise.py
├── tests/
├── docs/
├── config.yaml
├── Makefile
└── README.md
```

## What Makes Girivinity Different
| Capability | Girivinity | GPT-4 (hosted) | Claude (hosted) | Gemini (hosted) |
|---|---|---|---|---|
| Self-hostable full stack | Yes | No (closed hosted) | No (closed hosted) | No (closed hosted) |
| Autonomous agent forge/reuse/adapt | Yes | Tool-dependent | Tool-dependent | Tool-dependent |
| Built-in training queue + adapter updates | Yes | Not user-owned | Not user-owned | Not user-owned |
| Security-mode APIs and incident controls | Yes | Limited app-side only | Limited app-side only | Limited app-side only |
| Domain-router + India-specific legal/education emphasis | Yes | General | General | General |
| Successor model governance endpoints | Yes | No | No | No |

