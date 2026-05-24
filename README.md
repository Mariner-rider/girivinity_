# Girivinity — Autonomous Self-Learning AI Platform

## What Girivinity Is
Girivinity is a FastAPI-based autonomous intelligence platform that unifies secure query handling, retrieval-augmented reasoning, adaptive response generation, memory, and continuous self-learning. It combines classic application engineering with model-centric workflows so the system can serve users now while steadily improving itself from observed usage.

It is different from standard chat wrappers because it is architected as a full stack: CyberShield at the HTTP boundary, RASP at runtime/hardware level, domain-aware routing, agent orchestration, skill generation, successor-model governance, and background learning pipelines. In practice, that means one deployment can answer queries, defend itself, improve over time, and manage lifecycle controls from one codebase.

Girivinity is built for India-first and global use: education, legal, finance, engineering, research, and enterprise workloads where trust, provenance, adaptability, and cost-aware operation matter. It is designed for developers, startups, institutions, and public-facing systems that need practical autonomy with operational guardrails.

## The Complete System Architecture
```text
                                 ┌──────────────────────────┐
                                 │      Clients / Apps      │
                                 └────────────┬─────────────┘
                                              │ HTTP
                           ┌──────────────────▼──────────────────┐
                           │      FastAPI (app/main.py)          │
                           │  Routers + Middleware + Startup     │
                           └───────┬───────────────┬──────────────┘
                                   │               │
                    ┌──────────────▼───────┐   ┌──▼────────────────────┐
                    │ CyberShield (HTTP)   │   │ RASP (Runtime/Host)   │
                    │ boundary protection   │   │ hardware/process/file │
                    └──────────────┬───────┘   │ interceptor/self-heal │
                                   │           └───────────┬───────────┘
                            ┌──────▼────────────────────────▼───────┐
                            │            Chat Ingress                │
                            │         /chat/message                  │
                            └──────────────┬─────────────────────────┘
                                           │
                     ┌─────────────────────▼─────────────────────┐
                     │ Threat Reasoner + Jailbreak Classifier    │
                     └─────────────────────┬─────────────────────┘
                                           │
                               ┌───────────▼───────────┐
                               │ Agent Request Check    │
                               └───────┬────────┬───────┘
                                       │yes     │no
         ┌─────────────────────────────▼──┐   ┌─▼────────────────────────────────┐
         │ Autonomous Agent Orchestrator  │   │ Sentiment + Social + Memory Recall│
         │ forge/reuse/adapt/run/save     │   └──────────────┬────────────────────┘
         └──────────────────────┬─────────┘                  │
                                │                            ▼
                                │                 QueryRouter (KB/Web)
                                │                            │
                                │                            ▼
                                │                    Cognitive enrichment
                                │                            │
                                │                            ▼
                                │             LLMSynthesiser + DomainRouter
                                │                            │
                                └──────────────┬─────────────┘
                                               ▼
                                   Truth/Citation/Grounding
                                               │
                                               ▼
                                          Response

Background daemons:
- SelfTrainer (LoRA/QLoRA updates)
- SuccessorEngine (generation roadmap)
- RASPEngine (runtime self-protection)
- SkillForge + memory/analytics async updates
```

## Core Intelligence Pipeline
1. Query ingress (`/chat/message`).
2. Security checks (AI threat + jailbreak).
3. Agent mode detection.
4. Sentiment profiling.
5. Social/user-model update.
6. Long-term memory recall.
7. Cognitive reasoning enrichment.
8. Retrieval pathing (knowledge base vs web).
9. Synthesis via LLM + domain prompts.
10. Truth/grounding/citation shaping.
11. Structured response + async learning hooks.

## Self-Learning Pipeline
1. Web retrieval and chunking.
2. Training poison guard screening.
3. Quality scoring and filtering.
4. Deduplication by content/hash identity.
5. Curriculum queue construction.
6. Replay batch preparation.
7. QLoRA/LoRA adapter training cycle.
8. Adapter promotion and continuous rollout.

## Autonomous Agent Mode
Autonomous Agent Mode lets users request specialized agents in natural language; the platform decides whether to forge new, reuse existing, or adapt prior agents, then runs and stores them.

### Agent types
- Research
- Code
- Data Analysis
- Monitoring
- Writing
- Legal Research
- Financial
- Teaching
- Security
- Custom

### Lifecycle
forge/reuse/adapt → run → learn → rest

## CyberShield — 7 Security Layers
1. Boundary middleware request inspection.
2. AI threat-intent reasoning.
3. Jailbreak and prompt-injection detection.
4. Threat-signature detection (SSRF/LFI/XSS/SQLi patterns).
5. Model extraction / abuse detection.
6. Training-data poisoning defense.
7. Security-mode escalation controls (observe/guard/contain/emergency).

## Intelligence Engines (4 layers)
1. **Cognitive**: reasoning framing and deliberation scaffolding.
2. **Sentiment**: tone and user-emotion adaptation.
3. **Social**: per-user interaction model and personalization.
4. **Memory**: recall/write of persistent user context.

## Knowledge Domains (25 domains)
| # | Domain | Description |
|---|---|---|
| 1 | cuda_kernels | CUDA kernels, GPU architecture, optimization |
| 2 | space_astronomy | ISRO/NASA, astrophysics, orbital systems |
| 3 | computer_science | Algorithms, systems, networking, compilers |
| 4 | three_d_design | 3D modeling, rendering, game pipelines |
| 5 | artificial_intelligence | ML, DL, LLMs, CV, NLP |
| 6 | medical_clinical | Clinical reasoning and medical concepts |
| 7 | indian_legal | BNS/BNSS/IPC and Indian legal contexts |
| 8 | international_law | Treaties, comparative law, global legal frameworks |
| 9 | business_strategy | Startup strategy, GTM, planning, growth |
|10 | accounting_finance | Tax, compliance, accounting, finance |
|11 | mathematics | Pure/applied mathematics and statistics |
|12 | education_pedagogy | Teaching workflows, exam prep, pedagogy |
|13 | research_academia | Literature review, methods, citation workflows |
|14 | history_geopolitics | Historical and geopolitical analysis |
|15 | economics | Macroeconomics, policy, markets |
|16 | cybersecurity_ops | SOC, detection engineering, DFIR |
|17 | devops_sre | CI/CD, reliability, observability |
|18 | cloud_architecture | Cloud design, governance, resilience |
|19 | data_engineering | ETL/ELT, warehouse, orchestration |
|20 | product_management | Discovery, roadmap, prioritization |
|21 | ux_hci | UX research, IA, interaction design |
|22 | public_policy | Governance and policy analysis |
|23 | agriculture_food | Agriculture systems and food value chains |
|24 | energy_climate | Energy systems and climate adaptation |
|25 | manufacturing_industry4 | Automation, industrial analytics, QA |

## Model Architecture — GirivinityModel
Custom decoder-only transformer, built from scratch in PyTorch.

| Component | Spec |
|-----------|------|
| Parameters | 2.983B (v1) → growing |
| Dimensions | 3072 |
| Layers | 28 |
| Attention heads | 24 (GQA, 8 KV heads) |
| Context window | 4096 tokens |
| FFN | SwiGLU, dim=8192 |
| Position encoding | RoPE (theta=500,000) |
| Normalisation | RMSNorm |
| KV sharing | Cross-layer from layer 14 (Gemma 4) |
| Per-layer embeddings | ple_dim=64 (Gemma 4) |
| Residual streams | mHC n=4 (DeepSeek V4) |
| Inference | Q4_K_M GGUF via llama-cpp-python |

## Generation Roadmap — The Model Grows Itself
| Generation | Parameters | Trigger | Description |
|-----------|-----------|---------|-------------|
| 1.0 | 3B | Deploy | Foundation model |
| 2.0 | 7B | 100,000 chunks | First Evolution |
| 3.0 | 13B | 500,000 chunks | Second Evolution |
| 4.0 | 30B | 2,000,000 chunks | Third Evolution |
| 5.0 | 70B | 10,000,000 chunks | Fourth Evolution |

Each generation is retrained from scratch on the full corpus.
Admin approval required before any generation goes live.

## API Reference
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/message` | POST | Send query, get answer |
| `/chat/message/stream` | POST | Streaming response |
| `/agents/run` | POST | Create and run agent |
| `/agents/list` | GET | List saved agents |
| `/agents/{id}` | GET | Get agent details |
| `/agents/{id}/run` | POST | Run existing agent |
| `/security/status` | GET | Security mode + stats |
| `/security/events` | GET | Recent security events |
| `/security/mode/{mode}` | POST | Set observe/guard/contain/emergency |
| `/security/rasp/status` | GET | RASP hardware status |
| `/security/rasp/hardware` | GET | Live hardware metrics |
| `/security/rasp/integrity-check` | GET | File integrity status |
| `/skills/` | GET | List all skills |
| `/skills/{slug}` | GET | Get skill details |
| `/cuda/generate` | POST | Generate CUDA kernel |
| `/cuda/benchmark` | POST | Benchmark kernel |
| `/admin/notifications` | GET | Successor model alerts |
| `/admin/approve-successor/{v}` | POST | Deploy new generation |
| `/admin/generation-roadmap` | GET | Growth progress |
| `/tenant/security/config` | GET/PUT | Per-customer security |
| `/health` | GET | Basic health |
| `/health/deep` | GET | ChromaDB + embedder |

## Deployment
### Requirements
- RAM: 4GB minimum (8GB recommended)
- CPU: 2+ cores
- Storage: 20GB
- OS: Ubuntu 22.04
- GPU: Optional (T4 for training, CPU for inference)

### Steps
```bash
git clone https://github.com/Mariner-rider/girivinity_
cd girivinity_
pip install -r requirements.txt
bash scripts/setup_postgres.sh
make train-tokeniser
make crawl-domains
make build-dataset
make train-model
make quantise
make run
```

### The model then:
- Starts serving at `http://0.0.0.0:8000`
- Begins self-learning from first query
- Runs RASP daemon (hardware monitoring)
- Runs SelfTrainer daemon (LoRA updates)
- Runs SuccessorEngine daemon (generation tracking)

## Repository Structure
```text
girivinity_/
├── app/
│   ├── main.py
│   ├── api/routes/
│   │   ├── chat.py
│   │   ├── agents.py
│   │   ├── rasp.py
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
│   │   └── migrations.py
│   ├── security/
│   │   ├── cyber_shield.py
│   │   ├── ai_threat_reasoner.py
│   │   ├── jailbreak_classifier.py
│   │   ├── threat_detector.py
│   │   ├── model_steal_detector.py
│   │   ├── training_poison_guard.py
│   │   └── rasp/
│   │       ├── hardware_monitor.py
│   │       ├── process_guard.py
│   │       ├── runtime_interceptor.py
│   │       ├── self_healer.py
│   │       └── rasp_engine.py
├── model/
├── tests/
├── config.yaml
└── README.md
```
