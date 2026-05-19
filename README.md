# Girivinity

Girivinity is a self-learning AI platform with retrieval, synthesis, security checks, memory, analytics, and now **Autonomous Agent Mode**.

## Highlights
- Chat pipeline with retrieval + synthesis + memory.
- Continuous self-training pipeline (`SelfTrainer`) from live usage data.
- Skill generation (`SkillForge`) from useful learned chunks.
- Security layer (threat reasoning + jailbreak classification).
- **Autonomous Agent Mode**: create, run, save, reuse, adapt, and learn from custom agents.

## Autonomous Agent Mode

### What it does
Users can ask for an agent in natural language, such as:
- “Create an agent to research EV battery trends.”
- “Build an agent that tracks legal updates on Section 302.”
- “Make an agent to generate Python scripts for CSV cleanup.”

The platform will:
1. Detect that it is an agent request.
2. Classify the request type (research, code, legal, finance, etc.).
3. Reuse/adapt a similar agent if one exists.
4. Otherwise forge a new agent definition.
5. Run the agent step-by-step.
6. Save it for future reuse.
7. Feed the result back into the self-learning pipeline.

### Core components
- `app/core/agent_forge.py` — request classification + agent creation/adaptation.
- `app/core/agent_registry.py` — persistent storage and similarity-based reuse.
- `app/core/agent_runner.py` — step execution using Girivinity tools.
- `app/core/agent_orchestrator.py` — create/reuse/adapt decision flow.
- `app/api/routes/agents.py` — REST API for run/list/get/delete/re-run.

### Agent lifecycle
`ready → running → resting`

### Learning loop integration
After each agent run:
- Result chunks are queued to `SelfTrainer` for future LoRA updates.
- `SkillForge` generates new skills from useful outputs.
- `MemoryEngine` stores summarised execution traces.

## API

### Chat
- `POST /chat/message`
- `POST /chat/message/stream`

If chat query looks like an agent request, chat auto-routes to agent orchestration and returns agent output.

### Agents
- `POST /agents/run` — run/create/reuse/adapt agent from natural-language request.
- `GET /agents/list` — list saved agents.
- `GET /agents/{agent_id}` — fetch one agent definition.
- `DELETE /agents/{agent_id}` — delete saved agent.
- `POST /agents/{agent_id}/run` — run a saved agent again.

## Configuration

Add/update in `config.yaml`:

```yaml
agent_mode:
  agents_dir: data/agents
  similarity_threshold: 0.5
  max_steps_per_agent: 10
  learn_from_results: true
```

## Development

### Run checks
```bash
python -m py_compile app/core/agent_forge.py app/core/agent_registry.py app/core/agent_runner.py app/core/agent_orchestrator.py app/api/routes/agents.py
ruff check app/core/agent_forge.py app/core/agent_registry.py app/core/agent_runner.py app/core/agent_orchestrator.py app/api/routes/agents.py
pytest tests/test_agent_mode.py -v
```

## Project structure (relevant)
- `app/main.py` — app setup and router wiring.
- `app/api/routes/chat.py` — chat orchestration path.
- `app/api/routes/agents.py` — agent endpoints.
- `app/core/*` — retrieval, synthesis, learning, and autonomous agent components.
- `tests/test_agent_mode.py` — autonomous agent mode tests.
