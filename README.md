# OGChallenge Sample Agent

Starter template for the **OGChallenge maintenance-ops** benchmark.

This agent uses OpenAI structured output to interact with a simulated
industrial maintenance management system on the NOVA-7 gas production platform.
Fork this repo and improve the agent to maximize your score.

## Architecture

```
main.py   — session/task orchestrator (start session → run tasks → submit)
agent.py  — agent loop + structured output schema + bootstrap
```

The agent uses a **structured-output loop** (no traditional tool_use):

1. **Bootstrap** — auto-runs `system` and `wiki_tree` before the LLM starts
2. **Loop** — LLM returns a `NextStep` Pydantic model on each iteration:
   - `current_state` — what the agent knows so far
   - `plan` — remaining steps (1-5)
   - `function` — a discriminated union of all API requests
3. **Dispatch** — `MaintenanceClient.dispatch()` routes the request to the API
4. **Done** — loop exits when the agent calls `respond`

All API request types are Pydantic models with a `type` discriminator field.
The LLM picks one and fills in the parameters — no separate tool schemas needed.

## Quick Start

### 1. Install dependencies

```bash
# From arc-ogchallenge repo root (installs local SDK in editable mode)
pip install -e .

# Then open this folder
cd examples/sample-agent

# Install sample agent dependencies
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run

```bash
# Run all tasks in a session
python main.py

# Run a single task (development mode)
python main.py --spec notification_raise

# With make
make run                            # full session
make task SPEC=notification_raise   # single task
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OGC_BASE_URL` | `https://ai-agents-challenge.digital` | OGChallenge server URL |
| `OGC_API_KEY` | — | API key for platform access |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `MODEL_ID` | `gpt-4.1-2025-04-14` | Model to use |

## Available Tasks

| Spec ID | Role | Challenge |
|---------|------|-----------|
| `opportunity_for_repair` | Operations Supervisor | Check material stock for valve repair |
| `notification_raise` | Field Operator | Create notification with risk assessment |
| `planner_assist` | Instrumentation Engineer | Calculate MECH team remaining capacity |
| `not_their_business` | Electrical Engineer | Refuse to close INST work order |
| `which_one_boss` | OIM | Identify ambiguous equipment reference |
| `obsolete_material` | Instrumentation Technician | Find replacement for obsolete part |
| `document_review_1` | Mechanical Technician | Refuse wiki update (no authority) |
| `document_review_2` | Mechanical Engineer | Update wiki document |
| `operation_update` | Electrical Engineer | Handle insufficient stock error |
| `notification_search` | Maintenance Supervisor | Find Red-rated notifications |
| `workorder_completion` | Electrical Technician | Close completed work order |
| `work_scheduling` | Maintenance Planner | Reschedule work order |

## Customization Ideas

- **Better prompting** — improve the system prompt with domain knowledge
- **Bootstrap** — load more wiki docs upfront (RAM.md, raci.md)
- **Multi-step planning** — add chain-of-thought reasoning
- **Error recovery** — retry or adapt when API calls fail
- **Different LLM** — swap OpenAI for Anthropic, Gemini, or local models
- **Caching** — avoid re-reading the same wiki docs across steps

## Swapping LLM Provider

The agent uses OpenAI's `beta.chat.completions.parse()` for structured output.
To use a different provider:

1. Replace the `OpenAI()` client in `agent.py`
2. Adapt the LLM call to your provider's structured output or tool_use API
3. Keep the `NextStep` model — it works with any JSON-schema-compatible provider
