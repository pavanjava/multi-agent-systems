# Multi-Agent Systems Workshop

A hands-on collection of Python examples for learning how to build AI agents — from a single agent following instructions, all the way up to teams of agents and multi-step workflows with memory, tools, and human oversight.

Everything here is built on [**agno**](https://github.com/agno-agi/agno), a Python framework for building agents, teams of agents, and agentic workflows.

## Who this is for

Students working through a multi-agent systems workshop who want runnable, progressively more advanced examples rather than slides. Read the folders in order — each one builds on ideas from the last.

## How the repo is organized

```
basic_agents/              1. A single agent: instructions, and pausing for human input
agents_with_memory/        2. Giving an agent short-term + long-term memory
agents_with_tools_and_mcp/ 3. Tools, MCP servers, and approval workflows
multi_agent_teams/         4. Multiple agents collaborating as a team
multi_agent_workflows/     5. Multi-step pipelines (sequential, parallel, conditional)
semantic_memory/           Shared building blocks: vector search & memory storage
data/                      Sample PDFs / datasets used by the RAG examples
```

### 1. `basic_agents/` — Start here
- **`agent_with_instructions.py`** — The simplest possible agent: give it instructions, ask a question, stream the response. This is the "Hello World" of agno.
- **`agent_with_hil.py`** — A travel-planning agent that pauses mid-conversation to ask *you* clarifying questions (human-in-the-loop), then resumes once you answer.

### 2. `agents_with_memory/`
- **`medical_agent_with_memory.py`** — A research agent that breaks a question into sub-questions, only searches trusted medical sources, and remembers things about you across sessions using Redis (short-term) and Qdrant (long-term) memory.

### 3. `agents_with_tools_and_mcp/`
- **`agent_with_approvals.py`** — Shows the full lifecycle of a tool call that requires human approval before it's allowed to run: pause → review → approve/reject → resume.
- **`agent_with_pg_mcp_and_hil.py`** — Connects an agent to a live Postgres database through an **MCP** (Model Context Protocol) server, and requires your confirmation before it runs any schema-inspecting tool.

### 4. `multi_agent_teams/`
- **`delegation_in_teams.py`** — Two agents (a Reddit researcher and a Hacker News researcher) discuss a topic as a team until they reach consensus.
- **`coordinated_reasoning_rag.py`** — A four-agent team that retrieves information from a document knowledge base (RAG), reasons over it step by step, and cross-checks its own evidence before answering.
- **`file_system_context_provider.py`** — Instead of hardcoding an agent's tools/instructions, this loads them dynamically from files on disk.
- **`tool_call_compression_with_manager.py`** — A research team that automatically summarizes old tool results so long conversations don't blow up the context window.

### 5. `multi_agent_workflows/`
- **`clinical_diagnostic_support.py`** — A sequential pipeline: gather input → research team investigates → a diagnostic agent writes a report → the result is saved to long-term memory.
- **`financial_and_risk_advisory_workflow.py`** — A workflow that conditionally runs different analysis steps (fundamentals, news, risk) depending on what the user actually asked.
- **`legal_advisory_workflow.py`** — Runs two research agents *in parallel*, then feeds their combined output into sequential analysis and compliance-review steps.
- **`ltm_recall_test.py`** — A test script that checks whether an agent actually remembers something from a previous session by rephrasing an earlier question.

### `semantic_memory/` — Shared infrastructure
The reusable memory layer that the examples above depend on:
- **`qdrant_db.py`** — Hybrid (dense + sparse) vector search over Qdrant.
- **`memory_util.py`** — Factories for short-term (Redis) and long-term (Qdrant) memory.
- **`ltm_tools.py`** — Wraps long-term memory as tools an agent can call directly.

## Prerequisites

- Python **3.13+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management (this repo uses `pyproject.toml` + `uv.lock`)
- Running instances of the services the examples connect to, depending on which script you run:
  - **Redis** (short-term memory)
  - **Qdrant** (long-term / vector memory)
  - **PostgreSQL** (used by some workflow and MCP examples)
- API keys for the model/tool providers you plan to use (Google Gemini, OpenAI, Tavily, etc.)

## Setup

```bash
uv sync
```

Create a `.env` file in the project root with the keys the examples need. At minimum:

```bash
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
TAVILY_API_KEY=...
DATABASE_URL=...
REDIS_URL=...
QDRANT_URL=...
QDRANT_API_KEY=...
```

Some memory-related scripts use additional `LTM_*` and `TEMPORAL_MEMORY_*` variables — check the top of the script you're running for the exact names it expects.

## Running an example

```bash
uv run python basic_agents/agent_with_instructions.py
```

Swap in the path to any other script in the same way. Read the file itself first — most are short and heavily illustrate one specific concept.

## Suggested learning path

1. `basic_agents/agent_with_instructions.py` — understand the core `Agent` API
2. `basic_agents/agent_with_hil.py` — add human-in-the-loop
3. `agents_with_memory/medical_agent_with_memory.py` — add memory
4. `agents_with_tools_and_mcp/agent_with_approvals.py` and `agent_with_pg_mcp_and_hil.py` — add tools, MCP, and approvals
5. `multi_agent_teams/*` — move from one agent to a team
6. `multi_agent_workflows/*` — chain teams/agents into multi-step pipelines
