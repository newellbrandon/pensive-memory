# Pensive — LangGraph + MongoDB Demo

A minimal AI agent demo using **LangGraph** for workflow orchestration, **MongoDB** for short-term conversation state, and the **Atlas Agent Engine memory service** (`agent-engine-sdk-memory`) for long-term memory:

- **Short-term memory** — LangGraph checkpoints (`MongoDBSaver`) persist conversation state per thread
- **Long-term memory** — the Atlas Agent Engine memory service: Pensive records turns with `Memory.record_turn` and distils durable semantic/episodic/taxonomic/procedural facts itself (the service's own extraction only runs for turns its runtime records), then assembles prompt context with `Memory.build_context`

The API is **OpenAI-compatible** so you can connect [OpenWebUI](https://docs.openwebui.com/) or a coding harness (e.g. OpenCode) and chat with the agent. Workflow decisions render as native reasoning **Thought** blocks; the assistant message contains only the final answer.

## Architecture

### System overview

OpenWebUI or a coding harness sends chat requests to the FastAPI layer, which invokes a compiled LangGraph workflow. MongoDB holds short-term thread state (checkpointer); long-term memory is the **Atlas Agent Engine memory service**, reached through the `agent-engine-sdk-memory` SDK. On top of the core chat loop, Pensive runs a **server-tool layer** (subagent delegation, project browsing, Glean enterprise search), an **agent lab** (isolated, runnable/rerunnable agents), a **document knowledge base** (contextualized chunk embeddings — agents ground answers in uploaded PDFs via `documents_search` on demand), and a **management console** at `/ui` — with every agent workflow recorded as sortable, metric-bearing run history.

Pensive exposes both OpenAI-compatible protocols: **Chat Completions** (`POST /v1/chat/completions`) and the **Responses API** (`POST /v1/responses`). OpenWebUI can be pointed at either connection type.

```mermaid
flowchart TD
  client[OpenWebUI / coding harness] -->|"POST /v1/chat/completions · /v1/responses"| api[FastAPI]
  api -->|"ainvoke / astream"| lg[Compiled LangGraph]
  lg --> llm["OpenAI-compatible LLM"]
  lg -.->|"MongoDBSaver · every superstep"| checkpoints["checkpoints<br/>checkpoint_writes"]
  lg -->|"Memory.record_turn · Memory.build_context (SDK)"| memsvc[("Atlas Agent Engine<br/>memory service")]

  lg -.->|"pensive_subagent · digest only"| sub["Isolated subagent<br/>memory agent:sub-* · own workspace"]
  sub -->|"runs the same graph"| lg
  lg -.->|"glean_* tools · MCP + OAuth2"| glean[("Glean enterprise search")]
  lg -.->|"project_list / project_read"| projects[("~/projects · read-only")]
  lg -.->|"documents_search · on demand"| dstore[("document_chunks<br/>voyage-context-4 · 1024d")]

  subgraph console["Pensive Console — /ui"]
    ui["Memories · Agents lab · Documents · Glean · System Prompt"]
  end
  ui -->|"run_agent · same graph"| lg
  ui -->|"records executed workflows"| runs[("pensive_agent_runs<br/>pensive/test tags · metrics · 1500/30d retention")]
  ui -->|"upload → extract → embed (contextualized)"| dstore
  api --> ui
```

### LangGraph workflow

Every chat message — and every agent-lab/subagent run — runs the four nodes below. The graph also carries a **server-tool loop inside `generate_response`** (offered only in main-context chats): the model may delegate via `pensive_subagent`, browse host projects (`project_list` / `project_read`), or call the discovered Glean tools (`glean_*`); results are executed server-side and fed back until the model produces a final answer. OpenWebUI streams the **decision** from each step before the final answer — the second line in each node label matches what you see in the demo screenshots.

The checkpointer persists graph state after every superstep (not only these four nodes), which is why a single message produces multiple `checkpoints` and `checkpoint_writes` documents.

```mermaid
flowchart TD
  startNode((START)) --> analyze["analyze_intent<br/>Classify Intent"]

  analyze -->|"needs_memory"| search["search_long_term_memory<br/>Search Long Term Memory"]
  analyze -->|"skip"| generate["generate_response<br/>Generate Response"]

  search -->|"Memory.build_context · long-term sources"| memsvc[("Agent Engine<br/>memory service")]
  search --> generate

  generate -.->|"OpenAI Responses API"| llmNode[["LLM"]]
  generate -.->|"server-tool loop: pensive_subagent · project_* · glean_*"| serverTools["server tools<br/>budgeted · results fed back"]
  generate --> save["save_memory<br/>Save Memory"]

  save -->|"Memory.record_turn (user + assistant)"| memsvc
  save --> endNode((END))

  agentLab["Agents lab — run_agent<br/>memory agent:&lt;id&gt; scope"] -->|"same graph"| startNode
  agentLab -.->|"workflow history"| runs2[("pensive_agent_runs<br/>pensive/test · tokens · duration")]
  runs2 --> agentLab

  analyze -.-> cpNote["checkpoints / checkpoint_writes"]
  search -.-> cpNote
  generate -.-> cpNote
  save -.-> cpNote
```

**Demo paths** (see [Demo Results](#demo-results-openwebui)):

| Step | Session 1 — "My name is Brandon." | Session 2 — "Do you remember me?" |
| ---- | ----------------------------------- | --------------------------------- |
| Classify Intent | `memory` — personal context | `memory` — recall |
| Search Long Term Memory | `0 results` | `1 results` |
| Generate Response | `respond` — no retrieved context | `respond` — uses 1 memory |
| Save Memory | `saved` — turns recorded | `saved` — turns recorded |

## Coding-Harness / Persistent-Memory Context

Pensive can serve as an OpenAI-compatible model for a coding harness (e.g. OpenCode)
while providing persistent memory for important context:

- **Full context is honored.** Every incoming request's messages (system prompt,
  conversation history, tool results) are passed through to the model; Pensive does
  not discard anything the client sends.
- **Tool calling works.** Tool definitions from the client are forwarded to the
  model; tool calls the model requests are relayed back on the native channel
  (`message.tool_calls` / `delta.tool_calls` on `/v1/chat/completions`,
  `function_call` output items / `response.function_call_arguments.delta` on
  `/v1/responses`), and tool results replay as `function_call_output` items — so
  a coding harness can drive read/write/execute tools through Pensive.
- **Persistent memory runs on top.** Every turn is recorded to the Atlas Agent
  Engine memory service (scoped by `user_id` and session), and Pensive distils
  durable facts from it with its own LLM; each turn's prompt is grounded with
  `build_context` over those facts, so they carry across sessions, chats, and
  workflows.
- **Subagent delegation keeps the primary context clean.** Main-context chats are
  offered a server-side `pensive_subagent` tool: the model can delegate a
  self-contained task (research, builds, long tool chains) to an isolated
  subagent with its own memory scope (`agent:sub-…`), workspace, and tools. Only
  a short digest of the subagent's final answer returns to the conversation (at
  most 2 calls per turn, no nesting — subagents never see the delegation tool).
  Full artifacts land in the subagent's workspace and run history. The tool is
  named `pensive_subagent` on purpose: a coding harness (e.g. OpenCode) may offer
  its **own** `subagent` tool, whose calls Pensive relays back to the client. When
  a client offers one, Pensive adds a one-line system reminder so the model uses
  `pensive_subagent` for in-Pensive delegation.
- **Read access to your projects (incl. the OKF bundle).** The host's
  `~/projects` directory (or `PROJECTS_HOST_DIR`) is mounted read-only into the
  container at `/projects`, and main-context chats plus all agents get
  `project_list` / `project_read` tools (path-escape-safe, capped at 40k chars
  per read, disabled when `PENSIVE_PROJECTS_ROOT` is unset). The OKF knowledge
  bundle lives under `okf/`.
- **Glean enterprise search (MCP, no agentengine).** Pensive connects to the
  Glean MCP server (`GLEAN_MCP_URL`) with a native OAuth2 dynamic-client flow
  (DCR + PKCE; the `MCP` scope is required by Glean) and a bind-mounted token
  cache. Run `python scripts/glean_auth.py` once — or use the console's
  **Glean** tab (Connect button, the OAuth callback rides the existing
  `localhost:8000` port) — and the ~13 discovered Glean tools become
  server-side tools in main chats (`glean_search`, `glean_chat`,
  `glean_read_document`, …). The discovered tools are also appended to the
  tool set of **every agent run — lab agents and spawned subagents alike**
  (and the `web_search` description steers to `glean_search` for internal
  content), so "search exclusively via Glean" tasks never fall back to the
  public web tools. Calls are permission-scoped to the signed-in
  user. **Boundaries:** this is a single-user local-dev credential — never
  ship or upload the token cache (` .glean-tokens.json`).
- Connect any OpenAI chat/responses client to `http://<host>:8000/v1` with model
  `pensive` (see [OpenWebUI Configuration](#openwebui-configuration) and the OpenCode
  provider config in the OpenCode docs).

## Pensive Console (Web UI)

The management UI lives at **`http://<host>:8000/ui`** (MongoDB visual theme — dark navy
canvas, mongo-green accents) and powers five tabs:

| Tab | What you can do |
| --- | --------------- |
| **Memories** | Search a user's memory in the **Atlas Agent Engine memory service** and curate it. Pick a user and type a query — the SDK's ranked search (`Memory.search`) returns the relevant **Semantic** (user facts and preferences), **Episodic** (session events, decisions, outcomes), **Taxonomic** (domain vocabulary and reusable concepts like corporate policies), and **Procedural** (reusable workflows and agent runbooks — skills) memories with scores. **Semantic** facts are editable in place and **re-embedded automatically** on save (`Memory.save_semantic` upserts by label). There is no manual delete or merge: the memory service versions and supersedes memories itself as it extracts them in the background. |
| **System Prompt** | View and change the assistant persona injected at the top of every main-context request. Persisted in MongoDB; leave empty to restore the built-in default. |
| **Agents** | Deploy a **test agent** with its own system prompt and tools, then run a task in **complete isolation** — memory is scoped to `agent:<id>` so it can't pollute (or be polluted by) the main Pensive context. The tab is split into three segmented views: **Test Agents** (create form + registry; leave the name blank to **auto-generate** one like `coral-scribe-3621`, and tools empty for the built-in sandboxed set `write_file`, `append_file`, `read_file`, `run_command`, `web_search`, `web_extract`, `documents_search`), **Test Agent History** (every lab-agent run, locked to `test`, sortable), and **Agent Workflows** (every workflow incl. `pensive` subagent runs — aligned Source · Agent · When · Tokens · Tool calls · Runtime columns with clickable filter chips and sort chips). Clicking any row expands the full task, answer, step transcripts, a **Reuse as specialist agent** action, and the workspace files (assets) — a capable subagent (say, a great Glean researcher) can be promoted into a permanent, rerunnable agent with its own memory scope. **Edit any agent** (Edit button loads its name/prompt/tools back into the form), and open **Runs & artifacts** on a card to review past runs — each run shows a **step-by-step transcript** (model response, decisions, and every executed tool call with full arguments and results), a one-click **Run again** button, and a **workspace file browser** (list + content preview via `GET /agents/{id}/workspace`). |
| **Documents** | Build a **queryable knowledge base** from uploaded files (PDF / txt / md) using **contextualized chunk embeddings** (`voyage-context-4`) — extract text (pypdf), split into bounded segments, and embed each chunk **in the context of its full document** (server-side auto-chunking, 512-token chunks, char-budgeted batches to respect the API's per-request token cap). The library lists every document with its status (`ingesting → ready | failed`), pages, chunk count, tokens, and model — plus **Chunk preview** (text per chunk), **Re-embed** (retry a failed ingest), and **Delete** (drops chunks + the stored original). A **semantic search** box embeds the query (`input_type=query`) and returns the top-k chunks with scores, optionally restricted to one document. The **`documents_search` tool** then makes the corpus available to every agent on demand — it is never injected into prompts, so queries that don't need it pay nothing. |
| **Glean** | Connect Pensive to the Glean MCP server (MongoDB enterprise search) with a native **OAuth2 dynamic-client flow** (DCR + PKCE; the `MCP` scope is required by Glean): status + token expiry, **Connect** (opens the authorization screen; the callback rides the existing `localhost:8000` port), **Refresh token**, **Disconnect**, and a **smoke-test box** that calls any discovered tool (`search: …` runs `glean_search`). Once connected, the ~13 discovered Glean tools are available to main chats **and every agent run** as `glean_*` server tools, permission-scoped to the signed-in user. The token cache (` .glean-tokens.json`) is gitignored and bind-mounted — a single-user dev credential, never to be shipped. |

**Agent tool execution.** Agent tool calls are actually *executed*: file and shell tools run inside a per-agent workspace directory (`./agent-workspaces/<agent_id>/`, mounted into the container so built apps survive restarts), results are fed back to the model in a loop (up to 24 steps), and transient upstream failures or broken tool-call payloads are retried automatically. Every tool failure becomes a model-visible `[error]` result — never a crash — with actionable guidance (invalid JSON is reported with reissue instructions; large files should be built with `append_file`). Completed runs are persisted (`pensive_agent_runs`) with their answer, decisions, and executed tool calls so past agents can be reviewed and re-run — the counts **aggregate every executed call, server-tool executions included** (glean/spawn/project work inside the generate loop is folded into the step transcripts), and legacy runs derive their call/step totals from the stored transcripts. The workflow list is **retained to at most 1500 runs and none older than 30 days** — whichever limit is hit first, enforced after every recorded run.

Backing REST API (all under `/api/admin`): `memory-types`, `memories` (`GET` search with
`?user_id=&query=&mem_type=` + `PUT /memories/{mem_type}/{label}`),
`system-prompt` (GET/PUT), `agents` (CRUD + `POST /agents/{id}/run` + `GET /agents/{id}/runs`
+ `GET /agents/{id}/workspace` + `GET /agents/{id}/workspace/file?path=…`), `runs`
(global workflow feed, tagged `pensive` vs `test`, filterable via
`?executor=` and sortable by `created_at`
/ `tokens_total` / `duration_ms` / tool-call and step counts via
`?sort_by=&order=`), and `documents` (`GET` library + `POST /upload` (multipart,
background ingest) + `DELETE /{id}` + `POST /{id}/reembed` + `GET /{id}/chunks`
+ `POST /search`).

## Document Knowledge Base (contextualized chunk embeddings)

Documents uploaded on the **Documents** tab become a searchable corpus agents can
ground answers in — on demand, never ambient:

1. **Extract** — PDFs are read with `pypdf` (page count kept); `.txt`/`.md` pass
   straight through. Originals are stored under `document-uploads/` (bind-mounted —
   deliberately outside the `documents/` python package) so re-embedding works
   after restarts.
2. **Segment** — text is split into bounded segments (≤ 90K chars, preferring
   paragraph boundaries) and sent in **char-budgeted batches** — the contextualized
   embedding API caps a whole request at 120K tokens, so batches stay well inside
   the budget.
3. **Embed** — each batch goes to `POST {CONTEXT_EMBEDDING_URI}`
   (`https://ai.mongodb.com/v1/contextualizedembeddings` by default, same key as
   `LLM_EMBEDDING_KEY`) with `model=voyage-context-4`, `input_type=document` and
   **server-side auto-chunking** (512-token chunks): every chunk is encoded in the
   context of its full document, so vectors carry book-level context without any
   manual metadata augmentation. Chunk text + 1024-dim vectors land in
   `document_chunks`; status bookkeeping (`ingesting → ready | failed`, chunk
   count, tokens, pages, model) in `documents`.
4. **Query** — the same endpoint encodes the query (`input_type=query`); the top-k
   chunks are found with cosine similarity over the stored vectors (pure-Python
   scan — at lab scale no Atlas Search index is needed; swap in `$vectorSearch`
   when the corpus grows). Optionally restrict to one document.

Every agent's toolset includes **`documents_search(query, k?, document?)`** — a
**memory-first** knowledge lookup: Pensive's long-term memory is searched
*before* the corpus via the Agent Engine SDK (`Memory.search`) — the run's own
scope and the console user's (`default`), merged by score. When a confident
match exists (similarity ≥ 0.80 — this information was already retrieved and
stored, by this agent or you) the memory facts answer the query, the tool says
so, and the corpus is **not** queried at all. Only weak or empty memory matches
fall through to the corpus, which costs one query embedding + scan.
The corpus is never injected into prompts automatically. When a run did
retrieve chunks, its answer ends with a deterministic one-line **Sources**
suffix — just the name(s) of the ingested resource(s), deduped and best-first
(e.g. `Sources: High Performance MongoDB — Asya Kamsky`) — parsed from the
tool digests stored in the run transcript, so the citation sits at the end and
the answer text stays clean to copy-paste.

## Long-Term Memory (Agent Engine)

Pensive's long-term memory **is** the [Atlas Agent Engine memory
service](https://www.mongodb.com/docs/agentengine/sdk/python/packages/agent-engine-sdk-memory/),
reached through the official `agent-engine-sdk-memory` SDK. The service distils
conversations into the same four shapes it always uses:

| Type        | Meaning                                              |
| ----------- | ---------------------------------------------------- |
| `semantic`  | Stable facts and preferences about the user          |
| `episodic`  | Specific past events / interactions                  |
| `taxonomic` | Categories or domain knowledge the user relies on    |
| `procedural`| How the user does things (steps, output preferences) |

The adapter (`memory/agent_engine.py`) is the only seam that talks to the
service; everything else calls it:

1. **Intent planning** (`analyze_intent`, in `memory/memories.py`) — an LLM turns
   the latest message into a search `query`, a `needs_memory` flag, and a
   `research` classification that drives deterministic subagent delegation.
2. **Context assembly** (`search_long_term_memory`) — `Memory.build_context`
   retrieves over the long-term sources (`semantic`, `episodic`, `taxonomic`,
   `procedural`), ranks and de-duplicates across them, trims to a token budget,
   and returns the context. **STM is deliberately excluded**: the harness
   transcript already carries the current conversation, so re-injecting recent
   turns would duplicate it.
3. **Generation** — the retrieved facts are folded into ONE injected system
   block (background context, never a fake user/assistant turn), each line
   carrying its **real ISO 8601 date** (`- [2026-11-13T18:46:07.978116+00:00,
   episodic] …`, newest first) so the model can reason about *when* things
   happened. The service's own `timestamp` is the write time, so Pensive prefers
   the original `recorded_at` (preserved in metadata for migrated memories).
4. **Write-back** (`save_memory`) — Pensive records the turn
   (`Memory.record_turn`, idempotency-keyed so a retried request never
   double-records) and extracts durable facts from it with its own LLM,
   persisting each as a typed memory (`save_semantic` / `save_episode` /
   `save_taxonomic` / `save_procedure`). It extracts itself because the service's
   background extraction only runs for turns the Agent Engine runtime records.

### Where memory is stored

`docker compose` runs the Agent Engine memory service itself — `memory-server`
plus the `memory-oe` proxy — against **Pensive's own MongoDB** (`MONGODB_URI`),
so there is a single database instance. The server keeps its collections
(`memory_semantic`, `memory_episodic`, `memory_taxonomic`, `memory_procedural`,
`memory_short_term`, …) in the `pensive_memory` database and provisions its own
Vector Search indexes on startup. Pensive reaches it at `http://memory-oe:8000`
(the compose default); a host-run Pensive can use `http://localhost:8010`.

To use the hosted platform instead, set `AGENTIC_MEMORY_PROJECT_ID` +
`AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN` (see `env.example`). With no backend — or
when it is unreachable — memory degrades to a no-op: a chat turn never fails
because memory is down.

### Extraction is Pensive's

The service normally distils conversations into durable memories itself, but that
path is driven by the orchestration engine's consolidation scanner, which sweeps
the **orchestration engine's own session records** — records Pensive never
creates (it writes turns straight through the memory API). So Pensive distils the
facts itself: `save_memory` extracts durable, typed memories from the turn with
Pensive's LLM and persists each through the matching typed SDK write
(`save_semantic` / `save_episode` / `save_taxonomic` / `save_procedure`). It also
records the turn (`Memory.record_turn`) so the conversation lives on the service.
The orchestration engine is used only as the SDK's routing proxy — no Agent
Engine app runs.

### Console

The **Memories** tab **browses** a user's memory (newest first) and, when you type a query, ranks it by relevance (`Memory.search`). Semantic facts are editable in place (`Memory.save_semantic`, upsert by label — re-embedded automatically). There is no manual delete or merge: the service versions and supersedes memories itself.

The browse read uses the service's list route through the SDK's own transport: the SDK exposes no public list method for semantic/taxonomic/procedural memories, so `agent_engine.list_memories` reaches the same endpoint the SDK's typed reads use (same base URL, auth, and project routing).

![Long-term memory in the Pensive Console: search results across the four types](console_long_term_memory.png)

### Migrating pre-existing memories

If you ran Pensive before this change, its old memories are still in the
`agent_memories` collection. Move them into the service with:

```bash
docker compose exec pensive-api python scripts/migrate_memories.py --dry-run
docker compose exec pensive-api python scripts/migrate_memories.py
```

It maps each legacy record to the matching typed write (slugifying procedural
names, which the service validates), and records every migrated key in a
`pensive_memory_migration` ledger so re-runs are safe — the typed writes are
create-only, so without the ledger a re-run would duplicate episodes, taxonomic
terms, and procedures. The legacy collection is never modified. Use `--user` /
`--type` to narrow the scope.

**Dates are preserved.** The service stamps every write with its own time and
accepts no timestamp, so each migrated memory keeps its original `recorded_at`
(ISO 8601) in custom metadata. Pensive reads that back and renders it — in the
prompt's memory block and in the Console — so temporal reasoning still works.

### Memory pays off on repeat work (tokens, tools, runtime)

Every agent run in the **Agents** tab — lab agents and chat-spawned subagents alike — is recorded in **Agent Workflows** with sortable **tokens**, **tool calls**, and **runtime**. That history makes it easy to see when scoped long-term memory (`agent:<id>`) is doing its job: later runs on the same task need less exploration, fewer tool round-trips, and finish faster because facts and procedures already live in the store instead of being rediscovered every time.

![Agent Workflows: memory-backed runs trend toward fewer tokens, tool calls, and shorter runtime](agent_workflows_memory_efficiency.png)

The same test agent (`55b371ea16dd`) on **11/3/2026** shows the pattern clearly as its memory namespace fills in:

| When (local) | Tokens | Tool calls | Runtime | What it suggests |
| ------------ | ------ | ---------- | ------- | ---------------- |
| 2:41:07 PM | 27.9k | 4 | 59.5s | Cold run — full tool chain to complete the task |
| 2:41:57 PM | 26.6k | 3 | 49.9s | Partial reuse — one fewer tool hop |
| **2:45:06 PM** | **11.7k** | **1** | **14.3s** | **Warm run — memory-backed answer path (~58% fewer tokens, ~76% faster)** |

Sort the workflow table by **Fewest tokens** or **Fastest** to surface agents worth promoting via **Reuse as specialist agent** — the runs that stay cheap are usually the ones that learned something durable on the first pass.

## Requirements

- Python 3.13+
- MongoDB with Vector Search (local 8.x or Atlas)
- OpenAI-compatible LLM endpoint. Pensive speaks the OpenAI **Responses API** (`LLM_URI`, `LLM_KEY`, `LLM_MODEL`) so it works correctly with reasoning models; the final answer is read from the response's `output` message items and any model reasoning/"thinking" is discarded. Models that only speak Chat Completions (the gateway answers `model_not_found` on `/responses`, e.g. `deepseek-*`, `glm-*`) are retried automatically against `/chat/completions` with a translated payload.
- Voyage embeddings via MongoDB AI (`LLM_EMBEDDING_KEY`) — custom `VoyageEmbeddings` client in `utils/llm.py`
- Document ingestion uses the same key against the **contextualized embeddings** endpoint (`CONTEXT_EMBEDDING_URI`, default `https://ai.mongodb.com/v1/contextualizedembeddings`, model `CONTEXT_EMBEDDING_MODEL`, default `voyage-context-4`) — the Files tab uploads land here with server-side auto-chunking
- An **Atlas Agent Engine memory** backend for long-term memory (`AGENTIC_MEMORY_BASE_URL`, or `AGENTIC_MEMORY_PROJECT_ID` + `AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN`). Optional: with none set, Pensive runs without long-term memory.

## Setup

1. Copy environment template:

```bash
cp env.example .env
```

1. Edit `.env` with your MongoDB URI, LLM credentials, and embedding key.
2. Create and activate the virtual environment:

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

1. Start the API:

```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

Verify health: `curl http://localhost:8000/health`

## Agent Skills

This project is a good fit for official agent skills that guide coding agents (Cursor, Claude Code, and others) when extending or debugging the workflow.

### [MongoDB Agent Skills](https://github.com/mongodb/agent-skills)

Schema design, connection setup, vector search, natural-language querying, and MCP server configuration for MongoDB.

**Cursor** — install from the marketplace or run:

```
/add-plugin mongodb
```

**Any agent** (via [skills.sh](https://skills.sh/)):

```bash
npx skills add mongodb/agent-skills
```

Relevant skills for this repo: `mongodb-connection`, `mongodb-schema-design`, `mongodb-search-and-ai`, `mongodb-natural-language-querying`.

### [LangChain Skills](https://github.com/langchain-ai/langchain-skills)

LangGraph workflows, checkpointing, persistence, and LangChain agent patterns.

```bash
npx skills add langchain-ai/langchain-skills --skill '*' --yes
```

Relevant skills for this repo: `langgraph-fundamentals`, `langgraph-persistence`, `langchain-fundamentals`, `langchain-dependencies`.

## Observability

`user_id`, `chat_id`, and `session_id` come from the OpenWebUI session — not a static env var. Enable header forwarding on your OpenWebUI instance:

```bash
ENABLE_FORWARD_USER_INFO_HEADERS=True
```

OpenWebUI then sends `X-OpenWebUI-User-Id`, `X-OpenWebUI-Chat-Id`, and `X-OpenWebUI-Message-Id` on each request to Pensive. Pensive maps `chat_id` to the LangGraph `thread_id` (checkpoint scope per conversation) and forwards all session fields to LangSmith `metadata`.

For `curl` without OpenWebUI, set optional `USER_ID` in `.env` or pass `"user": "demo-user"` in the JSON body.

LangSmith tracing is optional and controlled by environment variables in `.env`:

| Variable | Purpose |
| -------- | ------- |
| `LANGSMITH_TRACING` | Set to `true` to enable tracing |
| `LANGSMITH_API_KEY` | Your LangSmith API key |
| `LANGSMITH_PROJECT` | Project name (default: `pensive`) |
| `LANGSMITH_ENDPOINT` | API endpoint (default: `https://api.smith.langchain.com`) |
| `USER_ID` | Identity pin — when set, every client shares this memory scope (overrides forwarded OpenWebUI user headers and body `user`); leave empty for per-user memory |

When `LANGSMITH_TRACING=true` and a valid `LANGSMITH_API_KEY` is set, traces appear at [smith.langchain.com](https://smith.langchain.com) under the configured project. Each chat request produces a top-level run with tags `pensive` and `chat`, `metadata.user_id` from the OpenWebUI user, plus `chat_id` / `session_id` when forwarded, including nested LangGraph node spans and OpenAI Responses API LLM calls (memory retrieval happens inside the Agent Engine service, outside the trace).

If tracing is disabled or the API key is missing, the API starts normally without reporting to LangSmith.

## Docker

Run the API in a container (MongoDB stays on the host):

1. Set `MONGODB_URI` in `.env` to use `host.docker.internal` instead of `localhost`:

```bash
# Example — replace credentials with your values
MONGODB_URI=mongodb://username:password@host.docker.internal:27017/?authSource=admin&directConnection=true
```

1. Build and start:

```bash
docker compose up -d --build
```

1. Verify:

```bash
curl http://localhost:8000/health
```

Stop with `docker compose down`. Long-term memory is stored by the memory service in Pensive's MongoDB (database `pensive_memory`); it provisions its own Vector Search indexes on startup, so there is nothing to create by hand (document embeddings keep their own `document_chunks` index).

## OpenWebUI Configuration

In OpenWebUI → **Settings → Connections → OpenAI**:


| Field   | Value                   |
| ------- | ----------------------- |
| API URL | `http://<host>:8000/v1` |
| API Key | any non-empty string    |
| Model   | `pensive`               |

Connection type: **Chat Completions** (default) or **Responses** — pensive serves both `POST /v1/chat/completions` and `POST /v1/responses`.

On the OpenWebUI server, enable session header forwarding so Pensive receives the logged-in user and chat context:

```bash
ENABLE_FORWARD_USER_INFO_HEADERS=True
```

See [OpenWebUI environment variables](https://docs.openwebui.com/getting-started/env-configuration/) for details. Without this, Pensive cannot resolve `user_id` from the OpenWebUI session (unless you set optional `USER_ID` in `.env` for local testing).

Enable streaming in chat settings for the best experience — the final answer streams token-by-token.

The agent's workflow decisions (**Classify Intent → Search Long Term Memory → Generate Response → Save Memory**) **stream live as they are made** on the **native reasoning channel** — OpenAI-compatible clients render the growing thought as a collapsible **Thought** block (e.g. OpenCode) or a thinking pane (e.g. OpenWebUI): `/v1/responses` emits a `reasoning` output item with `response.reasoning_summary_text.delta` events; `/v1/chat/completions` sends `delta.reasoning_content`. The same trace is mirrored as structured metadata (`metadata.workflow` on `/v1/responses`, a top-level `workflow` field on `/v1/chat/completions`). The assistant message itself always contains only the final answer.

## Demo Results (OpenWebUI)

These screenshots show Pensive connected to OpenWebUI with streaming enabled. The streamed response is the final answer; the workflow trace below is shown for reference:

### Session 1 — Save a fact

**Prompt:** "My name is Brandon."

![Session 1: telling the agent your name](session_1.png)

Workflow:

- **Classify Intent** → `memory` — query references personal context
- **Search Long Term Memory** → `0 results` — no prior memories found
- **Generate Response** → `respond` — answer generated without retrieved context
- **Save Memory** → `saved` — stored "User's name is Brandon"

**Answer:** "Nice to meet you, Brandon! How can I help you today?"

#### MongoDB after this interaction

A single Session 1 message on a fresh database produces:

![MongoDB collection stats after Session 1](session_1_docs.png)


| Collection          | Documents | Role                                                   |
| ------------------- | --------- | ------------------------------------------------------ |
| `checkpoints`       | 6         | Thread-scoped graph state snapshots (~1 per superstep) |
| `checkpoint_writes` | 18        | State-channel writes across those steps                |

(Long-term facts live in the `pensive_memory` database on the same MongoDB, written by the memory service.)


LangGraph persists checkpoint state at each graph superstep, not just the four workflow nodes visible in OpenWebUI — so **6 checkpoints and ~18 writes per message is expected**, not a leak.

### Session 2 — Recall across sessions

**Prompt:** "Do you remember me?" (new chat, ~1 minute after Session 1)

![Session 2: asking if the agent remembers you](session_2.png)

Workflow:

- **Classify Intent** → `memory` — query references personal context or recall
- **Search Long Term Memory** → `1 results` — found the name saved in Session 1
- **Generate Response** → `respond` — answer generated using 1 retrieved memory
- **Save Memory** → `saved` — this turn's messages recorded to the memory service

**Answer:** "Yes, Brandon! Good to hear from you again. How can I help?"

Session 2 demonstrates **long-term memory**: the fact was recorded to the Agent Engine memory service in Session 1, distilled into a durable memory, and retrieved via `Memory.build_context` in a separate OpenWebUI chat. Session 2 adds another checkpoint thread (~6 checkpoints, ~18 writes).

## API Endpoints


| Endpoint                    | Description                          |
| --------------------------- | ------------------------------------ |
| `GET /health`               | Liveness check (503 if MongoDB down) |
| `GET /v1/models`            | Lists model `pensive`                |
| `POST /v1/chat/completions` | Chat (supports `stream: true`)       |
| `POST /v1/responses`        | OpenAI **Responses API** chat (supports `stream: true`) |


### Example (streaming)

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pensive",
    "stream": true,
    "user": "demo-user",
    "messages": [{"role": "user", "content": "Remember that my favorite database is MongoDB"}]
  }'
```

## Project Structure

```
pensive/
├── Dockerfile              # container image (+ pytest for the agent lab)
├── docker-compose.yml      # pensive-api + memory-server + memory-oe, all on MONGODB_URI
├── memory-config.yaml      # Agent Engine memory server config (voyage + extraction LLM)
├── main.py                 # uvicorn entry
├── config.py               # environment configuration
├── agent/
│   ├── graph.py            # LangGraph workflow (+ subagent wiring)
│   ├── nodes.py            # analyze_intent, search, generate (server-tool loop), save
│   └── state.py            # AgentState schema (incl. token accounting)
├── admin/
│   ├── executor.py         # sandboxed agent tools (files, shell, web, projects)
│   └── service.py          # agent registry, run loop + metrics, memory curation, run history
├── memory/
│   ├── agent_engine.py     # Atlas Agent Engine memory adapter (agent-engine-sdk-memory)
│   └── memories.py         # per-turn intent planning + durable-fact extraction
├── glean/
│   ├── oauth.py            # MCP dynamic-client OAuth2 (discovery, DCR, PKCE, refresh)
│   ├── client.py           # MCP client wrapper (one session per operation)
│   ├── token_cache.py      # bind-mounted token cache
│   └── transport.py
├── api/
│   ├── routes.py           # OpenAI-compatible FastAPI routes
│   ├── admin_routes.py     # /api/admin REST API (UI + agent lab + run feed)
│   ├── glean_routes.py     # /api/admin/glean + OAuth callback at /glean/callback
│   └── models.py           # request/response schemas
├── persistence/
│   └── mongo.py            # MongoDBSaver (short-term thread state)
├── static/                 # Pensive Console UI (MongoDB theme)
├── tests/                  # graph, Agent Engine memory adapter, glean OAuth/client, agents, API
├── utils/
│   └── llm.py              # OpenAI Responses API (+ usage) + Voyage embeddings clients
├── agent-workspaces/       # per-agent sandboxes (gitignored, host-mounted)
└── scripts/
    ├── glean_auth.py       # one-shot host-side Glean OAuth (loopback + fragment relay)
    └── migrate_memories.py # move legacy agent_memories into the Agent Engine service
```

## Background

This project was invoked with the agent prompt in [PROMPT.md](PROMPT.md) — a Cursor session using LangGraph and MongoDB agent skills to build a minimal checkpoint + long-term memory demo with streamed workflow decisions and an OpenAI-compatible API.

Inspired by the memory architecture described in [Beyond the Chatbot: Escaping the "Groundhog Day" Loop with Agentic Memory](https://www.rosston.dev/blog/groundhog-day). 