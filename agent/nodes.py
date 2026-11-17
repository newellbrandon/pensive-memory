"""LangGraph agent nodes with four-type long-term memory and streamed decisions."""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from admin import service as admin
from admin.executor import PROJECT_TOOL_NAMES, execute_project_tool_call
from glean.client import TOOL_PREFIX, get_glean_client
from agent.state import AgentState
from memory import agent_engine, memories
from utils.llm import get_responses_client

logger = logging.getLogger(__name__)


def _emit_decision(step: str, decision: str, reason: str) -> None:
    writer = get_stream_writer()
    writer({"type": "decision", "step": step, "decision": decision, "reason": reason})


def _latest_user_message(state: AgentState) -> str:
    return memories.latest_user_text(state["messages"])


def _iso_date(value) -> str:
    """The real date as ISO 8601, or 'date unknown'."""
    if not value:
        return "date unknown"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _memory_messages(memory_context: list[dict]) -> list:
    """Render retrieved memory as ONE dated system block (newest first).

    Memory is background context, not the user's words, so it is folded into a
    single system message. Each fact carries its REAL ISO 8601 date (the
    migration preserved original dates in metadata) so the model can reason
    about when things happened.
    """
    records = [
        record
        for record in (memory_context or [])
        if isinstance(record, dict) and str(record.get("text") or "").strip()
    ]
    if not records:
        return []
    records.sort(
        key=lambda r: str(r.get("recorded_at") or r.get("timestamp") or ""),
        reverse=True,
    )
    lines = [
        "Relevant long-term memory about this user, newest first "
        "(retrieved automatically; the user did not ask for it):"
    ]
    for record in records:
        when = _iso_date(record.get("recorded_at") or record.get("timestamp"))
        lines.append(f"- [{when}, {record.get('source') or 'memory'}] {record['text']}")
    return [SystemMessage(content="\n".join(lines))]


# Model context budget. The client transcript is the HARNESS CONTRACT and is
# always sent in full: windowing it hid earlier turns, so the model repeated
# work it had already done ("the context sticks and doesn't evolve"). Only the
# server-tool exchanges THIS loop appends are windowed — that is where
# unbounded growth came from.
_MODEL_WINDOW = 6
# Cap for a single tool result replayed into the model context.
_MODEL_RESULT_CAP = 4000


def _text_of(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content or "")


def _context_summary(messages: list) -> str:
    """Compact, hashable description of a model context (telemetry).

    The hash makes context drift — or a context that stops evolving across
    turns — visible in the logs at a glance.
    """
    roles = [getattr(message, "type", "?") for message in messages]
    total = sum(len(_text_of(message)) for message in messages)
    blob = "".join(
        f"{role}\x1f{_text_of(message)}\x1e" for role, message in zip(roles, messages)
    )
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    return (
        f"{len(messages)} messages / {total} chars / "
        f"tail={','.join(roles[-4:]) or '-'} / sha={digest}"
    )


def _bounded_loop_tail(growth: list, limit: int = _MODEL_WINDOW) -> list:
    """Most recent loop growth, trimmed to a complete tool-call boundary.

    A windowed tail must never begin with an orphan tool result: the upstream
    gateway rejects function-call outputs with no matching call.
    """
    if len(growth) <= limit:
        return list(growth)
    tail = list(growth[-limit:])
    while tail and not getattr(tail[0], "tool_calls", None):
        tail = tail[1:]
    if not tail:
        # Degenerate growth (no call in the tail): keep the last exchange
        # whole rather than dropping the newest result.
        return list(growth[-2:])
    return tail


def _loop_model_messages(prompt_messages: list, base_len: int) -> list:
    """Full harness transcript + a bounded tail of this loop's own growth."""
    if len(prompt_messages) <= base_len:
        return list(prompt_messages)
    return list(prompt_messages[:base_len]) + _bounded_loop_tail(
        prompt_messages[base_len:]
    )


def _cap_model_result(text: str) -> str:
    """Bound a tool result replayed into the model context."""
    return str(text or "")[:_MODEL_RESULT_CAP]


async def analyze_intent(state: AgentState) -> dict:
    """Decide whether long-term memory is needed and build a search query.

    Also classifies research turns (current/external/multi-source/verifiable
    information needed): those set ``delegate_research`` so the generate node
    deterministically delegates to a subagent before answering.
    """
    context = state.get("prompt_messages") or state["messages"]
    client = get_responses_client()
    plan = await memories.plan_memory_query(client, context)
    research = plan.intent.strip().lower() == "research"
    intent = "research" if research else ("memory" if plan.needs_memory else "general")
    _emit_decision("classify_intent", intent, plan.rationale)
    decision = {
        "step": "classify_intent",
        "decision": intent,
        "reason": plan.rationale,
    }
    return {
        "intent": intent,
        "memory_needed": plan.needs_memory,
        "memory_query": plan.query,
        "delegate_research": research,
        "decisions": [decision],
    }


def route_after_intent(state: AgentState) -> Literal["search_long_term_memory", "generate_response"]:
    """Route to memory search when the intent analysis decided it is needed."""
    if state.get("memory_needed") and state.get("user_id"):
        return "search_long_term_memory"
    return "generate_response"


async def search_long_term_memory(state: AgentState, runtime: Runtime | None = None) -> dict:
    """Assemble long-term memory context for the current user.

    The Agent Engine memory service does the retrieval (vector + text ranking,
    dedup and token budgeting) behind ``Memory.build_context``; STM is excluded
    because the harness transcript already carries the current conversation.
    """
    user_id = state.get("user_id") or ""
    session_id = state.get("session_id") or ""
    query = state.get("memory_query") or ""

    context = (
        agent_engine.build_context(user_id, session_id, query) if query else []
    )

    reason = (
        f"Assembled memory context ({len(context)} message(s))."
        if context
        else "No relevant long-term memory found."
    )
    _emit_decision("search_long_term_memory", f"{len(context)} results", reason)
    decision = {
        "step": "search_long_term_memory",
        "decision": str(len(context)),
        "reason": reason,
    }
    return {
        "memory_context": context,
        "retrieved_memories": context,
        "decisions": [decision],
    }


async def generate_response(state: AgentState) -> dict:
    """Generate an assistant response using the full incoming context + retrieved memory.

    Tool definitions from the harness are forwarded to the model so it can
    request read/write/execute operations; any tool calls it emits are relayed
    back to the client in the response.

    When ``offer_server_tools`` is set (main-context chats only), the model is
    also offered ``pensive_subagent`` (delegate to an isolated subagent with its
    own memory scope, workspace, and tools) and the read-only project tools
    (``project_list`` / ``project_read`` over the host projects directory,
    including the OKF bundle). server-executed results are fed back to the
    model; only non-server tool calls are relayed to the client. Subagent runs
    have the flag off, so nesting is impossible.
    """
    query = _latest_user_message(state)
    memory_context = state.get("memory_context") or []
    offer_server = bool(state.get("offer_server_tools"))

    # Honoring the full request transcript (harness context) is what lets this
    # act as an OpenCode coding model: the client's system prompt, tool results,
    # and conversation history all flow to the model. Pensive adds persistent
    # memory on top and does not override the caller's instructions.
    transcript = state.get("prompt_messages") or state["messages"]
    prompt_messages: list = []
    system_prompt = state.get("system_prompt") or ""
    if system_prompt:
        prompt_messages.append(SystemMessage(content=system_prompt))
    prompt_messages.extend(_memory_messages(memory_context))
    if offer_server:
        # Disambiguate Pensive's own delegation tool from a client harness's
        # `subagent` tool (only present when the client offers one).
        note = _subagent_disambiguation(state.get("tools"))
        if note:
            prompt_messages.append(SystemMessage(content=note))
    prompt_messages.extend(transcript)

    client = get_responses_client()
    if offer_server:
        # Warm the Glean tool snapshot (cheap: once per minute; no-op when the
        # deployment is not connected) so glean_* schemas reach the model.
        try:
            await get_glean_client().refresh_tools_if_stale()
        except (Exception, BaseExceptionGroup):  # noqa: BLE001
            logger.warning("Glean snapshot refresh failed before tool build", exc_info=True)
    tools = _server_tools(state.get("tools") or None, offer_server)
    tool_choice = state.get("tool_choice")
    spawn_budget = _SPAWN_BUDGET
    server_budget = _SERVER_TOOL_BUDGET
    spawn_used = 0

    content = ""
    tool_calls: list[dict] = []
    tokens_in = 0
    tokens_out = 0
    server_log: list[dict] = []
    # Everything at or before this point is the harness transcript (sent in
    # full every call); only growth appended below is windowed.
    base_len = len(prompt_messages)
    logger.info("model context (harness): %s", _context_summary(prompt_messages))

    # Deterministic research routing: intent analysis classified this turn as
    # research (current/multi-source/verifiable information needed), so
    # delegate to a subagent UP FRONT — its digest grounds the final answer
    # instead of the model answering from memory alone.
    if offer_server and state.get("delegate_research"):
        task = _latest_user_message(state)
        if task:
            auto_call = {
                "id": f"auto_spawn_{uuid.uuid4().hex[:12]}",
                "name": SPAWN_SUBAGENT_NAME,
                "arguments": json.dumps({"task": task}),
            }
            digest = await _execute_server_call(auto_call, state)
            server_log.append(
                {
                    "name": SPAWN_SUBAGENT_NAME,
                    "arguments": str(task)[:12000],
                    "output": str(digest or "")[:8000],
                }
            )
            spawn_used += 1
            server_budget -= 1
            prompt_messages.append(
                AIMessage(content="", tool_calls=[_to_langchain_tool_call(auto_call)])
            )
            prompt_messages.append(
                ToolMessage(
                    content=_cap_model_result(digest),
                    tool_call_id=auto_call.get("id") or "",
                )
            )
            _emit_decision(
                "pensive_subagent",
                "ok",
                f"research turn auto-delegated to a subagent "
                f"({len(str(digest or ''))} chars digest)",
            )

    for iteration in range(_SERVER_ITERATION_LIMIT):
        model_messages = _loop_model_messages(prompt_messages, base_len)
        logger.info(
            "model call %d context: %s", iteration + 1, _context_summary(model_messages)
        )
        try:
            result = await client.acomplete(
                model_messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            logger.error("LLM generation failed: %s", detail)
            content = (
                "Sorry, I couldn't reach the language model. "
                f"Error: {detail[:300]}"
            )
            reason = f"Generation failed: {detail[:200]}"
            _emit_decision("generate_response", "respond", reason)
            return _generate_result(
                content,
                [],
                [{"step": "generate_response", "decision": "respond", "reason": reason}],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        tokens_in += getattr(result, "tokens_in", 0)
        tokens_out += getattr(result, "tokens_out", 0)
        content = result.text
        tool_calls = result.tool_calls or []
        server_calls = [c for c in tool_calls if _is_server_call(c)]
        relay_calls = [c for c in tool_calls if not _is_server_call(c)]
        if not server_calls:
            break

        # Execute server tools (spawns + project browsing) and feed results back.
        handled: list[dict] = []
        exhausted: list[dict] = []
        for call in server_calls:
            is_spawn = _call_name(call) == SPAWN_SUBAGENT_NAME
            if server_budget > 0 and (not is_spawn or spawn_used < spawn_budget):
                handled.append(call)
                server_budget -= 1
                spawn_used += int(is_spawn)
            else:
                exhausted.append(call)
        for call in handled:
            digest = await _execute_server_call(call, state)
            server_log.append(
                {
                    "name": _call_name(call),
                    "arguments": str(call.get("arguments") or "")[:12000],
                    "output": str(digest or "")[:8000],
                }
            )
            prompt_messages.append(
                AIMessage(content="", tool_calls=[_to_langchain_tool_call(call)])
            )
            prompt_messages.append(
                ToolMessage(
                    content=_cap_model_result(digest), tool_call_id=call.get("id") or ""
                )
            )
        for call in exhausted:
            prompt_messages.append(
                AIMessage(content="", tool_calls=[_to_langchain_tool_call(call)])
            )
            prompt_messages.append(
                ToolMessage(
                    content=(
                        "[error] server tool budget exhausted: at most 2 "
                        "pensive_subagent and 8 server tool calls per turn; "
                        "combine tasks or finish with what you have."
                    ),
                    tool_call_id=call.get("id") or "",
                )
            )
            _emit_decision(
                "server_tools",
                "skipped",
                "budget exhausted: at most 2 spawns and 8 server tool calls per turn",
            )

    if tool_calls:
        reason = f"Requested {len(tool_calls)} tool call(s)."
    else:
        reason = f"Generated answer ({len(content)} chars)."
    _emit_decision("generate_response", "tool_calls" if tool_calls else "respond", reason)
    decision = {
        "step": "generate_response",
        "decision": "tool_calls" if tool_calls else "respond",
        "reason": reason,
    }
    return _generate_result(
        content,
        relay_calls,
        [decision],
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        server_tool_log=server_log,
    )


def _call_name(call: dict) -> str:
    return str(call.get("name") or "")


SPAWN_SUBAGENT_NAME = "pensive_subagent"
# A coding harness (e.g. OpenCode) may offer its own tool with this exact name;
# Pensive relays those calls back to the client instead of executing them.
CLIENT_SUBAGENT_TOOL = "subagent"
_SPAWN_BUDGET = 2
_SERVER_TOOL_BUDGET = 8
# More LLM rounds than the server-tool budget, so a model that pages through
# multiple tool results can actually finish; the budget still bounds total work.
_SERVER_ITERATION_LIMIT = _SERVER_TOOL_BUDGET + 4
_SPAWN_DIGEST_ANSWER_LIMIT = 4000
_SPAWN_DIGEST_OUTPUT_LIMIT = 200

SPAWN_DEFAULT_PROMPT = (
    "You are a focused subagent that completes the delegated task using "
    "the available tools. Work autonomously, verify your result, and give "
    "a concise final answer. "
    "Pensive's own knowledge is reachable through the provided tools "
    "(documents_search, pensive_api, glean_*); never call Pensive's API "
    "over HTTP with curl or shell/network commands. "
    "Converge fast: prefer a few targeted searches or extractions over "
    "exhaustive coverage; once you have enough to answer, stop and give "
    "the final answer directly — do not keep re-searching the same topic."
)

SERVER_TOOL_NAMES = frozenset({SPAWN_SUBAGENT_NAME, *PROJECT_TOOL_NAMES})

SPAWN_SUBAGENT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": SPAWN_SUBAGENT_NAME,
        "description": (
            "Pensive's own delegation tool: run a self-contained task in an "
            "isolated subagent that has its own memory scope, workspace, and "
            "tools (file editing, shell, web search) so this conversation stays "
            "clean; only a short digest of its final answer returns. Pass "
            "tools: [] for the defaults. At most 2 calls per turn. USE THIS for "
            "open-ended research and verification tasks that need current, "
            "multi-source, or verifiable information (research questions, deep "
            "dives, comparisons, fact-checking) — do not answer such research "
            "tasks from memory alone. If the client also offers a tool named "
            "'subagent', that is the client harness's own tool; use "
            "pensive_subagent to delegate inside Pensive."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The full task for the subagent"},
                "system_prompt": {"type": "string", "description": "Optional persona/instructions for the subagent"},
                "tools": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional OpenAI function definitions; empty = built-in sandbox + web tools",
                },
            },
            "required": ["task"],
        },
    },
}


def _named_tool(name: str) -> dict:
    for tool in admin.DEFAULT_AGENT_TOOLS:
        if (tool.get("function") or {}).get("name") == name:
            return tool
    raise KeyError(f"missing built-in tool: {name}")


PROJECT_LIST_TOOL = _named_tool("project_list")
PROJECT_READ_TOOL = _named_tool("project_read")


def _client_tool_names(client_tools: list[dict] | None) -> set[str]:
    return {
        str((t.get("function") or {}).get("name") or t.get("name") or "")
        for t in (client_tools or [])
        if isinstance(t, dict)
    }


def _subagent_disambiguation(client_tools: list[dict] | None) -> str:
    """Reminder added only when the client offers its own ``subagent`` tool.

    Without it the model reads "spawn a subagent" as the client's tool (whose
    calls Pensive relays back to the harness) instead of Pensive's own
    server-side delegation.
    """
    if CLIENT_SUBAGENT_TOOL not in _client_tool_names(client_tools):
        return ""
    return (
        f"This client offers its own `{CLIENT_SUBAGENT_TOOL}` tool — a child "
        f"session in the client harness. Pensive's own delegation tool is "
        f"`{SPAWN_SUBAGENT_NAME}`: use it when you want Pensive to run an "
        f"isolated subagent (its own workspace, web search, document and Glean "
        f"tools) and return a digest."
    )


def _server_tools(client_tools: list[dict] | None, offer_server: bool) -> list[dict]:
    """Client tool defs plus the server tools (unless the client defines its own)."""
    tools = [t for t in (client_tools or []) if isinstance(t, dict)]
    if not offer_server:
        return tools
    defined = {
        (t.get("function") or {}).get("name") or t.get("name")
        for t in tools
    }
    for tool in (SPAWN_SUBAGENT_TOOL, PROJECT_LIST_TOOL, PROJECT_READ_TOOL):
        name = (tool.get("function") or {}).get("name")
        if name not in defined:
            tools.append(tool)
    # Glean tools (discovered MCP surface, prefixed glean_*) — only when the
    # deployment is connected; failures surface as [error] results at call time.
    for tool in get_glean_client().tool_schemas():
        name = (tool.get("function") or {}).get("name")
        if name and name not in defined:
            tools.append(tool)
    return tools


def _is_server_call(call: dict) -> bool:
    name = _call_name(call)
    return name in SERVER_TOOL_NAMES or name.startswith(TOOL_PREFIX)


async def _execute_server_call(call: dict, state: AgentState) -> str:
    """Execute one server tool call and return the result fed back to the model."""
    name = _call_name(call)
    if name == SPAWN_SUBAGENT_NAME:
        return await _execute_spawn(call, state)
    if name.startswith(TOOL_PREFIX):
        try:
            args = json.loads(call.get("arguments") or "{}")
            if not isinstance(args, dict):
                args = {}
        except (ValueError, TypeError):
            return (
                "[error] glean tool arguments are not valid JSON; reissue the "
                "call with valid JSON"
            )
        try:
            await get_glean_client().refresh_tools_if_stale()
        except Exception as exc:  # noqa: BLE001
            logger.warning("glean refresh failed: %s", exc)
        logger.info("glean server call %s args=%s", name, json.dumps(args)[:400])
        return (await get_glean_client().call_tool(name, args))["output"]
    executed = execute_project_tool_call(call)
    return executed["output"]


async def _execute_spawn(call: dict, state: AgentState) -> str:
    """Run one pensive_subagent call; returns the digest fed back to the model."""
    # Imported here to avoid the agent.graph <-> agent.nodes import cycle.
    from agent.graph import active_graph

    try:
        args = json.loads(call.get("arguments") or "{}")
        if not isinstance(args, dict):
            args = {}
    except (ValueError, TypeError):
        return (
            "[error] pensive_subagent arguments are not valid JSON; reissue the "
            "call with valid JSON (task is a string, tools optional)."
        )

    task = str(args.get("task") or "").strip()
    if not task:
        return "[error] pensive_subagent requires a non-empty 'task' string"
    graph = active_graph()
    if graph is None:
        return "[error] subagent execution unavailable: the graph is not ready"

    raw_tools = args.get("tools")
    tools = (
        [t for t in raw_tools if isinstance(t, dict)][:12]
        if isinstance(raw_tools, list)
        else None
    )
    agent_id = f"sub-{uuid.uuid4().hex[:12]}"
    subagent = {
        "id": agent_id,
        "name": admin.autogen_agent_name(),
        "system_prompt": str(args.get("system_prompt") or "").strip()
        or SPAWN_DEFAULT_PROMPT,
        "tools": tools or admin.DEFAULT_AGENT_TOOLS,
    }
    try:
        result = await admin.run_agent(subagent, task, graph)
    except Exception as exc:
        logger.exception("subagent execution failed")
        return f"[error] subagent run failed: {type(exc).__name__}: {exc}"
    try:
        admin.record_run(
            agent_id,
            task,
            result,
            agent_def={
                "name": subagent["name"],
                "system_prompt": subagent["system_prompt"],
                "tools": subagent["tools"],
            },
            executor="pensive",
        )
    except Exception:
        logger.exception("could not record subagent run history")

    answer = str(result.get("answer") or "")[:_SPAWN_DIGEST_ANSWER_LIMIT]
    calls = result.get("tool_calls") or []
    summary = "\n".join(
        f"- {c.get('name')}: {str(c.get('output') or '')[: _SPAWN_DIGEST_OUTPUT_LIMIT]}"
        for c in calls
    )
    _emit_decision(
        "pensive_subagent",
        "ok",
        f"subagent {agent_id} finished: {len(calls)} tool call(s), "
        f"answer {len(answer)} chars",
    )
    return (
        f"[ok] subagent {agent_id} ({subagent['name']}) finished after "
        f"{len(calls)} executed tool call(s); full artifacts are in its workspace "
        f"({result.get('workspace') or 'n/a'}).\n"
        f"SUBAGENT ANSWER:\n{answer}\n"
        + (f"TOOL ACTIVITY:\n{summary}\n" if summary else "")
    )


def _to_langchain_tool_call(call: dict) -> dict:
    """Convert a model tool call (id/name/arguments) to LangChain form for replay."""
    try:
        args = json.loads(call.get("arguments") or "{}")
    except (ValueError, TypeError):
        args = {}
    return {
        "id": call.get("id") or "",
        "name": call.get("name") or "",
        "args": args if isinstance(args, dict) else {},
    }


def _generate_result(
    content: str,
    tool_calls: list[dict],
    decisions: list[dict],
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    server_tool_log: list[dict] | None = None,
) -> dict:
    """Build the generate node's return value."""
    assistant_message = AIMessage(content=content)
    if tool_calls:
        assistant_message = AIMessage(
            content=content,
            tool_calls=[
                {
                    "id": call.get("id", ""),
                    "name": call.get("name", ""),
                    "args": _parse_arguments(call.get("arguments", "")),
                }
                for call in tool_calls
            ],
        )
    return {
        "response": content,
        "tool_calls": tool_calls,
        "messages": [assistant_message],
        "decisions": decisions,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "server_tool_log": server_tool_log or [],
    }


def _parse_arguments(arguments: str) -> dict:
    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


async def save_memory(state: AgentState, runtime: Runtime | None = None) -> dict:
    """Append this turn's user and assistant messages to Agent Engine memory.

    The service extracts durable facts from recorded turns in the background,
    so Pensive no longer runs its own extraction pass.
    """
    user_id = state.get("user_id") or ""
    session_id = state.get("session_id") or ""
    query = _latest_user_message(state)
    response = state.get("response") or ""

    saved = 0
    if agent_engine.record_turn(user_id, session_id, "user", query):
        saved += 1
    if agent_engine.record_turn(user_id, session_id, "assistant", response):
        saved += 1

    if saved:
        reason = f"Recorded {saved} conversation turn(s)."
        decision_label = "saved"
    else:
        reason = "No turn recorded (memory unavailable or empty)."
        decision_label = "skipped"

    _emit_decision("save_memory", decision_label, reason)
    return {
        "decisions": [
            {"step": "save_memory", "decision": decision_label, "reason": reason}
        ]
    }