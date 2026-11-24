"""LangGraph agent nodes with four-type long-term memory and streamed decisions."""
from __future__ import annotations

import hashlib
import json
import logging
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

    """
    context = state.get("prompt_messages") or state["messages"]
    client = get_responses_client()
    plan = await memories.plan_memory_query(client, context)
    intent = "memory" if plan.needs_memory else "general"
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
    also offered the read-only project tools (``project_list`` /
    ``project_read`` over the host projects directory, including the OKF
    bundle) and the Glean tools. Server-executed results are fed back to the
    model; every other tool call — including the harness's own ``subagent``
    tool — is relayed to the client. Pensive runs no subagents itself.
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
    server_budget = _SERVER_TOOL_BUDGET

    content = ""
    tool_calls: list[dict] = []
    tokens_in = 0
    tokens_out = 0
    server_log: list[dict] = []
    # Everything at or before this point is the harness transcript (sent in
    # full every call); only growth appended below is windowed.
    base_len = len(prompt_messages)
    logger.info("model context (harness): %s", _context_summary(prompt_messages))

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

        # Execute server tools (project browsing + Glean) and feed results back.
        handled: list[dict] = []
        exhausted: list[dict] = []
        for call in server_calls:
            if server_budget > 0:
                handled.append(call)
                server_budget -= 1
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
                        "[error] server tool budget exhausted: at most 8 "
                        "server tool calls per turn; finish with what you have."
                    ),
                    tool_call_id=call.get("id") or "",
                )
            )
            _emit_decision(
                "server_tools",
                "skipped",
                "budget exhausted: at most 8 server tool calls per turn",
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


_SERVER_TOOL_BUDGET = 8
# More LLM rounds than the server-tool budget, so a model that pages through
# multiple tool results can actually finish; the budget still bounds total work.
_SERVER_ITERATION_LIMIT = _SERVER_TOOL_BUDGET + 4

SERVER_TOOL_NAMES = frozenset(PROJECT_TOOL_NAMES)

def _named_tool(name: str) -> dict:
    for tool in admin.DEFAULT_AGENT_TOOLS:
        if (tool.get("function") or {}).get("name") == name:
            return tool
    raise KeyError(f"missing built-in tool: {name}")


PROJECT_LIST_TOOL = _named_tool("project_list")
PROJECT_READ_TOOL = _named_tool("project_read")


def _server_tools(client_tools: list[dict] | None, offer_server: bool) -> list[dict]:
    """Client tool defs plus the server tools (unless the client defines its own)."""
    tools = [t for t in (client_tools or []) if isinstance(t, dict)]
    if not offer_server:
        return tools
    defined = {
        (t.get("function") or {}).get("name") or t.get("name")
        for t in tools
    }
    for tool in (PROJECT_LIST_TOOL, PROJECT_READ_TOOL):
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

    The service's consolidation pipeline promotes idle sessions to snapshots
    and extracts durable facts from them, so Pensive runs no extraction itself.
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