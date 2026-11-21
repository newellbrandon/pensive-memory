"""Intent planning for long-term memory.

Storage and retrieval are the Atlas Agent Engine memory service now (see
``memory/agent_engine.py``). What remains here is the one thing the service
does not do: deciding, per turn, whether memory is relevant, what to search
for, and whether the turn is a research task that should be delegated.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from utils.llm import get_responses_client

logger = logging.getLogger(__name__)

# The four long-term memory types the Agent Engine memory server extracts.
MEMORY_TYPES = ("semantic", "episodic", "taxonomic", "procedural")
ALL_TYPES = list(MEMORY_TYPES)

INTENT_PROMPT = (
    "You plan long-term memory lookups for an assistant. Read the latest user "
    "message in the context of the recent conversation and decide whether remembered "
    "facts about this user (name, interests, goals, location, past conversations, "
    "preferences, skills, how they like things done) could improve the reply.\n"
    "Lean toward needs_memory=true for anything that can be personalized. Set it to "
    "false only for mechanical requests such as today's date or a factual web lookup.\n"
    "Classify `intent` as \"research\" when the request asks for current, external, "
    "multi-source, or verifiable information about the world (research questions, "
    "deep dives, comparisons, fact-checking, how something works in the latest "
    "release) — such turns are delegated to a research subagent. Otherwise use a "
    "short category such as \"general\" or \"memory\".\n"
    "Write `query` as a standalone semantic search query describing the user facts "
    "that would help, not a restatement of the request.\n"
    'Return JSON only: {"intent": str, "needs_memory": bool, "query": str, "rationale": str}.'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

class MemoryPlan(BaseModel):
    intent: str = Field(description="Short description of what the user wants.")
    needs_memory: bool = Field(description="Whether remembered user facts could help.")
    query: str = Field(description="Standalone semantic search query for long-term memory.")
    rationale: str = Field(description="Why memory is or is not needed.")


def latest_user_text(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _extract_json(text: str, pattern) -> Any | None:
    if not text:
        return None
    match = pattern.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (ValueError, TypeError) as exc:
        logger.warning("Could not parse JSON from LLM reply: %s", exc)
        return None


def _intent_context(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Text-only context for intent analysis: the latest user message and the
    most recent assistant reply.

    Tool calls and results are excluded. Truncating a tool-heavy transcript
    would split ``function_call``/``function_call_output`` pairs, which the
    upstream AI gateway rejects ("No tool call found for function call output")
    — and intent classification has no use for tool payloads anyway.
    """
    picked: list[AnyMessage] = []
    for message in reversed(messages):
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, ToolMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        content = getattr(message, "content", None)
        if not content:
            continue
        picked.append(message)
        if len(picked) >= 2:
            break
    return list(reversed(picked))


async def plan_memory_query(client: Any, messages: Sequence[AnyMessage]) -> MemoryPlan:
    """Ask the LLM for intent and a semantic search query (JSON response)."""
    recent = _intent_context(messages)
    try:
        if recent:
            raw = await client.agenerate([SystemMessage(content=INTENT_PROMPT), *recent])
            data = _extract_json(str(raw), _JSON_OBJECT)
            if isinstance(data, dict):
                return MemoryPlan(
                    intent=str(data.get("intent") or ""),
                    needs_memory=bool(data.get("needs_memory")),
                    query=str(data.get("query") or ""),
                    rationale=str(data.get("rationale") or ""),
                )
    except Exception:
        logger.exception("Memory intent analysis failed; skipping memory search")
    text = latest_user_text(messages)
    # A failed analysis must not claim memory is needed: this turn proceeds as
    # a general query (the answer is still grounded in the harness context).
    return MemoryPlan(
        intent=text,
        needs_memory=False,
        query=text,
        rationale="Intent analysis unavailable",
    )


def default_client():
    """Return the configured OpenAI Responses client used for memory planning."""
    return get_responses_client()
