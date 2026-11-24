"""Tests for the long-term memory intent planner.

Storage and retrieval are the Agent Engine memory service now
(``tests/test_agent_engine.py`` covers the adapter). What remains in
``memory/memories.py`` is per-turn intent planning: whether memory is needed,
and what to search for.
"""
from __future__ import annotations

import asyncio
import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from memory import memories as mem
from memory.memories import MemoryPlan


def _run(coro):
    return asyncio.run(coro)


class FakeClient:
    def __init__(self, result: str | Exception):
        self.result = result
        self.calls: list = []

    async def agenerate(self, messages):
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class MemTypeTest(unittest.TestCase):
    def test_all_four_types(self):
        self.assertEqual(
            set(mem.MEMORY_TYPES), {"semantic", "episodic", "taxonomic", "procedural"}
        )


class PlanTest(unittest.TestCase):
    def test_intent_prompt_does_not_route_to_subagents(self):
        # Delegation is the harness's job; the planner only plans memory.
        self.assertNotIn("subagent", mem.INTENT_PROMPT.lower())
        self.assertNotIn("delegat", mem.INTENT_PROMPT.lower())

    def test_falls_back_to_no_memory_search_on_llm_failure(self):
        client = FakeClient(RuntimeError("boom"))
        plan = _run(mem.plan_memory_query(client, [HumanMessage(content="inspire me")]))
        self.assertIsInstance(plan, MemoryPlan)
        # A failed analysis must not claim memory is needed.
        self.assertFalse(plan.needs_memory)
        self.assertEqual(plan.query, "inspire me")
        self.assertEqual(plan.rationale, "Intent analysis unavailable")

    def test_classify_context_is_text_only_across_tool_turns(self):
        client = FakeClient(
            json.dumps({
                "intent": "general",
                "needs_memory": False,
                "query": "",
                "rationale": "simple",
            })
        )
        transcript = [
            HumanMessage(content="read the file"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "read", "args": {"path": "README.md"}}],
            ),
            ToolMessage(content="<file>", tool_call_id="call_1"),
            HumanMessage(content="what does it say?"),
        ]
        plan = _run(mem.plan_memory_query(client, transcript))
        self.assertEqual(plan.intent, "general")
        sent = client.calls[-1]
        # The classify context must be text-only: truncated tool turns would
        # split function_call/function_call_output pairs and fail upstream (the
        # gateway rejects outputs without a matching call).
        self.assertTrue(all(not getattr(m, "tool_calls", None) for m in sent))
        self.assertTrue(all(not isinstance(m, ToolMessage) for m in sent))
        self.assertTrue(
            any("what does it say?" in str(getattr(m, "content", "")) for m in sent)
        )

    def test_parses_json_plan(self):
        payload = json.dumps({
            "intent": "wants inspiration",
            "needs_memory": True,
            "query": "user interests",
            "rationale": "r",
        })
        client = FakeClient(payload)
        plan = _run(mem.plan_memory_query(client, [HumanMessage(content="inspire me")]))
        self.assertEqual(plan.intent, "wants inspiration")
        self.assertTrue(plan.needs_memory)
        self.assertEqual(plan.query, "user interests")


if __name__ == "__main__":
    unittest.main()
