"""Tests for the agent nodes: server-tool loop, memory nodes, context window."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from admin import service as admin
from agent import nodes
from memory import memories
from memory.memories import MemoryPlan


def _run(coro):
    return asyncio.run(coro)


def _state(offer=True, tools=None):
    return {
        "messages": [HumanMessage(content="delegate the research")],
        "prompt_messages": [HumanMessage(content="delegate the research")],
        "decisions": [],
        "user_id": "u1",
        "session_id": "thread-1",
        "intent": "",
        "retrieved_memories": [],
        "response": "",
        "memory_needed": False,
        "memory_query": "",
        "memory_context": [],
        "tools": tools or [],
        "tool_choice": None,
        "tool_calls": [],
        "system_prompt": "",
        "offer_server_tools": offer,
    }


class _Result:
    def __init__(self, text, tool_calls, tokens_in=0, tokens_out=0):
        self.text = text
        self.tool_calls = tool_calls
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out


class _FakeClient:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    async def acomplete(self, messages, tools=None, tool_choice=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        step = self.steps[min(len(self.calls) - 1, len(self.steps) - 1)]
        return _Result(step.get("text", ""), step.get("tool_calls", []))


def _project_call(call_id, path="okf/index.md"):
    return {"id": call_id, "name": "project_read", "arguments": json.dumps({"path": path})}


CLIENT_SUBAGENT = {
    "type": "function",
    "function": {
        "name": "subagent",
        "description": "Spawns an agent in a child session.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class ServerToolLoopTest(unittest.TestCase):
    """generate_response's server-tool loop (project browsing + Glean).

    Pensive no longer spawns its own subagents: delegation belongs to the client
    harness's ``subagent`` tool, whose calls are relayed back to the client."""

    def setUp(self):
        self.decisions = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, "PENSIVE_PROJECTS_ROOT", None)

        def capture(step, decision, reason):
            self.decisions.append((step, decision, reason))

        self.client_patcher = mock.patch("agent.nodes.get_responses_client")
        self.mock_get_client = self.client_patcher.start()
        self.addCleanup(self.client_patcher.stop)
        self.emit_patcher = mock.patch("agent.nodes._emit_decision", side_effect=capture)
        self.emit_patcher.start()
        self.addCleanup(self.emit_patcher.stop)

    def run_node(self, state, steps):
        self.mock_get_client.return_value = _FakeClient(steps)
        return _run(nodes.generate_response(state)), self.mock_get_client.return_value

    def tool_names(self, client):
        names = []
        for tool in client.calls[0]["tools"]:
            if isinstance(tool, dict):
                names.append((tool.get("function") or {}).get("name") or tool.get("name"))
        return names

    def _system_text(self, client):
        return "\n".join(
            str(getattr(m, "content", ""))
            for m in client.calls[0]["messages"]
            if isinstance(m, SystemMessage)
        )

    def _project_root(self):
        root = Path(self.tmp.name)
        os.environ["PENSIVE_PROJECTS_ROOT"] = str(root)
        (root / "okf").mkdir(parents=True, exist_ok=True)
        (root / "okf" / "index.md").write_text("# Bundle index\n\nconcept docs live here")
        return root

    def test_server_tools_offered_without_pensive_subagent(self):
        result, client = self.run_node(_state(), [{"text": "ok"}])
        names = self.tool_names(client)
        self.assertIn("project_list", names)
        self.assertIn("project_read", names)
        self.assertNotIn("pensive_subagent", names)
        self.assertEqual(result["response"], "ok")
        self.assertEqual(result["tool_calls"], [])

    def test_no_server_tool_without_flag(self):
        client_tool = {"type": "function", "function": {"name": "edit", "description": "d"}}
        _, client = self.run_node(_state(offer=False, tools=[client_tool]), [{"text": "ok"}])
        names = self.tool_names(client)
        self.assertNotIn("project_list", names)
        self.assertIn("edit", names)

    def test_harness_subagent_offered_and_relayed_to_client(self):
        call = {"id": "h1", "name": "subagent", "arguments": json.dumps({"prompt": "dig in"})}
        result, client = self.run_node(
            _state(tools=[CLIENT_SUBAGENT]), [{"text": "", "tool_calls": [call]}]
        )
        names = self.tool_names(client)
        self.assertIn("subagent", names)
        self.assertNotIn("pensive_subagent", names)
        self.assertEqual(result["tool_calls"], [call])
        # No Pensive-side delegation reminder is injected any more.
        self.assertNotIn("pensive_subagent", self._system_text(client))
        self.assertEqual(len(client.calls), 1)

    def test_pensive_subagent_call_is_not_executed_server_side(self):
        # A stale model asking for the removed tool is just relayed, never run.
        call = {"id": "s1", "name": "pensive_subagent", "arguments": json.dumps({"task": "x"})}
        result, client = self.run_node(_state(), [{"text": "", "tool_calls": [call]}])
        self.assertEqual(result["tool_calls"], [call])
        self.assertEqual(len(client.calls), 1)

    def test_client_tool_calls_still_relayed(self):
        call = {"id": "c1", "name": "read_file", "arguments": json.dumps({"path": "x.txt"})}
        result, _ = self.run_node(_state(tools=[]), [{"text": "", "tool_calls": [call]}])
        self.assertEqual(result["tool_calls"], [call])

    def test_project_tool_executed_and_result_fed_back(self):
        self._project_root()
        steps = [
            {"text": "", "tool_calls": [_project_call("p1")]},
            {"text": "The bundle index says concept docs live there.", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps)
        self.assertEqual(result["response"], "The bundle index says concept docs live there.")
        self.assertEqual(result["tool_calls"], [])
        tool_msgs = [m for m in client.calls[1]["messages"] if isinstance(m, ToolMessage)]
        self.assertTrue(any("concept docs live here" in m.content for m in tool_msgs))

    def test_project_tool_disabled_when_root_unset_returns_error(self):
        steps = [
            {"text": "", "tool_calls": [_project_call("p1")]},
            {"text": "not available", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps)
        self.assertEqual(result["response"], "not available")
        tool_msgs = [m for m in client.calls[1]["messages"] if isinstance(m, ToolMessage)]
        self.assertTrue(any("not configured" in m.content for m in tool_msgs))

    def test_glean_tools_offered_when_connected(self):
        fake = mock.Mock()
        fake.tool_schemas.return_value = [
            {"type": "function", "function": {"name": "glean_search", "description": "d", "parameters": {"type": "object"}}}
        ]
        with mock.patch("agent.nodes.get_glean_client", return_value=fake):
            result, client = self.run_node(_state(), [{"text": "ok"}])
        self.assertIn("glean_search", self.tool_names(client))
        self.assertEqual(result["response"], "ok")

    def test_glean_tool_executed_and_result_fed_back(self):
        fake = mock.Mock()
        fake.tool_schemas.return_value = []
        fake.refresh_tools_if_stale = mock.AsyncMock()
        fake.call_tool = mock.AsyncMock(
            return_value={"name": "glean_search", "output": "[ok] glean search returned:\nhit: doc1"}
        )
        call = {"id": "g1", "name": "glean_search", "arguments": json.dumps({"query": "deep learning"})}
        steps = [{"text": "", "tool_calls": [call]}, {"text": "found doc1", "tool_calls": []}]
        with mock.patch("agent.nodes.get_glean_client", return_value=fake):
            result, client = self.run_node(_state(), steps)
        self.assertEqual(result["response"], "found doc1")
        self.assertEqual(fake.call_tool.await_args.args[0], "glean_search")
        tool_msgs = [m for m in client.calls[1]["messages"] if isinstance(m, ToolMessage)]
        self.assertTrue(any("hit: doc1" in m.content for m in tool_msgs))

    def test_glean_tool_executed_even_without_server_flag(self):
        # Agent-lab runs have offer_server_tools=False but must still be able
        # to use Glean when the tool is in their (default) tool list.
        fake = mock.Mock()
        fake.tool_schemas.return_value = []
        fake.refresh_tools_if_stale = mock.AsyncMock()
        fake.call_tool = mock.AsyncMock(
            return_value={"name": "glean_search", "output": "[ok] glean search returned:\ninternal hit"}
        )
        glean_tool = {
            "type": "function",
            "function": {"name": "glean_search", "description": "d", "parameters": {"type": "object"}},
        }
        call = {"id": "g1", "name": "glean_search", "arguments": json.dumps({"query": "x"})}
        steps = [{"text": "", "tool_calls": [call]}, {"text": "internal hit it is", "tool_calls": []}]
        with mock.patch("agent.nodes.get_glean_client", return_value=fake):
            result, client = self.run_node(_state(offer=False, tools=[glean_tool]), steps)
        self.assertEqual(result["response"], "internal hit it is")
        tool_msgs = [m for m in client.calls[1]["messages"] if isinstance(m, ToolMessage)]
        self.assertTrue(any("internal hit" in m.content for m in tool_msgs))

    def test_generate_logs_executed_server_calls(self):
        self._project_root()
        steps = [{"text": "", "tool_calls": [_project_call("p1")]}, {"text": "done", "tool_calls": []}]
        result, _ = self.run_node(_state(), steps)
        log = result.get("server_tool_log") or []
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["name"], "project_read")
        self.assertIn("arguments", log[0])

    def test_generate_loop_accumulates_usage_tokens(self):
        self._project_root()
        steps = [
            {"text": "", "tool_calls": [_project_call("p1")], "tokens_in": 110, "tokens_out": 22},
            {"text": "done", "tool_calls": [], "tokens_in": 330, "tokens_out": 44},
        ]

        class _TokenClient(_FakeClient):
            async def acomplete(self, messages, tools=None, tool_choice=None):
                step = self.steps[min(len(self.calls), len(self.steps) - 1)]
                self.calls.append({"messages": list(messages), "tools": tools})
                return _Result(
                    step.get("text", ""),
                    step.get("tool_calls", []),
                    tokens_in=step.get("tokens_in", 0),
                    tokens_out=step.get("tokens_out", 0),
                )

        self.mock_get_client.return_value = _TokenClient(steps)
        result = _run(nodes.generate_response(_state()))
        self.assertEqual(result["tokens_in"], 440)
        self.assertEqual(result["tokens_out"], 66)

    def test_server_tool_budget_caps_at_eight(self):
        self._project_root()
        steps = [{"text": "", "tool_calls": [_project_call(f"p{i}")]} for i in range(9)]
        steps.append({"text": "enough", "tool_calls": []})
        result, client = self.run_node(_state(), steps)
        self.assertEqual(result["response"], "enough")
        last = client.calls[-1]["messages"]
        err = [m for m in last if isinstance(m, ToolMessage) and "budget exhausted" in m.content]
        self.assertTrue(err)
        self.assertNotIn("pensive_subagent", err[0].content)
        self.assertTrue(any(d[0] == "server_tools" and d[1] == "skipped" for d in self.decisions))


class SubagentRemovalTest(unittest.TestCase):
    """Pensive's own subagents are gone; the harness's subagent tool is used."""

    def test_nodes_exposes_no_subagent_machinery(self):
        for name in (
            "SPAWN_SUBAGENT_NAME", "SPAWN_SUBAGENT_TOOL", "SPAWN_DEFAULT_PROMPT",
            "CLIENT_SUBAGENT_TOOL", "_subagent_disambiguation", "_execute_spawn",
        ):
            self.assertFalse(hasattr(nodes, name), name)
        self.assertNotIn("pensive_subagent", nodes.SERVER_TOOL_NAMES)

    def test_graph_has_no_subagent_registry(self):
        from agent import graph
        self.assertFalse(hasattr(graph, "active_graph"))
        self.assertFalse(hasattr(graph, "set_active_graph"))

    def test_analyze_intent_never_delegates(self):
        plan = MemoryPlan(intent="research", needs_memory=False, query="q", rationale="r")
        with mock.patch(
            "agent.nodes.memories.plan_memory_query", new=mock.AsyncMock(return_value=plan)
        ), mock.patch("agent.nodes.get_responses_client"), mock.patch(
            "agent.nodes._emit_decision"
        ):
            result = _run(nodes.analyze_intent(_state()))
        self.assertNotIn("delegate_research", result)

    def test_prompts_do_not_mention_pensive_subagents(self):
        from config import Config
        self.assertNotIn("pensive_subagent", Config.SYSTEM_PROMPT)
        self.assertNotIn("subagent", memories.INTENT_PROMPT.lower())


class ModelWindowTest(ServerToolLoopTest):
    """The client transcript is the harness contract: it must always reach the
    model IN FULL. Windowing applies only to the server-tool exchanges the
    generate loop itself appends (windowing the transcript hid earlier turns
    and made the model repeat work — 'the context sticks')."""

    def _harness(self, count):
        return [HumanMessage(content=f"turn {i}") for i in range(count)]

    def _growth(self, pairs):
        growth = []
        for i in range(pairs):
            growth.append(
                AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "glean_search", "args": {}}])
            )
            growth.append(ToolMessage(content=f"result {i}", tool_call_id=f"c{i}"))
        return growth

    def test_loop_messages_keep_full_harness_transcript(self):
        harness = self._harness(20)
        growth = self._growth(4)
        msgs = nodes._loop_model_messages(harness + growth, len(harness))
        for message in harness:
            self.assertIn(message, msgs)
        self.assertLessEqual(len(msgs), len(harness) + nodes._MODEL_WINDOW)

    def test_first_call_sends_full_transcript(self):
        harness = self._harness(12)
        self.assertEqual(nodes._loop_model_messages(harness, len(harness)), harness)

    def test_loop_tail_starts_at_a_tool_call_boundary(self):
        # A windowed tail must not begin with an orphan tool result — the
        # upstream gateway rejects outputs with no matching call.
        harness = self._harness(1)
        growth = self._growth(5)
        msgs = nodes._loop_model_messages(harness + growth, 1)
        first_growth = msgs[len(harness)]
        self.assertTrue(getattr(first_growth, "tool_calls", None))

    def test_server_tool_digests_capped_before_replay(self):
        self.assertEqual(len(nodes._cap_model_result("x" * 99999)), 4000)
        self.assertEqual(nodes._cap_model_result("short"), "short")

    def test_generate_loop_keeps_full_client_transcript(self):
        # End to end through the node: a 14-message client transcript must be
        # fully present in EVERY model call, not just the first.
        transcript = [HumanMessage(content=f"m{i}") for i in range(14)]
        call = {"id": "g1", "name": "glean_search", "arguments": json.dumps({"query": "q"})}
        steps = [{"text": "", "tool_calls": [call]}, {"text": "", "tool_calls": [call]}, {"text": "done"}]
        state = _state()
        state["prompt_messages"] = transcript
        state["messages"] = [transcript[-1]]
        result, client = self.run_node(state, steps)
        self.assertEqual(result["response"], "done")
        for recorded in client.calls:
            sent = recorded["messages"]
            for message in transcript:
                self.assertIn(message, sent)

    def test_context_summary_reports_size_and_stable_hash(self):
        from langchain_core.messages import ToolMessage as TM

        msgs = [HumanMessage(content="hello"), TM(content="world", tool_call_id="c1")]
        summary = nodes._context_summary(msgs)
        self.assertIn("2 messages", summary)
        self.assertIn("sha=", summary)
        # Identical context -> identical hash (so "sticking" is visible in logs)
        self.assertEqual(summary, nodes._context_summary(list(msgs)))
        self.assertNotEqual(summary, nodes._context_summary([HumanMessage(content="different")]))


class AgentEngineMemoryNodeTest(unittest.TestCase):
    """The memory nodes talk to the Agent Engine memory service through the
    adapter: reads assemble context with ``build_context``, writes append turns
    with ``record_turn``. The service's consolidation pipeline extracts durable
    facts from those turns, so Pensive runs no extraction of its own."""

    def setUp(self):
        self.emit = mock.patch("agent.nodes._emit_decision").start()
        self.addCleanup(mock.patch.stopall)

    def test_search_long_term_memory_uses_build_context(self):
        state = _state()
        state["memory_query"] = "what does the user prefer"
        context = [{"role": "system", "content": "Brandon prefers Glean."}]
        with mock.patch.object(
            nodes.agent_engine, "build_context", return_value=context
        ) as build:
            result = _run(nodes.search_long_term_memory(state, None))
        self.assertEqual(result["memory_context"], context)
        self.assertEqual(result["retrieved_memories"], context)
        args = build.call_args.args
        self.assertEqual(args[:2], ("u1", "thread-1"))
        self.assertEqual(args[2], "what does the user prefer")

    def test_search_skips_when_no_query(self):
        state = _state()
        state["memory_query"] = ""
        with mock.patch.object(nodes.agent_engine, "build_context") as build:
            result = _run(nodes.search_long_term_memory(state, None))
        build.assert_not_called()
        self.assertEqual(result["memory_context"], [])

    def test_save_memory_records_turns_without_extracting(self):
        state = _state()
        state["response"] = "Done."
        with mock.patch.object(
            nodes.agent_engine, "record_turn", return_value=True
        ) as rec, mock.patch("agent.nodes.get_responses_client") as get_client:
            _run(nodes.save_memory(state, None))
        calls = [(c.args[2], c.args[3]) for c in rec.call_args_list]
        self.assertEqual(
            calls, [("user", "delegate the research"), ("assistant", "Done.")]
        )
        self.assertEqual(rec.call_args_list[0].args[:2], ("u1", "thread-1"))
        # The service extracts; Pensive makes no extra LLM call per turn.
        get_client.assert_not_called()
        self.assertFalse(hasattr(nodes.memories, "extract_memories"))
        self.assertFalse(hasattr(nodes.agent_engine, "save_facts"))
        emitted = [(c.args[0], c.args[1]) for c in self.emit.call_args_list]
        self.assertIn(("save_memory", "saved"), emitted)

    def test_save_memory_reports_skipped_when_nothing_recorded(self):
        state = _state()
        state["response"] = ""
        with mock.patch.object(
            nodes.agent_engine, "record_turn", return_value=False
        ):
            _run(nodes.save_memory(state, None))
        emitted = [(c.args[0], c.args[1]) for c in self.emit.call_args_list]
        self.assertIn(("save_memory", "skipped"), emitted)

    def test_generate_injects_memory_context_as_messages(self):
        state = _state(offer=False)
        state["memory_context"] = [
            {
                "text": "Brandon prefers Glean.",
                "source": "semantic",
                "recorded_at": "2026-11-13T18:46:07.978116+00:00",
            }
        ]
        with mock.patch("agent.nodes.get_responses_client") as get_client:
            get_client.return_value = _FakeClient([{"text": "ok"}])
            _run(nodes.generate_response(state))
        sent = get_client.return_value.calls[0]["messages"]
        block = next(
            m.content for m in sent
            if isinstance(m, SystemMessage) and "Brandon prefers Glean." in m.content
        )
        # The real ISO date must reach the model so it can reason about time.
        self.assertIn("2026-11-13T18:46:07.978116+00:00", block)
        self.assertIn("semantic", block)

    def test_generate_without_memory_injects_nothing_extra(self):
        state = _state(offer=False)
        state["memory_context"] = []
        with mock.patch("agent.nodes.get_responses_client") as get_client:
            get_client.return_value = _FakeClient([{"text": "ok"}])
            _run(nodes.generate_response(state))
        sent = get_client.return_value.calls[0]["messages"]
        self.assertEqual([m for m in sent if isinstance(m, SystemMessage)], [])


class MemoryBlockTest(unittest.TestCase):
    """The injected memory block is dated ISO 8601, newest first, so the model
    can reason about when things happened (the service's own timestamp is the
    write time, wrong for migrated memories)."""

    def test_renders_iso_dates_newest_first(self):
        records = [
            {"text": "old", "source": "semantic", "recorded_at": "2026-11-01T09:00:00+00:00"},
            {"text": "new", "source": "episodic", "recorded_at": "2026-11-13T18:46:07.978116+00:00"},
            {"text": "undated", "source": "semantic", "recorded_at": None, "timestamp": None},
        ]
        block = nodes._memory_messages(records)[0].content
        lines = [l for l in block.splitlines() if l.startswith("- ")]
        self.assertEqual(lines, [
            "- [2026-11-13T18:46:07.978116+00:00, episodic] new",
            "- [2026-11-01T09:00:00+00:00, semantic] old",
            "- [date unknown, semantic] undated",
        ])

    def test_falls_back_to_service_timestamp_when_no_recorded_at(self):
        records = [{"text": "fresh", "source": "semantic", "recorded_at": None,
                    "timestamp": "2026-11-17T22:16:54.827000"}]
        block = nodes._memory_messages(records)[0].content
        self.assertIn("[2026-11-17T22:16:54.827000, semantic] fresh", block)

    def test_empty_when_no_records(self):
        self.assertEqual(nodes._memory_messages([]), [])
