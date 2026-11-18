"""Tests for generate_response's server-side pensive_subagent loop."""
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


class _FakeSubGraph:
    """Graph the subagent runs on (driven by admin.run_agent)."""

    def __init__(self, response):
        self.response = response
        self.invocations = []

    async def astream(self, input_state, config=None, stream_mode=None):
        self.invocations.append({"input_state": input_state, "config": config})
        yield ("values", self.response)


class _BoomGraph:
    async def astream(self, input_state, config=None, stream_mode=None):
        raise RuntimeError("subgraph exploded")


def _spawn_call(call_id, task, **extra):
    args = {"task": task, **extra}
    return {"id": call_id, "name": "pensive_subagent", "arguments": json.dumps(args)}


class SpawnSubagentTest(unittest.TestCase):
    def setUp(self):
        self.decisions = []
        self.recorded_runs = []
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
        self.graph_patcher = mock.patch("agent.graph.active_graph")
        self.mock_graph = self.graph_patcher.start()
        self.addCleanup(self.graph_patcher.stop)
        # record_run would hit the real Mongo connection on the host — stub it.
        self.record_patcher = mock.patch(
            "agent.nodes.admin.record_run",
            side_effect=lambda agent_id, task, result, agent_def=None, executor=None: self.recorded_runs.append(
                (agent_id, task, agent_def, executor)
            ),
        )
        self.record_patcher.start()
        self.addCleanup(self.record_patcher.stop)

    def run_node(self, state, steps, graph=None):
        self.mock_get_client.return_value = _FakeClient(steps)
        self.mock_graph.return_value = graph
        return _run(nodes.generate_response(state)), self.mock_get_client.return_value

    def tool_names(self, client):
        names = []
        for tool in client.calls[0]["tools"]:
            if isinstance(tool, dict):
                names.append((tool.get("function") or {}).get("name") or tool.get("name"))
        return names

    def test_server_tool_offered_when_flag_on(self):
        result, client = self.run_node(_state(), [{"text": "ok"}])
        names = self.tool_names(client)
        self.assertIn("pensive_subagent", names)
        self.assertIn("project_list", names)
        self.assertIn("project_read", names)
        self.assertEqual(result["response"], "ok")
        self.assertEqual(result["tool_calls"], [])

    def test_no_server_tool_without_flag(self):
        client_tool = {"type": "function", "function": {"name": "edit", "description": "d"}}
        result, client = self.run_node(_state(offer=False, tools=[client_tool]), [{"text": "ok"}])
        names = self.tool_names(client)
        self.assertNotIn("pensive_subagent", names)
        self.assertNotIn("project_list", names)
        self.assertIn("edit", names)

    def test_client_tool_calls_still_relayed(self):
        call = {"id": "c1", "name": "read_file", "arguments": json.dumps({"path": "x.txt"})}
        result, _ = self.run_node(_state(tools=[]), [{"text": "", "tool_calls": [call]}])
        self.assertEqual(result["tool_calls"], [call])

    def test_project_tool_executed_and_result_fed_back(self):
        root = Path(self.tmp.name)
        os.environ["PENSIVE_PROJECTS_ROOT"] = str(root)
        (root / "okf").mkdir(parents=True)
        (root / "okf" / "index.md").write_text("# Bundle index\n\nconcept docs live here")

        call = {"id": "p1", "name": "project_read", "arguments": json.dumps({"path": "okf/index.md"})}
        steps = [
            {"text": "", "tool_calls": [call]},
            {"text": "The bundle index says concept docs live there.", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps)
        self.assertEqual(result["response"], "The bundle index says concept docs live there.")
        self.assertEqual(result["tool_calls"], [])
        second = client.calls[1]["messages"]
        tool_msgs = [m for m in second if isinstance(m, ToolMessage)]
        self.assertTrue(any("Bundle index" in m.content for m in tool_msgs))
        self.assertTrue(any("concept docs live here" in m.content for m in tool_msgs))

    def test_project_tool_disabled_when_root_unset_returns_error(self):
        call = {"id": "p1", "name": "project_read", "arguments": json.dumps({"path": "okf/index.md"})}
        steps = [
            {"text": "", "tool_calls": [call]},
            {"text": "not available", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps)
        self.assertEqual(result["response"], "not available")
        second = client.calls[1]["messages"]
        tool_msgs = [m for m in second if isinstance(m, ToolMessage)]
        self.assertTrue(any("not configured" in m.content for m in tool_msgs))

    def test_glean_tools_offered_when_connected(self):
        fake = mock.Mock()
        fake.tool_schemas.return_value = [
            {"type": "function", "function": {"name": "glean_search", "description": "d", "parameters": {"type": "object"}}}
        ]
        with mock.patch("agent.nodes.get_glean_client", return_value=fake):
            result, client = self.run_node(_state(), [{"text": "ok"}])
        names = self.tool_names(client)
        self.assertIn("glean_search", names)
        self.assertIn("pensive_subagent", names)
        self.assertEqual(result["response"], "ok")

    def test_glean_tool_executed_and_result_fed_back(self):
        fake = mock.Mock()
        fake.tool_schemas.return_value = []
        fake.refresh_tools_if_stale = mock.AsyncMock()
        fake.call_tool = mock.AsyncMock(
            return_value={"name": "glean_search", "output": "[ok] glean search returned:\nhit: doc1"}
        )
        call = {"id": "g1", "name": "glean_search", "arguments": json.dumps({"query": "deep learning"})}
        steps = [
            {"text": "", "tool_calls": [call]},
            {"text": "found doc1", "tool_calls": []},
        ]
        with mock.patch("agent.nodes.get_glean_client", return_value=fake):
            result, client = self.run_node(_state(), steps)
        self.assertEqual(result["response"], "found doc1")
        self.assertEqual(fake.call_tool.await_args.args[0], "glean_search")
        second = client.calls[1]["messages"]
        tool_msgs = [m for m in second if isinstance(m, ToolMessage)]
        self.assertTrue(any("hit: doc1" in m.content for m in tool_msgs))

    def test_glean_tool_executed_even_without_server_flag(self):
        # Subagents/agent-lab runs have offer_server_tools=False but must still
        # be able to use Glean when the tool is in their (default) tool list.
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
        steps = [
            {"text": "", "tool_calls": [call]},
            {"text": "internal hit it is", "tool_calls": []},
        ]
        with mock.patch("agent.nodes.get_glean_client", return_value=fake):
            result, client = self.run_node(_state(offer=False, tools=[glean_tool]), steps)
        self.assertEqual(result["response"], "internal hit it is")
        self.assertEqual(fake.call_tool.await_args.args[0], "glean_search")
        second = client.calls[1]["messages"]
        tool_msgs = [m for m in second if isinstance(m, ToolMessage)]
        self.assertTrue(any("internal hit" in m.content for m in tool_msgs))

    def test_generate_logs_executed_server_calls(self):
        # server-tool executions (spawn/glean/project) happen inside the
        # generate loop; they must surface in the run history as tool calls
        graph = _FakeSubGraph({"response": "digest ok", "tool_calls": [], "decisions": []})
        steps = [
            {"text": "", "tool_calls": [_spawn_call("s1", "research x")]},
            {"text": "done", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps, graph=graph)
        log = result.get("server_tool_log") or []
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["name"], "pensive_subagent")
        self.assertIn("[ok] subagent", log[0]["output"])
        self.assertIn("arguments", log[0])

    def test_generate_loop_accumulates_usage_tokens(self):
        graph = _FakeSubGraph({"response": "r", "tool_calls": [], "decisions": []})
        steps = [
            {"text": "", "tool_calls": [_spawn_call("s1", "t1")], "tokens_in": 110, "tokens_out": 22},
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
        self.mock_graph.return_value = graph
        result = _run(nodes.generate_response(_state()))
        self.assertEqual(result["tokens_in"], 440)
        self.assertEqual(result["tokens_out"], 66)

    def test_spawn_executes_isolated_subagent_and_returns_digest(self):
        graph = _FakeSubGraph({"response": "research digest: HNSW is the index", "tool_calls": [], "decisions": []})
        steps = [
            {"text": "", "tool_calls": [_spawn_call("s1", "research vector search")]},
            {"text": "Here is the summary of the research.", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps, graph=graph)
        self.assertEqual(result["response"], "Here is the summary of the research.")
        self.assertEqual(result["tool_calls"], [])

        # subagent ran in complete isolation: agent:sub-* memory scope + thread
        inv = graph.invocations[0]["input_state"]
        self.assertTrue(inv["user_id"].startswith("agent:sub-"))
        thread = graph.invocations[0]["config"]["configurable"]["thread_id"]
        self.assertTrue(thread.startswith("agent:sub-"), thread)
        # subagents never see the spawn tool (no nesting)
        self.assertFalse(inv["offer_server_tools"])
        self.assertEqual(inv["tools"], admin.DEFAULT_AGENT_TOOLS)

        # digest was fed back to the main model as a ToolMessage
        second = client.calls[1]["messages"]
        tool_msgs = [m for m in second if isinstance(m, ToolMessage)]
        self.assertTrue(any("[ok] subagent" in m.content for m in tool_msgs))
        self.assertTrue(any("HNSW is the index" in m.content for m in tool_msgs))
        # decision streamed for the UI
        self.assertTrue(any(d[0] == "pensive_subagent" and d[1] == "ok" for d in self.decisions))
        # run history recorded under the ephemeral subagent id
        self.assertEqual(len(self.recorded_runs), 1)
        self.assertTrue(self.recorded_runs[0][0].startswith("sub-"))
        self.assertEqual(self.recorded_runs[0][1], "research vector search")
        # ... and the subagent definition is recorded with the run, so a good
        # subagent can later be promoted to a reusable specialist agent
        agent_def = self.recorded_runs[0][2]
        self.assertEqual(set(agent_def.keys()), {"name", "system_prompt", "tools"})
        self.assertTrue(agent_def["name"])
        self.assertTrue(agent_def["system_prompt"])
        self.assertEqual(agent_def["tools"], admin.DEFAULT_AGENT_TOOLS)
        # ... tagged as a "pensive"-executed workflow (vs "test" console runs)
        self.assertEqual(self.recorded_runs[0][3], "pensive")

    def test_spawn_budget_caps_at_two(self):
        graph = _FakeSubGraph({"response": "r", "tool_calls": [], "decisions": []})
        steps = [
            {"text": "", "tool_calls": [_spawn_call("s1", "t1")]},
            {"text": "", "tool_calls": [_spawn_call("s2", "t2")]},
            {"text": "", "tool_calls": [_spawn_call("s3", "t3")]},
            {"text": "all delegated", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps, graph=graph)
        self.assertEqual(result["response"], "all delegated")
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(len(graph.invocations), 2)  # only two spawns executed
        fourth = client.calls[3]["messages"]
        err = [m for m in fourth if isinstance(m, ToolMessage) and "budget exhausted" in m.content]
        self.assertTrue(err)
        self.assertTrue(any(d[0] == "server_tools" and d[1] == "skipped" for d in self.decisions))

    def test_missing_task_returns_error_result(self):
        graph = _FakeSubGraph({"response": "r", "tool_calls": [], "decisions": []})
        call = {"id": "s1", "name": "pensive_subagent", "arguments": "{}"}
        steps = [{"text": "", "tool_calls": [call]}, {"text": "done"}]
        result, client = self.run_node(_state(), steps, graph=graph)
        self.assertEqual(result["response"], "done")
        self.assertEqual(len(graph.invocations), 0)
        second = client.calls[1]["messages"]
        err = [m for m in second if isinstance(m, ToolMessage) and "task" in m.content]
        self.assertTrue(err)

    def test_spawn_failure_becomes_error_result(self):
        steps = [
            {"text": "", "tool_calls": [_spawn_call("s1", "t")]},
            {"text": "recovered here", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps, graph=_BoomGraph())
        self.assertEqual(result["response"], "recovered here")
        second = client.calls[1]["messages"]
        err = [m for m in second if isinstance(m, ToolMessage) and "subagent run failed" in m.content]
        self.assertTrue(err)

class SpawnToolDefinitionTest(unittest.TestCase):
    def test_spawn_description_guides_research_delegation(self):
        # The model must recognize open-ended research as a delegation case:
        # research tasks need pensive_subagent, not a memory-only answer.
        description = nodes.SPAWN_SUBAGENT_TOOL["function"]["description"].lower()
        self.assertIn("research", description)
        self.assertIn("open-ended", description)

    def test_tool_name_is_namespaced_for_pensive(self):
        # The tool must NOT be called "subagent": a coding harness (e.g. OpenCode)
        # offers its own `subagent` tool, and the exact-name clash made the model
        # delegate to the harness instead of Pensive.
        self.assertEqual(nodes.SPAWN_SUBAGENT_NAME, "pensive_subagent")
        self.assertEqual(
            nodes.SPAWN_SUBAGENT_TOOL["function"]["name"], "pensive_subagent"
        )

    def test_description_disambiguates_from_client_subagent(self):
        description = nodes.SPAWN_SUBAGENT_TOOL["function"]["description"].lower()
        # It names the harness tool and says this one is Pensive's own.
        self.assertIn("subagent", description)
        self.assertIn("pensive", description)


class SubagentDisambiguationTest(SpawnSubagentTest):
    """When the client harness offers its own `subagent` tool, Pensive adds a
    one-line system reminder so the model delegates inside Pensive."""

    CLIENT_SUBAGENT = {
        "type": "function",
        "function": {
            "name": "subagent",
            "description": "Spawns an agent in a child session.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    def _system_text(self, client):
        messages = client.calls[0]["messages"]
        return "\n".join(
            str(getattr(m, "content", ""))
            for m in messages
            if isinstance(m, SystemMessage)
        )

    def test_disambiguation_added_when_client_offers_subagent(self):
        _, client = self.run_node(
            _state(tools=[self.CLIENT_SUBAGENT]), [{"text": "ok"}]
        )
        text = self._system_text(client)
        self.assertIn("pensive_subagent", text)
        self.assertIn("subagent", text)
        # Pensive's own tool is still offered alongside the client's.
        self.assertIn("pensive_subagent", self.tool_names(client))
        self.assertIn("subagent", self.tool_names(client))

    def test_no_disambiguation_without_client_subagent(self):
        _, client = self.run_node(_state(tools=[]), [{"text": "ok"}])
        self.assertNotIn("pensive_subagent", self._system_text(client))


class ResearchDelegationTest(SpawnSubagentTest):
    """Research-classified turns must deterministically delegate to a subagent
    before the model answers — the digest grounds the final response."""

    def _plan(self, intent: str, needs_memory: bool = False) -> MemoryPlan:
        return MemoryPlan(
            intent=intent,
            needs_memory=needs_memory,
            query="research query",
            rationale="rationale",
        )

    def test_research_intent_sets_delegate_flag(self):
        with mock.patch(
            "agent.nodes.memories.plan_memory_query",
            new=mock.AsyncMock(return_value=self._plan("research")),
        ):
            result = _run(nodes.analyze_intent(_state()))
        self.assertTrue(result.get("delegate_research"))
        self.assertEqual(result["intent"], "research")

    def test_non_research_intent_does_not_delegate(self):
        with mock.patch(
            "agent.nodes.memories.plan_memory_query",
            new=mock.AsyncMock(return_value=self._plan("general")),
        ):
            result = _run(nodes.analyze_intent(_state()))
        self.assertFalse(result.get("delegate_research"))
        self.assertEqual(result["intent"], "general")

    def test_research_turn_auto_spawns_before_generating(self):
        state = _state()
        state["delegate_research"] = True
        graph = _FakeSubGraph(
            {"response": "subagent research output", "tool_calls": [], "decisions": []}
        )
        result, client = self.run_node(
            state, [{"text": "Answer grounded in research"}], graph=graph
        )

        # One auto-spawn executed and recorded as a pensive subagent run.
        self.assertEqual(len(graph.invocations), 1)
        self.assertEqual(len(self.recorded_runs), 1)
        agent_id, task, _, executor = self.recorded_runs[0]
        self.assertTrue(agent_id.startswith("sub-"))
        self.assertEqual(task, "delegate the research")
        self.assertEqual(executor, "pensive")

        # The subagent digest is fed back BEFORE the model's first call, so
        # the answer is grounded in actual research.
        messages = client.calls[0]["messages"]
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        self.assertTrue(any("SUBAGENT ANSWER" in m.content for m in tool_msgs))
        self.assertEqual(result["response"], "Answer grounded in research")
        self.assertIn(
            ("pensive_subagent", "ok"), [(s, d) for s, d, _ in self.decisions]
        )

    def test_no_research_flag_no_auto_spawn(self):
        graph = _FakeSubGraph({"response": "r", "tool_calls": [], "decisions": []})
        result, _ = self.run_node(_state(), [{"text": "plain answer"}], graph=graph)
        self.assertEqual(len(graph.invocations), 0)
        self.assertEqual(self.recorded_runs, [])
        self.assertEqual(result["response"], "plain answer")


class SpawnPromptConvergenceTest(unittest.TestCase):
    def test_spawn_default_prompt_pushes_convergence(self):
        # Subagents must converge fast: few targeted searches, then answer —
        # not exhaustive re-searching that burns tokens and time.
        prompt = nodes.SPAWN_DEFAULT_PROMPT.lower()
        self.assertIn("converge", prompt)
        self.assertIn("answer directly", prompt)


class ModelWindowTest(SpawnSubagentTest):
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
        graph = _FakeSubGraph({"response": "r", "tool_calls": [], "decisions": []})
        result, client = self.run_node(state, steps, graph=graph)
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
    adapter: reads assemble context with ``build_context``, writes record the
    turn and persist durable facts with the typed SDK writes (the service's own
    background extraction does not run for externally-written turns)."""

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

    def test_save_memory_records_turn_and_saves_extracted_facts(self):
        state = _state()
        state["response"] = "Done."
        fact = {"type": "semantic", "text": "Likes teal.", "label": "colour"}
        with mock.patch.object(
            nodes.agent_engine, "record_turn", return_value=True
        ) as rec, mock.patch(
            "agent.nodes.get_responses_client"
        ), mock.patch.object(
            nodes.memories, "extract_memories",
            new=mock.AsyncMock(return_value=[fact]),
        ) as extract, mock.patch.object(
            nodes.agent_engine, "save_facts", return_value=1
        ) as save:
            _run(nodes.save_memory(state, None))
        calls = [(c.args[2], c.args[3]) for c in rec.call_args_list]
        self.assertEqual(
            calls, [("user", "delegate the research"), ("assistant", "Done.")]
        )
        self.assertEqual(rec.call_args_list[0].args[:2], ("u1", "thread-1"))
        extract.assert_awaited_once()
        self.assertEqual(save.call_args.args[0], "u1")
        self.assertEqual(save.call_args.args[1], [fact])
        self.assertEqual(save.call_args.kwargs["session_id"], "thread-1")
        emitted = [(c.args[0], c.args[1]) for c in self.emit.call_args_list]
        self.assertIn(("save_memory", "saved"), emitted)

    def test_save_memory_skips_when_nothing_durable(self):
        state = _state()
        state["response"] = ""
        with mock.patch.object(
            nodes.agent_engine, "record_turn", return_value=True
        ), mock.patch("agent.nodes.get_responses_client"), mock.patch.object(
            nodes.memories, "extract_memories", new=mock.AsyncMock(return_value=[]),
        ), mock.patch.object(
            nodes.agent_engine, "save_facts", return_value=0
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
