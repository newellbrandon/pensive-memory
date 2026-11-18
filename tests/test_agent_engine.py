"""Tests for the Atlas Agent Engine memory adapter (agent-engine-sdk-memory).

Pensive's long-term memory is the Agent Engine memory service; the adapter is
the only seam that talks to it. These tests pin the contract: prompt context is
assembled from long-term sources only (the harness transcript already carries
the current conversation), writes append turns with an idempotency key, and
every call degrades to a no-op when memory is unconfigured or unreachable.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest import mock

from memory import agent_engine as ae


class FakeChunk:
    def __init__(self, content, source="semantic", *, id="c1", label="name",
                 similarity_score=0.9, timestamp=None, metadata=None):
        self.id = id
        self.content = content
        self.source = source
        self.similarity_score = similarity_score
        self.timestamp = timestamp
        self.metadata = {"label": label, **(metadata or {})}


class FakeContextResponse:
    def __init__(self, formatted_context=None, selected_memories=None):
        self.formatted_context = formatted_context
        self.selected_memories = selected_memories


class FakeResult:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeHandle:
    def __init__(self, *, chunks=None, formatted=None, memories=None, semantic=None, error=None):
        self.chunks = chunks or []
        self.formatted = formatted if formatted is not None else []
        self.memories = memories
        self.semantic = semantic
        self.error = error
        self.turns: list[dict] = []
        self.context_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self.saved: list[dict] = []

    def record_turn(self, **kwargs):
        if self.error:
            raise self.error
        self.turns.append(kwargs)
        return FakeResult(id="t1")

    def build_context(self, **kwargs):
        if self.error:
            raise self.error
        self.context_calls.append(kwargs)
        return FakeContextResponse(self.formatted, self.memories)

    def search(self, query, **kwargs):
        if self.error:
            raise self.error
        self.search_calls.append({"query": query, **kwargs})
        return self.chunks

    def save_semantic(self, **kwargs):
        if self.error:
            raise self.error
        self.saved.append(kwargs)
        return FakeResult(id="s1", label=kwargs.get("label"))

    def get_semantic(self, label, **kwargs):
        if self.error:
            raise self.error
        return self.semantic


class FakeMemory:
    def __init__(self, handle, *, list_client=None, base_path="/api/v1/memory"):
        self.handle = handle
        self.bound: list = []
        if list_client is not None:
            transport = type("T", (), {"_client": list_client})()
            self._runtime = type(
                "R", (), {"_transport": transport, "_base_path": base_path}
            )()

    def bind(self, ctx):
        self.bound.append(ctx)
        return self.handle


class AdapterTest(unittest.TestCase):
    def setUp(self):
        ae.reset()
        self.addCleanup(ae.reset)

    def _install(self, handle):
        memory = FakeMemory(handle)
        patcher = mock.patch.object(ae, "get_memory", lambda: memory)
        patcher.start()
        self.addCleanup(patcher.stop)
        return memory

    def test_unconfigured_is_a_noop(self):
        with mock.patch.object(ae, "is_configured", lambda: False):
            self.assertFalse(ae.record_turn("u1", "s1", "user", "hi"))
            self.assertEqual(ae.build_context("u1", "s1", "q"), [])
            self.assertEqual(ae.search("u1", "q"), [])
            self.assertIsNone(ae.save_semantic("u1", "name", "x"))

    def test_record_turn_binds_identity_and_dedupes(self):
        handle = FakeHandle()
        memory = self._install(handle)
        ok = ae.record_turn("u1", "thread-9", "user", "remember TOKEN-42")
        self.assertTrue(ok)
        self.assertEqual(memory.bound[0].user_id, "u1")
        self.assertEqual(memory.bound[0].session_id, "thread-9")
        call = handle.turns[0]
        self.assertEqual(call["role"], "user")
        self.assertEqual(call["content"], "remember TOKEN-42")
        self.assertTrue(call["idempotency_key"])
        # same turn -> same key (a retried request must not double-record)
        ae.record_turn("u1", "thread-9", "user", "remember TOKEN-42")
        self.assertEqual(
            handle.turns[0]["idempotency_key"], handle.turns[1]["idempotency_key"]
        )

    def test_record_turn_skips_blank_content(self):
        handle = FakeHandle()
        self._install(handle)
        self.assertFalse(ae.record_turn("u1", "s1", "assistant", "   "))
        self.assertEqual(handle.turns, [])

    def test_build_context_returns_dated_records_from_selected_memories(self):
        chunk = FakeChunk(
            "Brandon prefers Glean.",
            "semantic",
            metadata={"metadata": {"recorded_at": "2026-11-13T18:46:07.978116+00:00"}},
        )
        handle = FakeHandle(memories=[chunk])
        self._install(handle)
        records = ae.build_context("u1", "s1", "search tooling", max_tokens=1234)
        self.assertEqual(
            records,
            [{
                "text": "Brandon prefers Glean.",
                "source": "semantic",
                "recorded_at": "2026-11-13T18:46:07.978116+00:00",
                "timestamp": None,
            }],
        )
        call = handle.context_calls[0]
        self.assertEqual(call["query"], "search tooling")
        self.assertEqual(call["max_tokens"], 1234)
        # Chunks are requested so Pensive can render the REAL date.
        self.assertTrue(call["include_memories"])
        # STM is excluded: the harness transcript already carries the conversation.
        self.assertNotIn("stm", call["enabled_sources"])
        self.assertEqual(set(call["enabled_sources"]), set(ae.LONG_TERM_SOURCES))

    def test_build_context_falls_back_to_formatted_context(self):
        handle = FakeHandle(formatted=[{"role": "system", "content": "Brandon prefers Glean."}])
        self._install(handle)
        records = ae.build_context("u1", "s1", "q")
        self.assertEqual(records[0]["text"], "Brandon prefers Glean.")
        self.assertIsNone(records[0]["recorded_at"])

    def test_build_context_returns_empty_on_error(self):
        handle = FakeHandle(error=RuntimeError("memory down"))
        self._install(handle)
        self.assertEqual(ae.build_context("u1", "s1", "q"), [])

    def test_build_context_requires_a_query(self):
        handle = FakeHandle()
        self._install(handle)
        self.assertEqual(ae.build_context("u1", "s1", "  "), [])
        self.assertEqual(handle.context_calls, [])

    def test_search_normalizes_chunks(self):
        chunk = FakeChunk(
            "Brandon prefers Glean.",
            "semantic",
            id="c9",
            label="search-pref",
            similarity_score=0.77,
            timestamp=datetime(2026, 11, 1, tzinfo=timezone.utc),
        )
        handle = FakeHandle(chunks=[chunk])
        self._install(handle)
        hits = ae.search("u1", "glean", top_k=3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["text"], "Brandon prefers Glean.")
        self.assertEqual(hits[0]["source"], "semantic")
        self.assertEqual(hits[0]["label"], "search-pref")
        self.assertEqual(hits[0]["id"], "c9")
        self.assertEqual(hits[0]["similarity_score"], 0.77)
        self.assertIn("2026-11-01", hits[0]["timestamp"])
        self.assertEqual(handle.search_calls[0]["top_k"], 3)

    def test_search_surfaces_custom_recorded_at(self):
        chunk = FakeChunk(
            "fact", "semantic",
            metadata={"metadata": {"recorded_at": "2026-11-13T00:00:00+00:00"}},
        )
        handle = FakeHandle(chunks=[chunk])
        self._install(handle)
        hits = ae.search("u1", "q")
        self.assertEqual(hits[0]["recorded_at"], "2026-11-13T00:00:00+00:00")

    def test_save_semantic_upserts_by_label(self):
        handle = FakeHandle()
        self._install(handle)
        record = ae.save_semantic("u1", "seat-pref", "Prefers window seats.")
        self.assertEqual(record["label"], "seat-pref")
        self.assertEqual(record["content"], "Prefers window seats.")
        self.assertEqual(handle.saved[0]["label"], "seat-pref")
        self.assertEqual(handle.saved[0]["text"], "Prefers window seats.")

    def test_save_semantic_rejects_blank(self):
        handle = FakeHandle()
        self._install(handle)
        self.assertIsNone(ae.save_semantic("u1", "seat-pref", "   "))
        self.assertIsNone(ae.save_semantic("u1", "  ", "text"))
        self.assertEqual(handle.saved, [])

    def test_sdk_error_never_raises(self):
        handle = FakeHandle(error=RuntimeError("boom"))
        self._install(handle)
        self.assertFalse(ae.record_turn("u1", "s1", "user", "hi"))
        self.assertEqual(ae.search("u1", "q"), [])
        self.assertIsNone(ae.save_semantic("u1", "l", "t"))


class FakeResponse:
    def __init__(self, entries):
        self._entries = entries

    def raise_for_status(self):
        return None

    def json(self):
        return {"entries": self._entries}


class FakeListClient:
    """Stands in for the SDK's own httpx.Client (list endpoints)."""

    def __init__(self, entries_by_kind=None, error=None):
        self.entries_by_kind = entries_by_kind or {}
        self.error = error
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        if self.error:
            raise self.error
        kind = url.rsplit("/", 1)[-1]
        return FakeResponse(self.entries_by_kind.get(kind, []))


class ListMemoriesTest(unittest.TestCase):
    """Browsing: the SDK has no public list for these types, so the adapter
    reads the service's list route through the SDK's own transport."""

    def setUp(self):
        ae.reset()
        self.addCleanup(ae.reset)

    def _install(self, memory):
        patcher = mock.patch.object(ae, "get_memory", lambda: memory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_lists_all_types_sorted_newest_first(self):
        client = FakeListClient({
            "semantic": [{"id": "s1", "content": "Name is Brandon", "label": "name",
                          "timestamp": "2026-11-01T10:00:00"}],
            "episodic": [{"id": "e1", "content": "Fixed UI", "label": "ui",
                          "timestamp": "2026-11-02T10:00:00"}],
        })
        self._install(FakeMemory(FakeHandle(), list_client=client))
        hits = ae.list_memories("u1")
        self.assertEqual([h["id"] for h in hits], ["e1", "s1"])  # newest first
        self.assertEqual({h["source"] for h in hits}, {"semantic", "episodic"})
        self.assertEqual(
            {url for url, _ in client.calls},
            {f"/api/v1/memory/{k}" for k in ae.LONG_TERM_SOURCES},
        )
        self.assertEqual(client.calls[0][1], {"user_id": "u1"})

    def test_lists_one_type_when_filtered(self):
        client = FakeListClient({"taxonomic": [
            {"id": "t1", "content": "c", "label": "Cloud", "timestamp": "x"}
        ]})
        self._install(FakeMemory(FakeHandle(), list_client=client))
        hits = ae.list_memories("u1", "taxonomic")
        self.assertEqual([h["source"] for h in hits], ["taxonomic"])
        self.assertEqual(len(client.calls), 1)

    def test_returns_empty_when_unconfigured(self):
        self._install(None)
        self.assertEqual(ae.list_memories("u1"), [])

    def test_uses_type_specific_name_fields_for_labels(self):
        client = FakeListClient({
            "episodic": [{"id": "e1", "content": "c", "title": "Rewrote memory system", "timestamp": "t"}],
            "taxonomic": [{"id": "t1", "content": "c", "term": "cluster cost", "timestamp": "t"}],
            "procedural": [{"id": "p1", "content": "c", "procedure": "okf-pref", "timestamp": "t"}],
        })
        self._install(FakeMemory(FakeHandle(), list_client=client))
        labels = {h["source"]: h["label"] for h in ae.list_memories("u1")}
        self.assertEqual(labels["episodic"], "Rewrote memory system")
        self.assertEqual(labels["taxonomic"], "cluster cost")
        self.assertEqual(labels["procedural"], "okf-pref")

    def test_list_entries_carry_recorded_at_from_metadata(self):
        client = FakeListClient({"semantic": [
            {"id": "s1", "content": "c", "label": "name", "timestamp": "2026-11-17T22:16:54.827000",
             "metadata": {"recorded_at": "2026-11-13T18:51:14.122559+00:00"}}
        ]})
        self._install(FakeMemory(FakeHandle(), list_client=client))
        hits = ae.list_memories("u1", "semantic")
        self.assertEqual(hits[0]["recorded_at"], "2026-11-13T18:51:14.122559+00:00")
        self.assertEqual(hits[0]["timestamp"], "2026-11-17T22:16:54.827000")

    def test_swallows_transport_errors(self):
        client = FakeListClient(error=RuntimeError("down"))
        self._install(FakeMemory(FakeHandle(), list_client=client))
        self.assertEqual(ae.list_memories("u1"), [])


class RecordedAtTest(unittest.TestCase):
    """Migrated memories keep their real date in custom metadata (the service
    stamps its own write time and cannot be told otherwise)."""

    def test_prefers_nested_custom_recorded_at(self):
        self.assertEqual(
            ae._recorded_at({"metadata": {"recorded_at": "2026-11-13T18:46:07.978116+00:00"}}),
            "2026-11-13T18:46:07.978116+00:00",
        )

    def test_reads_flat_custom_recorded_at(self):
        self.assertEqual(
            ae._recorded_at({"recorded_at": "2026-11-11T20:00:45.141230+00:00"}),
            "2026-11-11T20:00:45.141230+00:00",
        )

    def test_none_when_absent(self):
        self.assertIsNone(ae._recorded_at({"label": "name"}))
        self.assertIsNone(ae._recorded_at(None))


class SaveFactsTest(unittest.TestCase):
    """Pensive distils durable facts itself (the service's own extraction only
    runs for turns its runtime records) and writes them with typed SDK calls."""

    def setUp(self):
        ae.reset()
        self.addCleanup(ae.reset)

    def _install(self, memory):
        patcher = mock.patch.object(ae, "get_memory", lambda: memory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_maps_each_type_to_its_typed_write(self):
        calls: list = []

        class FakeMemory:
            def save_semantic(self, **kw): calls.append(("semantic", kw))
            def save_episode(self, **kw): calls.append(("episodic", kw))
            def save_taxonomic(self, **kw): calls.append(("taxonomic", kw))
            def save_procedure(self, **kw): calls.append(("procedural", kw))

        self._install(FakeMemory())
        facts = [
            {"type": "semantic", "text": "Likes teal.", "label": "colour"},
            {"type": "episodic", "text": "Met at the Atlanta office.", "label": "atlanta"},
            {"type": "taxonomic", "text": "Strydn is a platform.", "label": "strydn"},
            {"type": "procedural", "text": "Prefers short answers.", "label": "answer_style"},
        ]
        saved = ae.save_facts("u1", facts, session_id="thread-9")
        self.assertEqual(saved, 4)
        self.assertEqual([k for k, _ in calls], ["semantic", "episodic", "taxonomic", "procedural"])
        by_kind = dict(calls)
        self.assertEqual(by_kind["semantic"]["label"], "colour")
        self.assertEqual(by_kind["episodic"]["session_id"], "thread-9")
        self.assertEqual(by_kind["episodic"]["title"], "atlanta")
        self.assertEqual(by_kind["taxonomic"]["domain"], "general")
        self.assertEqual(by_kind["taxonomic"]["visibility"], "private")
        # The service validates procedure names: slug it.
        self.assertEqual(by_kind["procedural"]["procedure"], "answer-style")
        self.assertRegex(by_kind["procedural"]["procedure"], ae.PROCEDURE_PATTERN)

    def test_skips_unknown_type_and_blank_text(self):
        calls: list = []

        class FakeMemory:
            def save_semantic(self, **kw): calls.append(kw)

        self._install(FakeMemory())
        saved = ae.save_facts("u1", [
            {"type": "bogus", "text": "x", "label": "l"},
            {"type": "semantic", "text": "   ", "label": "l"},
        ])
        self.assertEqual(saved, 0)
        self.assertEqual(calls, [])

    def test_duplicate_writes_do_not_count_as_saved(self):
        class FakeMemory:
            def save_semantic(self, **kw):
                raise RuntimeError("Duplicate document: uniqueness constraint violated")

        self._install(FakeMemory())
        saved = ae.save_facts("u1", [{"type": "semantic", "text": "x", "label": "l"}])
        self.assertEqual(saved, 0)

    def test_unconfigured_is_a_noop(self):
        self._install(None)
        self.assertEqual(
            ae.save_facts("u1", [{"type": "semantic", "text": "x", "label": "l"}]), 0
        )


class ConfiguredTest(unittest.TestCase):
    def setUp(self):
        ae.reset()
        self.addCleanup(ae.reset)

    def test_is_configured_false_without_backend(self):
        with mock.patch.object(ae.Config, "AGENTIC_MEMORY_BASE_URL", ""), mock.patch.object(
            ae.Config, "AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN", ""
        ):
            self.assertFalse(ae.is_configured())
            self.assertIsNone(ae.get_memory())

    def test_is_configured_true_with_base_url(self):
        with mock.patch.object(
            ae.Config, "AGENTIC_MEMORY_BASE_URL", "http://localhost:56423"
        ), mock.patch.object(ae.Config, "AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN", ""):
            self.assertTrue(ae.is_configured())


if __name__ == "__main__":
    unittest.main()
