"""Tests for the legacy-memory -> Agent Engine migration."""
from __future__ import annotations

import re
import unittest
from unittest import mock

from memory import migration as mig


def _doc(key, user_id="u1", mem_type="semantic", label="name", content="Brandon", **extra):
    value = {
        "content": content,
        "user_id": user_id,
        "type": mem_type,
        "label": label,
        "recorded_at": "2026-10-30T19:47:33+00:00",
        **extra,
    }
    return {
        "_id": key,
        "namespace": ["users", user_id, "memories", mem_type],
        "value": value,
    }


class FakeCollection:
    def __init__(self, docs):
        self.docs = list(docs)

    def find(self, _query):
        return list(self.docs)


class FakeLedger:
    def __init__(self):
        self.docs = {}

    def find_one(self, query):
        return self.docs.get(query["_id"])

    def update_one(self, query, update, upsert=False):
        self.docs.setdefault(query["_id"], {}).update(update.get("$set", {}))


def _mem(mem_type="semantic", label="name", content="Brandon", user_id="u1", key="k1"):
    return mig.LegacyMemory(
        key=key, user_id=user_id, type=mem_type, label=label, content=content,
        recorded_at="2026-10-30T19:47:33+00:00",
    )


class SlugifyTest(unittest.TestCase):
    def test_slugify_matches_the_service_pattern(self):
        for value in ("response_preference", "Memory curation UI fix", "Cloud/Arch"):
            slug = mig.slugify(value)
            self.assertRegex(slug, mig.PROCEDURE_PATTERN, value)

    def test_slugify_falls_back_when_empty(self):
        self.assertEqual(mig.slugify(""), "memory")
        self.assertEqual(mig.slugify("  --  "), "memory")
        self.assertRegex(mig.slugify("!!!"), mig.PROCEDURE_PATTERN)


class PlanWriteTest(unittest.TestCase):
    def test_semantic_maps_to_save_semantic(self):
        kind, kw = mig.plan_write(_mem("semantic", label="name"))
        self.assertEqual(kind, "semantic")
        self.assertEqual(kw["text"], "Brandon")
        self.assertEqual(kw["label"], "name")
        self.assertEqual(kw["user_id"], "u1")
        self.assertEqual(kw["metadata"]["migrated_from"], "agent_memories")
        self.assertEqual(kw["metadata"]["recorded_at"], "2026-10-30T19:47:33+00:00")

    def test_untyped_maps_to_semantic_with_fallback_label(self):
        kind, kw = mig.plan_write(_mem("", label=""))
        self.assertEqual(kind, "semantic")
        self.assertTrue(kw["label"].startswith("migrated-"))

    def test_episodic_sets_a_session_id(self):
        kind, kw = mig.plan_write(_mem("episodic", label="ui_fix", content="Fixed the UI."))
        self.assertEqual(kind, "episodic")
        self.assertEqual(kw["title"], "ui_fix")
        self.assertEqual(kw["content"], "Fixed the UI.")
        self.assertEqual(kw["session_id"], "migrated")

    def test_taxonomic_is_private_with_a_domain(self):
        kind, kw = mig.plan_write(_mem("taxonomic", label="Cloud considerations", content="c"))
        self.assertEqual(kind, "taxonomic")
        self.assertEqual(kw["domain"], "general")
        self.assertEqual(kw["term"], "Cloud considerations")
        self.assertEqual(kw["definition"], "c")
        self.assertEqual(kw["visibility"], "private")

    def test_procedural_slugifies_the_name(self):
        kind, kw = mig.plan_write(_mem("procedural", label="response_preference", content="Short answers."))
        self.assertEqual(kind, "procedural")
        self.assertEqual(kw["procedure"], "response-preference")
        self.assertRegex(kw["procedure"], mig.PROCEDURE_PATTERN)
        self.assertEqual(kw["content"], "Short answers.")

    def test_blank_content_is_skipped(self):
        self.assertIsNone(mig.plan_write(_mem("semantic", content="   ")))

    def test_missing_user_is_skipped(self):
        self.assertIsNone(mig.plan_write(_mem("semantic", user_id="")))


class IterLegacyTest(unittest.TestCase):
    def test_skips_empty_and_superseded_and_uncuratable(self):
        coll = FakeCollection([
            _doc("a"),
            _doc("b", content="   "),
            _doc("c", superseded_by_version_id="v2"),
            {"_id": "d", "value": "legacy-string"},
        ])
        mems = list(mig.iter_legacy_memories(coll))
        self.assertEqual([m.key for m in mems], ["a"])

    def test_falls_back_to_namespace_for_user(self):
        doc = _doc("a")
        del doc["value"]["user_id"]
        mems = list(mig.iter_legacy_memories(FakeCollection([doc])))
        self.assertEqual(mems[0].user_id, "u1")


class MigrateTest(unittest.TestCase):
    def setUp(self):
        self.coll = FakeCollection([
            _doc("a", mem_type="semantic", label="name"),
            _doc("b", mem_type="procedural", label="response_preference"),
            _doc("c", user_id="u2", mem_type="semantic", label="name"),
        ])
        self.ledger = FakeLedger()
        self.calls = []

        def save(kind, kwargs):
            self.calls.append((kind, kwargs))
            return type("R", (), {"id": f"sdk-{len(self.calls)}"})()

        self.save = save

    def test_migrates_every_type_and_ledgers(self):
        stats = mig.migrate(self.coll, self.ledger, self.save)
        self.assertEqual(stats["scanned"], 3)
        self.assertEqual(stats["migrated"], 3)
        kinds = [k for k, _ in self.calls]
        self.assertEqual(kinds.count("semantic"), 2)
        self.assertEqual(kinds.count("procedural"), 1)
        self.assertEqual(set(self.ledger.docs), {"a", "b", "c"})

    def test_is_idempotent_via_the_ledger(self):
        mig.migrate(self.coll, self.ledger, self.save)
        second = mig.migrate(self.coll, self.ledger, self.save)
        self.assertEqual(second["migrated"], 0)
        self.assertEqual(second["skipped_ledger"], 3)
        self.assertEqual(len(self.calls), 3)  # nothing re-written

    def test_dry_run_writes_nothing(self):
        stats = mig.migrate(self.coll, self.ledger, None, dry_run=True)
        self.assertEqual(stats["planned"], 3)
        self.assertEqual(stats["migrated"], 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ledger.docs, {})

    def test_filters_by_user_and_type(self):
        stats = mig.migrate(self.coll, self.ledger, self.save, user_id="u2")
        self.assertEqual(stats["migrated"], 1)
        self.assertEqual(self.calls[0][1]["user_id"], "u2")

        self.calls.clear()
        self.ledger = FakeLedger()
        stats = mig.migrate(self.coll, self.ledger, self.save, mem_type="procedural")
        self.assertEqual(stats["migrated"], 1)
        self.assertEqual(self.calls[0][0], "procedural")

    def test_already_exists_counts_as_skipped_existing(self):
        def save(_kind, _kwargs):
            raise RuntimeError("A taxonomic memory with domain 'general' already exists")

        stats = mig.migrate(self.coll, self.ledger, save)
        self.assertEqual(stats["skipped_existing"], 3)
        self.assertEqual(stats["migrated"], 0)
        self.assertEqual(stats["failed"], 0)
        # Ledgered anyway so a re-run does not keep erroring.
        self.assertEqual(set(self.ledger.docs), {"a", "b", "c"})

    def test_duplicate_key_counts_as_skipped_existing(self):
        def save(_kind, _kwargs):
            raise RuntimeError(
                "Failed to create procedural memory: Duplicate document: "
                "uniqueness constraint violated"
            )

        stats = mig.migrate(self.coll, self.ledger, save)
        self.assertEqual(stats["skipped_existing"], 3)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(set(self.ledger.docs), {"a", "b", "c"})


class SdkSaverTest(unittest.TestCase):
    """The SDK's typed writes use irregular names (save_episode, save_procedure);
    the saver must dispatch by kind, not f'save_{kind}'."""

    def test_dispatches_to_the_irregular_sdk_method_names(self):
        calls = []

        class FakeMemory:
            def save_semantic(self, **kw):
                calls.append(("semantic", kw)); return "r"

            def save_episode(self, **kw):
                calls.append(("episodic", kw)); return "r"

            def save_taxonomic(self, **kw):
                calls.append(("taxonomic", kw)); return "r"

            def save_procedure(self, **kw):
                calls.append(("procedural", kw)); return "r"

        with mock.patch("memory.agent_engine.get_memory", return_value=FakeMemory()):
            save = mig.sdk_saver()
            for kind in mig.KNOWN_TYPES:
                save(kind, {"x": 1})
        self.assertEqual([k for k, _ in calls], list(mig.KNOWN_TYPES))

    def test_raises_without_a_backend(self):
        with mock.patch("memory.agent_engine.get_memory", return_value=None):
            with self.assertRaises(RuntimeError):
                mig.sdk_saver()


if __name__ == "__main__":
    unittest.main()
