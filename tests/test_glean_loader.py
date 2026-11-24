"""The on-demand Glean loader schema."""
from __future__ import annotations

import json
import unittest

from glean.client import LOADER_NAME, loader_schema


def _schema(name, description):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }}


class LoaderSchemaTest(unittest.TestCase):
    def test_none_when_no_tools(self):
        self.assertIsNone(loader_schema([]))

    def test_enum_lists_every_tool_and_stays_small(self):
        schemas = [_schema(f"glean_t{i}", "x " * 400) for i in range(21)]
        loader = loader_schema(schemas)
        self.assertEqual(loader["function"]["name"], LOADER_NAME)
        self.assertFalse(LOADER_NAME.startswith("glean_"))
        enum = loader["function"]["parameters"]["properties"]["tools"]["items"]["enum"]
        self.assertEqual(len(enum), 21)
        # Far smaller than the full schemas (~12k tokens for the real 21).
        self.assertLess(len(json.dumps(loader)), len(json.dumps(schemas)) // 10)
        self.assertLess(len(json.dumps(loader)) // 4, 400)


if __name__ == "__main__":
    unittest.main()
