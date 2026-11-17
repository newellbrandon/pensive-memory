#!/usr/bin/env python3
"""Migrate legacy Pensive memories (``agent_memories``) into the Agent Engine service.

Reads the old LangGraph-store collection and re-creates each record in the Atlas
Agent Engine memory service through the SDK. A ledger collection makes the run
resumable: episodic/taxonomic/procedural writes are create-only, so re-running
would otherwise duplicate them.

Usage:
  python scripts/migrate_memories.py --dry-run
  python scripts/migrate_memories.py
  python scripts/migrate_memories.py --user default --type semantic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from memory import migration  # noqa: E402
from persistence.mongo import get_client  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    parser.add_argument("--user", help="migrate only this user_id")
    parser.add_argument(
        "--type", dest="mem_type", choices=migration.KNOWN_TYPES,
        help="migrate only this memory type",
    )
    args = parser.parse_args(argv)

    Config.validate()
    client = get_client()
    collection = client[Config.MONGODB_DB][migration.LEGACY_COLLECTION]
    ledger = client[Config.MONGODB_DB][migration.LEDGER_COLLECTION]

    save = None if args.dry_run else migration.sdk_saver()
    stats = migration.migrate(
        collection,
        ledger,
        save,
        dry_run=args.dry_run,
        user_id=args.user,
        mem_type=args.mem_type,
    )
    print(json.dumps({"dry_run": args.dry_run, **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
