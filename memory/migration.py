"""Migrate legacy Pensive memories into the Atlas Agent Engine memory service.

Pensive's long-term memory used to live in the LangGraph ``agent_memories``
collection under per-type namespaces. This module reads those records and
re-creates each one in the Agent Engine service through the SDK.

Three of the four writes are **create-only** on HTTP backends (only
``save_semantic`` upserts), so a ledger collection records every migrated legacy
key: a re-run skips what is already done instead of duplicating episodes,
taxonomic terms, and procedures.

Mapping (legacy -> SDK):

============================  ==============================================
legacy type                   SDK write
============================  ==============================================
``semantic`` (or untyped)     ``save_semantic(text, label, user_id)``
``episodic``                  ``save_episode(title, content, user_id, session_id)``
``taxonomic``                 ``save_taxonomic(domain, term, definition, user_id)``
``procedural``                ``save_procedure(procedure=<slug>, description, content, user_id)``
============================  ==============================================

The legacy collection is never modified.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

LEGACY_COLLECTION = "agent_memories"
LEDGER_COLLECTION = "pensive_memory_migration"

# The service validates procedure names against this pattern.
PROCEDURE_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_NON_SLUG = re.compile(r"[^a-z0-9]+")

# Legacy types that map to a typed SDK write.
KNOWN_TYPES = ("semantic", "episodic", "taxonomic", "procedural")

# Plan "kind" -> SDK method name. Note the irregular plural forms.
SDK_METHOD = {
    "semantic": "save_semantic",
    "episodic": "save_episode",
    "taxonomic": "save_taxonomic",
    "procedural": "save_procedure",
}


@dataclass(frozen=True)
class LegacyMemory:
    """One curatable record read out of the legacy collection."""

    key: str
    user_id: str
    type: str
    label: str
    content: str
    recorded_at: str | None = None


def slugify(value: str, *, fallback: str = "memory") -> str:
    """Slug a name to the service's ``^[a-z0-9]+(-[a-z0-9]+)*$`` pattern."""
    slug = _NON_SLUG.sub("-", str(value or "").lower()).strip("-")
    return slug or fallback


def _fallback_label(mem: LegacyMemory) -> str:
    digest = hashlib.sha256(f"{mem.type}:{mem.content}".encode()).hexdigest()[:8]
    return f"migrated-{digest}"


def plan_write(mem: LegacyMemory) -> tuple[str, dict] | None:
    """Map a legacy memory to ``(sdk_kind, kwargs)``. ``None`` means skip."""
    content = str(mem.content or "").strip()
    if not content or not mem.user_id:
        return None
    label = str(mem.label or "").strip() or _fallback_label(mem)
    metadata: dict[str, Any] = {"migrated_from": LEGACY_COLLECTION, "legacy_key": mem.key}
    if mem.recorded_at:
        metadata["recorded_at"] = mem.recorded_at

    mem_type = (mem.type or "").lower()
    if mem_type in ("", "semantic"):
        return "semantic", dict(
            text=content, label=label, user_id=mem.user_id, metadata=metadata
        )
    if mem_type == "episodic":
        return "episodic", dict(
            title=label,
            content=content,
            user_id=mem.user_id,
            session_id="migrated",
            metadata=metadata,
        )
    if mem_type == "taxonomic":
        return "taxonomic", dict(
            domain="general",
            term=label,
            definition=content,
            user_id=mem.user_id,
            visibility="private",
            metadata=metadata,
        )
    if mem_type == "procedural":
        return "procedural", dict(
            procedure=slugify(label),
            description=content[:200],
            content=content,
            user_id=mem.user_id,
            metadata=metadata,
        )
    logger.warning("Skipping legacy memory %s: unknown type %r", mem.key, mem.type)
    return None


def _namespace_user(doc: dict) -> str:
    namespace = doc.get("namespace") or []
    if isinstance(namespace, list) and len(namespace) > 1:
        return str(namespace[1])
    return ""


def iter_legacy_memories(collection) -> Iterable[LegacyMemory]:
    """Yield the curatable records of the legacy collection (skips junk)."""
    for doc in collection.find({}):
        value = doc.get("value")
        if not isinstance(value, dict):
            continue  # legacy string-encoded rows are not curatable
        if value.get("superseded_by_version_id") or value.get("superseded_by_snapshot"):
            continue
        content = str(value.get("content") or "")
        user_id = str(value.get("user_id") or "") or _namespace_user(doc)
        if not content.strip() or not user_id:
            continue
        recorded_at = value.get("recorded_at")
        yield LegacyMemory(
            key=str(doc.get("_id") or ""),
            user_id=user_id,
            type=str(value.get("type") or "").lower(),
            label=str(value.get("label") or ""),
            content=content,
            recorded_at=str(recorded_at) if recorded_at else None,
        )


def sdk_saver() -> Callable[[str, dict], Any]:
    """Build a saver that writes through the Agent Engine memory SDK."""
    from memory import agent_engine

    memory = agent_engine.get_memory()
    if memory is None:
        raise RuntimeError(
            "No Agent Engine memory backend configured "
            "(set AGENTIC_MEMORY_BASE_URL or AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN)."
        )

    def save(kind: str, kwargs: dict) -> Any:
        return getattr(memory, SDK_METHOD[kind])(**kwargs)

    return save


def migrate(
    collection,
    ledger,
    save: Callable[[str, dict], Any] | None,
    *,
    dry_run: bool = False,
    user_id: str | None = None,
    mem_type: str | None = None,
) -> dict:
    """Re-create legacy memories in the Agent Engine service.

    ``save`` may be ``None`` for a dry run. Returns per-outcome counts.
    """
    stats = {
        "scanned": 0,
        "planned": 0,
        "migrated": 0,
        "skipped_ledger": 0,
        "skipped_existing": 0,
        "failed": 0,
    }
    for mem in iter_legacy_memories(collection):
        stats["scanned"] += 1
        if user_id and mem.user_id != user_id:
            continue
        if mem_type and mem.type != mem_type:
            continue
        plan = plan_write(mem)
        if plan is None:
            continue
        stats["planned"] += 1
        if ledger is not None and ledger.find_one({"_id": mem.key}):
            stats["skipped_ledger"] += 1
            continue
        if dry_run:
            continue
        kind, kwargs = plan
        result = None
        try:
            result = save(kind, kwargs)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if (
                "already exists" in message
                or "duplicate" in message
                or "uniqueness constraint" in message
            ):
                # Already present from a prior (possibly partial) run — the
                # service's typed writes are create-only, and procedures are
                # unique per org, so slug-equal legacy labels collapse.
                stats["skipped_existing"] += 1
            else:
                stats["failed"] += 1
                logger.warning(
                    "Migration failed for %s (%s): %s", mem.key, mem.type, exc
                )
                continue
        else:
            stats["migrated"] += 1
        if ledger is not None:
            ledger.update_one(
                {"_id": mem.key},
                {
                    "$set": {
                        "user_id": mem.user_id,
                        "type": mem.type or "semantic",
                        "sdk_id": str(getattr(result, "id", "") or ""),
                    }
                },
                upsert=True,
            )
    return stats
