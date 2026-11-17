"""Atlas Agent Engine long-term memory adapter (`agent-engine-sdk-memory`).

Pensive's long-term memory IS the Agent Engine memory service; this module is
the only seam that talks to it. Prompt context is assembled with
``Memory.build_context`` over the long-term sources, conversation turns are
appended with ``Memory.record_turn`` (the server extracts durable facts in the
background), and the console searches with ``Memory.search`` and edits with
``Memory.save_semantic``.

Short-term memory (``stm``) is deliberately NOT read into the prompt: Pensive
is a harness model backend whose client already sends the full conversation
transcript, so injecting recent turns again would duplicate them. Turns are
still *written* via ``record_turn`` — that is what feeds extraction.

Every call degrades to a no-op when the backend is unconfigured or unreachable:
a chat turn must never fail because memory is down.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional, Sequence

from agent_engine_sdk_memory import Memory, MemoryRequestContext

from config import Config

logger = logging.getLogger(__name__)

# Long-term memory types read into the prompt (mirrors the server's four
# extraction types). STM is intentionally excluded — see the module docstring.
LONG_TERM_SOURCES = ("semantic", "episodic", "taxonomic", "procedural")
ALL_SOURCES = LONG_TERM_SOURCES

_memory: Optional[Memory] = None


def reset() -> None:
    """Drop the cached client (tests / configuration changes)."""
    global _memory
    _memory = None


def is_configured() -> bool:
    """Whether a memory backend is configured (base URL or service account)."""
    return bool(
        Config.AGENTIC_MEMORY_BASE_URL.strip()
        or Config.AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN.strip()
    )


def get_memory() -> Optional[Memory]:
    """Return the shared ``Memory`` client, or None when memory is off."""
    global _memory
    if not is_configured():
        return None
    if _memory is None:
        try:
            _memory = Memory(
                base_url=Config.AGENTIC_MEMORY_BASE_URL.strip() or None,
                project_id=Config.AGENTIC_MEMORY_PROJECT_ID.strip() or None,
                service_account_token=(
                    Config.AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN.strip() or None
                ),
            )
        except Exception:  # noqa: BLE001 — memory must never break a request
            logger.exception("Could not construct the Agent Engine memory client")
            return None
    return _memory


def _bind(user_id: str, session_id: str | None = None):
    """Bind a request scope, or None when memory is unavailable."""
    memory = get_memory()
    if memory is None or not str(user_id or "").strip():
        return None
    try:
        return memory.bind(
            MemoryRequestContext(
                user_id=str(user_id), session_id=session_id or None
            )
        )
    except Exception:  # noqa: BLE001
        logger.warning("Could not bind the memory identity", exc_info=True)
        return None


def _turn_key(session_id: str | None, role: str, content: str) -> str:
    """Stable idempotency key so a retried request does not double-record."""
    blob = f"{session_id or ''}\x1f{role}\x1f{content}"
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def record_turn(user_id: str, session_id: str, role: str, content: str) -> bool:
    """Append a conversation turn; the server extracts durable facts async."""
    text = str(content or "")
    if not text.strip():
        return False
    handle = _bind(user_id, session_id)
    if handle is None:
        return False
    try:
        handle.record_turn(
            role=role,
            content=text,
            idempotency_key=_turn_key(session_id, role, text),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Memory turn write failed (%s: %s)", type(exc).__name__, exc
        )
        return False


def normalize_context(formatted: Any) -> list[dict]:
    """Normalize ``ContextResponse.formatted_context`` to ``[{role, content}]``.

    The server returns OpenAI-style messages for an OpenAI-family model and a
    plain string otherwise; both reduce to a list of role-tagged messages.
    """
    if not formatted:
        return []
    if isinstance(formatted, str):
        text = formatted.strip()
        return [{"role": "system", "content": text}] if text else []
    messages: list[dict] = []
    for item in formatted:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if content:
            messages.append(
                {"role": str(item.get("role") or "system"), "content": str(content)}
            )
    return messages


def _recorded_at(metadata: Any) -> str | None:
    """The memory's REAL date, ISO 8601.

    Migrated memories carry their original date in custom metadata because the
    service stamps its own write time and cannot be told otherwise. Custom
    metadata is nested under ``metadata`` on search chunks and flat on list
    entries, so both shapes are checked.
    """
    if not isinstance(metadata, dict):
        return None
    nested = metadata.get("metadata")
    if isinstance(nested, dict) and nested.get("recorded_at"):
        return str(nested["recorded_at"])
    if metadata.get("recorded_at"):
        return str(metadata["recorded_at"])
    return None


def _chunk_record(chunk: Any) -> dict:
    """A selected memory chunk as a dated record for the prompt."""
    metadata = getattr(chunk, "metadata", None) or {}
    source = getattr(chunk, "source", None)
    timestamp = getattr(chunk, "timestamp", None)
    return {
        "text": str(getattr(chunk, "content", "") or ""),
        "source": str(getattr(source, "value", source) or "memory"),
        "recorded_at": _recorded_at(metadata),
        "timestamp": str(timestamp) if timestamp else None,
    }


def build_context(
    user_id: str,
    session_id: str,
    query: str,
    *,
    max_tokens: int | None = None,
    sources: Sequence[str] | None = None,
) -> list[dict]:
    """Retrieve long-term memory for the prompt as dated records.

    The service ranks, de-duplicates and token-budgets the selection
    (``build_context``); Pensive then renders the block itself so each fact
    carries its REAL ISO 8601 date — the service's own ``timestamp`` is the
    write time, which for migrated memories is not when the fact happened.
    """
    if not str(query or "").strip():
        return []
    handle = _bind(user_id, session_id)
    if handle is None:
        return []
    try:
        response = handle.build_context(
            query=query,
            enabled_sources=set(sources or LONG_TERM_SOURCES),
            max_tokens=max_tokens or Config.AGENTIC_MEMORY_MAX_TOKENS,
            include_memories=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Memory context build failed (%s: %s)", type(exc).__name__, exc
        )
        return []
    chunks = getattr(response, "selected_memories", None)
    if chunks:
        return [_chunk_record(chunk) for chunk in chunks]
    # Fallback: the server did not return chunks — keep the content, undated.
    return [
        {"text": message["content"], "source": "", "recorded_at": None, "timestamp": None}
        for message in normalize_context(getattr(response, "formatted_context", None))
    ]


def _hit(chunk: Any) -> dict:
    metadata = getattr(chunk, "metadata", None) or {}
    source = getattr(chunk, "source", None)
    timestamp = getattr(chunk, "timestamp", None)
    return {
        "text": str(getattr(chunk, "content", "") or ""),
        "source": str(getattr(source, "value", source) or "memory"),
        "label": str(metadata.get("label") or ""),
        "id": str(getattr(chunk, "id", "") or ""),
        "similarity_score": getattr(chunk, "similarity_score", None),
        "recorded_at": _recorded_at(metadata),
        "timestamp": str(timestamp) if timestamp else None,
    }


def search(
    user_id: str,
    query: str,
    *,
    sources: Sequence[str] | None = None,
    top_k: int = 8,
) -> list[dict]:
    """Ranked memory records for a query (normalized for tools + the console)."""
    if not str(query or "").strip():
        return []
    handle = _bind(user_id)
    if handle is None:
        return []
    try:
        chunks = handle.search(
            query, sources=list(sources or LONG_TERM_SOURCES), top_k=top_k
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory search failed (%s: %s)", type(exc).__name__, exc)
        return []
    return [_hit(chunk) for chunk in chunks or []]


def _entry_hit(entry: dict, kind: str) -> dict:
    """Normalize a service list entry to the same shape as ``_hit``.

    The name lives in a different field per type (``label`` / ``title`` /
    ``term`` / ``procedure``).
    """
    label = (
        entry.get("label")
        or entry.get("title")
        or entry.get("term")
        or entry.get("procedure")
        or ""
    )
    return {
        "text": str(entry.get("content") or ""),
        "source": kind,
        "label": str(label),
        "id": str(entry.get("id") or ""),
        "similarity_score": None,
        "recorded_at": _recorded_at(entry.get("metadata")),
        "timestamp": str(entry.get("timestamp")) if entry.get("timestamp") else None,
    }


def _list_transport(memory: Memory):
    """The SDK's own client + base path.

    The SDK has no public list method for semantic/taxonomic/procedural, so the
    console's browse view reads the service's list route through the SDK's own
    transport — inheriting its base URL, auth, and project routing.
    """
    runtime = getattr(memory, "_runtime", None)
    transport = getattr(runtime, "_transport", None)
    client = getattr(transport, "_client", None)
    prefix = getattr(runtime, "_base_path", None)
    if client is None or not prefix:
        return None, None
    return client, prefix


def list_memories(user_id: str, mem_type: str | None = None) -> list[dict]:
    """List a user's memories newest-first (browse, not ranked search)."""
    if not str(user_id or "").strip():
        return []
    memory = get_memory()
    if memory is None:
        return []
    client, prefix = _list_transport(memory)
    if client is None:
        return []
    kinds = [mem_type] if mem_type in LONG_TERM_SOURCES else list(LONG_TERM_SOURCES)
    hits: list[dict] = []
    for kind in kinds:
        try:
            response = client.get(f"{prefix}/{kind}", params={"user_id": str(user_id)})
            response.raise_for_status()
            entries = response.json().get("entries", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Memory list failed for %s (%s: %s)", kind, type(exc).__name__, exc
            )
            continue
        hits.extend(_entry_hit(entry, kind) for entry in entries)
    hits.sort(key=lambda hit: hit.get("timestamp") or "", reverse=True)
    return hits


def save_semantic(
    user_id: str, label: str, text: str, *, visibility: str = "private"
) -> Optional[dict]:
    """Save (or replace, by label) a semantic fact. Returns the record or None."""
    if not str(label or "").strip() or not str(text or "").strip():
        return None
    handle = _bind(user_id)
    if handle is None:
        return None
    try:
        result = handle.save_semantic(
            text=str(text), label=str(label), visibility=visibility
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory save failed (%s: %s)", type(exc).__name__, exc)
        return None
    return {
        "id": str(getattr(result, "id", "") or ""),
        "label": str(label),
        "content": str(text),
    }


def get_semantic(user_id: str, label: str) -> Optional[dict]:
    """Fetch a semantic memory by label (for the console), or None."""
    if not str(label or "").strip():
        return None
    handle = _bind(user_id)
    if handle is None:
        return None
    try:
        entry = handle.get_semantic(str(label))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory fetch failed (%s: %s)", type(exc).__name__, exc)
        return None
    if not entry:
        return None
    if isinstance(entry, dict):
        return entry
    return {
        "id": str(getattr(entry, "id", "") or ""),
        "label": str(getattr(entry, "label", label) or label),
        "content": str(getattr(entry, "content", "") or ""),
    }
