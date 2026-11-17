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


def build_context(
    user_id: str,
    session_id: str,
    query: str,
    *,
    max_tokens: int | None = None,
    sources: Sequence[str] | None = None,
) -> list[dict]:
    """Assemble long-term memory context for a query (list of message dicts)."""
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
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Memory context build failed (%s: %s)", type(exc).__name__, exc
        )
        return []
    return normalize_context(getattr(response, "formatted_context", None))


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
