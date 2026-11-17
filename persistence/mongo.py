"""MongoDB client and LangGraph checkpointer.

Long-term memory no longer lives here: it is the Atlas Agent Engine memory
service, reached through the ``agent-engine-sdk-memory`` adapter
(``memory/agent_engine.py``). MongoDB now holds only short-term conversation
state (the LangGraph checkpointer) and Pensive's own admin records.
"""
from __future__ import annotations

import logging
from typing import Optional

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import Config

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_checkpointer: Optional[MongoDBSaver] = None


def get_client() -> MongoClient:
    """Return a shared MongoClient, creating it on first use."""
    global _client
    if _client is None:
        if not Config.MONGODB_URI:
            raise RuntimeError(
                "MONGODB_URI is not set. Copy env.example to .env and configure your connection string."
            )
        logger.info("Connecting to MongoDB database=%s", Config.MONGODB_DB)
        _client = MongoClient(Config.MONGODB_URI)
    return _client


def verify_connection() -> None:
    """Ping MongoDB; raise on failure."""
    try:
        get_client().admin.command("ping")
    except PyMongoError as exc:
        logger.error("MongoDB connection failed: %s", type(exc).__name__)
        raise RuntimeError(
            "MongoDB is unavailable. Check .env, verify the cluster is running, and confirm credentials."
        ) from exc


def get_checkpointer() -> MongoDBSaver:
    """Return the LangGraph MongoDB checkpointer."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MongoDBSaver(get_client(), db_name=Config.MONGODB_DB)
    return _checkpointer


def close() -> None:
    """Release MongoDB resources."""
    global _client, _checkpointer
    if _checkpointer is not None:
        _checkpointer.close()
        _checkpointer = None
    if _client is not None:
        _client.close()
        _client = None
