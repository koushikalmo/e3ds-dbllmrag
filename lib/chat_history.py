# ============================================================
# lib/chat_history.py — Persistent Chat History in MongoDB
# ============================================================
# Stores every successful query the user runs in a dedicated
# MongoDB collection so they can revisit past queries across
# sessions and browser refreshes.
#
# COLLECTION:
#   Database:   stream-datastore (same DB as the session data)
#   Collection: _QUERY_HISTORY_  (underscore prefix keeps it
#               separate from monthly session collections)
#
# EACH DOCUMENT:
#   {
#     "_id":            ObjectId (MongoDB auto-assigned),
#     "question":       "Which cities had the most sessions?",
#     "collection":     "Apr_2026",
#     "result_count":   42,
#     "result_label":   "Top Cities by Session Count",
#     "explanation":    "Groups sessions by city...",
#     "elapsed_seconds": 1.3,
#     "timestamp":      ISODate("2026-04-12T10:30:00Z")
#   }
#
# RETENTION:
#   We keep the most recent MAX_HISTORY entries. When that cap
#   is exceeded, the oldest entries are deleted automatically.
#   This keeps the collection from growing unboundedly.
# ============================================================

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from lib.mongodb import get_stream_db

logger = logging.getLogger(__name__)

# Collection name — underscore prefix distinguishes it from the
# monthly session collections (Apr_2026, Mar_2026, etc.)
HISTORY_COLLECTION = "_QUERY_HISTORY_"

# Maximum entries to keep in the history collection.
# When exceeded, oldest entries are trimmed automatically.
MAX_HISTORY = 500


async def save_query(
    question:        str,
    collection:      str,
    result_count:    int,
    result_label:    str  = "",
    explanation:     str  = "",
    elapsed_seconds: float = 0.0,
) -> str:
    """
    Persists a successful query to the history collection.

    Called automatically by main.py after every successful
    /api/query execution that returned at least one result.

    Args:
        question:        The user's natural-language question.
        collection:      Which monthly collection was queried.
        result_count:    How many documents were returned.
        result_label:    Short UI label from the LLM (e.g. "Top Cities").
        explanation:     One-sentence LLM explanation of what it did.
        elapsed_seconds: Total time from request to response.

    Returns:
        The inserted document's _id as a hex string.
    """
    db  = get_stream_db()
    doc = {
        "question":        question,
        "collection":      collection,
        "result_count":    result_count,
        "result_label":    result_label,
        "explanation":     explanation,
        "elapsed_seconds": elapsed_seconds,
        "timestamp":       datetime.now(timezone.utc),
    }

    result = await asyncio.shield(db[HISTORY_COLLECTION].insert_one(doc))

    # Trim old entries if we've exceeded the cap.
    # We do a count-then-delete rather than a capped collection
    # so that each document can be individually removed via the UI.
    try:
        count = await db[HISTORY_COLLECTION].count_documents({})
        if count > MAX_HISTORY:
            excess   = count - MAX_HISTORY
            # Find the oldest `excess` entries and delete them
            cursor   = (
                db[HISTORY_COLLECTION]
                .find({}, {"_id": 1})
                .sort("timestamp", 1)
                .limit(excess)
            )
            old_ids = [doc["_id"] async for doc in cursor]
            if old_ids:
                await db[HISTORY_COLLECTION].delete_many({"_id": {"$in": old_ids}})
                logger.debug(f"[chat_history] Trimmed {len(old_ids)} old entries.")
    except Exception as e:
        # Trim failure is non-fatal — don't surface it to the user
        logger.warning(f"[chat_history] Trim failed (non-fatal): {e}")

    return str(result.inserted_id)


async def get_history(limit: int = 100) -> list[dict]:
    """
    Fetches recent history entries from MongoDB, newest first.

    Args:
        limit: Max entries to return (default 100, max 500).

    Returns:
        List of serializable dicts, each representing one past query.
        The "_id" ObjectId is converted to a plain string "id" field.
    """
    db     = get_stream_db()
    limit  = min(limit, MAX_HISTORY)

    cursor = (
        db[HISTORY_COLLECTION]
        .find({})
        .sort("timestamp", -1)   # newest first
        .limit(limit)
    )

    entries = []
    async for doc in cursor:
        entries.append({
            "id":              str(doc["_id"]),
            "question":        doc.get("question",        ""),
            "collection":      doc.get("collection",      ""),
            "result_count":    doc.get("result_count",    0),
            "result_label":    doc.get("result_label",    ""),
            "explanation":     doc.get("explanation",     ""),
            "elapsed_seconds": doc.get("elapsed_seconds", 0.0),
            # ISO 8601 string so the frontend can parse it with new Date()
            "timestamp": (
                doc["timestamp"].isoformat()
                if isinstance(doc.get("timestamp"), datetime)
                else str(doc.get("timestamp", ""))
            ),
        })

    return entries


async def delete_entry(entry_id: str) -> bool:
    """
    Deletes a single history entry by its MongoDB _id.

    Args:
        entry_id: The hex string representation of the ObjectId.

    Returns:
        True if the document was found and deleted, False if not found.
    """
    try:
        oid    = ObjectId(entry_id)
    except Exception:
        return False   # invalid ObjectId format

    db     = get_stream_db()
    result = await db[HISTORY_COLLECTION].delete_one({"_id": oid})
    return result.deleted_count == 1


async def clear_all() -> int:
    """
    Deletes all history entries from the collection.

    Returns:
        The number of documents deleted.
    """
    db     = get_stream_db()
    result = await db[HISTORY_COLLECTION].delete_many({})
    logger.info(f"[chat_history] Cleared {result.deleted_count} history entries.")
    return result.deleted_count
