import asyncio
import logging
from datetime import datetime, timezone

from bson import ObjectId
from lib.mongodb import get_stream_db

logger = logging.getLogger(__name__)

HISTORY_COLLECTION = "_QUERY_HISTORY_"
MAX_HISTORY        = 500


async def save_query(
    question:        str,
    collection:      str,
    result_count:    int,
    result_label:    str   = "",
    explanation:     str   = "",
    elapsed_seconds: float = 0.0,
) -> str:
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
    # asyncio.shield so the write survives if the HTTP client disconnects mid-response
    result = await asyncio.shield(db[HISTORY_COLLECTION].insert_one(doc))

    try:
        count = await db[HISTORY_COLLECTION].count_documents({})
        if count > MAX_HISTORY:
            excess  = count - MAX_HISTORY
            cursor  = db[HISTORY_COLLECTION].find({}, {"_id": 1}).sort("timestamp", 1).limit(excess)
            old_ids = [doc["_id"] async for doc in cursor]
            if old_ids:
                await db[HISTORY_COLLECTION].delete_many({"_id": {"$in": old_ids}})
    except Exception as e:
        logger.warning(f"[chat_history] Trim failed (non-fatal): {e}")

    return str(result.inserted_id)


async def get_history(limit: int = 100) -> list[dict]:
    db     = get_stream_db()
    cursor = db[HISTORY_COLLECTION].find({}).sort("timestamp", -1).limit(min(limit, MAX_HISTORY))

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
            "timestamp": (
                doc["timestamp"].isoformat()
                if isinstance(doc.get("timestamp"), datetime)
                else str(doc.get("timestamp", ""))
            ),
        })
    return entries


async def delete_entry(entry_id: str) -> bool:
    try:
        oid = ObjectId(entry_id)
    except Exception:
        return False
    result = await get_stream_db()[HISTORY_COLLECTION].delete_one({"_id": oid})
    return result.deleted_count == 1


async def clear_all() -> int:
    result = await get_stream_db()[HISTORY_COLLECTION].delete_many({})
    logger.info(f"[chat_history] Cleared {result.deleted_count} entries.")
    return result.deleted_count
