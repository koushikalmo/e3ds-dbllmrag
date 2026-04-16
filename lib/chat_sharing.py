
import asyncio
import uuid
from datetime import datetime, timezone

from lib.mongodb import get_stream_db

SHARE_COLLECTION = "_SHARED_CHATS_"

# Shareable chat generate and store in db


async def create_share(turns: list, title: str = "") -> str:
    """
    Saves a chat session snapshot to MongoDB and returns a unique share ID.

    Args:
        turns: List of turn dicts, each with question/data/error/ts fields.
        title: Human-readable title for the shared chat (first question).

    Returns:
        16-character hex share ID (e.g. 'a3f9c1d2e4b78605').
    """
    share_id = uuid.uuid4().hex[:16]
    db = get_stream_db()
    await asyncio.shield(db[SHARE_COLLECTION].insert_one({
        "share_id":   share_id,
        "title":      title or "Shared Chat",
        "turns":      turns,
        "created_at": datetime.now(timezone.utc),
        "view_count": 0,
    }))
    return share_id

# Shareable chat access using link  

async def get_share(share_id: str) -> dict | None:
    """
    Retrieves a shared chat by ID and increments its view counter.

    Args:
        share_id: The 16-character hex ID returned by create_share().

    Returns:
        Dict with share_id, title, turns, created_at, view_count.
        None if not found.
    """
    db  = get_stream_db()
    doc = await db[SHARE_COLLECTION].find_one({"share_id": share_id})
    if not doc:
        return None

    await db[SHARE_COLLECTION].update_one(
        {"share_id": share_id},
        {"$inc": {"view_count": 1}},
    )

    return {
        "share_id":   doc["share_id"],
        "title":      doc.get("title", "Shared Chat"),
        "turns":      doc.get("turns", []),
        "created_at": (
            doc["created_at"].isoformat()
            if isinstance(doc.get("created_at"), datetime)
            else str(doc.get("created_at", ""))
        ),
        "view_count": doc.get("view_count", 0) + 1,
    }
