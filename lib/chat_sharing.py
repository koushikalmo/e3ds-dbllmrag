from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from lib.mongodb import get_stream_db

logger = logging.getLogger(__name__)

SHARE_COLLECTION = "_SHARED_CHATS_"

async def create_share(turns: list, title: str = "") -> str:
    """Make a 16-char share id and return it right away — the Mongo insert
    runs in the background because the aux cluster sometimes stalls writes
    and nothing about the id depends on the write finishing.
    """
    share_id = uuid.uuid4().hex[:16]
    doc = {
        "share_id":   share_id,
        "title":      title or "Shared Chat",
        "turns":      turns,
        "created_at": datetime.now(timezone.utc),
        "view_count": 0,
    }
    asyncio.create_task(_persist_share(doc))
    return share_id


async def _persist_share(doc: dict) -> None:
    try:
        db = get_stream_db()
        await db[SHARE_COLLECTION].insert_one(doc)
    except Exception as e:
        logger.error(f"[share] background insert failed for id={doc.get('share_id')}: {e}")


async def _bump_view_count(share_id: str) -> None:
    try:
        db = get_stream_db()
        await db[SHARE_COLLECTION].update_one(
            {"share_id": share_id},
            {"$inc": {"view_count": 1}},
        )
    except Exception as e:
        logger.error(f"[share] background view_count bump failed for {share_id}: {e}")

async def get_share(share_id: str) -> dict | None:
    """Look up a shared chat and bump its view counter. None if not found."""
    db  = get_stream_db()
    # 5s ceiling — if the aux cluster is stalled, showing "not found" beats
    # hanging the browser tab
    doc = await db[SHARE_COLLECTION].find_one(
        {"share_id": share_id},
        max_time_ms=5_000,
    )
    if not doc:
        return None

    asyncio.create_task(_bump_view_count(share_id))

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
