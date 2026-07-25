from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone

from bson import ObjectId

logger = logging.getLogger(__name__)

FEEDBACK_COLLECTION = "_QUERY_FEEDBACK_"


async def save_feedback(
    session_id:         str,
    question:           str,
    query_obj:          dict,
    result_count:       int,
    rating:             str,           # "good" | "bad"
    correction_note:    str  = "",
    corrected_pipeline: list | None = None,
    auto_warnings:      list | None = None,
) -> str:
    """Queue a feedback write and return immediately. The aux cluster can
    stall inserts for tens of seconds, so the _id is generated locally and
    the insert (plus the RAG follow-up) runs in background tasks.
    """
    doc_id_obj = ObjectId()
    doc_id     = str(doc_id_obj)

    doc = {
        "_id":                doc_id_obj,
        "session_id":         session_id,
        "question":           question,
        "query_obj":          query_obj,
        "result_count":       result_count,
        "rating":             rating,
        "correction_note":    correction_note or "",
        "corrected_pipeline": corrected_pipeline or [],
        "auto_warnings":      auto_warnings or [],
        "created_at":         datetime.now(timezone.utc),
        "used_in_rag":        False,
    }

    asyncio.create_task(_persist_feedback(doc))

    # trainer saves already wrote their own trainer_gold RAG entry — running
    # the normal pathway here would overwrite it with a weaker user_verified one
    is_trainer_audit = (correction_note or "").startswith("TRAINER_GOLD:")

    if not is_trainer_audit:
        asyncio.create_task(
            _process_feedback_for_rag(
                doc_id             = doc_id,
                rating             = rating,
                question           = question,
                query_obj          = query_obj,
                result_count       = result_count,
                corrected_pipeline = corrected_pipeline or [],
            )
        )

    logger.info(f"[feedback] '{rating}' queued for: '{question[:60]}' (id={doc_id})")
    return doc_id


async def _persist_feedback(doc: dict) -> None:
    from lib.mongodb import get_stream_db
    try:
        db = get_stream_db()
        await db[FEEDBACK_COLLECTION].insert_one(doc)
    except Exception as e:
        logger.error(f"[feedback] background insert failed for id={doc.get('_id')}: {e}")


async def _process_feedback_for_rag(
    doc_id:             str,
    rating:             str,
    question:           str,
    query_obj:          dict,
    result_count:       int,
    corrected_pipeline: list,
) -> None:
    from lib.mongodb import get_stream_db
    from lib.query_examples import (
        add_verified_example, add_corrected_example,
        find_similar_examples_vector, bump_example,
    )

    try:
        db_hint = _infer_db_hint(query_obj)

        # bump the examples that would have been retrieved for this question
        # (good = success, bad = failure). Has to run before add_*_example,
        # otherwise the fresh entry we write below gets counted too.
        try:
            retrieved = await find_similar_examples_vector(question, db_hint, top_n=3)
            for r in (retrieved or []):
                ex_id = r.get("id")
                if not ex_id:
                    continue
                bump_example(ex_id, success=(rating == "good"))
        except Exception as e:
            logger.error(f"[feedback] bump retrieved failed: {e}")

        if rating == "good":
            await add_verified_example(question, query_obj, result_count, db_hint)
            logger.info(f"[feedback] Embedded verified example (weight=2.0) for: '{question[:60]}'")

        elif rating == "bad" and corrected_pipeline:
            corrected = dict(query_obj)
            qt = query_obj.get("queryType", "single")
            if qt == "single":
                corrected["pipeline"] = corrected_pipeline
            elif qt == "dual" and query_obj.get("queries"):
                queries = [dict(q) for q in query_obj["queries"]]
                queries[0]["pipeline"] = corrected_pipeline
                corrected["queries"] = queries

            await add_corrected_example(question, corrected, result_count, db_hint)
            logger.info(f"[feedback] Embedded corrected example (weight=2.5) for: '{question[:60]}'")

        from bson import ObjectId
        db = get_stream_db()
        await db[FEEDBACK_COLLECTION].update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"used_in_rag": True}},
        )

    except Exception as e:
        logger.error(f"[feedback] RAG processing failed for id={doc_id}: {e}")


def _infer_db_hint(query_obj: dict) -> str:
    if query_obj.get("queryType") == "dual":
        return "both"
    db = query_obj.get("database", "stream-datastore")
    return "appconfigs" if db == "appConfigs" else "stream"


async def get_feedback_stats() -> dict:
    try:
        from lib.mongodb import get_stream_db
        db    = get_stream_db()
        total = await db[FEEDBACK_COLLECTION].count_documents({})
        good  = await db[FEEDBACK_COLLECTION].count_documents({"rating": "good"})
        bad   = await db[FEEDBACK_COLLECTION].count_documents({"rating": "bad"})
        return {"total": total, "good": good, "bad": bad}
    except Exception:
        return {"total": 0, "good": 0, "bad": 0}
