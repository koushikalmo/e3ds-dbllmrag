from __future__ import annotations
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_EXAMPLES_FILE = Path(__file__).parent.parent / "data" / "query_examples.json"
MAX_EXAMPLES   = 500
TOP_N          = 3


BOOTSTRAP_EXAMPLES = [
    {
        "question": "How many sessions per country this month?",
        "query": {
            "queryType": "single", "database": "stream-datastore", "collection": "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False}},
                {"$group": {"_id": "$clientInfo.country_name", "sessions": {"$sum": 1}}},
                {"$sort": {"sessions": -1}}, {"$limit": 50},
            ],
            "explanation": "Groups real user sessions by country and counts them descending.",
            "resultLabel": "Sessions by Country",
        },
        "result_count": 45, "db_hint": "stream",
    },
    {
        "question": "What is the average load time per owner?",
        "query": {
            "queryType": "single", "database": "stream-datastore", "collection": "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False, "loadTime": {"$gt": 0}}},
                {"$group": {"_id": "$appInfo.owner", "avgLoadTime": {"$avg": "$loadTime"}, "sessions": {"$sum": 1}}},
                {"$sort": {"avgLoadTime": -1}}, {"$limit": 50},
            ],
            "explanation": "Calculates average load time in seconds per owner, slowest first.",
            "resultLabel": "Average Load Time by Owner",
        },
        "result_count": 30, "db_hint": "stream",
    },
    {
        "question": "Show sessions with average round trip time over 200ms",
        "query": {
            "queryType": "single", "database": "stream-datastore", "collection": "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False, "webRtcStatsData.avgRoundTripTime": {"$exists": True}}},
                {"$addFields": {"rttFloat": {"$toDouble": "$webRtcStatsData.avgRoundTripTime"}}},
                {"$match": {"rttFloat": {"$gt": 0.2}}},
                {"$project": {"appInfo.owner": 1, "clientInfo.city": 1, "clientInfo.country_name": 1, "rttFloat": 1, "loadTime": 1}},
                {"$sort": {"rttFloat": -1}}, {"$limit": 50},
            ],
            "explanation": "Finds sessions with RTT >200ms, converting the string field to number first.",
            "resultLabel": "High Latency Sessions",
        },
        "result_count": 22, "db_hint": "stream",
    },
    {
        "question": "What is the total streaming time per user this month?",
        "query": {
            "queryType": "single", "database": "stream-datastore", "collection": "Apr_2026",
            "pipeline": [
                {"$match": {"e3ds_employee": {"$ne": True}, "VideoStreamStartedAt_Timestamp": {"$exists": True}}},
                {"$addFields": {"streamEndTs": {"$ifNull": ["$VideoStreamContinuedAt_Timestamp", "$DataChannelHeartBeatReceivedAt_Timestamp"]}}},
                {"$addFields": {"durationMs": {"$subtract": ["$streamEndTs", "$VideoStreamStartedAt_Timestamp"]}}},
                {"$match": {"durationMs": {"$gt": 0}}},
                {"$group": {
                    "_id": "$loggedInUserData.name",
                    "userName": {"$first": "$loggedInUserData.name"},
                    "sessionCount": {"$sum": 1},
                    "totalMinutesStreamed": {"$sum": {"$divide": ["$durationMs", 60000]}},
                }},
                {"$sort": {"totalMinutesStreamed": -1}}, {"$limit": 50},
            ],
            "explanation": "Duration = VideoStreamContinuedAt_Timestamp (fallback DataChannelHeartBeatReceivedAt_Timestamp) - VideoStreamStartedAt_Timestamp / 60000.",
            "resultLabel": "Streaming Time by User",
        },
        "result_count": 30, "db_hint": "stream",
    },
    {
        "question": "What is the average session duration per city?",
        "query": {
            "queryType": "single", "database": "stream-datastore", "collection": "Apr_2026",
            "pipeline": [
                {"$match": {"e3ds_employee": {"$ne": True}, "VideoStreamStartedAt_Timestamp": {"$exists": True}}},
                {"$addFields": {"streamEndTs": {"$ifNull": ["$VideoStreamContinuedAt_Timestamp", "$DataChannelHeartBeatReceivedAt_Timestamp"]}}},
                {"$addFields": {"durationMs": {"$subtract": ["$streamEndTs", "$VideoStreamStartedAt_Timestamp"]}}},
                {"$match": {"durationMs": {"$gt": 0}}},
                {"$group": {"_id": "$clientInfo.city", "avgDurationMinutes": {"$avg": {"$divide": ["$durationMs", 60000]}}, "sessions": {"$sum": 1}}},
                {"$sort": {"sessions": -1}}, {"$limit": 50},
            ],
            "explanation": "Average session duration in minutes per city. Uses VideoStreamStartedAt_Timestamp and DisconnectTime_Timestamp (both milliseconds).",
            "resultLabel": "Session Duration by City",
        },
        "result_count": 80, "db_hint": "stream",
    },
    {
        "question": "Which browsers are most commonly used by viewers?",
        "query": {
            "queryType": "single", "database": "stream-datastore", "collection": "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False}},
                {"$group": {"_id": "$userDeviceInfo.client.name", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}, {"$limit": 20},
            ],
            "explanation": "Groups sessions by browser name, sorted by most popular.",
            "resultLabel": "Sessions by Browser",
        },
        "result_count": 8, "db_hint": "stream",
    },
    {
        "question": "Which servers (computers) are handling the most sessions?",
        "query": {
            "queryType": "single", "database": "stream-datastore", "collection": "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False}},
                {"$group": {"_id": "$elInfo.computerName", "sessions": {"$sum": 1}, "avgBitrate": {"$avg": "$webRtcStatsData.avgBitrate"}}},
                {"$sort": {"sessions": -1}}, {"$limit": 20},
            ],
            "explanation": "Groups sessions by server hostname, showing most-used servers.",
            "resultLabel": "Sessions by Server",
        },
        "result_count": 15, "db_hint": "stream",
    },
    {
        "question": "Show me the subscription status for owner eduardo",
        "query": {
            "queryType": "single", "database": "appConfigs", "collection": "eduardo",
            "pipeline": [
                {"$match": {"_id": "usersinfo"}},
                {"$project": {"maxUserLimit": 1, "shouldAutoRenew": 1, "paidMinutes": 1, "paidSecondsUsage": 1, "products": 1,
                              "SubscriptionEndDate._seconds": 1, "SubscriptionStartDate._seconds": 1,
                              "apiKeys.apiKey": 0, "streamingApiKeys.apiKey": 0}},
            ],
            "explanation": "Fetches subscription and billing data for owner 'eduardo'.",
            "resultLabel": "Subscription Status: eduardo",
        },
        "result_count": 1, "db_hint": "appconfigs",
    },
    {
        "question": "Which sessions had packet loss above 500?",
        "query": {
            "queryType": "single", "database": "stream-datastore", "collection": "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False, "webRtcStatsData.packetsLost": {"$gt": 500}}},
                {"$project": {"appInfo.owner": 1, "clientInfo.city": 1, "clientInfo.country_name": 1,
                              "webRtcStatsData.packetsLost": 1, "webRtcStatsData.avgBitrate": 1, "loadTime": 1}},
                {"$sort": {"webRtcStatsData.packetsLost": -1}}, {"$limit": 50},
            ],
            "explanation": "Finds sessions with packet loss >500, sorted worst-first.",
            "resultLabel": "High Packet Loss Sessions",
        },
        "result_count": 18, "db_hint": "stream",
    },
    {
        "question": "Which owners had sessions this month and what is their subscription limit?",
        "query": {
            "queryType": "dual",
            "queries": [
                {"database": "stream-datastore", "collection": "Apr_2025", "pipeline": [
                    {"$match": {"e3ds_employee": False}},
                    {"$group": {"_id": "$appInfo.owner", "sessions": {"$sum": 1}, "avgLoad": {"$avg": "$loadTime"}}},
                    {"$sort": {"sessions": -1}}, {"$limit": 50},
                ]},
                {"database": "appConfigs", "collection": "eduardo", "pipeline": [
                    {"$match": {"_id": "usersinfo"}},
                    {"$project": {"maxUserLimit": 1, "SubscriptionEndDate._seconds": 1, "apiKeys.apiKey": 0, "streamingApiKeys.apiKey": 0}},
                ]},
            ],
            "mergeKey": "owner",
            "explanation": "Session counts from stream-datastore merged with subscription limits from appConfigs.",
            "resultLabel": "Owner Sessions + Subscription Limits",
        },
        "result_count": 20, "db_hint": "both",
    },
    {
        "question": "Give all users name and streaming time of april 16 2026",
        "query": {
            "queryType": "single", "database": "stream-datastore", "collection": "Apr_2026",
            "operation": "aggregate",
            "pipeline": [
                {"$match": {
                    "e3ds_employee": {"$ne": True},
                    "VideoStreamStartedAt_Timestamp": {"$gte": 1776297600000, "$lt": 1776384000000},
                }},
                {"$addFields": {
                    "streamEndTs": {"$ifNull": ["$VideoStreamContinuedAt_Timestamp", "$DataChannelHeartBeatReceivedAt_Timestamp"]},
                }},
                {"$addFields": {
                    "durationMs": {"$subtract": ["$streamEndTs", "$VideoStreamStartedAt_Timestamp"]},
                }},
                {"$match": {"durationMs": {"$gt": 0}}},
                {"$group": {
                    "_id": "$loggedInUserData.name",
                    "userName": {"$first": "$loggedInUserData.name"},
                    "sessionCount": {"$sum": 1},
                    "totalMinutesStreamed": {"$sum": {"$divide": ["$durationMs", 60000]}},
                }},
                {"$sort": {"totalMinutesStreamed": -1}},
                {"$limit": 200},
            ],
            "explanation": "Duration = VideoStreamContinuedAt_Timestamp (or DataChannelHeartBeatReceivedAt_Timestamp fallback) - VideoStreamStartedAt_Timestamp, divided by 60000 for minutes.",
            "resultLabel": "User Streaming Time — Apr 16 2026",
        },
        "result_count": 13, "db_hint": "stream",
    },
    {
        "question": "Show active subscriptions with sessions count for each owner",
        "query": {
            "queryType": "dual",
            "queries": [
                {"database": "stream-datastore", "collection": "Apr_2025", "pipeline": [
                    {"$match": {"e3ds_employee": False}},
                    {"$group": {"_id": "$appInfo.owner", "sessions": {"$sum": 1}}},
                    {"$sort": {"sessions": -1}}, {"$limit": 50},
                ]},
                {"database": "appConfigs", "collection": "eduardo", "pipeline": [
                    {"$match": {"_id": "usersinfo"}},
                    {"$project": {"maxUserLimit": 1, "SubscriptionEndDate._seconds": 1,
                                  "SubscriptionStartDate._seconds": 1, "apiKeys.apiKey": 0, "streamingApiKeys.apiKey": 0}},
                ]},
            ],
            "mergeKey": "owner",
            "explanation": "Session counts per owner joined with subscription data from appConfigs.",
            "resultLabel": "Sessions + Subscription Status",
        },
        "result_count": 15, "db_hint": "both",
    },
]


def _load_examples() -> list[dict]:
    if not _EXAMPLES_FILE.exists():
        _EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
        _save_examples(BOOTSTRAP_EXAMPLES)
        return list(BOOTSTRAP_EXAMPLES)
    try:
        with open(_EXAMPLES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"[query_examples] Failed to load: {e}")
        return list(BOOTSTRAP_EXAMPLES)


def _save_examples(examples: list[dict]) -> None:
    try:
        _EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_EXAMPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(examples[:MAX_EXAMPLES], f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.error(f"[query_examples] Failed to save: {e}")


def _tokenize(text: str) -> set[str]:
    import re
    stopwords = {"the", "a", "an", "is", "are", "was", "of", "in", "for", "to", "and", "or",
                 "by", "with", "this", "that", "what", "which", "how", "many", "show", "me",
                 "get", "list", "find", "from", "all", "any"}
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in stopwords}


# cap on the effective-weight multiplier, otherwise a good example keeps
# boosting itself through its own accuracy updates and drowns out everything
_EFFECTIVE_WEIGHT_CAP = 5.0


def _days_since(iso_ts: str | None) -> float:
    if not iso_ts:
        return 10_000.0  # treat unknown as ancient
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 10_000.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    return delta.total_seconds() / 86400.0


def effective_weight(m: dict) -> float:
    base     = float(m.get("weight", 1.0))

    s        = int(m.get("success_count", 0))
    f        = int(m.get("failure_count", 0))
    rate     = (s + 1) / (s + f + 2)                  # Laplace-smoothed ∈ (0, 1)

    a_sum    = float(m.get("accuracy_sum", 0.0))
    a_n      = int(m.get("accuracy_n", 0))
    acc_mean = (a_sum + 50) / (a_n + 1) / 100.0       # Laplace-smoothed ∈ (0, 1)

    age_days = _days_since(m.get("last_used") or m.get("created_at") or m.get("timestamp"))
    if   age_days <  30: decay = 1.0
    elif age_days <  90: decay = 0.9
    elif age_days < 180: decay = 0.75
    else:                decay = 0.5

    weighted = base * (0.5 + rate) * (0.5 + acc_mean) * decay
    return min(weighted, _EFFECTIVE_WEIGHT_CAP)


async def find_similar_examples_vector(question: str, db_hint: str = "stream", top_n: int = TOP_N) -> list[dict] | None:
    from lib.embeddings   import embed
    from lib.vector_store import VectorStore

    store = VectorStore("examples")
    if store.count() == 0:
        return None

    q_emb = await embed(question)
    if q_emb is None:
        return None

    def _filter(item: dict) -> bool:
        hint = item["metadata"].get("db_hint", "stream")
        return hint == db_hint or hint == "both" or db_hint == "both"

    # pull a wider pool so a high-weight older example can beat a newer one
    candidates = store.search(q_emb, top_k=top_n * 4, filter_fn=_filter, min_score=0.3)
    if not candidates:
        return None

    # rank by cosine similarity x effective weight
    weighted = sorted(
        candidates,
        key=lambda r: r["score"] * effective_weight(r["metadata"]),
        reverse=True,
    )
    top_results = weighted[:top_n]

    return [
        {
            "id":               r["id"],
            "question":         r["text"],
            "query":            r["metadata"].get("query", {}),
            "result_count":     r["metadata"].get("result_count", 0),
            "db_hint":          r["metadata"].get("db_hint", "stream"),
            "weight":           r["metadata"].get("weight", 1.0),
            "effective_weight": round(effective_weight(r["metadata"]), 3),
        }
        for r in top_results
    ]


def bump_example(example_id: str, *, success: bool, accuracy: int | None = None) -> bool:
    from lib.vector_store import VectorStore
    return VectorStore("examples").bump(example_id, success=success, accuracy=accuracy)


def rag_stats(top_n: int = 10) -> dict:
    # snapshot of the example store for the admin stats endpoint
    from lib.vector_store import VectorStore
    store  = VectorStore("examples")
    items  = store.all_items()
    total  = len(items)
    if total == 0:
        return {"total": 0, "by_source": {}, "avg_weight": 0.0, "avg_accuracy": 0.0,
                "top_10": [], "bottom_10": [], "stale": 0}

    by_source: dict[str, int] = {}
    weight_sum   = 0.0
    acc_sum_all  = 0.0
    acc_n_all    = 0
    stale        = 0
    ranked: list[tuple[float, dict]] = []

    for it in items:
        m   = it.get("metadata", {}) or {}
        src = m.get("source", "auto")
        by_source[src] = by_source.get(src, 0) + 1

        weight_sum  += float(m.get("weight", 1.0))
        acc_sum_all += float(m.get("accuracy_sum", 0.0))
        acc_n_all   += int(m.get("accuracy_n", 0))

        if _days_since(m.get("last_used") or m.get("created_at") or m.get("timestamp")) > 90:
            stale += 1

        ew = effective_weight(m)
        ranked.append((ew, {
            "id":               it.get("id", ""),
            "question":         (it.get("text") or "")[:120],
            "source":           src,
            "weight":           float(m.get("weight", 1.0)),
            "effective_weight": round(ew, 3),
            "success_count":    int(m.get("success_count", 0)),
            "failure_count":    int(m.get("failure_count", 0)),
            "accuracy_n":       int(m.get("accuracy_n", 0)),
            "accuracy_mean":    round((float(m.get("accuracy_sum", 0.0)) / int(m.get("accuracy_n", 0)))
                                      if int(m.get("accuracy_n", 0)) > 0 else 0.0, 1),
            "last_used":        m.get("last_used") or m.get("created_at") or m.get("timestamp"),
        }))

    ranked.sort(key=lambda x: x[0], reverse=True)
    top    = [r[1] for r in ranked[:top_n]]
    bottom = [r[1] for r in ranked[-top_n:][::-1]]

    avg_acc = round(acc_sum_all / acc_n_all, 1) if acc_n_all > 0 else 0.0

    return {
        "total":        total,
        "by_source":    by_source,
        "avg_weight":   round(weight_sum / total, 3),
        "avg_accuracy": avg_acc,
        "stale":        stale,
        "top_10":       top,
        "bottom_10":    bottom,
    }


def prune_examples(dry_run: bool = False) -> dict:
    """Drop stale or repeatedly-failing examples. First matching rule wins:
    trainer_gold never goes; user_verified only after a year unused; auto
    entries go after 3 failures with no success, 180 days unused with no
    success, or 5+ accuracy samples averaging under 40.
    """
    from lib.vector_store import VectorStore
    store = VectorStore("examples")
    items = list(store.all_items())   # snapshot — we mutate via remove()

    removed: list[dict] = []
    kept                = 0

    def _verdict(m: dict) -> tuple[str, str | None]:
        src       = m.get("source", "auto")
        last_used = m.get("last_used") or m.get("created_at") or m.get("timestamp")
        s         = int(m.get("success_count", 0))
        f         = int(m.get("failure_count", 0))
        a_n       = int(m.get("accuracy_n", 0))
        a_mean    = (float(m.get("accuracy_sum", 0.0)) / a_n) if a_n > 0 else 0.0

        if src == "trainer_gold":
            return ("keep", "trainer_gold")
        if src == "user_verified" and _days_since(last_used) > 365:
            return ("remove", "verified_but_unused_>365d")
        if f >= 3 and s == 0:
            return ("remove", "failures_without_successes")
        if _days_since(last_used) > 180 and s == 0:
            return ("remove", "unused_>180d_no_success")
        if a_n >= 5 and a_mean < 40:
            return ("remove", "low_accuracy_after_evidence")
        return ("keep", None)

    for it in items:
        m = it.get("metadata", {}) or {}
        verdict, reason = _verdict(m)
        if verdict == "remove":
            entry = {
                "id":       it.get("id", ""),
                "question": (it.get("text") or "")[:100],
                "source":   m.get("source", "auto"),
                "reason":   reason,
            }
            removed.append(entry)
            if not dry_run:
                store.remove(it["id"])
        else:
            kept += 1

    if not dry_run and removed:
        # keep query_examples.json in sync with the vector store
        removed_questions = {entry["question"] for entry in removed}
        local = _load_examples()
        filtered = [
            e for e in local
            if (e.get("question") or "")[:100] not in removed_questions
        ]
        if len(filtered) != len(local):
            _save_examples(filtered)

    logger.info(f"[prune] {'dry-run ' if dry_run else ''}removed={len(removed)} kept={kept}")
    return {"removed": len(removed), "kept": kept, "dry_run": dry_run, "details": removed}


def migrate_phase4_metadata() -> int:
    """One-off stamp for entries created before timestamps existed. Without it
    the age-decay treats them as ancient and the pruner deletes them. Skips
    anything that already has created_at, so it's safe on every startup.
    """
    from lib.vector_store import VectorStore
    store = VectorStore("examples")
    now   = datetime.now(timezone.utc).isoformat()
    migrated = 0
    for it in store.all_items():
        m = it.setdefault("metadata", {})
        if "created_at" in m and "last_used" in m:
            continue
        m.setdefault("created_at",    m.get("timestamp") or now)
        m.setdefault("last_used",     m.get("timestamp") or now)
        m.setdefault("success_count", 0)
        m.setdefault("failure_count", 0)
        m.setdefault("accuracy_sum",  0.0)
        m.setdefault("accuracy_n",    0)
        migrated += 1
    if migrated:
        store._save()
        logger.info(f"[migrate_phase4] stamped {migrated} legacy entries with timestamps")
    return migrated


async def _prune_scheduler(interval_seconds: int = 6 * 3600) -> None:
    # background prune loop, started from the lifespan hook in main.py
    import asyncio
    # don't prune right at startup — it competes with cache warming
    await asyncio.sleep(interval_seconds)
    while True:
        try:
            result = prune_examples(dry_run=False)
            logger.info(f"[prune_scheduler] removed={result['removed']} kept={result['kept']}")
        except Exception as e:
            logger.error(f"[prune_scheduler] failed: {e}")
        await asyncio.sleep(interval_seconds)


def find_similar_examples(question: str, db_hint: str = "stream", top_n: int = TOP_N) -> list[dict]:
    examples      = _load_examples()
    question_words = _tokenize(question)
    if not examples or not question_words:
        return []

    scored = []
    for ex in examples:
        ex_words = _tokenize(ex.get("question", ""))
        if not ex_words:
            continue
        score = len(question_words & ex_words) / len(question_words)
        if ex.get("db_hint", "stream") == db_hint:
            score += 0.05
        if score > 0:
            scored.append((score, ex))

    scored.sort(key=lambda x: (x[0], x[1].get("result_count", 0)), reverse=True)
    return [ex for _, ex in scored[:top_n]]


def add_example(
    question:       str,
    query_obj:      dict,
    result_count:   int,
    db_hint:        str        = "stream",
    accuracy_score: int | None = None,
) -> None:
    if result_count == 0:
        return

    existing = find_similar_examples(question, db_hint, top_n=1)
    if existing:
        similarity = len(_tokenize(question) & _tokenize(existing[0].get("question", ""))) / max(len(_tokenize(question)), 1)
        if similarity > 0.85:
            return

    new_example = {
        "question": question, "query": query_obj,
        "result_count":   result_count,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "db_hint":        db_hint,
        "accuracy_score": accuracy_score,
    }
    examples = _load_examples()
    examples.insert(0, new_example)
    _save_examples(examples)

    import asyncio
    asyncio.create_task(_index_example_async(
        question, query_obj, result_count, db_hint,
        accuracy_score=accuracy_score,
    ))


async def _index_example_async(
    question:       str,
    query_obj:      dict,
    result_count:   int,
    db_hint:        str,
    weight:         float      = 1.0,  # 1.0=auto, 2.0=user verified, 2.5=user corrected
    source:         str        = "auto",
    accuracy_score: int | None = None,
) -> None:
    import hashlib
    from lib.embeddings   import embed
    from lib.vector_store import VectorStore

    emb = await embed(question)
    if emb is None:
        return

    store   = VectorStore("examples")
    item_id = hashlib.sha1(question.encode()).hexdigest()
    now     = datetime.now(timezone.utc).isoformat()

    # upsert: keep the success/failure counters if this question already exists
    prev = next((i for i in store.all_items() if i["id"] == item_id), None)
    prev_meta = (prev or {}).get("metadata", {}) if prev else {}

    store.upsert(
        id        = item_id,
        text      = question,
        embedding = emb,
        metadata  = {
            "query":          query_obj,
            "result_count":   result_count,
            "db_hint":        db_hint,
            "weight":         weight,
            "source":         source,
            "accuracy_score": accuracy_score,
            "created_at":     prev_meta.get("created_at", now),
            "last_used":      prev_meta.get("last_used", now),
            "last_good_at":   prev_meta.get("last_good_at"),
            "success_count":  int(prev_meta.get("success_count", 0)),
            "failure_count": int(prev_meta.get("failure_count", 0)),
            "accuracy_sum":  float(prev_meta.get("accuracy_sum", 0.0)),
            "accuracy_n":    int(prev_meta.get("accuracy_n", 0)),
        },
    )
    store.trim_to(MAX_EXAMPLES)


def format_examples_for_prompt(examples: list[dict]) -> str:
    if not examples:
        return ""

    lines = [
        "─────────────────────────────────────────────────────────────",
        "SIMILAR EXAMPLES — correct pipelines for similar questions:",
        "(Use these as a reference pattern for the current question.)",
        "─────────────────────────────────────────────────────────────",
    ]
    for i, ex in enumerate(examples, 1):
        try:
            query_json = json.dumps(ex.get("query", {}), separators=(",", ":"))
            if len(query_json) > 800:
                query_json = query_json[:797] + "..."
        except Exception:
            query_json = str(ex.get("query", ""))[:800]
        lines.append(f"\nExample {i}:")
        lines.append(f"  Question: {ex.get('question', '')}")
        lines.append(f"  Correct answer ({ex.get('result_count', 0)} results): {query_json}")

    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("")
    return "\n".join(lines)


async def add_verified_example(question: str, query_obj: dict, result_count: int, db_hint: str = "stream") -> None:
    _update_example_weight_in_file(question, 2.0)
    await _index_example_async(question, query_obj, result_count, db_hint, weight=2.0, source="user_verified")


async def add_trainer_gold_example(
    question:     str,
    query_obj:    dict,
    result_count: int,
    db_hint:      str = "stream",
) -> str:
    """Trainer-authored gold example: weight 3.0. The JSON write is quick and
    happens inline; the embedding takes seconds so it's pushed to a background
    task. Returns the sha1 id of the entry.
    """
    import asyncio, hashlib
    new_example = {
        "question":     question,
        "query":        query_obj,
        "result_count": result_count,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "db_hint":      db_hint,
        "weight":       3.0,
        "source":       "trainer_gold",
    }
    examples = _load_examples()
    q_tokens = _tokenize(question)
    examples = [e for e in examples if _tokenize(e.get("question", "")) != q_tokens]
    examples.insert(0, new_example)
    _save_examples(examples)

    asyncio.create_task(_index_example_async(
        question, query_obj, result_count, db_hint,
        weight=3.0, source="trainer_gold",
    ))
    return hashlib.sha1(question.encode()).hexdigest()


async def add_corrected_example(question: str, query_obj: dict, result_count: int, db_hint: str = "stream") -> None:
    new_example = {
        "question":     question,
        "query":        query_obj,
        "result_count": result_count,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "db_hint":      db_hint,
        "weight":       2.5,
        "source":       "user_corrected",
    }
    examples = _load_examples()
    # replace any older entry for the same question
    examples = [e for e in examples if _tokenize(e.get("question", "")) != _tokenize(question)]
    examples.insert(0, new_example)
    _save_examples(examples)
    await _index_example_async(question, query_obj, result_count, db_hint, weight=2.5, source="user_corrected")


def _update_example_weight_in_file(question: str, weight: float) -> None:
    q_tokens = _tokenize(question)
    examples = _load_examples()
    changed = False
    for ex in examples:
        if _tokenize(ex.get("question", "")) == q_tokens:
            ex["weight"] = weight
            ex["source"] = "user_verified"
            changed = True
            break
    if changed:
        _save_examples(examples)


def get_example_count() -> int:
    return len(_load_examples())


async def index_all_examples_async() -> int:
    import hashlib
    from lib.embeddings   import embed
    from lib.vector_store import VectorStore

    examples = _load_examples()
    if not examples:
        return 0

    store   = VectorStore("examples")
    indexed = 0

    for ex in examples:
        question = ex.get("question", "")
        if not question:
            continue
        item_id = hashlib.sha1(question.encode()).hexdigest()
        if item_id in store.ids():
            continue
        emb = await embed(question)
        if emb is None:
            break
        store.upsert(
            id        = item_id,
            text      = question,
            embedding = emb,
            metadata  = {
                "query":        ex.get("query", {}),
                "result_count": ex.get("result_count", 0),
                "db_hint":      ex.get("db_hint", "stream"),
                "weight":       ex.get("weight", 1.0),
                "source":       ex.get("source", "auto"),
            },
        )
        indexed += 1

    if indexed > 0:
        logger.info(f"[query_examples] Indexed {indexed} example(s).")
    return indexed
