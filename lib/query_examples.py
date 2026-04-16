# lib/query_examples.py — RAG few-shot example store
# Saves successful queries and retrieves similar ones to prepend to LLM prompts.
# Primary search: semantic vector similarity. Fallback: keyword overlap.

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
        "question": "What is the average session duration per city?",
        "query": {
            "queryType": "single", "database": "stream-datastore", "collection": "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False, "DisconnectTime_Timestamp": {"$exists": True}, "startTimeStamp": {"$exists": True}}},
                {"$addFields": {"durationSeconds": {"$subtract": ["$DisconnectTime_Timestamp", "$startTimeStamp"]}}},
                {"$match": {"durationSeconds": {"$gt": 0}}},
                {"$group": {"_id": "$clientInfo.city", "avgDuration": {"$avg": "$durationSeconds"}, "sessions": {"$sum": 1}}},
                {"$sort": {"sessions": -1}}, {"$limit": 50},
            ],
            "explanation": "Calculates average session duration in seconds per city.",
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


async def find_similar_examples_vector(question: str, db_hint: str = "stream", top_n: int = TOP_N) -> list[dict] | None:
    """Semantic search via embeddings. Returns None if embedding model is unavailable."""
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

    results = store.search(q_emb, top_k=top_n, filter_fn=_filter, min_score=0.4)
    if not results:
        return None

    return [
        {"question": r["text"], "query": r["metadata"].get("query", {}),
         "result_count": r["metadata"].get("result_count", 0), "db_hint": r["metadata"].get("db_hint", "stream")}
        for r in results
    ]


def find_similar_examples(question: str, db_hint: str = "stream", top_n: int = TOP_N) -> list[dict]:
    """Keyword overlap fallback when vector search is unavailable."""
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


def add_example(question: str, query_obj: dict, result_count: int, db_hint: str = "stream") -> None:
    """Save a successful query as a future few-shot example. Skips 0-result queries and near-duplicates."""
    if result_count == 0:
        return

    existing = find_similar_examples(question, db_hint, top_n=1)
    if existing:
        similarity = len(_tokenize(question) & _tokenize(existing[0].get("question", ""))) / max(len(_tokenize(question)), 1)
        if similarity > 0.85:
            return  # near-duplicate already stored

    new_example = {
        "question": question, "query": query_obj,
        "result_count": result_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "db_hint": db_hint,
    }
    examples = _load_examples()
    examples.insert(0, new_example)
    _save_examples(examples)

    import asyncio
    asyncio.create_task(_index_example_async(question, query_obj, result_count, db_hint))


async def _index_example_async(question: str, query_obj: dict, result_count: int, db_hint: str) -> None:
    """Background task: embed question and store in vector store for semantic search."""
    import hashlib
    from lib.embeddings   import embed
    from lib.vector_store import VectorStore

    emb = await embed(question)
    if emb is None:
        return

    store = VectorStore("examples")
    store.upsert(
        id       = hashlib.sha1(question.encode()).hexdigest(),
        text     = question,
        embedding= emb,
        metadata = {"query": query_obj, "result_count": result_count, "db_hint": db_hint},
    )
    store.trim_to(MAX_EXAMPLES)


def format_examples_for_prompt(examples: list[dict]) -> str:
    """Format retrieved examples as a few-shot block for the LLM prompt."""
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


def get_example_count() -> int:
    return len(_load_examples())


async def index_all_examples_async() -> int:
    """Index all examples into the vector store on startup."""
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
            id       = item_id,
            text     = question,
            embedding= emb,
            metadata = {"query": ex.get("query", {}), "result_count": ex.get("result_count", 0), "db_hint": ex.get("db_hint", "stream")},
        )
        indexed += 1

    if indexed > 0:
        logger.info(f"[query_examples] Indexed {indexed} example(s).")
    return indexed
