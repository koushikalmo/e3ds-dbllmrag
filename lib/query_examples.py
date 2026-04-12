# ============================================================
# lib/query_examples.py — RAG Few-Shot Query Example Store
# ============================================================
# This module implements a simple RAG (Retrieval-Augmented
# Generation) system for MongoDB query generation.
#
# THE PROBLEM IT SOLVES:
# ───────────────────────
#   Local LLMs (7B models) sometimes struggle with complex
#   MongoDB aggregation patterns they haven't seen before.
#   Even a perfect schema description doesn't always convey
#   HOW to combine stages correctly for a given question type.
#
# THE SOLUTION — FEW-SHOT EXAMPLES:
# ───────────────────────────────────
#   When a user asks a question, we search our library of
#   previously-successful queries for similar questions.
#   The 2-3 most similar past examples are prepended to the
#   LLM's prompt as demonstrations:
#
#     "Here are examples of questions and their correct pipelines:
#
#      Question: How many sessions per country this month?
#      Answer: [{ "$group": { "_id": "$clientInfo.country_name",
#                             "count": { "$sum": 1 } } }, ...]
#
#      Now answer this question: ..."
#
#   Seeing a concrete example of the exact pattern needed
#   dramatically improves accuracy for complex queries.
#
# WHERE EXAMPLES COME FROM:
# ──────────────────────────
#   1. AUTOMATIC: Every successful query is automatically saved.
#      "Successful" means: executed without error AND returned
#      at least 1 result (empty results might indicate a bad query).
#
#   2. MANUAL: The file data/query_examples.json can be edited
#      directly. This is how you bootstrap good examples before
#      the system has run many queries.
#
#   3. BOOTSTRAP: We ship a set of hand-crafted examples for
#      common query patterns. See BOOTSTRAP_EXAMPLES below.
#
# SIMILARITY SEARCH:
# ───────────────────
#   We use a simple but effective keyword overlap approach:
#   - Split both the stored question and the new question into words
#   - Count how many words (lowercased) appear in both
#   - Return the top-N examples by overlap score
#
#   This works well for our domain because question vocabulary
#   is fairly limited (stream, session, country, owner, etc.).
#   We don't need embeddings for this scale.
#
# STORAGE FORMAT:
# ────────────────
#   JSON file at data/query_examples.json.
#   Each entry: {
#     "question":     "How many sessions per country this month?",
#     "query":        { "queryType": "single", "pipeline": [...], ... },
#     "result_count": 42,
#     "timestamp":    "2026-04-12T10:30:00Z",
#     "db_hint":      "stream"    ← which DB this is for
#   }
# ============================================================

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Storage path ───────────────────────────────────────────────
# Stored outside lib/ so it persists across deploys/updates.
_EXAMPLES_FILE = Path(__file__).parent.parent / "data" / "query_examples.json"

# Max examples to keep in the store (oldest are dropped when exceeded)
MAX_EXAMPLES = 500

# How many similar examples to return per query
TOP_N = 3


# ────────────────────────────────────────────────────────────
# BOOTSTRAP EXAMPLES
# ────────────────────────────────────────────────────────────
# Hand-crafted examples for common query patterns.
# These seed the store before any real queries have run.
# They demonstrate the most important patterns the LLM needs
# to know: counting by group, duration calculation, cross-db
# lookup, avgRoundTripTime string conversion, etc.

BOOTSTRAP_EXAMPLES = [
    {
        "question": "How many sessions per country this month?",
        "query": {
            "queryType":   "single",
            "database":    "stream-datastore",
            "collection":  "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False}},
                {"$group": {"_id": "$clientInfo.country_name", "sessions": {"$sum": 1}}},
                {"$sort": {"sessions": -1}},
                {"$limit": 50},
            ],
            "explanation": "Groups real user sessions by country and counts them descending.",
            "resultLabel": "Sessions by Country",
        },
        "result_count": 45,
        "db_hint": "stream",
    },
    {
        "question": "What is the average load time per owner?",
        "query": {
            "queryType":   "single",
            "database":    "stream-datastore",
            "collection":  "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False, "loadTime": {"$gt": 0}}},
                {"$group": {
                    "_id": "$appInfo.owner",
                    "avgLoadTime": {"$avg": "$loadTime"},
                    "sessions":    {"$sum": 1},
                }},
                {"$sort": {"avgLoadTime": -1}},
                {"$limit": 50},
            ],
            "explanation": "Calculates average load time in seconds per owner, sorted by slowest first.",
            "resultLabel": "Average Load Time by Owner",
        },
        "result_count": 30,
        "db_hint": "stream",
    },
    {
        "question": "Show sessions with average round trip time over 200ms",
        "query": {
            "queryType":  "single",
            "database":   "stream-datastore",
            "collection": "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False, "webRtcStatsData.avgRoundTripTime": {"$exists": True}}},
                {"$addFields": {"rttFloat": {"$toDouble": "$webRtcStatsData.avgRoundTripTime"}}},
                {"$match": {"rttFloat": {"$gt": 0.2}}},
                {"$project": {
                    "appInfo.owner": 1,
                    "clientInfo.city": 1,
                    "clientInfo.country_name": 1,
                    "rttFloat": 1,
                    "loadTime": 1,
                }},
                {"$sort": {"rttFloat": -1}},
                {"$limit": 50},
            ],
            "explanation": "Finds sessions with high round-trip time (>200ms), converting the string RTT field to a number first.",
            "resultLabel": "High Latency Sessions",
        },
        "result_count": 22,
        "db_hint": "stream",
    },
    {
        "question": "What is the average session duration per city?",
        "query": {
            "queryType":  "single",
            "database":   "stream-datastore",
            "collection": "Apr_2025",
            "pipeline": [
                {"$match": {
                    "e3ds_employee": False,
                    "DisconnectTime_Timestamp": {"$exists": True},
                    "startTimeStamp": {"$exists": True},
                }},
                {"$addFields": {
                    "durationSeconds": {"$subtract": ["$DisconnectTime_Timestamp", "$startTimeStamp"]}
                }},
                {"$match": {"durationSeconds": {"$gt": 0}}},
                {"$group": {
                    "_id": "$clientInfo.city",
                    "avgDuration": {"$avg": "$durationSeconds"},
                    "sessions": {"$sum": 1},
                }},
                {"$sort": {"sessions": -1}},
                {"$limit": 50},
            ],
            "explanation": "Calculates average session duration in seconds per city, filtering out zero-duration sessions.",
            "resultLabel": "Session Duration by City",
        },
        "result_count": 80,
        "db_hint": "stream",
    },
    {
        "question": "Which browsers are most commonly used by viewers?",
        "query": {
            "queryType":  "single",
            "database":   "stream-datastore",
            "collection": "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False}},
                {"$group": {"_id": "$userDeviceInfo.client.name", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 20},
            ],
            "explanation": "Groups sessions by browser name and counts each, sorted by most popular.",
            "resultLabel": "Sessions by Browser",
        },
        "result_count": 8,
        "db_hint": "stream",
    },
    {
        "question": "Which servers (computers) are handling the most sessions?",
        "query": {
            "queryType":  "single",
            "database":   "stream-datastore",
            "collection": "Apr_2025",
            "pipeline": [
                {"$match": {"e3ds_employee": False}},
                {"$group": {
                    "_id": "$elInfo.computerName",
                    "sessions":   {"$sum": 1},
                    "avgBitrate": {"$avg": "$webRtcStatsData.avgBitrate"},
                }},
                {"$sort": {"sessions": -1}},
                {"$limit": 20},
            ],
            "explanation": "Groups sessions by server hostname and counts them, showing most-used servers.",
            "resultLabel": "Sessions by Server",
        },
        "result_count": 15,
        "db_hint": "stream",
    },
    {
        "question": "Show me the subscription status for owner eduardo",
        "query": {
            "queryType":  "single",
            "database":   "appConfigs",
            "collection": "eduardo",
            "pipeline": [
                {"$match": {"_id": "usersinfo"}},
                {"$project": {
                    "maxUserLimit": 1,
                    "shouldAutoRenew": 1,
                    "paidMinutes": 1,
                    "paidSecondsUsage": 1,
                    "products": 1,
                    "SubscriptionEndDate._seconds": 1,
                    "SubscriptionStartDate._seconds": 1,
                    "apiKeys.apiKey": 0,
                    "streamingApiKeys.apiKey": 0,
                }},
            ],
            "explanation": "Fetches subscription and billing data for the owner 'eduardo' from appConfigs.",
            "resultLabel": "Subscription Status: eduardo",
        },
        "result_count": 1,
        "db_hint": "appconfigs",
    },
    {
        "question": "Which sessions had packet loss above 500?",
        "query": {
            "queryType":  "single",
            "database":   "stream-datastore",
            "collection": "Apr_2025",
            "pipeline": [
                {"$match": {
                    "e3ds_employee": False,
                    "webRtcStatsData.packetsLost": {"$gt": 500},
                }},
                {"$project": {
                    "appInfo.owner": 1,
                    "clientInfo.city": 1,
                    "clientInfo.country_name": 1,
                    "webRtcStatsData.packetsLost": 1,
                    "webRtcStatsData.avgBitrate":  1,
                    "loadTime": 1,
                }},
                {"$sort": {"webRtcStatsData.packetsLost": -1}},
                {"$limit": 50},
            ],
            "explanation": "Finds sessions with high packet loss (>500 packets), sorted worst-first.",
            "resultLabel": "High Packet Loss Sessions",
        },
        "result_count": 18,
        "db_hint": "stream",
    },
]


# ────────────────────────────────────────────────────────────
# LOAD / SAVE
# ────────────────────────────────────────────────────────────

def _load_examples() -> list[dict]:
    """
    Loads the stored examples from disk.

    On first run (file doesn't exist), seeds the store with
    the hand-crafted BOOTSTRAP_EXAMPLES and saves them.

    Returns:
        List of example dicts, newest first.
    """
    if not _EXAMPLES_FILE.exists():
        _EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
        _save_examples(BOOTSTRAP_EXAMPLES)
        logger.info(
            f"[query_examples] Seeded example store with "
            f"{len(BOOTSTRAP_EXAMPLES)} bootstrap examples."
        )
        return list(BOOTSTRAP_EXAMPLES)

    try:
        with open(_EXAMPLES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"[query_examples] Failed to load examples: {e}")
        return list(BOOTSTRAP_EXAMPLES)


def _save_examples(examples: list[dict]) -> None:
    """
    Persists the example list to disk.

    Trims to MAX_EXAMPLES (keeping the most recent) before saving
    so the file doesn't grow unboundedly.
    """
    # Keep most recent examples (list is newest-first by convention)
    trimmed = examples[:MAX_EXAMPLES]
    try:
        _EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_EXAMPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.error(f"[query_examples] Failed to save examples: {e}")


# ────────────────────────────────────────────────────────────
# SIMILARITY SEARCH
# ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """
    Splits text into lowercase words for overlap comparison.

    Simple but effective for our domain: MongoDB query questions
    have a limited vocabulary (country, session, owner, load,
    bitrate, etc.) so word overlap captures question similarity well.
    """
    import re
    words = re.findall(r"[a-z0-9]+", text.lower())
    # Filter out very short or common words that add noise
    stopwords = {"the", "a", "an", "is", "are", "was", "of", "in",
                 "for", "to", "and", "or", "by", "with", "this", "that",
                 "what", "which", "how", "many", "show", "me", "get",
                 "list", "find", "from", "all", "any"}
    return {w for w in words if len(w) > 2 and w not in stopwords}


async def find_similar_examples_vector(
    question: str,
    db_hint:  str = "stream",
    top_n:    int = TOP_N,
) -> list[dict] | None:
    """
    Semantic vector search for similar past examples.

    Embeds the question and retrieves the top_n most semantically
    similar examples from the vector store. Returns None if the
    embedding model is unavailable (caller should fall back to
    keyword search).

    Args:
        question: The user's current question.
        db_hint:  "stream", "appconfigs", or "both".
        top_n:    Max examples to return.

    Returns:
        List of example dicts (same format as keyword search), or
        None if embeddings are unavailable.
    """
    from lib.embeddings   import embed
    from lib.vector_store import VectorStore

    store = VectorStore("examples")
    if store.count() == 0:
        return None   # not indexed yet

    q_emb = await embed(question)
    if q_emb is None:
        return None   # embedding model unavailable

    def _filter(item: dict) -> bool:
        hint = item["metadata"].get("db_hint", "stream")
        # Include examples for the same DB, or "both" examples always
        return hint == db_hint or hint == "both" or db_hint == "both"

    results = store.search(q_emb, top_k=top_n, filter_fn=_filter, min_score=0.4)
    if not results:
        return None

    # Convert vector store results back to the example dict format
    # that format_examples_for_prompt() expects
    examples = []
    for r in results:
        meta = r["metadata"]
        examples.append({
            "question":     r["text"],
            "query":        meta.get("query", {}),
            "result_count": meta.get("result_count", 0),
            "db_hint":      meta.get("db_hint", "stream"),
        })
    return examples


def find_similar_examples(
    question: str,
    db_hint:  str = "stream",
    top_n:    int = TOP_N,
) -> list[dict]:
    """
    Keyword overlap search for similar past examples.

    This is the SYNCHRONOUS fallback used when the embedding model
    is unavailable. For better accuracy, prefer find_similar_examples_vector()
    when embeddings are available — it finds semantic matches that
    share meaning but not necessarily the same words.

    Args:
        question: The user's current question.
        db_hint:  "stream", "appconfigs", or "both".
        top_n:    How many examples to return.

    Returns:
        List of up to top_n example dicts, most similar first.
    """
    examples = _load_examples()
    if not examples:
        return []

    question_words = _tokenize(question)
    if not question_words:
        return []

    scored = []
    for ex in examples:
        ex_words = _tokenize(ex.get("question", ""))
        if not ex_words:
            continue

        overlap = len(question_words & ex_words)
        score   = overlap / len(question_words)

        # Small boost for same-database examples
        if ex.get("db_hint", "stream") == db_hint:
            score += 0.05

        if score > 0:
            scored.append((score, ex))

    scored.sort(key=lambda x: (x[0], x[1].get("result_count", 0)), reverse=True)
    return [ex for _, ex in scored[:top_n]]


# ────────────────────────────────────────────────────────────
# ADDING EXAMPLES
# ────────────────────────────────────────────────────────────

def add_example(
    question:     str,
    query_obj:    dict,
    result_count: int,
    db_hint:      str = "stream",
) -> None:
    """
    Saves a successful query as a new example for future use.

    Called automatically by query_generator.py after each
    successful query execution that returned results.

    We don't save:
    - Queries that returned 0 results (likely too narrow/wrong)
    - Queries we already have a very similar example for
      (avoids redundant near-duplicates in the store)

    Args:
        question:     The user's original question.
        query_obj:    The validated query dict that was executed.
        result_count: How many documents the query returned.
        db_hint:      "stream", "appconfigs", or "both".
    """
    if result_count == 0:
        return  # 0 results might mean the query was too narrow or wrong

    # Deduplicate: if we already have a very similar question, skip
    existing = find_similar_examples(question, db_hint, top_n=1)
    if existing:
        best_score_words = _tokenize(question) & _tokenize(existing[0].get("question", ""))
        similarity = len(best_score_words) / max(len(_tokenize(question)), 1)
        if similarity > 0.85:
            # Very similar question already stored — update its result_count instead
            logger.debug(
                f"[query_examples] Skipping near-duplicate example "
                f"(similarity={similarity:.2f}): '{question[:50]}'"
            )
            return

    new_example = {
        "question":     question,
        "query":        query_obj,
        "result_count": result_count,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "db_hint":      db_hint,
    }

    examples = _load_examples()
    examples.insert(0, new_example)
    _save_examples(examples)

    # Also index into the vector store for semantic search.
    # This is async (needs Ollama embed), so we schedule it as a
    # background task — saving to JSON above is synchronous and
    # ensures the example is persisted even if embedding fails.
    import asyncio
    asyncio.create_task(_index_example_async(question, query_obj, result_count, db_hint))

    logger.debug(
        f"[query_examples] Saved new example: '{question[:60]}' "
        f"({result_count} results)"
    )


async def _index_example_async(
    question:     str,
    query_obj:    dict,
    result_count: int,
    db_hint:      str,
) -> None:
    """
    Background task: embeds the question and stores it in the
    vector store for future semantic similarity search.

    Silently skips if the embedding model is unavailable.
    """
    import hashlib
    from lib.embeddings   import embed
    from lib.vector_store import VectorStore

    emb = await embed(question)
    if emb is None:
        return

    store   = VectorStore("examples")
    item_id = hashlib.sha1(question.encode()).hexdigest()

    store.upsert(
        id       = item_id,
        text     = question,
        embedding= emb,
        metadata = {
            "query":        query_obj,
            "result_count": result_count,
            "db_hint":      db_hint,
        },
    )
    # Trim to prevent unbounded growth
    store.trim_to(MAX_EXAMPLES)


# ────────────────────────────────────────────────────────────
# PROMPT FORMATTING
# ────────────────────────────────────────────────────────────

def format_examples_for_prompt(examples: list[dict]) -> str:
    """
    Formats retrieved examples as a few-shot block for the LLM prompt.

    The format shows the question and the full working pipeline,
    so the model can see exactly what pattern to follow.

    Args:
        examples: List of example dicts from find_similar_examples().

    Returns:
        A multi-line string to prepend to the user message, or ""
        if examples is empty.
    """
    if not examples:
        return ""

    lines = [
        "─────────────────────────────────────────────────────────────",
        "SIMILAR EXAMPLES — correct pipelines for similar questions:",
        "(Use these as a reference pattern for the current question.)",
        "─────────────────────────────────────────────────────────────",
    ]

    for i, ex in enumerate(examples, 1):
        q     = ex.get("question", "")
        query = ex.get("query", {})
        count = ex.get("result_count", 0)

        # Serialize pipeline compactly — we want it readable but not huge
        try:
            query_json = json.dumps(query, separators=(",", ":"))
            # If the JSON is very long, truncate at 800 chars
            if len(query_json) > 800:
                query_json = query_json[:797] + "..."
        except Exception:
            query_json = str(query)[:800]

        lines.append(f"\nExample {i}:")
        lines.append(f"  Question: {q}")
        lines.append(f"  Correct answer ({count} results): {query_json}")

    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("")

    return "\n".join(lines)


def get_example_count() -> int:
    """Returns how many examples are stored in the JSON file."""
    examples = _load_examples()
    return len(examples)


async def index_all_examples_async() -> int:
    """
    Indexes all examples from the JSON file into the vector store.

    Called once on startup to ensure the bootstrap examples (and any
    examples saved in previous sessions) are available for vector
    search from the very first query.

    Returns:
        Number of newly indexed examples (0 if all were already indexed).
    """
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
            continue   # already indexed

        emb = await embed(question)
        if emb is None:
            break      # model not available — stop trying

        store.upsert(
            id       = item_id,
            text     = question,
            embedding= emb,
            metadata = {
                "query":        ex.get("query", {}),
                "result_count": ex.get("result_count", 0),
                "db_hint":      ex.get("db_hint", "stream"),
            },
        )
        indexed += 1

    if indexed > 0:
        logger.info(
            f"[query_examples] Indexed {indexed} example(s) into vector store."
        )
    return indexed
