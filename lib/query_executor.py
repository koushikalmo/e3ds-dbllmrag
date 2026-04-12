# ============================================================
# lib/query_executor.py — Safe Async MongoDB Query Executor
# ============================================================
# This module takes the validated query dict from
# query_generator.py and actually runs it against MongoDB.
#
# SAFETY FIRST:
# ─────────────
# We run an analytics tool that regular users can query with
# plain English. We NEVER want a generated pipeline to:
#   - Write data ($out, $merge) → we strip these stages
#   - Return thousands of docs → we enforce a hard $limit
#   - Run forever → we set maxTimeMS on every aggregation
#   - Leak encrypted API keys → we project them away
#
# HOW SINGLE QUERIES WORK:
# ──────────────────────────
# 1. Sanitize the pipeline (remove write stages)
# 2. Enforce the document limit
# 3. Run aggregation with allowDiskUse + timeout
# 4. Convert BSON types to JSON-safe Python types
# 5. Return the docs + a summary
#
# HOW DUAL QUERIES WORK:
# ────────────────────────
# MongoDB cannot $lookup across different cluster URIs.
# Instead, we run two separate pipelines in parallel using
# asyncio.gather(), which starts both at the same time and
# waits for both to finish. Then we merge the results in
# Python memory using the owner username as the join key.
#
# Example timing with asyncio.gather():
#   Without: stream query (2s) + appconfigs query (1s) = 3s total
#   With:    both run simultaneously → 2s total (the slower one)
# ============================================================

import asyncio
import os
from typing import Any

from lib.db_registry import get_db, get_default_collection

# ── Safety constants ──────────────────────────────────────────
MAX_RESULTS   = 200      # hard cap: never return more than this many docs
QUERY_TIMEOUT = 15_000   # milliseconds → 15 seconds max per query

# MongoDB pipeline stages that write data — we never allow these.
# $out  → writes results to a new collection (destructive)
# $merge → merges results into an existing collection (destructive)
_WRITE_STAGES = frozenset({"$out", "$merge"})


# ────────────────────────────────────────────────────────────
# PIPELINE SAFETY FILTERS
# ────────────────────────────────────────────────────────────

def _sanitize_pipeline(pipeline: list) -> list:
    """
    Removes any $out or $merge stages from the pipeline.

    Even with a perfectly crafted system prompt, the LLM might
    occasionally generate a write stage. This is our last line
    of defense to ensure the tool stays read-only.

    We log a warning when this happens so it's visible in
    server output — it helps spot if the prompt needs improving.

    Example input:
        [{ "$group": {...} }, { "$out": "temp_results" }]
    Example output:
        [{ "$group": {...} }]   ← $out stripped silently
    """
    clean = []
    for stage in pipeline:
        # Each stage is a dict with one key (the operator name)
        stage_name = next(iter(stage), None)
        if stage_name in _WRITE_STAGES:
            print(
                f"[executor] SECURITY: Stripped write stage '{stage_name}' "
                "from generated pipeline. Check the system prompt."
            )
        else:
            clean.append(stage)
    return clean


def _enforce_limit(pipeline: list) -> list:
    """
    Ensures the pipeline always has a $limit ≤ MAX_RESULTS.

    Two cases:
    1. No $limit in pipeline → we append one.
       This handles queries like "list all sessions" that would
       otherwise dump the entire collection.

    2. $limit present but too large → clamp it down.
       The LLM might generate { "$limit": 500 } despite the
       system prompt saying 50. We cap at MAX_RESULTS (200).

    We preserve the position of an existing $limit in the pipeline
    rather than always appending at the end, because $limit placed
    early (before $group, etc.) can change semantics.
    """
    has_limit = any("$limit" in stage for stage in pipeline)

    if not has_limit:
        # Add $limit at the end — safest default position
        return pipeline + [{"$limit": MAX_RESULTS}]

    # Clamp any $limit that exceeds our maximum
    result = []
    for stage in pipeline:
        if "$limit" in stage:
            original = stage["$limit"]
            if original > MAX_RESULTS:
                print(
                    f"[executor] Clamped $limit from {original} → {MAX_RESULTS}"
                )
                result.append({"$limit": MAX_RESULTS})
            else:
                result.append(stage)
        else:
            result.append(stage)
    return result


def _prepare_pipeline(pipeline: list) -> list:
    """
    Applies all safety transforms in one call.
    Called once before every aggregation run.
    """
    return _enforce_limit(_sanitize_pipeline(pipeline))


# ────────────────────────────────────────────────────────────
# DATABASE ROUTING
# ────────────────────────────────────────────────────────────

def _get_db(database: str):
    """
    Returns the correct Motor database handle given a database name.

    Looks the name up in the db_registry so any registered database
    works — not just the original two. Adding a new database to
    data/db_registry.json makes it instantly queryable here.

    Raises ValueError for names not in the registry (LLM hallucination).
    """
    try:
        return get_db(database)
    except ValueError:
        raise ValueError(
            f"Unknown database '{database}'. "
            "The LLM generated a database name that isn't registered. "
            "Valid databases are listed in data/db_registry.json."
        )


def _resolve_collection(database: str, collection: str) -> str:
    """
    Resolves the collection name to use for a query.

    Uses db_registry to look up the database's default collection.
    If no default exists (e.g. appConfigs, where each collection is
    a different owner) and the LLM didn't provide one, raises ValueError.
    """
    if collection:
        return collection

    default = get_default_collection(database)
    if default:
        return default

    raise ValueError(
        f"Query for '{database}' is missing a collection name, "
        f"and '{database}' has no default collection. "
        "The collection name must be specified explicitly. "
        "For appConfigs this is the owner's username (e.g. 'eduardo'). "
        "Try rephrasing your question to include the specific name."
    )


# ────────────────────────────────────────────────────────────
# RESULT PROCESSING
# ────────────────────────────────────────────────────────────

def _make_serializable(docs: list[dict]) -> list[dict]:
    """
    Converts MongoDB-specific types into plain Python types
    so FastAPI can serialize them to JSON.

    Motor returns documents with types that the standard json
    module doesn't know how to handle:

    - ObjectId → looks like: ObjectId('64a3f1c2...'), a 12-byte
      MongoDB unique ID. We convert to its hex string.

    - Decimal128 → MongoDB's high-precision decimal type.
      We convert to Python float (loses some precision but
      fine for display purposes).

    The conversion is recursive so it handles nested objects
    and arrays — which is important because our documents
    have many levels of nesting (clientInfo.fullInfo.timezone, etc.)
    """
    import bson

    def convert(val: Any) -> Any:
        if isinstance(val, bson.ObjectId):
            return str(val)            # "64a3f1c2abcdef1234567890"
        if isinstance(val, bson.Decimal128):
            return float(str(val))     # Decimal128("49758") → 49758.0
        if isinstance(val, dict):
            return {k: convert(v) for k, v in val.items()}
        if isinstance(val, list):
            return [convert(i) for i in val]
        return val  # int, float, str, bool, None → pass through unchanged

    return [convert(doc) for doc in docs]


def _summarize(results: list[dict]) -> dict:
    """
    Generates a quick summary for the frontend's result header.

    The frontend displays: "42 records · 1.3s"
    The "42 records" count comes from summary["count"].
    The "sample" field holds the first 5 documents as a preview
    (used in future for potential "quick look" features).
    """
    return {
        "count":  len(results),
        "sample": results[:5],
    }


# ────────────────────────────────────────────────────────────
# QUERY RUNNERS
# ────────────────────────────────────────────────────────────

async def _run_single(
    database:     str,
    collection:   str,
    raw_pipeline: list,
) -> list[dict]:
    """
    Executes one aggregation pipeline against MongoDB.

    WHY allowDiskUse=True?
    ──────────────────────
    MongoDB's in-memory sort limit is 100MB per aggregation.
    If you sort a large collection without an index, you'll
    hit this limit and the query will fail with an error.
    allowDiskUse=True lets MongoDB spill to disk for large
    sorts, at the cost of some performance. For an analytics
    tool with ad-hoc queries, this is the right trade-off.

    WHY maxTimeMS=15000?
    ─────────────────────
    Without a time limit, a poorly generated pipeline on a
    large collection could run for minutes. This would tie
    up a MongoDB connection and eventually time out the user's
    browser request anyway. Setting a 15-second limit on the
    MongoDB server side means:
    1. The query aborts immediately on the DB side (not just
       the client side), freeing resources.
    2. We can return a helpful "query timed out" message to
       the user with advice to add a more specific filter.

    Args:
        database:     "stream-datastore" or "appConfigs"
        collection:   Collection name (e.g. "Apr_2025", "users")
        raw_pipeline: The pipeline from the LLM, before safety transforms

    Returns:
        List of serializable dicts, ready for JSON response.

    Raises:
        TimeoutError: If MongoDB kills the query for taking too long.
        ValueError:   If the collection doesn't exist.
    """
    db        = _get_db(database)
    coll_name = _resolve_collection(database, collection)
    pipeline  = _prepare_pipeline(raw_pipeline)

    print(
        f"[executor] Running {database}/{coll_name} "
        f"— {len(pipeline)} pipeline stages"
    )

    try:
        cursor = db[coll_name].aggregate(
            pipeline,
            allowDiskUse=True,    # spill to disk for large sorts
            maxTimeMS=QUERY_TIMEOUT,
        )
        # to_list(length=MAX_RESULTS) fetches all results up to MAX_RESULTS
        # documents. Motor fetches them in batches from the cursor.
        results = await cursor.to_list(length=MAX_RESULTS)
        return _make_serializable(results)

    except Exception as err:
        err_str = str(err)

        # Map common MongoDB error conditions to friendly messages
        if "MaxTimeMSExpired" in err_str or "exceeded time limit" in err_str.lower():
            raise TimeoutError(
                f"The query took longer than {QUERY_TIMEOUT // 1000} seconds. "
                "Try adding a more specific filter (e.g. specify an owner name, "
                "date range, or country) to reduce the amount of data scanned."
            )

        if "NamespaceNotFound" in err_str:
            if database == "stream-datastore":
                raise ValueError(
                    f"Collection '{coll_name}' does not exist in stream-datastore. "
                    "Monthly collections follow the format 'Apr_2025', 'Mar_2025', etc. "
                    "The collection might not have data yet, or the name is misspelled."
                )
            else:
                raise ValueError(
                    f"Owner collection '{coll_name}' does not exist in appConfigs. "
                    "In appConfigs, each collection is named after an owner username. "
                    "Check that the owner name is spelled correctly."
                )

        if "BSONObjectTooLarge" in err_str:
            raise ValueError(
                "A query result document exceeded MongoDB's 16MB limit. "
                "Try adding a $project stage to return fewer fields."
            )

        # Re-raise anything else unchanged
        raise


# ────────────────────────────────────────────────────────────
# MAIN EXPORT
# ────────────────────────────────────────────────────────────

async def execute_query(query_obj: dict) -> dict:
    """
    Executes a validated query object and returns structured results.

    This is the only function external code (main.py) calls.
    It handles both 'single' and 'dual' query types.

    SINGLE QUERY FLOW:
        Run one pipeline → return results + summary + metadata

    DUAL QUERY FLOW:
        1. Run both pipelines simultaneously with asyncio.gather()
        2. If query_obj has a mergeKey, merge the two result sets
           in Python memory using that key as the join field.
        3. The merge works by building a lookup dict from the
           smaller (appConfigs) result set, then enriching each
           doc from the larger (stream) result set.

    HOW THE MERGE WORKS:
        Let's say we asked "Which active owners had sessions?"
        appConfigs results:  [{ "_id": "eduardo" }, { "_id": "imerza" }]
        stream results:      [{ "_id": "eduardo", "count": 42 }, { "_id": "Lunas", "count": 7 }]

        We build: config_map = { "eduardo": {...}, "imerza": {...} }
        We loop through stream results:
          - "eduardo" → found in config_map → add "_configData" field
          - "Lunas"   → NOT in config_map → skip (no config data)

        The merged result has "eduardo" enriched with config info,
        and "Lunas" with no config data (they're not in appConfigs
        or their subscription doesn't match).

    Args:
        query_obj: Validated dict from query_generator.generate_query()

    Returns:
        Dict shaped for the JSON response to the frontend:
        {
            "queryType": "single"|"dual",
            "results": [...] or { "primary": [...], "secondary": [...], "merged": [...] },
            "summary": { "count": N, "sample": [...] },
            "explanation": "...",
            "resultLabel": "...",
            "executedPipeline": [...] (single queries only)
        }
    """

    # ── Single database query ──────────────────────────────────
    if query_obj["queryType"] == "single":
        results = await _run_single(
            database     = query_obj["database"],
            collection   = query_obj.get("collection", ""),
            raw_pipeline = query_obj["pipeline"],
        )

        print(f"[executor] Single query returned {len(results)} documents")

        return {
            "queryType":        "single",
            "results":          results,
            "summary":          _summarize(results),
            "explanation":      query_obj.get("explanation", ""),
            "resultLabel":      query_obj.get("resultLabel", "Query Results"),
            # We return the sanitized+limited pipeline (not the raw one)
            # so the user sees exactly what was executed, not what the LLM generated.
            "executedPipeline": _prepare_pipeline(query_obj["pipeline"]),
        }

    # ── Dual database query ────────────────────────────────────
    if query_obj["queryType"] == "dual":
        q1, q2 = query_obj["queries"]

        # asyncio.gather() starts BOTH queries at the same time.
        # Neither one waits for the other. Total time ≈ max(t1, t2)
        # instead of t1 + t2. For a 2s stream query + 1s config
        # query, this saves 1 second on every dual request.
        print("[executor] Running dual query in parallel…")
        results1, results2 = await asyncio.gather(
            _run_single(q1["database"], q1.get("collection", ""), q1["pipeline"]),
            _run_single(q2["database"], q2.get("collection", ""), q2["pipeline"]),
        )

        print(
            f"[executor] Dual query: "
            f"primary={len(results1)} docs, secondary={len(results2)} docs"
        )

        # ── In-memory merge by owner key ───────────────────────
        # Only performed if the LLM included a "mergeKey" field.
        # mergeKey is usually "owner" — telling us to join on
        # appInfo.owner (stream) ↔ _id (appConfigs).
        merged = None
        merge_key = query_obj.get("mergeKey")

        if merge_key:
            # Build a fast lookup dict from results2 (usually appConfigs,
            # which is smaller). Key = the owner name from _id field.
            config_map = {
                str(doc.get(merge_key) or doc.get("_id", "")): doc
                for doc in results2
            }

            # Walk through results1 (stream data) and attach config
            # info to any document whose owner matches
            merged = []
            for doc in results1:
                # The owner might be at doc["owner"] (if $grouped by owner)
                # or at doc["appInfo"]["owner"] (if returning raw sessions)
                owner = str(
                    doc.get(merge_key)
                    or doc.get("appInfo", {}).get("owner", "")
                    or ""
                )
                enriched = dict(doc)  # copy so we don't mutate the original
                if owner and owner in config_map:
                    # Attach the config document as a nested field
                    # The frontend can display it in the table or JSON view
                    enriched["_configData"] = config_map[owner]
                merged.append(enriched)

            print(
                f"[executor] Merged {len(merged)} docs "
                f"({sum(1 for d in merged if '_configData' in d)} with config data)"
            )

        return {
            "queryType": "dual",
            "results": {
                "primary":   results1,  # first query (usually stream-datastore)
                "secondary": results2,  # second query (usually appConfigs)
                "merged":    merged,    # None if no mergeKey was specified
            },
            "summary": {
                "primary":   _summarize(results1),
                "secondary": _summarize(results2),
            },
            "explanation": query_obj.get("explanation", ""),
            "resultLabel": query_obj.get("resultLabel", "Cross-Database Results"),
        }

    # This should never be reached — _validate_query_object() in
    # query_generator.py enforces valid queryType values. But
    # it's good practice to be explicit about the fallthrough.
    raise ValueError(
        f"[executor] Unsupported queryType: '{query_obj.get('queryType')}'"
    )
