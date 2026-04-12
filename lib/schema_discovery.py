# ============================================================
# lib/schema_discovery.py — Live Schema Discovery from MongoDB
# ============================================================
# This module answers the question: "What if the database
# structure changes? Will the LLM still generate correct queries?"
#
# The static schema in schemas.py is a hand-written description.
# It works well but has a fundamental weakness: if a new field
# is added, a field type changes, or a new collection appears,
# the static schema goes stale and the LLM doesn't know about it.
#
# THIS MODULE SOLVES THAT:
# ────────────────────────
#   On startup (and periodically), we connect to both live
#   MongoDB databases, sample a handful of real documents, and
#   automatically extract:
#     - All field paths that actually exist
#     - The real data type of each field
#     - Example values so the LLM understands the data
#
#   The result is injected into the LLM prompt as a "live schema
#   supplement" that adds details the static schema might miss.
#
# HOW IT WORKS:
# ─────────────
#   1. For stream-datastore: sample 10 docs from the most recent
#      monthly collection. Extract all field paths recursively.
#
#   2. For appConfigs: list available owner collections, sample
#      one "usersinfo" and one "default" doc from several owners.
#      Extract field paths from those.
#
#   3. Cache the result for TTL_SECONDS (default 1 hour) to avoid
#      hitting MongoDB on every query.
#
#   4. If the set of discovered field paths changes between
#      refreshes, log a warning so admins know the schema evolved.
#
# WHAT IT ADDS TO PROMPTS:
# ─────────────────────────
#   When build_dynamic_supplement() is called, it returns a
#   compact text block that query_generator.py can append to
#   the system prompt. Example output:
#
#     "LIVE SCHEMA DISCOVERY (sampled 2026-04-12):
#      stream-datastore/Apr_2026 — 10 docs sampled
#      New fields found (not in static schema):
#        loggedInUserData.userId  (string) e.g. "user_abc123"
#        href_emulated            (string) e.g. "https://..."
#      appConfigs/eduardo — usersinfo doc:
#        products.gb: 5.0, products.ccu: 10, ..."
# ============================================================

import time
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from lib.mongodb import get_stream_db, get_appconfigs_db

logger = logging.getLogger(__name__)

# ── Cache settings ─────────────────────────────────────────────
# How long (in seconds) before we re-sample the database.
# 3600 = 1 hour — frequent enough to catch schema changes
# quickly, rare enough that it doesn't slow down requests.
TTL_SECONDS = 3600

# How many documents to sample per collection for schema extraction.
# 10 is enough to catch fields that appear in most docs, without
# adding too much startup latency.
SAMPLE_SIZE = 10

# How many owner collections to sample from appConfigs.
# We can't sample all 5,500+, so we grab a cross-section.
APPCONFIGS_SAMPLE_OWNERS = 5


# ── Cache state ────────────────────────────────────────────────
# Module-level cache avoids hitting MongoDB on every request.
# The _schema_cache dict holds the last discovery result.
# _cache_timestamp tracks when to refresh.

_schema_cache: dict = {}
_cache_timestamp: float = 0.0
_last_fingerprint: str = ""


# ────────────────────────────────────────────────────────────
# FIELD PATH EXTRACTION
# ────────────────────────────────────────────────────────────

def _extract_paths(
    obj: Any,
    prefix: str = "",
    max_depth: int = 5,
    paths: dict | None = None,
) -> dict:
    """
    Recursively walks a MongoDB document and extracts every
    field path with its Python type and one example value.

    Args:
        obj:       The document or nested value to walk.
        prefix:    Dot-notation path prefix built up during recursion.
        max_depth: How deep to recurse. 5 covers all our nested fields.
        paths:     Accumulator dict — modified in-place and returned.

    Returns:
        Dict mapping field_path → {"type": "str", "example": "value"}

    Example:
        {"appInfo": {"owner": "eduardo"}, "loadTime": 8.5}
        →  {
              "appInfo.owner": {"type": "str",   "example": "eduardo"},
              "loadTime":      {"type": "float", "example": 8.5}
           }
    """
    if paths is None:
        paths = {}

    if max_depth <= 0:
        return paths

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested objects
                _extract_paths(value, full_path, max_depth - 1, paths)
            elif isinstance(value, list):
                # For arrays: record the path as array type, and if
                # the first element is a dict, recurse into it.
                paths[full_path] = {
                    "type":    "array",
                    "example": f"[{len(value)} items]",
                }
                if value and isinstance(value[0], dict):
                    _extract_paths(value[0], f"{full_path}[]", max_depth - 1, paths)
            else:
                # Leaf value — record type and example
                type_name = type(value).__name__
                example   = value

                # Truncate long string examples
                if isinstance(value, str) and len(value) > 60:
                    example = value[:57] + "..."

                paths[full_path] = {"type": type_name, "example": example}

    return paths


def _merge_path_sets(all_path_sets: list[dict]) -> dict:
    """
    Merges path dicts from multiple documents.

    If a field appears in multiple docs, we keep the first
    non-null example. This handles fields that are null or
    missing in some documents but present in others.
    """
    merged = {}
    for paths in all_path_sets:
        for path, info in paths.items():
            if path not in merged:
                merged[path] = info
            elif merged[path]["example"] is None and info["example"] is not None:
                merged[path] = info
    return merged


# ────────────────────────────────────────────────────────────
# DATABASE SAMPLING
# ────────────────────────────────────────────────────────────

async def _sample_stream_schema(collection_name: str) -> dict:
    """
    Samples documents from a stream-datastore monthly collection
    and extracts all field paths.

    Returns:
        Dict with "fields" (path → type/example), "docs_sampled", "collection"
    """
    db = get_stream_db()

    try:
        cursor   = db[collection_name].aggregate([
            {"$match": {"e3ds_employee": {"$ne": True}}},
            {"$sample": {"size": SAMPLE_SIZE}},
        ])
        docs = await cursor.to_list(length=SAMPLE_SIZE)
    except Exception as e:
        logger.warning(f"[schema_discovery] Failed to sample stream/{collection_name}: {e}")
        return {"fields": {}, "docs_sampled": 0, "collection": collection_name}

    # Extract paths from each doc, then merge
    path_sets = [_extract_paths(doc) for doc in docs]
    merged    = _merge_path_sets(path_sets)

    return {
        "fields":       merged,
        "docs_sampled": len(docs),
        "collection":   collection_name,
    }


async def _sample_appconfigs_schema() -> dict:
    """
    Samples several owner collections from appConfigs and
    extracts field paths from their usersinfo and default docs.

    Since appConfigs has ~5,500 collections (one per owner),
    we can't sample all of them. We grab the first few collection
    names and sample one usersinfo + one default doc from each.

    Returns:
        Dict with "owners_sampled", "usersinfo_fields", "config_fields"
    """
    db = get_appconfigs_db()

    # List collection names — skip system/internal collections
    try:
        all_names   = await db.list_collection_names()
        skip_names  = {"system.indexes", "system.users", "system.version"}
        owner_names = [
            n for n in all_names
            if n not in skip_names and not n.startswith("system.")
        ][:APPCONFIGS_SAMPLE_OWNERS]
    except Exception as e:
        logger.warning(f"[schema_discovery] Failed to list appConfigs collections: {e}")
        return {"owners_sampled": 0, "usersinfo_fields": {}, "config_fields": {}}

    usersinfo_paths = []
    config_paths    = []

    for owner in owner_names:
        try:
            # Fetch the usersinfo document (billing/subscription data)
            usersinfo = await db[owner].find_one({"_id": "usersinfo"})
            if usersinfo:
                usersinfo_paths.append(_extract_paths(usersinfo))

            # Fetch the default config document
            default_cfg = await db[owner].find_one({"_id": "default"})
            if default_cfg:
                config_paths.append(_extract_paths(default_cfg))

        except Exception as e:
            logger.warning(f"[schema_discovery] Failed to sample appConfigs/{owner}: {e}")
            continue

    return {
        "owners_sampled":  len(owner_names),
        "usersinfo_fields": _merge_path_sets(usersinfo_paths) if usersinfo_paths else {},
        "config_fields":    _merge_path_sets(config_paths)    if config_paths    else {},
    }


# ────────────────────────────────────────────────────────────
# CACHE MANAGEMENT
# ────────────────────────────────────────────────────────────

async def refresh_schema_cache(
    stream_collection: str = "Apr_2025",
    force: bool = False,
) -> dict:
    """
    Refreshes the in-memory schema cache by sampling live data.

    Only re-samples if the cache is expired (TTL exceeded) or
    force=True. This is safe to call on every request — it will
    be a no-op if the cache is still fresh.

    Args:
        stream_collection: Which monthly collection to sample.
                           Should be the most recent one.
        force:             Skip TTL check and always re-sample.

    Returns:
        The (potentially refreshed) schema cache dict.
    """
    global _schema_cache, _cache_timestamp, _last_fingerprint

    now = time.monotonic()
    if not force and (now - _cache_timestamp) < TTL_SECONDS and _schema_cache:
        return _schema_cache  # cache is fresh

    logger.info(f"[schema_discovery] Sampling live schemas from MongoDB…")
    start = time.monotonic()

    # Run both samples concurrently
    stream_result, appconfigs_result = await asyncio.gather(
        _sample_stream_schema(stream_collection),
        _sample_appconfigs_schema(),
        return_exceptions=True,
    )

    # Handle exceptions from gather (return_exceptions=True means they
    # come back as exception objects rather than being re-raised)
    if isinstance(stream_result, Exception):
        logger.warning(f"[schema_discovery] Stream sampling failed: {stream_result}")
        stream_result = {"fields": {}, "docs_sampled": 0, "collection": stream_collection}

    if isinstance(appconfigs_result, Exception):
        logger.warning(f"[schema_discovery] AppConfigs sampling failed: {appconfigs_result}")
        appconfigs_result = {"owners_sampled": 0, "usersinfo_fields": {}, "config_fields": {}}

    elapsed = round(time.monotonic() - start, 2)

    new_cache = {
        "stream":      stream_result,
        "appconfigs":  appconfigs_result,
        "sampled_at":  datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": elapsed,
    }

    # Detect schema changes by comparing field path sets
    all_fields    = (
        sorted(stream_result["fields"].keys())
        + sorted(appconfigs_result.get("usersinfo_fields", {}).keys())
    )
    new_fingerprint = str(hash(tuple(all_fields)))

    if _last_fingerprint and new_fingerprint != _last_fingerprint:
        # Schema has changed since last sample — log it prominently
        old_fields  = set(json.loads(_schema_cache.get("_field_list_json", "[]")))
        new_fields  = set(all_fields)
        added       = new_fields - old_fields
        removed     = old_fields - new_fields
        if added:
            logger.warning(f"[schema_discovery] NEW FIELDS DETECTED: {sorted(added)}")
        if removed:
            logger.warning(f"[schema_discovery] REMOVED FIELDS DETECTED: {sorted(removed)}")

    new_cache["_field_list_json"] = json.dumps(all_fields)
    _schema_cache    = new_cache
    _cache_timestamp = now
    _last_fingerprint = new_fingerprint

    logger.info(
        f"[schema_discovery] Done in {elapsed}s — "
        f"stream: {stream_result['docs_sampled']} docs, "
        f"appconfigs: {appconfigs_result['owners_sampled']} owners sampled"
    )

    # ── Index discovered fields into the vector store ─────────
    # Do this in a separate background task so it doesn't block
    # the first query. If the embedding model isn't available,
    # _index_fields_async() is a no-op.
    asyncio.create_task(_index_fields_async(new_cache))

    return _schema_cache


async def _index_fields_async(cache: dict) -> None:
    """
    Embeds all discovered schema fields and stores them in the
    vector store for semantic retrieval.

    Called as a background task after refresh_schema_cache() so
    it doesn't delay the first query. Embeddings are reused on
    subsequent queries (the vector store persists to disk).

    When the embedding model is unavailable, this is a no-op and
    the app falls back to using the full static schema.
    """
    # Import here to avoid circular imports at module load time
    from lib.embeddings   import embed
    from lib.vector_store import VectorStore

    schema_store = VectorStore("schema")
    indexed = 0

    # ── Index stream-datastore fields ─────────────────────────
    stream_fields = cache.get("stream", {}).get("fields", {})
    for field_path, info in stream_fields.items():
        ftype   = info.get("type", "?")
        example = info.get("example", "")
        if example is None:
            example = "null"

        # Build a descriptive text the embedding model can
        # understand. "field_path (type): example value"
        text = f"{field_path} ({ftype}): {example}"

        item_id = f"stream::{field_path}"
        # Skip if already indexed (avoid re-embedding on every refresh)
        if item_id in schema_store.ids():
            continue

        emb = await embed(text)
        if emb:
            schema_store.upsert(
                id       = item_id,
                text     = text,
                embedding= emb,
                metadata = {
                    "db":    "stream-datastore",
                    "field": field_path,
                    "type":  ftype,
                },
            )
            indexed += 1

    # ── Index appConfigs fields ────────────────────────────────
    appconf = cache.get("appconfigs", {})
    for doc_type, fields in [
        ("usersinfo", appconf.get("usersinfo_fields", {})),
        ("config",    appconf.get("config_fields", {})),
    ]:
        for field_path, info in fields.items():
            ftype   = info.get("type", "?")
            example = info.get("example", "")
            if example is None:
                example = "null"

            text    = f"{field_path} ({ftype}): {example}  [appConfigs/{doc_type}]"
            item_id = f"appconfigs::{doc_type}::{field_path}"

            if item_id in schema_store.ids():
                continue

            emb = await embed(text)
            if emb:
                schema_store.upsert(
                    id       = item_id,
                    text     = text,
                    embedding= emb,
                    metadata = {
                        "db":       "appConfigs",
                        "doc_type": doc_type,
                        "field":    field_path,
                        "type":     ftype,
                    },
                )
                indexed += 1

    if indexed > 0:
        logger.info(
            f"[schema_discovery] Indexed {indexed} new field embeddings "
            f"into schema vector store."
        )


# ────────────────────────────────────────────────────────────
# PROMPT SUPPLEMENT BUILDER
# ────────────────────────────────────────────────────────────

def build_dynamic_supplement(
    include_stream: bool = True,
    include_appconfigs: bool = True,
    max_fields: int = 40,
) -> str:
    """
    Builds a concise text block describing LIVE schema data
    discovered from MongoDB. Intended to be appended to the
    main LLM system prompt.

    This bridges the gap between the static schema (which might
    be slightly outdated) and the current real database structure.

    Args:
        include_stream:     Include stream-datastore live fields.
        include_appconfigs: Include appConfigs live fields.
        max_fields:         Cap on fields to list (keep prompt short).

    Returns:
        A multi-line string to append to the LLM prompt, or ""
        if the cache is empty (hasn't been populated yet).
    """
    if not _schema_cache:
        return ""  # cache not populated yet — skip supplement

    lines = [
        "─────────────────────────────────────────────────────────────",
        f"LIVE SCHEMA DISCOVERY (sampled at {_schema_cache.get('sampled_at', 'unknown')})",
        "The following fields were found in the ACTUAL live database.",
        "If there is any conflict with the static schema above,",
        "TRUST THIS LIVE DATA instead.",
        "─────────────────────────────────────────────────────────────",
    ]

    if include_stream:
        stream = _schema_cache.get("stream", {})
        fields = stream.get("fields", {})
        docs   = stream.get("docs_sampled", 0)
        coll   = stream.get("collection", "?")

        if fields:
            lines.append(f"\nstream-datastore/{coll} — {docs} docs sampled, {len(fields)} fields:")
            # Show fields sorted, capped at max_fields
            for path, info in list(fields.items())[:max_fields]:
                example = info.get("example", "")
                ftype   = info.get("type", "?")
                if example is None:
                    example = "null"
                lines.append(f"  {path}  ({ftype})  e.g. {repr(example)}")
            if len(fields) > max_fields:
                lines.append(f"  … and {len(fields) - max_fields} more fields.")

    if include_appconfigs:
        appconf = _schema_cache.get("appconfigs", {})
        owners  = appconf.get("owners_sampled", 0)

        usersinfo = appconf.get("usersinfo_fields", {})
        if usersinfo:
            lines.append(f"\nappConfigs — usersinfo docs ({owners} owners sampled):")
            for path, info in list(usersinfo.items())[:max_fields]:
                example = info.get("example", "")
                ftype   = info.get("type", "?")
                if example is None:
                    example = "null"
                lines.append(f"  {path}  ({ftype})  e.g. {repr(example)}")

        cfg_fields = appconf.get("config_fields", {})
        if cfg_fields:
            lines.append(f"\nappConfigs — streaming config docs (default/_appName_):")
            for path, info in list(cfg_fields.items())[:max_fields]:
                example = info.get("example", "")
                ftype   = info.get("type", "?")
                if example is None:
                    example = "null"
                lines.append(f"  {path}  ({ftype})  e.g. {repr(example)}")

    if len(lines) <= 6:
        return ""  # nothing useful to add

    return "\n".join(lines)


async def retrieve_schema_context(
    question:           str,
    include_stream:     bool = True,
    include_appconfigs: bool = False,
    top_k:              int  = 20,
) -> str:
    """
    Retrieves the most relevant schema field descriptions for this
    question using semantic (vector) similarity.

    This is the RAG retrieval step for schema context. Instead of
    dumping all 200+ field descriptions into every prompt, we embed
    the question and find the top_k fields with the most similar
    descriptions. For "top countries by session count" this returns
    clientInfo.country_name, appInfo.owner, loadTime, etc. — not the
    500 GPU/CPU/WebRTC fields that are irrelevant to that question.

    Falls back to an empty string if the vector store is empty or
    the embedding model is unavailable (caller will use static rules).

    Args:
        question:           The user's question text.
        include_stream:     Include stream-datastore fields.
        include_appconfigs: Include appConfigs fields.
        top_k:              Max field descriptions to return.

    Returns:
        Multi-line string of relevant field descriptions, or "".
    """
    from lib.embeddings   import embed
    from lib.vector_store import VectorStore

    schema_store = VectorStore("schema")

    # ── Static fallback for appConfigs ─────────────────────────
    # If the vector store is empty (cold start before embeddings
    # are ready) AND the question needs appConfigs, inject the
    # critical appConfigs field list directly. Without this, the
    # LLM has no field context for the second database and dual
    # queries fail every time.
    _APPCONFIGS_STATIC = (
        "  appConfigs fields (each owner collection has these in the 'usersinfo' document):\n"
        "  maxUserLimit (integer): max concurrent users allowed for this owner\n"
        "  SubscriptionEndDate._seconds (integer): Unix timestamp of subscription expiry\n"
        "  SubscriptionStartDate._seconds (integer): Unix timestamp when subscription began\n"
        "  paidMinutes (number): total streaming minutes purchased\n"
        "  paidSecondsUsage (number): streaming seconds used so far\n"
        "  shouldAutoRenew (boolean): whether subscription auto-renews\n"
        "  products.ccu (number): concurrent user product limit\n"
        "  products.gb (number): bandwidth GB product\n"
        "  NOTE: In appConfigs each collection = one owner username (e.g. 'eduardo').\n"
        "  Query _id='usersinfo' for billing data. NEVER use 'users' as a collection name."
    )

    if schema_store.count() == 0:
        if include_appconfigs:
            return _APPCONFIGS_STATIC
        return ""   # not indexed yet — caller falls back to static rules

    q_emb = await embed(question)
    if q_emb is None:
        if include_appconfigs:
            return _APPCONFIGS_STATIC
        return ""   # embedding unavailable — caller falls back

    # Filter to only the relevant database(s)
    def _filter(item: dict) -> bool:
        db = item["metadata"].get("db", "")
        if include_stream     and db == "stream-datastore": return True
        if include_appconfigs and db == "appConfigs":       return True
        return False

    results = schema_store.search(q_emb, top_k=top_k, filter_fn=_filter, min_score=0.3)
    if not results:
        if include_appconfigs:
            return _APPCONFIGS_STATIC
        return ""

    lines = [f"  {r['text']}" for r in results]
    # If appConfigs was requested but no appConfigs results came back from
    # the vector store, append the static fallback so the LLM always has
    # field context for the second database.
    if include_appconfigs and not any(
        r["metadata"].get("db") == "appConfigs" for r in results
    ):
        lines.append(_APPCONFIGS_STATIC)

    return "\n".join(lines)


def get_cache_status() -> dict:
    """
    Returns status info about the schema cache for the health
    endpoint or debugging.
    """
    return {
        "populated":     bool(_schema_cache),
        "sampled_at":    _schema_cache.get("sampled_at"),
        "elapsed_sec":   _schema_cache.get("elapsed_sec"),
        "stream_docs":   _schema_cache.get("stream", {}).get("docs_sampled", 0),
        "owners_sampled": _schema_cache.get("appconfigs", {}).get("owners_sampled", 0),
    }
