import time
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from lib.mongodb import get_stream_db, get_appconfigs_db

logger = logging.getLogger(__name__)

TTL_SECONDS             = 3600  # re-sample every hour
SAMPLE_SIZE             = 10    # docs per collection
APPCONFIGS_SAMPLE_OWNERS = 5    # owner collections to sample from appConfigs

_schema_cache:    dict  = {}
_cache_timestamp: float = 0.0
_last_fingerprint: str  = ""


def _extract_paths(obj: Any, prefix: str = "", max_depth: int = 5, paths: dict | None = None) -> dict:
    if paths is None:
        paths = {}
    if max_depth <= 0:
        return paths

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _extract_paths(value, full_path, max_depth - 1, paths)
            elif isinstance(value, list):
                paths[full_path] = {"type": "array", "example": f"[{len(value)} items]"}
                if value and isinstance(value[0], dict):
                    _extract_paths(value[0], f"{full_path}[]", max_depth - 1, paths)
            else:
                example = value[:57] + "..." if isinstance(value, str) and len(value) > 60 else value
                paths[full_path] = {"type": type(value).__name__, "example": example}
    return paths


def _merge_path_sets(all_path_sets: list[dict]) -> dict:
    merged = {}
    for paths in all_path_sets:
        for path, info in paths.items():
            if path not in merged:
                merged[path] = info
            elif merged[path]["example"] is None and info["example"] is not None:
                merged[path] = info
    return merged


async def _sample_stream_schema(collection_name: str) -> dict:
    db = get_stream_db()
    try:
        cursor = db[collection_name].aggregate([
            {"$match": {"e3ds_employee": {"$ne": True}}},
            {"$sample": {"size": SAMPLE_SIZE}},
        ])
        docs = await cursor.to_list(length=SAMPLE_SIZE)
    except Exception as e:
        logger.warning(f"[schema_discovery] Failed to sample stream/{collection_name}: {e}")
        return {"fields": {}, "docs_sampled": 0, "collection": collection_name}

    return {
        "fields":       _merge_path_sets([_extract_paths(doc) for doc in docs]),
        "docs_sampled": len(docs),
        "collection":   collection_name,
    }


async def _sample_appconfigs_schema() -> dict:
    db = get_appconfigs_db()
    try:
        all_names   = await db.list_collection_names()
        owner_names = [n for n in all_names if not n.startswith("system.")][:APPCONFIGS_SAMPLE_OWNERS]
    except Exception as e:
        logger.warning(f"[schema_discovery] Failed to list appConfigs collections: {e}")
        return {"owners_sampled": 0, "usersinfo_fields": {}, "config_fields": {}}

    usersinfo_paths, config_paths = [], []
    for owner in owner_names:
        try:
            doc = await db[owner].find_one({"_id": "usersinfo"})
            if doc:
                usersinfo_paths.append(_extract_paths(doc))
            doc = await db[owner].find_one({"_id": "default"})
            if doc:
                config_paths.append(_extract_paths(doc))
        except Exception as e:
            logger.warning(f"[schema_discovery] Failed to sample appConfigs/{owner}: {e}")

    return {
        "owners_sampled":  len(owner_names),
        "usersinfo_fields": _merge_path_sets(usersinfo_paths) if usersinfo_paths else {},
        "config_fields":    _merge_path_sets(config_paths)    if config_paths    else {},
    }


async def refresh_schema_cache(stream_collection: str = "Apr_2025", force: bool = False) -> dict:
    global _schema_cache, _cache_timestamp, _last_fingerprint

    now = time.monotonic()
    if not force and (now - _cache_timestamp) < TTL_SECONDS and _schema_cache:
        return _schema_cache

    logger.info("[schema_discovery] Sampling live schemas…")
    start = time.monotonic()

    stream_result, appconfigs_result = await asyncio.gather(
        _sample_stream_schema(stream_collection),
        _sample_appconfigs_schema(),
        return_exceptions=True,
    )

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

    all_fields = sorted(stream_result["fields"].keys()) + sorted(appconfigs_result.get("usersinfo_fields", {}).keys())
    new_fingerprint = str(hash(tuple(all_fields)))

    if _last_fingerprint and new_fingerprint != _last_fingerprint:
        old_fields = set(json.loads(_schema_cache.get("_field_list_json", "[]")))
        new_fields = set(all_fields)
        added, removed = new_fields - old_fields, old_fields - new_fields
        if added:   logger.warning(f"[schema_discovery] NEW FIELDS: {sorted(added)}")
        if removed: logger.warning(f"[schema_discovery] REMOVED FIELDS: {sorted(removed)}")

    new_cache["_field_list_json"] = json.dumps(all_fields)
    _schema_cache      = new_cache
    _cache_timestamp   = now
    _last_fingerprint  = new_fingerprint

    logger.info(
        f"[schema_discovery] Done in {elapsed}s — "
        f"stream: {stream_result['docs_sampled']} docs, "
        f"appconfigs: {appconfigs_result['owners_sampled']} owners"
    )

    asyncio.create_task(_index_fields_async(new_cache))
    return _schema_cache


async def _index_fields_async(cache: dict) -> None:
    from lib.embeddings   import embed
    from lib.vector_store import VectorStore

    schema_store = VectorStore("schema")
    indexed = 0

    for field_path, info in cache.get("stream", {}).get("fields", {}).items():
        ftype   = info.get("type", "?")
        example = info.get("example", "") or "null"
        text    = f"{field_path} ({ftype}): {example}"
        item_id = f"stream::{field_path}"
        if item_id in schema_store.ids():
            continue
        emb = await embed(text)
        if emb:
            schema_store.upsert(item_id, text, emb, {"db": "stream-datastore", "field": field_path, "type": ftype})
            indexed += 1

    appconf = cache.get("appconfigs", {})
    for doc_type, fields in [("usersinfo", appconf.get("usersinfo_fields", {})), ("config", appconf.get("config_fields", {}))]:
        for field_path, info in fields.items():
            ftype   = info.get("type", "?")
            example = info.get("example", "") or "null"
            text    = f"{field_path} ({ftype}): {example}  [appConfigs/{doc_type}]"
            item_id = f"appconfigs::{doc_type}::{field_path}"
            if item_id in schema_store.ids():
                continue
            emb = await embed(text)
            if emb:
                schema_store.upsert(item_id, text, emb, {"db": "appConfigs", "doc_type": doc_type, "field": field_path, "type": ftype})
                indexed += 1

    if indexed > 0:
        logger.info(f"[schema_discovery] Indexed {indexed} new field embeddings.")


def build_dynamic_supplement(include_stream: bool = True, include_appconfigs: bool = True, max_fields: int = 40) -> str:
    if not _schema_cache:
        return ""

    lines = [
        "─────────────────────────────────────────────────────────────",
        f"LIVE SCHEMA DISCOVERY (sampled at {_schema_cache.get('sampled_at', 'unknown')})",
        "Trust this over the static schema if there is any conflict.",
        "─────────────────────────────────────────────────────────────",
    ]

    if include_stream:
        stream = _schema_cache.get("stream", {})
        fields = stream.get("fields", {})
        if fields:
            lines.append(f"\nstream-datastore/{stream.get('collection','?')} — {stream.get('docs_sampled',0)} docs, {len(fields)} fields:")
            for path, info in list(fields.items())[:max_fields]:
                example = info.get("example") or "null"
                lines.append(f"  {path}  ({info.get('type','?')})  e.g. {repr(example)}")
            if len(fields) > max_fields:
                lines.append(f"  … and {len(fields) - max_fields} more fields.")

    if include_appconfigs:
        appconf   = _schema_cache.get("appconfigs", {})
        usersinfo = appconf.get("usersinfo_fields", {})
        if usersinfo:
            lines.append(f"\nappConfigs — usersinfo docs ({appconf.get('owners_sampled',0)} owners sampled):")
            for path, info in list(usersinfo.items())[:max_fields]:
                lines.append(f"  {path}  ({info.get('type','?')})  e.g. {repr(info.get('example') or 'null')}")
        cfg_fields = appconf.get("config_fields", {})
        if cfg_fields:
            lines.append("\nappConfigs — streaming config docs (default/_appName_):")
            for path, info in list(cfg_fields.items())[:max_fields]:
                lines.append(f"  {path}  ({info.get('type','?')})  e.g. {repr(info.get('example') or 'null')}")

    return "\n".join(lines) if len(lines) > 6 else ""


async def retrieve_schema_context(
    question:           str,
    include_stream:     bool = True,
    include_appconfigs: bool = False,
    top_k:              int  = 20,
) -> str:
    from lib.embeddings   import embed
    from lib.vector_store import VectorStore

    schema_store = VectorStore("schema")

    # Static fallback for appConfigs fields when vector store is cold
    _APPCONFIGS_STATIC = (
        "  appConfigs fields (each owner collection has these in the 'usersinfo' document):\n"
        "  maxUserLimit (integer): max concurrent users allowed\n"
        "  SubscriptionEndDate._seconds (integer): Unix timestamp of subscription expiry\n"
        "  SubscriptionStartDate._seconds (integer): Unix timestamp when subscription began\n"
        "  paidMinutes (number): total streaming minutes purchased\n"
        "  paidSecondsUsage (number): streaming seconds used\n"
        "  shouldAutoRenew (boolean): whether subscription auto-renews\n"
        "  products.ccu (number): concurrent user product limit\n"
        "  products.gb (number): bandwidth GB product\n"
        "  NOTE: Each collection = one owner username. Query _id='usersinfo' for billing."
    )

    if schema_store.count() == 0:
        return _APPCONFIGS_STATIC if include_appconfigs else ""

    q_emb = await embed(question)
    if q_emb is None:
        return _APPCONFIGS_STATIC if include_appconfigs else ""

    def _filter(item: dict) -> bool:
        db = item["metadata"].get("db", "")
        if include_stream     and db == "stream-datastore": return True
        if include_appconfigs and db == "appConfigs":       return True
        return False

    results = schema_store.search(q_emb, top_k=top_k, filter_fn=_filter, min_score=0.3)
    if not results:
        return _APPCONFIGS_STATIC if include_appconfigs else ""

    lines = [f"  {r['text']}" for r in results]
    if include_appconfigs and not any(r["metadata"].get("db") == "appConfigs" for r in results):
        lines.append(_APPCONFIGS_STATIC)

    return "\n".join(lines)


def get_cache_status() -> dict:
    return {
        "populated":      bool(_schema_cache),
        "sampled_at":     _schema_cache.get("sampled_at"),
        "elapsed_sec":    _schema_cache.get("elapsed_sec"),
        "stream_docs":    _schema_cache.get("stream", {}).get("docs_sampled", 0),
        "owners_sampled": _schema_cache.get("appconfigs", {}).get("owners_sampled", 0),
    }
