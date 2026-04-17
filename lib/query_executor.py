import re
import asyncio
from typing import Any

from lib.db_registry import get_db, get_default_collection

MAX_RESULTS   = 200
QUERY_TIMEOUT = 15_000  # ms

_WRITE_STAGES = frozenset({"$out", "$merge"})

# These fields use regex matching so "Bogota" also finds "Bogotá"
_PARTIAL_MATCH_FIELDS = frozenset({
    "clientInfo.city", "clientInfo.country_name", "clientInfo.region",
    "appInfo.appName", "userDeviceInfo.os.name", "userDeviceInfo.client.name",
    "elInfo.computerName",
})

# This field is rarely present — filtering on it returns 0 results
_PROBLEMATIC_FIELDS = frozenset({"loggedInUserData"})

_DIACRITIC_MAP: dict[str, str] = {
    "a": "aáàâäãåā", "e": "eéèêëē", "i": "iíìîïī",
    "o": "oóòôöõøō", "u": "uúùûüū", "n": "nñ",
    "c": "cç",       "s": "sś",     "z": "zźż",
}


def _expand_diacritics(text: str) -> str:
    """"Bogota" → "b[oóòôöõøō]g[oóòôöõøō]t[aáàâäãåā]" for accent-insensitive matching."""
    parts = []
    for ch in text.lower():
        if ch in _DIACRITIC_MAP:
            parts.append(f"[{_DIACRITIC_MAP[ch]}]")
        elif ch.isalpha():
            parts.append(ch)
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def _normalize_match_query(query: dict) -> dict:
    if not isinstance(query, dict):
        return query

    result = {}
    for key, val in query.items():
        if key in _PROBLEMATIC_FIELDS:
            continue

        if key in ("$and", "$or", "$nor") and isinstance(val, list):
            result[key] = [_normalize_match_query(v) for v in val]
            continue

        if key in _PARTIAL_MATCH_FIELDS:
            if isinstance(val, str):
                result[key] = {"$regex": _expand_diacritics(val), "$options": "i"}
            elif isinstance(val, dict) and isinstance(val.get("$eq"), str):
                result[key] = {"$regex": _expand_diacritics(val["$eq"]), "$options": "i"}
            else:
                result[key] = val
        else:
            result[key] = val

    return result


def _normalize_pipeline(pipeline: list) -> list:
    return [{"$match": _normalize_match_query(s["$match"])} if "$match" in s else s for s in pipeline]


def _sanitize_pipeline(pipeline: list) -> list:
    # remove $out/$merge — read-only safety
    clean = []
    for stage in pipeline:
        op = next(iter(stage), None)
        if op in _WRITE_STAGES:
            print(f"[executor] SECURITY: stripped '{op}' stage")
        else:
            clean.append(stage)
    return clean


def _enforce_limit(pipeline: list) -> list:
    has_limit = any("$limit" in s for s in pipeline)
    if not has_limit:
        return pipeline + [{"$limit": MAX_RESULTS}]
    return [{"$limit": min(s["$limit"], MAX_RESULTS)} if "$limit" in s else s for s in pipeline]


def _prepare_pipeline(pipeline: list) -> list:
    return _enforce_limit(_sanitize_pipeline(pipeline))


def _get_db(database: str):
    try:
        return get_db(database)
    except ValueError:
        raise ValueError(f"Unknown database '{database}'. Check data/db_registry.json.")


def _resolve_collection(database: str, collection: str) -> str:
    if collection:
        return collection
    default = get_default_collection(database)
    if default:
        return default
    raise ValueError(
        f"No collection specified for '{database}' and no default configured. "
        "For appConfigs, provide the owner's username as the collection name."
    )


def _make_serializable(docs: list[dict]) -> list[dict]:
    import bson

    def convert(val: Any) -> Any:
        if isinstance(val, bson.ObjectId):   return str(val)
        if isinstance(val, bson.Decimal128): return float(str(val))
        if isinstance(val, dict):            return {k: convert(v) for k, v in val.items()}
        if isinstance(val, list):            return [convert(i) for i in val]
        return val

    return [convert(doc) for doc in docs]


def _summarize(results: list[dict]) -> dict:
    return {"count": len(results), "sample": results[:5]}


async def _run_aggregate(database: str, collection: str, raw_pipeline: list) -> list[dict]:
    db        = _get_db(database)
    coll_name = _resolve_collection(database, collection)
    pipeline  = _prepare_pipeline(_normalize_pipeline(raw_pipeline))
    print(f"[executor] aggregate {database}/{coll_name} ({len(pipeline)} stages)")
    try:
        cursor  = db[coll_name].aggregate(pipeline, allowDiskUse=True, maxTimeMS=QUERY_TIMEOUT)
        results = await cursor.to_list(length=MAX_RESULTS)
        return _make_serializable(results)
    except Exception as err:
        _raise_friendly(err, database, coll_name)


async def _run_count_documents(database: str, collection: str, query: dict) -> list[dict]:
    # count_documents() instead of aggregate+$count — avoids $limit interference
    db        = _get_db(database)
    coll_name = _resolve_collection(database, collection)
    clean     = _normalize_match_query(query or {})
    print(f"[executor] countDocuments {database}/{coll_name}")
    try:
        count = await db[coll_name].count_documents(clean, maxTimeMS=QUERY_TIMEOUT)
        return [{"count": count}]
    except Exception as err:
        _raise_friendly(err, database, coll_name)


async def _run_find(
    database: str, collection: str, query: dict,
    projection: dict | None = None, sort: dict | None = None, limit: int = 50,
) -> list[dict]:
    db        = _get_db(database)
    coll_name = _resolve_collection(database, collection)
    clean     = _normalize_match_query(query or {})
    limit     = min(limit or 50, MAX_RESULTS)
    print(f"[executor] find {database}/{coll_name} (limit={limit})")
    try:
        cursor = db[coll_name].find(clean, projection or {})
        if sort:
            cursor = cursor.sort(list(sort.items()))
        cursor  = cursor.max_time_ms(QUERY_TIMEOUT).limit(limit)
        results = await cursor.to_list(length=limit)
        return _make_serializable(results)
    except Exception as err:
        _raise_friendly(err, database, coll_name)


async def _run_distinct(
    database: str, collection: str, field: str, query: dict | None = None,
) -> list[dict]:
    db        = _get_db(database)
    coll_name = _resolve_collection(database, collection)
    clean     = _normalize_match_query(query or {})
    print(f"[executor] distinct '{field}' {database}/{coll_name}")
    try:
        values = await db[coll_name].distinct(field, clean)
        return [{"value": v} for v in values[:MAX_RESULTS] if v is not None]
    except Exception as err:
        _raise_friendly(err, database, coll_name)


def _raise_friendly(err: Exception, database: str, coll_name: str):
    s = str(err)
    if "MaxTimeMSExpired" in s or "exceeded time limit" in s.lower():
        raise TimeoutError(
            f"Query exceeded {QUERY_TIMEOUT // 1000}s. "
            "Add a more specific filter (owner, date range, country) to reduce data scanned."
        )
    if "NamespaceNotFound" in s:
        if database == "stream-datastore":
            raise ValueError(
                f"Collection '{coll_name}' doesn't exist. "
                "Monthly collections use format 'Apr_2025'. It may not have data yet."
            )
        raise ValueError(
            f"Owner collection '{coll_name}' doesn't exist in appConfigs. "
            "Check that the owner username is spelled correctly."
        )
    if "BSONObjectTooLarge" in s:
        raise ValueError("Result exceeded 16MB. Add a $project to return fewer fields.")
    raise err


_run_single = _run_aggregate  # backward-compat alias


async def get_existing_year_collections(database: str, year: int) -> list[str]:
    from lib.collection_resolver import all_collections_for_year
    db       = _get_db(database)
    existing = set(await db.list_collection_names())
    return [c for c in all_collections_for_year(year) if c in existing]


def build_year_pipeline(pipeline: list, extra_collections: list[str]) -> list:
    # Stages before $group/$count run per-collection inside each $unionWith.
    # $group and beyond run once after all months are unioned together.
    _REDUCE_OPS = {"$group", "$count", "$bucket", "$bucketAuto", "$facet"}
    split = next(
        (i for i, s in enumerate(pipeline) if isinstance(s, dict) and any(op in s for op in _REDUCE_OPS)),
        len(pipeline),
    )
    pre  = pipeline[:split]
    post = pipeline[split:]

    expanded = list(pre)
    for coll in extra_collections:
        expanded.append({"$unionWith": {"coll": coll, "pipeline": list(pre)}})
    expanded.extend(post)
    return expanded


async def execute_query(query_obj: dict) -> dict:
    if query_obj["queryType"] == "single":
        operation = query_obj.get("operation", "aggregate")
        db        = query_obj["database"]
        coll      = query_obj.get("collection", "")

        if operation == "countDocuments":
            results = await _run_count_documents(db, coll, query_obj.get("query", {}))
        elif operation == "find":
            results = await _run_find(
                db, coll,
                query      = query_obj.get("query", {}),
                projection = query_obj.get("projection"),
                sort       = query_obj.get("sort"),
                limit      = query_obj.get("limit", 50),
            )
        elif operation == "distinct":
            results = await _run_distinct(db, coll, query_obj.get("field", ""), query_obj.get("query", {}))
        else:
            results = await _run_aggregate(db, coll, query_obj["pipeline"])

        print(f"[executor] {operation} → {len(results)} result(s)")
        return {
            "queryType":        "single",
            "operation":        operation,
            "results":          results,
            "summary":          _summarize(results),
            "explanation":      query_obj.get("explanation", ""),
            "resultLabel":      query_obj.get("resultLabel", "Results"),
            "executedPipeline": query_obj.get("pipeline", []),
        }

    if query_obj["queryType"] == "dual":
        q1, q2 = query_obj["queries"]
        print("[executor] Running dual query in parallel…")
        results1, results2 = await asyncio.gather(
            _run_aggregate(q1["database"], q1.get("collection", ""), q1["pipeline"]),
            _run_aggregate(q2["database"], q2.get("collection", ""), q2["pipeline"]),
        )
        print(f"[executor] Dual: primary={len(results1)}, secondary={len(results2)}")

        merged = None
        if merge_key := query_obj.get("mergeKey"):
            config_map = {str(doc.get(merge_key) or doc.get("_id", "")): doc for doc in results2}
            merged = []
            for doc in results1:
                owner    = str(doc.get(merge_key) or doc.get("appInfo", {}).get("owner", "") or "")
                enriched = dict(doc)
                if owner and owner in config_map:
                    enriched["_configData"] = config_map[owner]
                merged.append(enriched)
            matched = sum(1 for d in merged if "_configData" in d)
            print(f"[executor] Merged {len(merged)} docs, {matched} with config data")

        return {
            "queryType":   "dual",
            "results":     {"primary": results1, "secondary": results2, "merged": merged},
            "summary":     {"primary": _summarize(results1), "secondary": _summarize(results2)},
            "explanation": query_obj.get("explanation", ""),
            "resultLabel": query_obj.get("resultLabel", "Cross-Database Results"),
        }

    raise ValueError(f"Unsupported queryType: '{query_obj.get('queryType')}'")
