import re
import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from lib.mongodb import get_stream_db, get_appconfigs_db

logger = logging.getLogger(__name__)

_TTL_DOCS   = 1_800  # 30 min
_TTL_VALUES = 3_600  # 60 min
_TTL_GLOBAL = 3_600  # 60 min

_N_DOCS    = 3
_N_COUNTRY = 30
_N_CITY    = 20
_N_OWNER   = 30
_N_APP     = 30
_N_OS      = 20
_N_BROWSER = 15
_N_APPCONF = 60

_MAX_DOC_CHARS = 400

_STRIP_TOP = frozenset({
    "apiKeys", "streamingApiKeys", "timeRecords",
    "iceConnectionStateChanges", "candidates_selected",
    "webRtcStatsData", "loggedInUserData", "_id",
})

_ELINFO_KEEP     = frozenset({"computerName", "city", "region", "country"})
_CLIENTINFO_KEEP = frozenset({"country_name", "city", "region", "timezone", "continent_code"})


@dataclass
class _CollectionData:
    documents: list[dict] = field(default_factory=list)
    countries: list[str]  = field(default_factory=list)
    cities:    list[str]  = field(default_factory=list)
    os_names:  list[str]  = field(default_factory=list)
    browsers:  list[str]  = field(default_factory=list)
    owners:    list[str]  = field(default_factory=list)
    app_names: list[str]  = field(default_factory=list)
    docs_ts:   float      = 0.0
    values_ts: float      = 0.0


@dataclass
class _GlobalData:
    collection_list: list[str] = field(default_factory=list)
    owner_list:      list[str] = field(default_factory=list)
    ts:              float     = 0.0


_collection_cache: dict[str, _CollectionData] = {}
_global_cache = _GlobalData()

# Tracks in-progress refreshes to prevent duplicate concurrent DB calls
_refresh_tasks: set[str] = set()


def _slim_doc(doc: dict) -> dict:
    result = {}
    for key, val in doc.items():
        if key in _STRIP_TOP:
            continue
        if key == "elInfo" and isinstance(val, dict):
            slim = {k: v for k, v in val.items() if k in _ELINFO_KEEP}
            if slim: result["elInfo"] = slim
            continue
        if key == "clientInfo" and isinstance(val, dict):
            slim: dict = {}
            for ck, cv in val.items():
                if ck in _CLIENTINFO_KEEP:
                    slim[ck] = cv
                elif ck == "fullInfo" and isinstance(cv, dict):
                    sec = cv.get("security")
                    if sec: slim["fullInfo"] = {"security": sec}
            if slim: result["clientInfo"] = slim
            continue
        if key == "userDeviceInfo" and isinstance(val, dict):
            slim = {}
            if isinstance(val.get("os"),     dict): slim["os"]     = {"name": val["os"].get("name", "")}
            if isinstance(val.get("client"), dict): slim["client"] = {"name": val["client"].get("name", "")}
            if slim: result["userDeviceInfo"] = slim
            continue
        result[key] = val
    return result


def _compact(doc: dict) -> str:
    raw = json.dumps(doc, default=str, separators=(",", ":"))
    return raw[:_MAX_DOC_CHARS - 3] + "..." if len(raw) > _MAX_DOC_CHARS else raw


async def _sample_documents(collection: str) -> list[dict]:
    db = get_stream_db()
    try:
        cursor = db[collection].aggregate([
            {"$match": {"e3ds_employee": {"$ne": True}}},
            {"$sample": {"size": _N_DOCS}},
        ], maxTimeMS=8_000)
        docs = await cursor.to_list(length=_N_DOCS)
        return [_slim_doc(d) for d in docs]
    except Exception as e:
        logger.warning(f"[live_ctx] Document sample failed ({collection}): {e}")
        return []


async def _top_values(collection: str, field_path: str, top_n: int) -> list[str]:
    db    = get_stream_db()
    match = {"e3ds_employee": {"$ne": True}, field_path: {"$exists": True, "$nin": [None, "", "null"]}}
    try:
        cursor = db[collection].aggregate([
            {"$match": match},
            {"$group": {"_id": f"${field_path}", "n": {"$sum": 1}}},
            {"$sort": {"n": -1}},
            {"$limit": top_n},
        ], maxTimeMS=10_000, allowDiskUse=True)
        docs = await cursor.to_list(length=top_n)
        return [str(d["_id"]) for d in docs if d.get("_id") not in (None, "", "null")]
    except Exception as e:
        logger.warning(f"[live_ctx] Top values failed ({collection}/{field_path}): {e}")
        return []


async def _fetch_collection_list() -> list[str]:
    _PAT = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_(\d{4})$")
    _MON = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    db = get_stream_db()
    try:
        names = await db.list_collection_names()
        valid = [n for n in names if _PAT.match(n)]
        valid.sort(key=lambda n: (int(_PAT.match(n).group(2)), _MON[_PAT.match(n).group(1)]), reverse=True)
        return valid
    except Exception as e:
        logger.warning(f"[live_ctx] Collection list failed: {e}")
        return []


async def _fetch_appconfigs_owners() -> list[str]:
    _SKIP = frozenset({"system.indexes", "system.users", "system.version"})
    db = get_appconfigs_db()
    try:
        names = await db.list_collection_names()
        return [n for n in names if n not in _SKIP and not n.startswith("system.")][:_N_APPCONF]
    except Exception as e:
        logger.warning(f"[live_ctx] AppConfigs owner list failed: {e}")
        return []


async def _populate_docs(collection: str) -> None:
    cache = _collection_cache.setdefault(collection, _CollectionData())
    docs  = await _sample_documents(collection)
    cache.documents = docs
    cache.docs_ts   = time.monotonic()
    logger.info(f"[live_ctx] Docs cached: {len(docs)} from {collection}")


async def _populate_values(collection: str) -> None:
    cache   = _collection_cache.setdefault(collection, _CollectionData())
    results = await asyncio.gather(
        _top_values(collection, "clientInfo.country_name",    _N_COUNTRY),
        _top_values(collection, "clientInfo.city",             _N_CITY),
        _top_values(collection, "userDeviceInfo.os.name",     _N_OS),
        _top_values(collection, "userDeviceInfo.client.name", _N_BROWSER),
        _top_values(collection, "appInfo.owner",              _N_OWNER),
        _top_values(collection, "appInfo.appName",            _N_APP),
        return_exceptions=True,
    )

    def _safe(r: Any) -> list[str]:
        return r if isinstance(r, list) else []

    cache.countries = _safe(results[0])
    cache.cities    = _safe(results[1])
    cache.os_names  = _safe(results[2])
    cache.browsers  = _safe(results[3])
    cache.owners    = _safe(results[4])
    cache.app_names = _safe(results[5])
    cache.values_ts = time.monotonic()
    logger.info(f"[live_ctx] Values cached for {collection}: {len(cache.countries)} countries, {len(cache.owners)} owners")


async def _populate_globals() -> None:
    coll_list, owner_list = await asyncio.gather(
        _fetch_collection_list(), _fetch_appconfigs_owners(), return_exceptions=True,
    )
    if isinstance(coll_list,  list): _global_cache.collection_list = coll_list
    if isinstance(owner_list, list): _global_cache.owner_list      = owner_list
    _global_cache.ts = time.monotonic()


def _docs_stale(col: str)   -> bool:
    c = _collection_cache.get(col)
    return c is None or (time.monotonic() - c.docs_ts) > _TTL_DOCS

def _values_stale(col: str) -> bool:
    c = _collection_cache.get(col)
    return c is None or (time.monotonic() - c.values_ts) > _TTL_VALUES

def _global_stale() -> bool:
    return (time.monotonic() - _global_cache.ts) > _TTL_GLOBAL


async def _bg_refresh(collection: str) -> None:
    to_run: dict[str, Any] = {}

    if _global_stale() and "globals" not in _refresh_tasks:
        _refresh_tasks.add("globals")
        to_run["globals"] = _populate_globals()

    key_docs = f"docs:{collection}"
    if _docs_stale(collection) and key_docs not in _refresh_tasks:
        _refresh_tasks.add(key_docs)
        to_run[key_docs] = _populate_docs(collection)

    key_vals = f"vals:{collection}"
    if _values_stale(collection) and key_vals not in _refresh_tasks:
        _refresh_tasks.add(key_vals)
        to_run[key_vals] = _populate_values(collection)

    if not to_run:
        return
    try:
        await asyncio.gather(*to_run.values(), return_exceptions=True)
    finally:
        _refresh_tasks.difference_update(to_run.keys())


async def warm_all_caches(collection: str) -> None:
    logger.info(f"[live_ctx] Warming caches for {collection}…")
    start = time.monotonic()
    await asyncio.gather(_populate_globals(), _populate_docs(collection), _populate_values(collection), return_exceptions=True)
    logger.info(f"[live_ctx] Cache warm done in {round(time.monotonic() - start, 2)}s")


async def get_live_context(collection: str, question: str = "") -> str:
    cache = _collection_cache.get(collection)

    if cache is None:
        # Cold start — wait up to 12s for an initial sample
        try:
            await asyncio.wait_for(
                asyncio.gather(_populate_globals(), _populate_docs(collection), _populate_values(collection), return_exceptions=True),
                timeout=12.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[live_ctx] Cold-start timed out for {collection}")
        cache = _collection_cache.get(collection)
    else:
        asyncio.create_task(_bg_refresh(collection))

    if not cache or (not cache.documents and not cache.countries):
        return ""

    parts: list[str] = []

    if _global_cache.collection_list:
        shown  = _global_cache.collection_list[:15]
        extra  = len(_global_cache.collection_list) - len(shown)
        suffix = f", ... (+{extra} more)" if extra else ""
        parts.append("Available stream collections (newest first): " + ", ".join(shown) + suffix)

    if cache.documents:
        parts.append(f"\nSample documents from {collection} (field names are EXACT — copy them):")
        for i, doc in enumerate(cache.documents, 1):
            parts.append(f"  [{i}] {_compact(doc)}")

    val_lines: list[str] = []
    if cache.countries:
        val_lines.append("  clientInfo.country_name: " + ", ".join(cache.countries))
    if cache.cities:
        shown  = cache.cities[:15]
        extra  = len(cache.cities) - len(shown)
        suffix = f", ... (+{extra} more)" if extra else ""
        val_lines.append("  clientInfo.city: " + ", ".join(shown) + suffix)
    if cache.os_names:
        val_lines.append("  userDeviceInfo.os.name: " + ", ".join(cache.os_names))
    if cache.browsers:
        val_lines.append("  userDeviceInfo.client.name: " + ", ".join(cache.browsers))
    if cache.owners:
        shown  = cache.owners[:20]
        extra  = len(cache.owners) - len(shown)
        suffix = f", ... (+{extra} more)" if extra else ""
        val_lines.append("  appInfo.owner (top active, also valid appConfigs collection names): " + ", ".join(shown) + suffix)
    if cache.app_names:
        shown  = cache.app_names[:20]
        extra  = len(cache.app_names) - len(shown)
        suffix = f", ... (+{extra} more)" if extra else ""
        val_lines.append("  appInfo.appName: " + ", ".join(shown) + suffix)

    if val_lines:
        parts.append(f"\nActual values in {collection} — use EXACT spelling, do not invent values:")
        parts.extend(val_lines)

    if not cache.owners and _global_cache.owner_list:
        parts.append("\nKnown appConfigs collection names (owner usernames): " + ", ".join(_global_cache.owner_list[:30]))

    if not parts:
        return ""

    header = (
        "═══════════════════════════════════════════════\n"
        "LIVE DATABASE CONTEXT (refreshed hourly from real data)\n"
        "═══════════════════════════════════════════════"
    )
    return header + "\n" + "\n".join(parts)


def get_cache_status() -> dict:
    now = time.monotonic()
    per_col = {
        col: {
            "docs":         len(d.documents),
            "countries":    len(d.countries),
            "owners":       len(d.owners),
            "docs_age_min": round((now - d.docs_ts)   / 60, 1) if d.docs_ts   else None,
            "vals_age_min": round((now - d.values_ts) / 60, 1) if d.values_ts else None,
        }
        for col, d in _collection_cache.items()
    }
    return {
        "collections":    len(_global_cache.collection_list),
        "appconf_owners": len(_global_cache.owner_list),
        "per_collection": per_col,
    }
