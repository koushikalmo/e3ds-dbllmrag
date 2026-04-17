"""
Persistent data digest — samples real documents every 3 days and stores field
names + example values to disk. Injected into the LLM system prompt so the
model always knows exact field spellings and real value formats.
"""

import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_DIGEST_FILE      = Path(__file__).parent.parent / "data" / "data_digest.json"
_REFRESH_DAYS     = 3
_REFRESH_INTERVAL = _REFRESH_DAYS * 24 * 3600
_SAMPLE_SIZE      = 10
_MAX_EXAMPLES     = 5  # unique values to keep per field
_MAX_STR_LEN      = 60

# Fields that must never appear in digest (PII / secrets)
_STRIP_FIELDS = frozenset({
    "apiKeys", "streamingApiKeys", "loggedInUserData",
    "timeRecords", "iceConnectionStateChanges", "candidates_selected",
    "elInfo.availableApps", "elInfo.xirysobj",
    "webRtcStatsData",
})

_digest: dict = {}


# ── helpers ───────────────────────────────────────────────────────────────────

def _should_strip(path: str) -> bool:
    return any(path == f or path.startswith(f + ".") or path.startswith(f + "[]") for f in _STRIP_FIELDS)


def _flatten(obj: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    if depth > 5 or not isinstance(obj, dict):
        return {}
    result = {}
    for key, val in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if _should_strip(path):
            continue
        if isinstance(val, dict):
            result.update(_flatten(val, path, depth + 1))
        elif isinstance(val, list):
            if val and isinstance(val[0], dict):
                result.update(_flatten(val[0], f"{path}[]", depth + 1))
        else:
            result[path] = val
    return result


def _build_field_summary(docs: list[dict]) -> dict[str, dict]:
    values: dict[str, list]  = {}
    types:  dict[str, str]   = {}

    for doc in docs:
        for path, val in _flatten(doc).items():
            if val is None or val == "" or val == "null":
                continue
            t = type(val).__name__
            types.setdefault(path, t)
            bucket = values.setdefault(path, [])
            if len(bucket) < _MAX_EXAMPLES:
                display = str(val)[:_MAX_STR_LEN] if isinstance(val, str) else val
                if display not in bucket:
                    bucket.append(display)

    return {
        path: {"type": types[path], "examples": values.get(path, [])}
        for path in types
    }


# ── sampling ──────────────────────────────────────────────────────────────────

async def _sample_stream(collection: str) -> dict:
    from lib.mongodb import get_stream_db
    db = get_stream_db()
    try:
        cursor = db[collection].aggregate([
            {"$match": {"e3ds_employee": {"$ne": True}}},
            {"$sort":  {"VideoStreamStartedAt_Timestamp": -1}},
            {"$limit": _SAMPLE_SIZE},
        ], maxTimeMS=15_000)
        docs = await cursor.to_list(length=_SAMPLE_SIZE)
        return {
            "collection":  collection,
            "docs_sampled": len(docs),
            "fields":      _build_field_summary(docs),
        }
    except Exception as e:
        logger.warning(f"[data_digest] Stream sample failed ({collection}): {e}")
        return {"collection": collection, "docs_sampled": 0, "fields": {}}


async def _sample_appconfigs(owner_count: int = 3) -> dict:
    from lib.mongodb import get_appconfigs_db
    db = get_appconfigs_db()
    try:
        all_names = await db.list_collection_names()
        owners    = [n for n in all_names if not n.startswith("system.")][:owner_count]
    except Exception as e:
        logger.warning(f"[data_digest] AppConfigs list failed: {e}")
        return {"owners_sampled": [], "usersinfo_fields": {}, "config_fields": {}}

    usersinfo_docs, config_docs = [], []
    for owner in owners:
        try:
            doc = await db[owner].find_one({"_id": "usersinfo"})
            if doc:
                usersinfo_docs.append(doc)
            doc = await db[owner].find_one({"_id": "default"})
            if doc:
                config_docs.append(doc)
        except Exception as e:
            logger.warning(f"[data_digest] AppConfigs sample failed ({owner}): {e}")

    return {
        "owners_sampled":   owners,
        "usersinfo_fields": _build_field_summary(usersinfo_docs),
        "config_fields":    _build_field_summary(config_docs),
    }


# ── refresh ───────────────────────────────────────────────────────────────────

async def refresh_digest(force: bool = False) -> None:
    global _digest

    age = time.time() - _digest.get("_ts", 0)
    if not force and _digest and age < _REFRESH_INTERVAL:
        return

    logger.info("[data_digest] Refreshing data digest…")
    start = time.monotonic()

    # figure out the most recent collection
    try:
        from lib.mongodb import get_stream_db
        db    = get_stream_db()
        names = await db.list_collection_names()
        import re
        _MON  = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        _PAT  = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_(\d{4})$")
        valid = [n for n in names if _PAT.match(n)]
        valid.sort(key=lambda n: (int(_PAT.match(n).group(2)), _MON[_PAT.match(n).group(1)]), reverse=True)
        latest_col = valid[0] if valid else "Apr_2026"
    except Exception:
        latest_col = "Apr_2026"

    stream_result, appconfigs_result = await asyncio.gather(
        _sample_stream(latest_col),
        _sample_appconfigs(owner_count=3),
        return_exceptions=True,
    )

    if isinstance(stream_result, Exception):
        logger.error(f"[data_digest] Stream sampling error: {stream_result}")
        stream_result = {"collection": latest_col, "docs_sampled": 0, "fields": {}}
    if isinstance(appconfigs_result, Exception):
        logger.error(f"[data_digest] AppConfigs sampling error: {appconfigs_result}")
        appconfigs_result = {"owners_sampled": [], "usersinfo_fields": {}, "config_fields": {}}

    _digest = {
        "refreshed_at":  datetime.now(timezone.utc).isoformat(),
        "next_refresh":  datetime.fromtimestamp(time.time() + _REFRESH_INTERVAL, tz=timezone.utc).isoformat(),
        "_ts":           time.time(),
        "stream":        stream_result,
        "appconfigs":    appconfigs_result,
    }

    _save_to_file()
    elapsed = round(time.monotonic() - start, 2)
    logger.info(
        f"[data_digest] Done in {elapsed}s — "
        f"{stream_result['docs_sampled']} stream docs, "
        f"{len(appconfigs_result['owners_sampled'])} appConfigs owners"
    )


def _save_to_file() -> None:
    try:
        _DIGEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DIGEST_FILE.write_text(json.dumps(_digest, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as e:
        logger.error(f"[data_digest] Save failed: {e}")


def load_from_file() -> None:
    global _digest
    if not _DIGEST_FILE.exists():
        return
    try:
        raw = _DIGEST_FILE.read_text(encoding="utf-8")
        _digest = json.loads(raw) if raw.strip() else {}
        if _digest:
            logger.info(f"[data_digest] Loaded digest from disk (refreshed {_digest.get('refreshed_at','?')})")
    except Exception as e:
        logger.warning(f"[data_digest] Could not load digest file: {e}")


# ── LLM prompt text ───────────────────────────────────────────────────────────

def get_digest_text() -> str:
    if not _digest:
        return ""

    refreshed = _digest.get("refreshed_at", "unknown")[:10]
    lines = [
        "═══════════════════════════════════════════════",
        f"REAL DATA DIGEST (sampled {refreshed}, refreshes every {_REFRESH_DAYS} days)",
        "Use these EXACT field names and spellings — copied from real documents.",
        "═══════════════════════════════════════════════",
    ]

    stream = _digest.get("stream", {})
    fields = stream.get("fields", {})
    if fields:
        lines.append(f"\nstream-datastore/{stream.get('collection','?')} — {stream.get('docs_sampled',0)} recent docs:")
        for path, info in fields.items():
            examples = info.get("examples", [])
            if not examples:
                continue
            example_str = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in examples[:_MAX_EXAMPLES])
            lines.append(f"  {path} ({info.get('type','?')}): {example_str}")

    appconf = _digest.get("appconfigs", {})
    owners  = appconf.get("owners_sampled", [])
    if owners:
        lines.append(f"\nappConfigs — sampled owners: {', '.join(owners)}")
        usersinfo = appconf.get("usersinfo_fields", {})
        if usersinfo:
            lines.append("  usersinfo document fields:")
            for path, info in usersinfo.items():
                examples = info.get("examples", [])
                if not examples:
                    continue
                example_str = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in examples[:3])
                lines.append(f"    {path} ({info.get('type','?')}): {example_str}")

    lines.append("═══════════════════════════════════════════════")
    return "\n".join(lines)


def get_digest_status() -> dict:
    return {
        "populated":    bool(_digest),
        "refreshed_at": _digest.get("refreshed_at"),
        "next_refresh": _digest.get("next_refresh"),
        "stream_docs":  _digest.get("stream", {}).get("docs_sampled", 0),
        "field_count":  len(_digest.get("stream", {}).get("fields", {})),
    }


# ── scheduler ─────────────────────────────────────────────────────────────────

async def _run_scheduler() -> None:
    while True:
        try:
            await refresh_digest()
        except Exception as e:
            logger.error(f"[data_digest] Scheduler error: {e}")
        await asyncio.sleep(_REFRESH_INTERVAL)


async def start_digest_scheduler() -> None:
    load_from_file()
    # refresh immediately if file is missing or stale
    age = time.time() - _digest.get("_ts", 0)
    if not _digest or age > _REFRESH_INTERVAL:
        asyncio.create_task(refresh_digest(force=True))
    asyncio.create_task(_run_scheduler())
