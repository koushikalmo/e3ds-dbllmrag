"""Change-stream watcher that keeps the schema cache in sync with reality.

We already re-sample the schema hourly in schema_discovery. That is fine as
a safety net, but up to an hour of drift is a lot when someone adds a field
and immediately starts asking about it. This module opens a Motor change
stream on both databases, flips a dirty flag when an event introduces a
path we've never seen, and lets a refresher loop force a schema refresh —
at most once per COOLDOWN_SEC (default 1 hour) no matter how many events
arrive. Change streams need a replica set; on a standalone we log a
warning and quietly step aside, and the hourly TTL sample carries the load.
"""
from __future__ import annotations
import os
import time
import asyncio
import logging
from typing import Any

from pymongo.errors import OperationFailure, PyMongoError

from lib.mongodb          import get_stream_db, get_appconfigs_db
from lib.schema_discovery import refresh_schema_cache
from lib                  import schema_discovery  # see _known_paths_for

logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    # Match the rest of the codebase: prefixed prints straight to stdout so
    # they show up under uvicorn / journalctl without extra logger config.
    print(f"[schema_watcher] {msg}", flush=True)

WAKE_SEC     = int(os.getenv("SCHEMA_WATCHER_WAKE_SEC",     "60"))
COOLDOWN_SEC = int(os.getenv("SCHEMA_WATCHER_COOLDOWN_SEC", "3600"))
MAX_DEPTH    = 4  # keep in step with schema_discovery's path extraction

_dirty:        bool  = False
_last_refresh: float = 0.0
_dirty_reason: str   = ""
_running:      bool  = False


def _paths_of(obj: Any, prefix: str = "", depth: int = 0, out: set[str] | None = None) -> set[str]:
    if out is None:
        out = set()
    if depth > MAX_DEPTH:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            out.add(path)
            if isinstance(v, dict):
                _paths_of(v, path, depth + 1, out)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                _paths_of(v[0], f"{path}[]", depth + 1, out)
    return out


def _known_paths_for(db_label: str) -> set[str]:
    # schema_discovery reassigns its _schema_cache dict on every refresh,
    # so `from schema_discovery import _schema_cache` would freeze on the
    # initial empty dict. Always go through the module.
    cache = schema_discovery._schema_cache
    if not cache:
        return set()
    if db_label == "stream-datastore":
        return set(cache.get("stream", {}).get("fields", {}).keys())
    if db_label == "appConfigs":
        appc = cache.get("appconfigs", {})
        return set(appc.get("usersinfo_fields", {}).keys()) | set(appc.get("config_fields", {}).keys())
    return set()


def _paths_in_event(change: dict) -> set[str]:
    op = change.get("operationType")
    if op in ("insert", "replace"):
        doc = change.get("fullDocument") or {}
        return _paths_of(doc)
    if op == "update":
        upd = change.get("updateDescription") or {}
        # updatedFields keys are already dot-paths for nested edits
        paths = set(upd.get("updatedFields", {}).keys())
        paths |= set(upd.get("removedFields", []) or [])
        return paths
    return set()


def _mark_dirty(reason: str) -> None:
    global _dirty, _dirty_reason
    if _dirty:
        return
    _dirty         = True
    _dirty_reason  = reason
    _log(f"dirty flag SET — {reason}")


async def _watch_database(db, db_label: str) -> None:
    # Filter server-side so we don't stream every operation over the wire.
    pipeline = [{"$match": {
        "operationType": {"$in": [
            "insert", "update", "replace",
            "drop", "rename", "dropDatabase", "invalidate",
        ]},
    }}]

    _log(f"opening change stream on {db_label}")
    try:
        async with db.watch(
            pipeline,
            full_document        = "updateLookup",
            max_await_time_ms    = 5_000,
        ) as stream:
            async for change in stream:
                # Already dirty? Nothing to gain from another comparison —
                # cheap backpressure for hot collections.
                if _dirty:
                    continue

                op = change.get("operationType")
                if op in ("drop", "rename", "dropDatabase"):
                    _mark_dirty(f"{db_label}: {op} on {change.get('ns',{})}")
                    continue
                if op == "invalidate":
                    # Cursor is dead after invalidate; break out and let the
                    # outer wrapper reopen it.
                    _mark_dirty(f"{db_label}: invalidate (cursor reset)")
                    break

                paths = _paths_in_event(change)
                if not paths:
                    continue
                known = _known_paths_for(db_label)
                if not known:
                    # First refresh hasn't populated the cache yet; without
                    # a baseline every path looks new. Skip until it lands.
                    continue
                new_paths = paths - known
                if new_paths:
                    sample = ", ".join(list(new_paths)[:5])
                    _mark_dirty(f"{db_label}: new field(s) {sample}"
                                + (f" (+{len(new_paths)-5} more)" if len(new_paths) > 5 else ""))
    except OperationFailure as e:
        # MongoDB rejects $changeStream on standalones with a message like
        # "The $changeStream stage is only supported on replica sets".
        # Bail permanently for this DB — no point in retrying.
        msg = str(e).lower()
        if "replica" in msg or "not supported" in msg or "requires" in msg:
            _log(
                f"WARN {db_label}: change streams unavailable "
                f"(cluster likely standalone): {e}. Watcher for this DB disabled."
            )
            return
        _log(f"ERROR {db_label}: operation failure: {e}")
    except PyMongoError as e:
        _log(f"ERROR {db_label}: pymongo error: {e}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        import traceback
        _log(f"ERROR {db_label}: crashed: {e}\n{traceback.format_exc()}")


async def _watch_database_forever(db, db_label: str) -> None:
    # Reopen the stream after transient network hiccups or invalidate events.
    # Exponential backoff so a broken cluster doesn't spam reconnects.
    backoff = 5
    while True:
        try:
            await _watch_database(db, db_label)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log(f"ERROR {db_label}: outer wrapper error: {e}")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 300)


async def _refresher_loop(default_stream_collection: str) -> None:
    global _dirty, _last_refresh, _dirty_reason
    while True:
        try:
            await asyncio.sleep(WAKE_SEC)
            if not _dirty:
                continue
            now = time.monotonic()
            if _last_refresh and (now - _last_refresh) < COOLDOWN_SEC:
                remaining = int(COOLDOWN_SEC - (now - _last_refresh))
                _log(f"dirty but cooldown active ({remaining}s left)")
                continue
            reason = _dirty_reason
            _log(f"refreshing schema cache → {reason}")
            try:
                await refresh_schema_cache(stream_collection=default_stream_collection, force=True)
                _last_refresh = time.monotonic()
                _dirty        = False
                _dirty_reason = ""
                _log(f"refresh complete (triggered by: {reason})")
            except Exception as e:
                # Leave the dirty flag set so we retry on the next tick.
                _log(f"ERROR refresh failed (will retry next tick): {e}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log(f"ERROR refresher loop error: {e}")


async def start_schema_watcher(default_stream_collection: str) -> None:
    global _running, _last_refresh
    if _running:
        _log("already running — skipping duplicate start")
        return
    if os.getenv("SCHEMA_WATCHER_ENABLED", "1").strip() in ("0", "false", "no"):
        _log("disabled via SCHEMA_WATCHER_ENABLED=0")
        return

    _running       = True
    # The startup refresh in main.py counts as "just refreshed" — don't let
    # the first event within COOLDOWN_SEC trigger another one immediately.
    _last_refresh  = time.monotonic()
    stream_db      = get_stream_db()
    appconfigs_db  = get_appconfigs_db()

    _log(f"starting (wake={WAKE_SEC}s, cooldown={COOLDOWN_SEC}s)")
    asyncio.create_task(_watch_database_forever(stream_db,     "stream-datastore"))
    asyncio.create_task(_watch_database_forever(appconfigs_db, "appConfigs"))
    asyncio.create_task(_refresher_loop(default_stream_collection))


def get_watcher_status() -> dict:
    return {
        "running":            _running,
        "dirty":              _dirty,
        "dirty_reason":       _dirty_reason,
        "last_refresh_ago":   round(time.monotonic() - _last_refresh, 1) if _last_refresh else None,
        "cooldown_sec":       COOLDOWN_SEC,
        "wake_sec":           WAKE_SEC,
    }
