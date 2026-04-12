# ============================================================
# lib/db_registry.py — Database Registry (Config-Driven Multi-DB)
# ============================================================
# Loads the list of databases from data/db_registry.json and
# manages Motor connections for each one.
#
# WHY THIS EXISTS:
# ─────────────────
# The original code hardcoded exactly two databases:
#   - "stream-datastore"  → hardcoded in query_executor.py
#   - "appConfigs"        → hardcoded in query_executor.py
#
# Adding a third database required editing two Python files.
# That's a code change, a review, and a redeploy for what is
# fundamentally a configuration change.
#
# With this registry, adding a new database is:
#   1. Add the URI to .env:  MONGODB_URI_NEWDB=mongodb+srv://...
#   2. Add an entry to data/db_registry.json
#   3. Restart the server (or POST /api/schema/refresh)
#   4. Done — schema discovery auto-indexes its fields,
#      the vector RAG learns its structure, the LLM can query it.
#
# HOW IT WORKS:
# ─────────────
# On first call to get_db(name), a Motor client is lazily
# created for that database. The connection is cached
# indefinitely (until server shutdown). This is identical to
# the pattern in the original mongodb.py.
#
# DATA FORMAT (data/db_registry.json):
# ──────────────────────────────────────
# [
#   {
#     "name":            "stream-datastore",
#     "env_uri":         "MONGODB_URI_STREAM",
#     "env_db_name":     "STREAM_DB_NAME",
#     "default_db_name": "stream-datastore",
#     "description":     "Monthly streaming sessions ...",
#     "default_collection": "Apr_2026"
#   },
#   ...
# ]
# ============================================================

import os
import json
import logging
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_REGISTRY_FILE = Path(__file__).parent.parent / "data" / "db_registry.json"


def _load_registry() -> list[dict]:
    """Reads and parses data/db_registry.json."""
    if not _REGISTRY_FILE.exists():
        logger.warning("[db_registry] data/db_registry.json not found — using empty registry")
        return []
    try:
        return json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"[db_registry] Failed to load registry: {e}")
        return []


# ── Module-level state ────────────────────────────────────────
# _registry is loaded once at import time.
# _clients is populated lazily on first get_db() call for each DB.

_registry: list[dict] = _load_registry()
_clients:  dict[str, AsyncIOMotorClient] = {}


def get_db_names() -> list[str]:
    """Returns the list of registered database names."""
    return [entry["name"] for entry in _registry]


def get_registry_entry(db_name: str) -> dict | None:
    """Returns the registry entry for a database, or None if not found."""
    for entry in _registry:
        if entry["name"] == db_name:
            return entry
    return None


def get_db(db_name: str) -> AsyncIOMotorDatabase:
    """
    Returns the Motor database handle for a registered database.

    Creates the Motor client on first call (lazy init), then caches
    and reuses it for all subsequent calls. This pattern avoids
    opening a new TCP connection pool on every request.

    Args:
        db_name: The logical database name from the registry
                 (e.g. "stream-datastore", "appConfigs").

    Raises:
        ValueError: If db_name is not in the registry.
        RuntimeError: If the URI env var is not set.
    """
    entry = get_registry_entry(db_name)
    if entry is None:
        known = get_db_names()
        raise ValueError(
            f"Unknown database '{db_name}'. "
            f"Registered databases: {known}. "
            f"Add an entry to data/db_registry.json to register a new database."
        )

    # Read the URI from the environment variable named in the registry
    uri_env_var = entry["env_uri"]
    uri         = os.getenv(uri_env_var)
    if not uri:
        raise RuntimeError(
            f"Environment variable '{uri_env_var}' is not set. "
            f"Add it to your .env file to connect to '{db_name}'."
        )

    # Lazy-create the Motor client for this database
    if db_name not in _clients:
        logger.info(f"[db_registry] Creating Motor client for '{db_name}'")
        _clients[db_name] = AsyncIOMotorClient(
            uri,
            maxPoolSize=10,
            serverSelectionTimeoutMS=8_000,
        )

    # Read the actual database name (may differ from the logical name)
    db_name_env = entry.get("env_db_name", "")
    actual_db   = (
        os.getenv(db_name_env)
        if db_name_env
        else entry.get("default_db_name", db_name)
    ) or entry.get("default_db_name", db_name)

    return _clients[db_name][actual_db]


def get_default_collection(db_name: str) -> str | None:
    """
    Returns the default collection for a database (may be None).

    stream-datastore has a meaningful default (most recent month).
    appConfigs has no default because each collection is a different owner.
    """
    entry = get_registry_entry(db_name)
    if entry:
        return entry.get("default_collection")
    return None


async def close_all() -> None:
    """Closes all Motor client connections. Called on server shutdown."""
    for name, client in _clients.items():
        client.close()
        logger.info(f"[db_registry] Closed connection to '{name}'")
    _clients.clear()


async def ping_all() -> dict[str, str]:
    """
    Pings all registered databases and returns their status.
    Used by GET /api/health.

    Returns:
        { "stream-datastore": "ok", "appConfigs": "ok", ... }
    """
    results = {}
    for name in get_db_names():
        try:
            db = get_db(name)
            await db.command("ping")
            results[name] = "ok"
        except Exception as e:
            results[name] = f"error: {str(e)[:100]}"
    return results


def get_all_descriptions() -> str:
    """
    Returns a formatted text block of all database descriptions.
    Injected into the LLM system prompt so it knows what databases exist.
    """
    lines = []
    for entry in _registry:
        lines.append(
            f"  {entry['name']}: {entry.get('description', '(no description)')}"
        )
    return "\n".join(lines)
