# lib/db_registry.py — Config-driven multi-database registry
# Add a new database: add URI to .env + add entry to data/db_registry.json + restart.

import os
import json
import logging
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_REGISTRY_FILE = Path(__file__).parent.parent / "data" / "db_registry.json"


def _load_registry() -> list[dict]:
    if not _REGISTRY_FILE.exists():
        logger.warning("[db_registry] data/db_registry.json not found — using empty registry")
        return []
    try:
        return json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"[db_registry] Failed to load registry: {e}")
        return []


_registry: list[dict] = _load_registry()
_clients:  dict[str, AsyncIOMotorClient] = {}


def get_db_names() -> list[str]:
    return [entry["name"] for entry in _registry]


def get_registry_entry(db_name: str) -> dict | None:
    for entry in _registry:
        if entry["name"] == db_name:
            return entry
    return None


def get_db(db_name: str) -> AsyncIOMotorDatabase:
    """Returns the Motor database handle for a registered database (lazy init)."""
    entry = get_registry_entry(db_name)
    if entry is None:
        raise ValueError(
            f"Unknown database '{db_name}'. "
            f"Registered: {get_db_names()}. "
            f"Add an entry to data/db_registry.json."
        )

    uri = os.getenv(entry["env_uri"])
    if not uri:
        raise RuntimeError(
            f"Environment variable '{entry['env_uri']}' is not set. Add it to .env."
        )

    if db_name not in _clients:
        logger.info(f"[db_registry] Creating Motor client for '{db_name}'")
        _clients[db_name] = AsyncIOMotorClient(uri, maxPoolSize=10, serverSelectionTimeoutMS=8_000)

    db_name_env = entry.get("env_db_name", "")
    actual_db = (
        os.getenv(db_name_env) if db_name_env else entry.get("default_db_name", db_name)
    ) or entry.get("default_db_name", db_name)

    return _clients[db_name][actual_db]


def get_default_collection(db_name: str) -> str | None:
    entry = get_registry_entry(db_name)
    return entry.get("default_collection") if entry else None


async def close_all() -> None:
    for name, client in _clients.items():
        client.close()
        logger.info(f"[db_registry] Closed '{name}'")
    _clients.clear()


async def ping_all() -> dict[str, str]:
    results = {}
    for name in get_db_names():
        try:
            await get_db(name).command("ping")
            results[name] = "ok"
        except Exception as e:
            results[name] = f"error: {str(e)[:100]}"
    return results


def get_all_descriptions() -> str:
    return "\n".join(
        f"  {e['name']}: {e.get('description', '(no description)')}"
        for e in _registry
    )
