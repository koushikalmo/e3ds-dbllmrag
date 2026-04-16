# lib/mongodb.py — Async MongoDB connection manager

import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from dotenv import load_dotenv

load_dotenv()

URI_STREAM     = os.getenv("MONGODB_URI_STREAM")
URI_APPCONFIGS = os.getenv("MONGODB_URI_APPCONFIGS")

STREAM_DB_NAME     = os.getenv("STREAM_DB_NAME",     "stream-datastore")
APPCONFIGS_DB_NAME = os.getenv("APPCONFIGS_DB_NAME", "appConfigs")

# Singletons — created on first use, reused until shutdown
_stream_client:     AsyncIOMotorClient | None = None
_appconfigs_client: AsyncIOMotorClient | None = None


def _build_client(uri: str) -> AsyncIOMotorClient:
    # 10 pooled connections, 8s timeout if server is unreachable
    return AsyncIOMotorClient(uri, maxPoolSize=10, serverSelectionTimeoutMS=8_000)


def get_stream_db() -> AsyncIOMotorDatabase:
    global _stream_client
    if not URI_STREAM:
        raise RuntimeError("MONGODB_URI_STREAM is not set. Copy .env.example to .env.")
    if _stream_client is None:
        _stream_client = _build_client(URI_STREAM)
    return _stream_client[STREAM_DB_NAME]


def get_appconfigs_db() -> AsyncIOMotorDatabase:
    global _appconfigs_client
    if not URI_APPCONFIGS:
        raise RuntimeError("MONGODB_URI_APPCONFIGS is not set. Copy .env.example to .env.")
    if _appconfigs_client is None:
        _appconfigs_client = _build_client(URI_APPCONFIGS)
    return _appconfigs_client[APPCONFIGS_DB_NAME]


async def close_connections() -> None:
    from lib.db_registry import close_all
    await close_all()

    global _stream_client, _appconfigs_client
    if _stream_client:
        _stream_client.close()
        _stream_client = None
    if _appconfigs_client:
        _appconfigs_client.close()
        _appconfigs_client = None


async def ping_databases() -> dict:
    """Returns {"stream": "ok", "appconfigs": "ok"} or error strings."""
    results = {}
    try:
        await get_stream_db().command("ping")
        results["stream"] = "ok"
    except Exception as e:
        results["stream"] = f"error: {e}"
    try:
        await get_appconfigs_db().command("ping")
        results["appconfigs"] = "ok"
    except Exception as e:
        results["appconfigs"] = f"error: {e}"
    return results
