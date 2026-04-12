# ============================================================
# lib/mongodb.py — Async MongoDB Connection Manager
# ============================================================
# This module is the only place in the entire codebase that
# knows how to connect to MongoDB. Every other module that
# needs a database just calls get_stream_db() or
# get_appconfigs_db() and gets back a ready-to-use object.
#
# WHY MOTOR INSTEAD OF PYMONGO?
# ─────────────────────────────
# FastAPI runs on asyncio — it handles many requests at the
# same time on a single thread by switching between them
# whenever one has to wait (e.g. waiting for a DB response).
#
# PyMongo is synchronous. If you used it inside an async
# route handler, it would BLOCK the entire thread during the
# query — freezing all other requests until that one finished.
# That turns your "concurrent" server into something worse
# than a simple single-threaded script.
#
# Motor is the official async wrapper around PyMongo. It uses
# the same MongoDB wire protocol under the hood but integrates
# with asyncio using `await`, so FastAPI can handle other
# requests while your MongoDB query is in flight.
#
# WHY MODULE-LEVEL SINGLETONS?
# ─────────────────────────────
# When you connect to MongoDB, the client opens a TCP
# connection pool (multiple open connections ready to use).
# Opening that pool is expensive — it involves DNS lookup,
# TLS handshake, authentication, etc.
#
# If we created a new client on every API request, we'd pay
# that cost thousands of times per minute. Instead we create
# one client per database at startup and reuse it forever.
#
# The pattern: None → created on first use → reused until
# server shutdown, where close_connections() is called.
# ============================================================

import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from dotenv import load_dotenv

# Load .env before we read any environment variables.
# This is a safety net — main.py also loads .env, but having
# it here means this module works even when imported directly
# in tests or scripts outside of the FastAPI server.
load_dotenv()

# ── Connection URIs from environment ─────────────────────────
# These are the full MongoDB connection strings, e.g.:
#   mongodb+srv://user:password@cluster.mongodb.net/
# They come from .env so secrets never touch source code.
URI_STREAM     = os.getenv("MONGODB_URI_STREAM")
URI_APPCONFIGS = os.getenv("MONGODB_URI_APPCONFIGS")

# ── Database names ────────────────────────────────────────────
# The database name is the part inside MongoDB that holds
# all the collections. The URI points to the cluster;
# the DB name selects which database within that cluster.
STREAM_DB_NAME     = os.getenv("STREAM_DB_NAME",     "stream-datastore")
APPCONFIGS_DB_NAME = os.getenv("APPCONFIGS_DB_NAME", "appConfigs")

# ── Module-level singletons (start as None) ──────────────────
# These will be set to real AsyncIOMotorClient objects the
# first time get_stream_db() or get_appconfigs_db() is called.
_stream_client:     AsyncIOMotorClient | None = None
_appconfigs_client: AsyncIOMotorClient | None = None


def _build_client(uri: str) -> AsyncIOMotorClient:
    """
    Creates a Motor client with production-safe settings.

    maxPoolSize=10
        Keep up to 10 concurrent TCP connections alive per
        worker process. Under load, queries share these
        connections instead of opening new ones each time.
        10 is a good default for a single-worker dev server;
        increase for production multi-worker deployments.

    serverSelectionTimeoutMS=8000
        If MongoDB is unreachable, give up after 8 seconds
        instead of hanging for the default 30 seconds.
        This surfaces infrastructure problems quickly instead
        of making every user request hang for half a minute.
    """
    return AsyncIOMotorClient(
        uri,
        maxPoolSize=10,
        serverSelectionTimeoutMS=8_000,
    )


def get_stream_db() -> AsyncIOMotorDatabase:
    """
    Returns the Motor database handle for stream-datastore.

    This is the main analytics database. Its collections are
    named by calendar month:
        "Apr_2025", "Mar_2025", "Feb_2025", etc.

    Each document in these collections represents one user
    streaming session. Here's the structure of a real document
    (simplified):

        {
          "_id":         "userInfo_000156c0-...",
          "appInfo":     { "owner": "eduardo", "appName": "FluidFluxProject" },
          "clientInfo":  { "city": "Itajaí", "country_name": "Brazil" },
          "userDeviceInfo": { "os": { "name": "Windows" }, ... },
          "loadTime":    8.514,           ← seconds to first video frame
          "startTimeStamp": 1745851775,   ← Unix seconds
          "DisconnectTime_Timestamp": 1745851874,
          "webRtcStatsData": {
              "avgBitrate": 50226,        ← kbps
              "packetsLost": 399,
              "avgRoundTripTime": "0.177" ← string, not float!
          },
          "elInfo": {                     ← the STREAMING SERVER info
              "computerName": "E3DS-S23",
              "systemInfo": { "cpu": {...}, "graphics": {...} }
          }
        }

    Usage example:
        db = get_stream_db()
        results = await db["Apr_2025"].aggregate([
            { "$match": { "clientInfo.country_name": "Brazil" } },
            { "$limit": 10 }
        ]).to_list(None)
    """
    global _stream_client

    if not URI_STREAM:
        raise RuntimeError(
            "MONGODB_URI_STREAM is not set.\n"
            "Copy .env.example to .env and fill in your MongoDB connection string."
        )

    # Lazy initialization: create only on first call, then reuse
    if _stream_client is None:
        _stream_client = _build_client(URI_STREAM)

    # Motor's [] operator returns a Database object.
    # No actual network call happens here — queries happen later.
    return _stream_client[STREAM_DB_NAME]


def get_appconfigs_db() -> AsyncIOMotorDatabase:
    """
    Returns the Motor database handle for appConfigs.

    This database holds account configuration for each owner.
    The default collection is "users" (set via APPCONFIGS_COLLECTION).

    Each document represents one account owner. Critically, the
    document's _id IS the owner's username — not an ObjectId.
    This makes cross-database joins easy: look up appInfo.owner
    from stream-datastore in the _id field here.

    Example document structure:
        {
          "_id":          "eduardo",          ← owner username
          "maxUserLimit": "5",                ← string, cast with $toInt
          "ccus": {
              "connector.eagle3dstreaming.com": 2  ← active connections
          },
          "SubscriptionStartDate": { "_seconds": 1666027056 },
          "SubscriptionEndDate":   { "_seconds": 1714435200 },
          "apiKeys": [
              { "isActive": true, "createdAt": { "_seconds": 1696013448 } }
              // apiKey field intentionally omitted — it's encrypted
          ],
          "streamingApiKeys": [
              { "isActive": true, "createdAt": { "_seconds": 1723836744 } }
          ]
        }

    Note on dates: MongoDB doesn't store these as native Date objects.
    Instead they're stored as { "_seconds": <unix_int> } objects
    (a Firebase Firestore timestamp format). To compare them:
        { "SubscriptionEndDate._seconds": { "$gte": 1714435200 } }
    """
    global _appconfigs_client

    if not URI_APPCONFIGS:
        raise RuntimeError(
            "MONGODB_URI_APPCONFIGS is not set.\n"
            "Copy .env.example to .env and fill in your MongoDB connection string."
        )

    if _appconfigs_client is None:
        _appconfigs_client = _build_client(URI_APPCONFIGS)

    return _appconfigs_client[APPCONFIGS_DB_NAME]


async def close_connections() -> None:
    """
    Gracefully closes all MongoDB connection pools.
    Delegates to db_registry which manages all registered databases.
    """
    from lib.db_registry import close_all
    await close_all()

    # Also close the legacy singletons (belt-and-suspenders)
    global _stream_client, _appconfigs_client
    if _stream_client:
        _stream_client.close()
        _stream_client = None
    if _appconfigs_client:
        _appconfigs_client.close()
        _appconfigs_client = None


async def ping_databases() -> dict:
    """
    Sends a lightweight ping command to both databases.

    The MongoDB "ping" command is the standard health-check —
    it verifies the connection is alive and the server responds,
    without doing any actual data work.

    Returns a dict like:
        { "stream": "ok", "appconfigs": "ok" }
    or on failure:
        { "stream": "error: <reason>", "appconfigs": "ok" }

    Used by GET /api/health so ops can monitor both connections
    with a single HTTP call (e.g. from Uptime Robot, Grafana, etc.)
    """
    results = {}

    try:
        db = get_stream_db()
        await db.command("ping")
        results["stream"] = "ok"
    except Exception as e:
        results["stream"] = f"error: {e}"

    try:
        db = get_appconfigs_db()
        await db.command("ping")
        results["appconfigs"] = "ok"
    except Exception as e:
        results["appconfigs"] = f"error: {e}"

    return results
