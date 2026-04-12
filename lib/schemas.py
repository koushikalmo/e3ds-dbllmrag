# ============================================================
# lib/schemas.py — Structural Rules for the LLM
# ============================================================
# This file contains ONLY structural rules and critical gotchas
# that the LLM needs for every query. It is intentionally small
# (~750 tokens) so the context window is not wasted.
#
# WHAT IS NOT HERE:
#   Field listings and descriptions are gone. Those now live in
#   the vector store (data/vectors/schema.json) and are retrieved
#   semantically per-query by lib/schema_discovery.py +
#   lib/vector_store.py. Only the 15-20 fields most relevant to
#   each question are injected — instead of all 200+.
#
# WHY THIS SPLIT:
#   Old approach: dump 4,500 tokens of schema on every query.
#   New approach: 750 tokens of rules + ~400 tokens of relevant
#   fields retrieved by semantic search = ~1,150 tokens.
#   This fits comfortably in 8192 num_ctx with room for output.
# ============================================================


# ── Structural rules ──────────────────────────────────────────
# These are included in every LLM call, regardless of which
# database the question targets.

_CORE_RULES = """You are a MongoDB aggregation pipeline expert for the Eagle 3D Streaming platform.
Convert the user's natural language question into a valid MongoDB aggregation pipeline.

═══════════════════════════════════════════════
OUTPUT FORMAT — FOLLOW EXACTLY
═══════════════════════════════════════════════
Respond with ONLY a raw JSON object. No prose, no markdown, no code fences.
First character must be "{". Last character must be "}".

For a SINGLE database query:
{
  "queryType":   "single",
  "database":    "stream-datastore" | "appConfigs",
  "collection":  "<collection name>",
  "pipeline":    [ ...aggregation stages... ],
  "explanation": "<one sentence: what this returns>",
  "resultLabel": "<short UI label, e.g. 'Top Countries by Sessions'>"
}

For a DUAL (cross-database) query:
{
  "queryType":   "dual",
  "queries": [
    { "database": "stream-datastore", "collection": "<month>", "pipeline": [...] },
    { "database": "appConfigs",       "collection": "<owner>", "pipeline": [...] }
  ],
  "mergeKey":    "owner",
  "explanation": "<one sentence>",
  "resultLabel": "<short UI label>"
}

═══════════════════════════════════════════════
SAFETY RULES
═══════════════════════════════════════════════
- Always include { "$limit": 50 } unless user asks for more. Hard cap: 200.
- NEVER include apiKey or streamingApiKeys[].apiKey in $project output.
- Use the default collection from user context unless a different month is specified.

═══════════════════════════════════════════════
STREAM-DATASTORE RULES
═══════════════════════════════════════════════
- Collections are named by month: "Apr_2026", "Mar_2026", "Feb_2026", etc.
- ALWAYS filter internal traffic first:
    { "$match": { "e3ds_employee": false } }
- ALL *_Timestamp fields are FLOATS (not ints). Subtraction works directly:
    { "$addFields": { "durationSeconds": { "$subtract": ["$DisconnectTime_Timestamp", "$startTimeStamp"] } } }
- webRtcStatsData.avgRoundTripTime is stored as a STRING:
    Always convert before sorting/comparing: { "$toDouble": "$webRtcStatsData.avgRoundTripTime" }

CRITICAL FIELD NAMES (use exactly as shown — wrong names return 0 results):
- Country:  "clientInfo.country_name"  ← full name like "Brazil". NEVER use "country_code".
- City:     "clientInfo.city"
- OS:       "userDeviceInfo.os.name"
- Browser:  "userDeviceInfo.client.name"
- Owner:    "appInfo.owner"
- App:      "appInfo.appName"

═══════════════════════════════════════════════
APPCONFIGS RULES — READ CAREFULLY
═══════════════════════════════════════════════
There is NO collection called "users". Do NOT use "users" as a collection name.

Structure: each collection is named after an owner's username.
  Database: appConfigs
    Collection: "eduardo"    ← the owner named "eduardo"
    Collection: "imerza"     ← the owner named "imerza"
    Collection: "Tridonic"   ← the owner named "Tridonic"

Each owner's collection has up to 4 document types (by _id):
  _id = "usersinfo"           → billing, subscription, limits
  _id = "InfoToConstructUrls" → streaming URL configuration
  _id = "default"             → default streaming app config
  _id = "<appName>"           → per-app config (e.g. "FluidFluxProject")

To query an owner's billing:  "collection": "<owner>", pipeline: [{ "$match": { "_id": "usersinfo" } }]
To query an owner's config:   "collection": "<owner>", pipeline: [{ "$match": { "_id": "default" } }]

- maxUserLimit in "usersinfo" is an INTEGER. No conversion needed.
- Subscription dates use Firebase format: { "_seconds": 1714435200, "_nanoseconds": 0 }
  To compare with current time: use SubscriptionEndDate._seconds vs Unix timestamp.
- NEVER project apiKeys.apiKey or streamingApiKeys.apiKey.

═══════════════════════════════════════════════
CROSS-DATABASE JOIN
═══════════════════════════════════════════════
MongoDB cannot $lookup across two connection strings. Use queryType="dual".

Join key:
  stream-datastore.appInfo.owner  ←→  appConfigs collection name
  e.g. appInfo.owner = "eduardo"  →   collection "eduardo" in appConfigs

For dual queries: the backend runs both pipelines concurrently and merges in Python.
Stream query should group by appInfo.owner.
AppConfigs query should set collection = the owner's username.

DUAL QUERY REQUIRED FIELDS — every sub-query MUST have all three:
  "database":   "stream-datastore" or "appConfigs"   ← REQUIRED
  "collection": "<month>" or "<owner username>"       ← REQUIRED
  "pipeline":   [ ...stages... ]                      ← REQUIRED

Example dual query structure:
{
  "queryType": "dual",
  "queries": [
    { "database": "stream-datastore", "collection": "Apr_2025", "pipeline": [...] },
    { "database": "appConfigs",       "collection": "eduardo",  "pipeline": [...] }
  ],
  "mergeKey": "owner",
  "explanation": "...",
  "resultLabel": "..."
}
"""


# ── Structural schema blocks ──────────────────────────────────
# Very short summaries of each DB. Field details come from RAG.

_STREAM_STRUCTURE = """
DATABASE: stream-datastore
One document = one user streaming session.
Collections by month: Apr_2026, Mar_2026, Feb_2026, Jan_2026, Dec_2025, ...back to Dec_2023.
Key join field: appInfo.owner links to the appConfigs collection name.
"""

_APPCONFIGS_STRUCTURE = """
DATABASE: appConfigs
~5,500+ collections, one per customer account.
Collection name = owner username (e.g. "eduardo", "imerza").
Each collection has up to 4 documents (usersinfo, InfoToConstructUrls, default, <appName>).
"""


def build_system_prompt(
    include_stream:     bool = True,
    include_appconfigs: bool = False,
    schema_context:     str  = "",
) -> str:
    """
    Assembles the LLM system prompt for this query.

    Args:
        include_stream:     Include the stream-datastore structure note.
        include_appconfigs: Include the appConfigs structure note.
        schema_context:     Retrieved-from-RAG field descriptions.
                            Pass the output of retrieve_schema_context()
                            from schema_discovery.py. If empty, the model
                            works from its general MongoDB knowledge and
                            whatever examples are in the user message.

    Returns:
        Complete system prompt string, ready for Ollama.
    """
    parts = [_CORE_RULES]

    if include_stream:
        parts.append(_STREAM_STRUCTURE)
    if include_appconfigs:
        parts.append(_APPCONFIGS_STRUCTURE)

    if schema_context:
        parts.append(
            "═══════════════════════════════════════════════\n"
            "RELEVANT SCHEMA FIELDS (retrieved for this question)\n"
            "═══════════════════════════════════════════════\n"
            + schema_context
        )

    return "\n".join(parts)


# ── Keyword sets for database routing ─────────────────────────
# query_generator.py scans the question for these keywords to
# decide which database(s) to target. Add here if questions
# are being routed to the wrong database.

STREAM_KEYWORDS = {
    "session", "stream", "connected", "connection", "disconnect", "spectator",
    "heartbeat", "reconnect", "streaming",
    "city", "country", "region", "continent", "timezone", "latitude", "longitude",
    "user", "viewer", "client", "vpn",
    "device", "desktop", "mobile", "tablet", "phone", "browser", "os",
    "windows", "macos", "mac", "linux", "android", "ios", "iphone",
    "chrome", "firefox", "safari", "edge",
    "app", "application", "owner", "project", "unreal",
    "load time", "loadtime", "loading", "bitrate", "fps", "framerate",
    "latency", "rtt", "round trip", "packets", "lost", "bandwidth", "quality",
    "resolution", "jitter", "ping", "buffer",
    "gpu", "cpu", "memory", "ram", "vram", "temperature", "server",
    "edge", "machine", "computer", "node", "worker",
    "webrtc", "ice", "turn", "stun", "peer",
    "april", "march", "february", "january", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "today", "week", "month", "recent", "latest", "this month",
    "top", "worst", "best", "average", "slow", "fast", "most", "least",
    "count", "total", "how many", "list",
}

APPCONFIGS_KEYWORDS = {
    "subscription", "expired", "expir", "active", "plan", "renew",
    "start date", "end date", "valid", "billing",
    "api key", "apikey", "key", "access", "token",
    "max user", "concurrent", "limit", "capacity", "ccus",
    "account", "config", "license", "sync", "quota", "minutes",
    "streaming key", "appconfig", "product", "paid",
}
