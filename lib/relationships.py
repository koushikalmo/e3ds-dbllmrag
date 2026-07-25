"""Cross-database relationship graph.

Declarative knowledge of how stream-datastore and appConfigs connect: primary
keys, foreign keys, discriminators, and user-facing aliases. Consumed by the
LLM prompt, the query classifier, and the response validator so the model does
not have to re-infer these relationships every call.
"""
from __future__ import annotations
import re
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


FIELD_ROLES = [
    {
        "db":         "stream-datastore",
        "collection": "*",
        "field":      "appInfo.owner",
        "role":       "foreign_key",
        "references": "appConfigs collection name",
        "notes":      "Value is the owner's username. That same username IS the collection name in appConfigs.",
    },
    {
        "db":         "stream-datastore",
        "collection": "*",
        "field":      "appInfo.appName",
        "role":       "foreign_key",
        "references": "appConfigs document _id (when _id names an app)",
        "notes":      "Owner+appName together identify a specific streaming app.",
    },
    {
        "db":         "stream-datastore",
        "collection": "*",
        "field":      "loggedInUserData.name",
        "role":       "end_user_identity",
        "notes":      "The end user who logged in to a session. NOT the owner. Sparse — null on most docs.",
    },
    {
        "db":         "appConfigs",
        "collection": "*",
        "field":      "<collection_name>",
        "role":       "primary_key",
        "notes":      "Each appConfigs collection IS one owner account. The collection name itself is the PK.",
    },
    {
        "db":         "appConfigs",
        "collection": "*",
        "field":      "_id",
        "role":       "discriminator",
        "notes":      "Identifies document type within an owner's collection.",
    },
]


JOINS = [
    {
        "id":         "owner-across-dbs",
        "kind":       "collection-name-fk",
        "from":       {"db": "stream-datastore", "collection": "*", "field": "appInfo.owner"},
        "to":         {"db": "appConfigs", "collection_is_key": True},
        "merge_key":  "owner",
        "description":
            "stream.appInfo.owner == appConfigs collection name. "
            "To join: stream aggregate grouped by appInfo.owner, then one appConfigs query per owner.",
    },
    {
        "id":         "appname-across-dbs",
        "kind":       "field-fk",
        "from":       {"db": "stream-datastore", "collection": "*", "field": "appInfo.appName"},
        "to":         {"db": "appConfigs", "collection_is_key": True, "field": "_id"},
        "merge_key":  "appName",
        "description":
            "stream.appInfo.appName == appConfigs.<owner>._id (when that _id is an app name, not a discriminator).",
    },
]


DISCRIMINATORS = [
    {
        "db":         "appConfigs",
        "collection": "*",
        "field":      "_id",
        "required":   True,
        "values": {
            "usersinfo":           "billing, subscription, maxUserLimit, paidMinutes, SubscriptionEndDate",
            "InfoToConstructUrls": "streaming URL templates",
            "default":             "default streaming app configuration",
            "<appName>":           "per-app configuration (e.g. _id='FluidFluxProject')",
        },
        "intent_map": [
            (("subscription", "billing", "maxuserlimit", "paid", "subscriptionenddate", "plan", "renew",
              "expired", "active subscription", "quota", "ccus"), "usersinfo"),
            (("streaming url", "url config", "infotoconstructurls"),                                     "InfoToConstructUrls"),
            (("app config", "default config", "per-app", "app configuration"),                           "default"),
        ],
    },
]


ALIASES = [
    {
        "term":     "owner",
        "meaning":  "appInfo.owner (stream-datastore) OR the collection name (appConfigs). NOT the logged-in user.",
        "db_hints": ["stream", "appconfigs"],
    },
    {
        "term":     "user",
        "meaning":  "loggedInUserData.name — the end user who logged in to the session. NOT the owner. Sparse.",
        "db_hints": ["stream"],
    },
    {
        "term":     "app",
        "meaning":  "appInfo.appName (stream) OR _id of the per-app document in appConfigs.<owner>.",
        "db_hints": ["stream", "appconfigs"],
    },
    {
        "term":     "current month",
        "meaning":  "Defaults to the DEFAULT_STREAM_COLLECTION env var (e.g. Apr_2026).",
        "db_hints": ["stream"],
    },
]


_STREAM_TRIGGERS = frozenset({
    "session", "sessions", "stream", "streaming", "heartbeat", "heartbeats",
    "connected", "connection", "disconnect", "disconnected", "spectator",
    "loadtime", "load time", "bitrate", "latency", "rtt", "packets",
    "city", "country", "region", "viewer", "viewers",
})
_APPCONFIGS_TRIGGERS = frozenset({
    "subscription", "subscriptions", "billing", "plan", "plans", "renew",
    "expired", "expires", "active subscription", "maxuserlimit", "max user limit",
    "paid", "paidminutes", "quota", "ccus", "apikey", "api key", "streaming key",
    "license", "concurrent",
})


_owner_names_cache: frozenset[str] = frozenset()
_owner_names_ts:    float          = 0.0
_OWNER_CACHE_TTL_S                 = 3600.0


async def _load_owner_names() -> frozenset[str]:
    """Return the lowercased set of appConfigs collection names (i.e. owner usernames)."""
    global _owner_names_cache, _owner_names_ts

    now = time.monotonic()
    if _owner_names_cache and (now - _owner_names_ts) < _OWNER_CACHE_TTL_S:
        return _owner_names_cache

    try:
        from lib.mongodb import get_appconfigs_db
        db    = get_appconfigs_db()
        names = await db.list_collection_names()
        _owner_names_cache = frozenset(n.lower() for n in names if n and not n.startswith("_"))
        _owner_names_ts    = now
        logger.info(f"[relationships] Cached {len(_owner_names_cache)} owner collection names")
    except Exception as e:
        logger.warning(f"[relationships] Failed to load owner names: {e}")

    return _owner_names_cache


@dataclass
class ClassificationResult:
    query_type:       str                   # "single" | "dual"
    db_hint:          str                   # "stream" | "appconfigs" | "both"
    owner_hint:       str | None            = None
    merge_key:        str | None            = None
    required_filters: list[dict]            = field(default_factory=list)
    reasoning:        list[str]             = field(default_factory=list)


async def classify_with_relationships(question: str) -> ClassificationResult:
    """Inspect the question through the relationship graph to route it."""
    q = question.lower()
    tokens = set(re.findall(r"[a-z0-9_]+", q))

    stream_hit     = any(t in _STREAM_TRIGGERS for t in tokens) or any(kw in q for kw in _STREAM_TRIGGERS)
    appconfigs_hit = any(t in _APPCONFIGS_TRIGGERS for t in tokens) or any(kw in q for kw in _APPCONFIGS_TRIGGERS)

    owner_hint = await _extract_owner(question)

    reasoning: list[str] = []
    if stream_hit:     reasoning.append("stream trigger word(s) matched")
    if appconfigs_hit: reasoning.append("appConfigs trigger word(s) matched")
    if owner_hint:     reasoning.append(f"owner hint extracted: '{owner_hint}'")

    required_filters: list[dict] = []
    if appconfigs_hit:
        disc = _pick_discriminator(q)
        if disc:
            required_filters.append({
                "db":     "appConfigs",
                "field":  "_id",
                "value":  disc,
                "reason": f"appConfigs queries require _id discriminator ('{disc}' inferred from question).",
            })
            reasoning.append(f"appConfigs discriminator: _id='{disc}'")

    if stream_hit and appconfigs_hit:
        return ClassificationResult(
            query_type       = "dual",
            db_hint          = "both",
            owner_hint       = owner_hint,
            merge_key        = JOINS[0]["merge_key"],
            required_filters = required_filters,
            reasoning        = reasoning + ["both sides triggered → dual query"],
        )

    if appconfigs_hit and not stream_hit:
        return ClassificationResult(
            query_type       = "single",
            db_hint          = "appconfigs",
            owner_hint       = owner_hint,
            required_filters = required_filters,
            reasoning        = reasoning + ["appConfigs-only single query"],
        )

    return ClassificationResult(
        query_type       = "single",
        db_hint          = "stream",
        owner_hint       = owner_hint,
        required_filters = required_filters,
        reasoning        = reasoning + ["stream-only single query (default)"],
    )


async def _extract_owner(question: str) -> str | None:
    """Find an owner-username mention in the question and validate against the known list."""
    q = question.strip()
    m = re.search(r"(?:owner|for|of|by)\s+['\"]?([a-zA-Z][a-zA-Z0-9_\-]{1,48})['\"]?", q, re.IGNORECASE)
    candidate = m.group(1).lower() if m else None

    if not candidate:
        for t in re.findall(r"['\"]([a-zA-Z][a-zA-Z0-9_\-]{1,48})['\"]", q):
            candidate = t.lower()
            break

    if not candidate:
        return None

    names = await _load_owner_names()
    if candidate in names:
        return candidate

    for n in names:
        if n.lower() == candidate:
            return n

    return None


def _pick_discriminator(question_lc: str) -> str | None:
    disc = DISCRIMINATORS[0]
    for keywords, value in disc["intent_map"]:
        if any(kw in question_lc for kw in keywords):
            return value
    return None


def render_relationship_block() -> str:
    """Render the relationship graph as a compact prompt section."""
    lines = [
        "═══════════════════════════════════════════════",
        "RELATIONSHIP GRAPH — cross-database join rules",
        "═══════════════════════════════════════════════",
        "",
        "PRIMARY / FOREIGN KEYS:",
        "  • stream-datastore.appInfo.owner  ──FK──▶  appConfigs.<collection name>",
        "    (The collection name ITSELF is the owner identifier.)",
        "  • stream-datastore.appInfo.appName  ──FK──▶  appConfigs.<owner>._id",
        "    (When _id is an app name, not a discriminator.)",
        "",
        "DISCRIMINATORS (appConfigs — REQUIRED in every $match):",
        "  _id = \"usersinfo\"            → billing, subscription, maxUserLimit, paidMinutes",
        "  _id = \"InfoToConstructUrls\"  → streaming URL templates",
        "  _id = \"default\"              → default streaming app config",
        "  _id = \"<appName>\"            → per-app configuration",
        "  ⚠ Querying appConfigs WITHOUT _id in $match returns mixed / wrong documents.",
        "",
        "TERMINOLOGY (disambiguate before querying):",
        "  • \"owner\"  = appInfo.owner (stream) OR collection name (appConfigs). NOT the end user.",
        "  • \"user\"   = loggedInUserData.name (stream). The end user who logged in. Sparse/null.",
        "  • \"app\"    = appInfo.appName (stream) OR _id (appConfigs, when _id is an app name).",
        "",
        "JOIN PATTERN (dual query):",
        "  1. Aggregate stream side grouped by appInfo.owner.",
        "  2. For the appConfigs side, pick the owner's collection name; add _id discriminator.",
        "  3. mergeKey: \"owner\"  (or \"appName\" when joining on app).",
        "═══════════════════════════════════════════════",
    ]
    return "\n".join(lines)


def check_relationships(query_obj: dict) -> list[dict]:
    """Validator checks derived from the relationship graph. Same shape as response_validator."""
    checks: list[dict] = []
    qt = query_obj.get("queryType", "single")

    if qt == "dual":
        if not query_obj.get("mergeKey"):
            checks.append(_err(
                "REL_NO_MERGE_KEY",
                "Dual query is missing 'mergeKey'. Must be 'owner' (most common) or 'appName'. "
                "Without a merge key the two result sets cannot be joined in Python.",
            ))
        elif query_obj["mergeKey"] not in {"owner", "appName"}:
            checks.append(_warn(
                "REL_UNKNOWN_MERGE_KEY",
                f"Unusual mergeKey '{query_obj['mergeKey']}'. Known keys: 'owner', 'appName'.",
            ))
        else:
            checks.append(_pass(
                "REL_MERGE_KEY_OK",
                f"Dual query uses mergeKey='{query_obj['mergeKey']}' — matches the known join.",
            ))

        for i, sub in enumerate(query_obj.get("queries", [])):
            if sub.get("database") == "appConfigs":
                if not _pipeline_has_id_match(sub.get("pipeline", [])):
                    checks.append(_err(
                        "REL_APPCONFIGS_NO_DISCRIMINATOR",
                        f"Dual query[{i}] targets appConfigs but has no _id discriminator in $match. "
                        "Add {\"$match\": {\"_id\": \"usersinfo\"}} (or the appropriate discriminator).",
                    ))

    elif qt == "single" and query_obj.get("database") == "appConfigs":
        pipe = query_obj.get("pipeline", [])
        match_filter = query_obj.get("query", {})
        if query_obj.get("operation", "aggregate") == "aggregate":
            if not _pipeline_has_id_match(pipe):
                checks.append(_err(
                    "REL_APPCONFIGS_NO_DISCRIMINATOR",
                    "appConfigs aggregate pipeline has no _id discriminator in $match. "
                    "Every appConfigs query must filter by _id (usersinfo / InfoToConstructUrls / default / <appName>).",
                ))
            else:
                checks.append(_pass(
                    "REL_DISCRIMINATOR_OK",
                    "appConfigs query correctly filters by _id discriminator.",
                ))
        else:
            if "_id" not in (match_filter or {}):
                checks.append(_warn(
                    "REL_APPCONFIGS_NO_DISCRIMINATOR",
                    "appConfigs query without _id filter may return mixed document types.",
                ))

    if qt == "single" and query_obj.get("database") == "stream-datastore":
        pipe_text = _stringify(query_obj)
        bad_username_refs = [
            name for name in ("username", "user_name", "userName", "userInfo.name", "user.name")
            if f'"{name}"' in pipe_text or f"'{name}'" in pipe_text or f"${name}" in pipe_text
        ]
        if bad_username_refs:
            checks.append(_err(
                "REL_WRONG_USERNAME_FIELD",
                f"Stream query references {bad_username_refs} — not a real field. "
                "The end user name is 'loggedInUserData.name' (sparse). "
                "The account operator is 'appInfo.owner'. Do not conflate them.",
            ))

    return checks


def _pipeline_has_id_match(pipeline: list) -> bool:
    if not isinstance(pipeline, list):
        return False
    for stage in pipeline:
        if not isinstance(stage, dict):
            continue
        if "$match" in stage:
            m = stage["$match"]
            if isinstance(m, dict) and "_id" in m:
                return True
    return False


def _stringify(obj) -> str:
    import json
    try:
        return json.dumps(obj)
    except Exception:
        return str(obj)


def _pass(code: str, message: str) -> dict: return {"level": "pass",    "code": code, "message": message}
def _warn(code: str, message: str) -> dict: return {"level": "warning", "code": code, "message": message}
def _err (code: str, message: str) -> dict: return {"level": "error",   "code": code, "message": message}
