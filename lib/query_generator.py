# lib/query_generator.py
# Converts plain-English questions into validated MongoDB query objects.
#
# Flow per query:
#   1. Keyword scan → which DB(s) does this need?
#   2. Vector RAG   → retrieve similar past examples as few-shot demos
#   3. Schema RAG   → retrieve relevant field descriptions
#   4. Live context → inject real document samples + categorical values
#   5. LLM call     → generate query JSON (Ollama, JSON mode)
#   6. Validation   → structure check + field name check
#   7. Auto-fix     → correct $limit placement regardless of LLM output
#   8. Retry loop   → up to MAX_ATTEMPTS with targeted correction prompts

import os
import json
import time
from datetime import datetime, timezone

from lib.schemas           import build_system_prompt, STREAM_KEYWORDS, APPCONFIGS_KEYWORDS
from lib.llm_provider      import generate_with_fallback
from lib.schema_discovery  import retrieve_schema_context
from lib.live_data_context import get_live_context
from lib.query_examples    import (
    find_similar_examples,
    find_similar_examples_vector,
    format_examples_for_prompt,
)

MAX_ATTEMPTS = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ── Field validation cache ────────────────────────────────────
# Populated from the schema vector store (schema_discovery.py).
# Lets us catch "appinfo.owner" before MongoDB rejects it.
_known_fields_cache: set[str] = set()
_known_fields_ts:    float    = 0.0
_KNOWN_FIELDS_TTL = 3600  # 1 hour — matches schema discovery TTL


def _get_known_fields() -> set[str]:
    """Returns valid field paths from the schema vector store, cached for 1h.

    Falls back to a hardcoded minimal set on cold start.
    """
    global _known_fields_cache, _known_fields_ts

    now = time.monotonic()
    if _known_fields_cache and (now - _known_fields_ts) < _KNOWN_FIELDS_TTL:
        return _known_fields_cache

    try:
        from lib.vector_store import VectorStore
        store = VectorStore("schema")
        if store.count() > 0:
            fields = {
                item["metadata"]["field"]
                for item in store.all_items()
                if "field" in item.get("metadata", {})
            }
            if fields:
                _known_fields_cache = fields
                _known_fields_ts    = now
                return _known_fields_cache
    except Exception:
        pass

    # Hardcoded fallback — used only before schema discovery has run
    return {
        "appInfo.owner", "appInfo.appName",
        "clientInfo.city", "clientInfo.country_name",
        "userDeviceInfo.os.name", "userDeviceInfo.client.name",
        "webRtcStatsData.avgBitrate", "webRtcStatsData.packetsLost",
        "webRtcStatsData.avgRoundTripTime",
        "elInfo.computerName", "elInfo.systemInfo.cpu.brand",
        "startTimeStamp", "DisconnectTime_Timestamp", "loadTime",
        "e3ds_employee", "maxUserLimit",
        "SubscriptionEndDate._seconds", "SubscriptionStartDate._seconds",
    }


# ── Database routing ──────────────────────────────────────────

def detect_relevant_databases(question: str) -> tuple[bool, bool]:
    """Keyword scan to decide which DB schemas to include in the prompt.

    Returns (needs_stream, needs_appconfigs).
    Defaults to stream-only when nothing matches.
    """
    q = question.lower()
    needs_stream     = any(kw in q for kw in STREAM_KEYWORDS)
    needs_appconfigs = any(kw in q for kw in APPCONFIGS_KEYWORDS)
    if not needs_stream and not needs_appconfigs:
        needs_stream = True
    return needs_stream, needs_appconfigs


# ── JSON parsing ──────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """Parse JSON from the LLM response. Handles stray markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        text = text[nl + 1:] if nl != -1 else text[3:]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON. Error: {e}\nRaw (first 600):\n{raw[:600]}")


# ── Pipeline limit fixer ──────────────────────────────────────

def _fix_pipeline_limits(pipeline: list) -> list:
    """Deterministically fix $limit placement regardless of what the LLM generated.

    Three cases:
    1. $count present → remove ALL $limit (count returns one doc, limit is wrong)
    2. $group present → remove $limit BEFORE $group, ensure one $limit AFTER
    3. Raw docs query → move $limit to the end, cap at 200
    """
    if not pipeline:
        return pipeline

    def op(s):    return next(iter(s), None) if isinstance(s, dict) else None
    def lim(s):   return s.get("$limit") if isinstance(s, dict) else None

    ops = [op(s) for s in pipeline]

    # Case 1: count
    if "$count" in ops:
        fixed = [s for s in pipeline if op(s) != "$limit"]
        if len(fixed) != len(pipeline):
            print(f"[pipeline-fix] Removed $limit from $count pipeline")
        return fixed

    # Case 2: group
    if "$group" in ops:
        gi = ops.index("$group")
        pre = [i for i, o in enumerate(ops) if o == "$limit" and i < gi]
        if pre:
            print(f"[pipeline-fix] Removed {len(pre)} $limit(s) before $group")
        fixed = [s for i, s in enumerate(pipeline) if i not in pre]
        # Cap any post-group $limit
        for s in fixed:
            if op(s) == "$limit" and isinstance(lim(s), int) and lim(s) > 200:
                s["$limit"] = 200
        # Add $limit after $group if none exists
        fops = [op(s) for s in fixed]
        if "$limit" not in fops[fops.index("$group"):]:
            fixed.append({"$limit": 50})
            print("[pipeline-fix] Added $limit: 50 after $group")
        return fixed

    # Case 3: raw documents
    lims = [i for i, o in enumerate(ops) if o == "$limit"]
    if not lims:
        return list(pipeline) + [{"$limit": 50}]

    val    = lim(pipeline[lims[-1]])
    capped = min(val, 200) if isinstance(val, int) and val > 0 else 50
    fixed  = [s for s in pipeline if op(s) != "$limit"]
    fixed.append({"$limit": capped})
    if len(lims) > 1 or lims[0] != len(pipeline) - 1:
        print(f"[pipeline-fix] Moved $limit: {capped} to end of pipeline")
    return fixed


def _fix_query_obj(query_obj: dict) -> dict:
    """Apply _fix_pipeline_limits to all aggregate pipelines in a query object.

    Skips countDocuments/find/distinct — they don't have pipelines.
    """
    if query_obj.get("queryType") == "single":
        if query_obj.get("operation", "aggregate") == "aggregate" and "pipeline" in query_obj:
            query_obj["pipeline"] = _fix_pipeline_limits(query_obj["pipeline"])
    elif query_obj.get("queryType") == "dual":
        for q in query_obj.get("queries", []):
            if "pipeline" in q:
                q["pipeline"] = _fix_pipeline_limits(q["pipeline"])
    return query_obj


# ── Structure validation ──────────────────────────────────────

_VALID_OPERATIONS = frozenset({"aggregate", "countDocuments", "find", "distinct"})
_VALID_DATABASES  = frozenset({"stream-datastore", "appConfigs"})


def _validate_structure(obj: dict) -> dict:
    """Validate that the LLM's JSON has the shape execute_query() expects.

    Normalizes missing fields to safe defaults. Raises ValueError with
    a clear message if the structure is unfixable.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")

    qt = obj.get("queryType")

    if qt == "single":
        if obj.get("database") not in _VALID_DATABASES:
            raise ValueError(
                f"Invalid database '{obj.get('database')}'. "
                f"Must be one of: {sorted(_VALID_DATABASES)}"
            )
        if "collection" not in obj:
            raise ValueError(f"Missing 'collection'. Keys: {list(obj.keys())}")

        operation = obj.get("operation", "aggregate")
        if operation not in _VALID_OPERATIONS:
            print(f"[validator] Unknown operation '{operation}' → defaulting to 'aggregate'")
            operation = "aggregate"
        obj["operation"] = operation

        if operation == "aggregate":
            if not isinstance(obj.get("pipeline"), list):
                raise ValueError(f"Missing or invalid 'pipeline' for aggregate. Keys: {list(obj.keys())}")

        elif operation == "distinct":
            if "field" not in obj:
                raise ValueError(f"Missing 'field' for distinct. Keys: {list(obj.keys())}")
            obj.setdefault("query", {})

        else:  # countDocuments or find
            obj.setdefault("query", {})

        return obj

    if qt == "dual":
        queries = obj.get("queries", [])
        if not isinstance(queries, list) or len(queries) != 2:
            raise ValueError(f"Dual query needs exactly 2 items in 'queries', got {len(queries) if isinstance(queries, list) else type(queries)}")
        for i, q in enumerate(queries):
            if q.get("database") not in _VALID_DATABASES:
                raise ValueError(f"Dual query[{i}] has invalid database '{q.get('database')}'")
            if not isinstance(q.get("pipeline"), list):
                raise ValueError(f"Dual query[{i}] missing 'pipeline'")
            if q["database"] == "appConfigs" and not q.get("collection"):
                raise ValueError(
                    f"Dual query[{i}] targets appConfigs but has no 'collection'. "
                    "Set it to the owner's username (e.g. 'eduardo')."
                )
        return obj

    raise ValueError(f"Unknown queryType '{qt}'. Must be 'single' or 'dual'. Keys: {list(obj.keys())}")


# ── Field name validation ─────────────────────────────────────

def _extract_field_references(obj, depth: int = 0) -> set[str]:
    """Recursively extract dotted field references from a pipeline or query dict."""
    refs = set()
    if depth > 10:
        return refs

    if isinstance(obj, dict):
        for k, v in obj.items():
            if "." in k and not k.startswith("$"):
                refs.add(k)
            if isinstance(v, str) and v.startswith("$") and "." in v:
                refs.add(v[1:])
            refs.update(_extract_field_references(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            refs.update(_extract_field_references(item, depth + 1))
    elif isinstance(obj, str) and obj.startswith("$") and "." in obj:
        refs.add(obj[1:])
    return refs


def _validate_field_names(query_obj: dict) -> list[str]:
    """Return a list of field references that don't exist in the known schema."""
    targets = []
    if query_obj.get("queryType") == "single":
        op = query_obj.get("operation", "aggregate")
        targets.append(query_obj.get("pipeline", []) if op == "aggregate" else query_obj.get("query", {}))
    elif query_obj.get("queryType") == "dual":
        targets.extend(q.get("pipeline", []) for q in query_obj.get("queries", []))

    known = _get_known_fields()
    suspicious = []
    for target in targets:
        for ref in _extract_field_references(target):
            if "." in ref and known and ref not in known:
                suspicious.append(ref)
    return suspicious


# ── Field correction hints ────────────────────────────────────

# Common LLM field name mistakes → correct field
_FIELD_ALIASES: dict[str, str] = {
    "clientinfo.country_code":  "clientInfo.country_name",
    "clientinfo.countrycode":   "clientInfo.country_name",
    "clientinfo.country":       "clientInfo.country_name",
    "appinfo.appname":          "appInfo.appName",
    "appinfo.app_name":         "appInfo.appName",
    "userdeviceinfo.os":        "userDeviceInfo.os.name",
    "webrtcstatsdata.bitrate":  "webRtcStatsData.avgBitrate",
    "webrtcstatsdata.rtt":      "webRtcStatsData.avgRoundTripTime",
}


def _find_closest_field(bad: str) -> str | None:
    """Return the closest known field name for a correction hint."""
    lower = bad.lower()
    if lower in _FIELD_ALIASES:
        return _FIELD_ALIASES[lower]
    known = _get_known_fields()
    # Exact case-insensitive match
    for f in known:
        if f.lower() == lower:
            return f
    # Leaf component match (e.g. ".country_name" suffix)
    parts = lower.split(".")
    if len(parts) >= 2:
        leaf = parts[-1]
        candidates = [f for f in known if f.lower().endswith(f".{leaf}")]
        if candidates:
            return candidates[0]
    return None


# ── Correction prompt ─────────────────────────────────────────

def _build_correction_prompt(
    original_question: str,
    collection:        str,
    failed_output:     str,
    error_message:     str,
    suspicious_fields: list[str],
    attempt:           int,
) -> str:
    """Build a retry prompt that shows the LLM what went wrong and why."""
    lines = [
        f'Default stream collection: "{collection}"',
        f"Current Unix timestamp: {int(time.time())}",
        "",
        "CORRECTION REQUEST:",
        f"The user asked: {original_question}",
        "",
        "Your previous response was rejected:",
        "---",
        failed_output[:800],
        "---",
        f"Rejection reason: {error_message}",
    ]
    if suspicious_fields:
        lines += ["", "FIELD NAME ERRORS:"]
        for f in suspicious_fields[:5]:
            lines.append(f"  - '{f}' does not exist.")
            hint = _find_closest_field(f)
            if hint:
                lines.append(f"    Did you mean '{hint}'?")
    lines += [""]
    if attempt >= MAX_ATTEMPTS - 1:
        lines += ["FINAL ATTEMPT. Output ONLY a JSON object. No prose, no markdown."]
    else:
        lines += ["Output a corrected JSON object only."]
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────

async def generate_query(
    question:         str,
    collection:       str = "Apr_2025",
    conversation_ctx: str = "",
) -> dict:
    """Convert a plain-English question into a validated MongoDB query object.

    Returns a dict ready for execute_query(). Raises ValueError after
    MAX_ATTEMPTS failed retries. Raises RuntimeError if Ollama is down.
    """
    needs_stream, needs_appconfigs = detect_relevant_databases(question)
    db_hint = (
        "both"       if needs_stream and needs_appconfigs else
        "appconfigs" if needs_appconfigs else
        "stream"
    )

    # Retrieve top-20 relevant field descriptions via vector search
    schema_ctx = await retrieve_schema_context(
        question           = question,
        include_stream     = needs_stream,
        include_appconfigs = needs_appconfigs,
        top_k              = 20,
    )

    system_prompt = build_system_prompt(
        include_stream     = needs_stream,
        include_appconfigs = needs_appconfigs,
        schema_context     = schema_ctx,
    )

    # Few-shot examples: try vector search, fall back to keyword overlap
    similar = await find_similar_examples_vector(question, db_hint, top_n=2)
    if similar is None:
        similar = find_similar_examples(question, db_hint, top_n=2)
    examples_text = format_examples_for_prompt(similar)

    # Live context: real document samples + categorical values from MongoDB
    live_ctx = await get_live_context(collection, question)
    live_ctx_block = f"{live_ctx}\n\n" if live_ctx else ""

    now_unix = int(time.time())
    now_iso  = datetime.now(timezone.utc).isoformat()

    initial_message = (
        f"{examples_text}"
        f"{conversation_ctx}"
        f'Default stream collection: "{collection}"\n'
        f"Current UTC time: {now_iso}\n"
        f"Current Unix timestamp: {now_unix}\n\n"
        f"{live_ctx_block}"
        f"Question: {question}"
    )

    print(
        f"[generator] '{question[:60]}' | "
        f"stream={needs_stream} appconfigs={needs_appconfigs} | "
        f"examples={len(similar)} live_ctx={'yes' if live_ctx else 'no'}"
    )

    last_error, last_raw, last_suspicious = "", "", []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        user_message = initial_message if attempt == 1 else _build_correction_prompt(
            original_question = question,
            collection        = collection,
            failed_output     = last_raw,
            error_message     = last_error,
            suspicious_fields = last_suspicious,
            attempt           = attempt,
        )

        print(f"[generator] Attempt {attempt}/{MAX_ATTEMPTS}…")

        try:
            raw, provider = await generate_with_fallback(system_prompt, user_message)
            print(f"[generator] Response from {provider} ({len(raw)} chars)")
        except RuntimeError:
            raise  # Ollama is down — don't swallow this

        last_raw = raw

        try:
            query_obj = _extract_json(raw)
        except ValueError as e:
            last_error = str(e)
            print(f"[generator] Attempt {attempt} — JSON parse failed: {e}")
            continue

        try:
            query_obj = _validate_structure(query_obj)
        except ValueError as e:
            last_error = str(e)
            print(f"[generator] Attempt {attempt} — Structure invalid: {e}")
            continue

        # Fix $limit placement regardless of what the LLM generated
        query_obj = _fix_query_obj(query_obj)

        suspicious = _validate_field_names(query_obj)
        if suspicious:
            last_suspicious = suspicious
            last_error = f"Unknown field references: {suspicious[:5]}"
            print(f"[generator] Attempt {attempt} — Suspicious fields: {suspicious}")
            if attempt < MAX_ATTEMPTS:
                continue

        if suspicious:
            print(f"[generator] Warning: unresolved fields after {MAX_ATTEMPTS} attempts: {suspicious}")
        else:
            print(f"[generator] Validated on attempt {attempt}")

        return query_obj

    raise ValueError(
        f"Failed to generate a valid query after {MAX_ATTEMPTS} attempts.\n"
        f"Last error: {last_error}\n"
        "Try rephrasing the question, or check if Ollama is running."
    )


def save_successful_query(question: str, query_obj: dict, result_count: int) -> None:
    """Save a successful query to the RAG example store for future few-shot use."""
    from lib.query_examples import add_example

    needs_stream, needs_appconfigs = detect_relevant_databases(question)
    db_hint = "both" if needs_stream and needs_appconfigs else \
              "appconfigs" if needs_appconfigs else "stream"
    try:
        add_example(question, query_obj, result_count, db_hint)
    except Exception as e:
        print(f"[generator] Failed to save example (non-fatal): {e}")
