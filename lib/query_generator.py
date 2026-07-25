from __future__ import annotations
import os
import json
import time
from datetime import datetime, timezone

from lib.schemas           import build_system_prompt, STREAM_KEYWORDS, APPCONFIGS_KEYWORDS
from lib.llm_provider      import generate_with_fallback
from lib.schema_discovery  import retrieve_schema_context
from lib.live_data_context import get_live_context
from lib.data_digest       import get_digest_text
from lib.query_examples    import (
    find_similar_examples,
    find_similar_examples_vector,
    format_examples_for_prompt,
)
from lib.relationships     import (
    classify_with_relationships,
    render_relationship_block,
    ClassificationResult,
)

MAX_ATTEMPTS = int(os.getenv("LLM_MAX_RETRIES", "10"))

_known_fields_cache: set[str] = set()
_known_fields_ts:    float    = 0.0
_KNOWN_FIELDS_TTL = 3600


def _get_known_fields() -> set[str]:
    global _known_fields_cache, _known_fields_ts

    now = time.monotonic()
    if _known_fields_cache and (now - _known_fields_ts) < _KNOWN_FIELDS_TTL:
        return _known_fields_cache

    try:
        from lib.vector_store import VectorStore
        store = VectorStore("schema")
        if store.count() > 0:
            fields = {item["metadata"]["field"] for item in store.all_items() if "field" in item.get("metadata", {})}
            if fields:
                _known_fields_cache = fields
                _known_fields_ts    = now
                return _known_fields_cache
    except Exception:
        pass

    # fallback until schema discovery has run at least once
    return {
        "appInfo.owner", "appInfo.appName",
        "loggedInUserData.name",
        "clientInfo.city", "clientInfo.country_name",
        "userDeviceInfo.os.name", "userDeviceInfo.client.name",
        "webRtcStatsData.avgBitrate", "webRtcStatsData.packetsLost",
        "webRtcStatsData.avgRoundTripTime",
        "elInfo.computerName", "elInfo.systemInfo.cpu.brand",
        "VideoStreamStartedAt_Timestamp",
        "VideoStreamContinuedAt_Timestamp",
        "DataChannelHeartBeatReceivedAt_Timestamp",
        "DisconnectTime_Timestamp", "loadTime",
        "e3ds_employee", "maxUserLimit",
        "SubscriptionEndDate._seconds", "SubscriptionStartDate._seconds",
    }


def detect_relevant_databases(question: str) -> tuple[bool, bool]:
    q = question.lower()
    needs_stream     = any(kw in q for kw in STREAM_KEYWORDS)
    needs_appconfigs = any(kw in q for kw in APPCONFIGS_KEYWORDS)
    if not needs_stream and not needs_appconfigs:
        needs_stream = True  # default to stream
    return needs_stream, needs_appconfigs


def _render_classification_hints(c: ClassificationResult) -> str:
    if not (c.owner_hint or c.merge_key or c.required_filters):
        return ""
    lines = ["CLASSIFICATION HINTS (from relationship graph — use these):"]
    lines.append(f"  • Query type:  {c.query_type}  (db_hint={c.db_hint})")
    if c.owner_hint:
        lines.append(f"  • Owner in question: \"{c.owner_hint}\"  (use as appConfigs collection name or in $match)")
    if c.merge_key:
        lines.append(f"  • mergeKey:    \"{c.merge_key}\"  (MUST be set on the dual query)")
    for f in c.required_filters:
        lines.append(f"  • REQUIRED filter on {f['db']}: {f['field']} = \"{f['value']}\"  — {f['reason']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        nl   = text.find("\n")
        text = text[nl + 1:] if nl != -1 else text[3:]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON. Error: {e}\nRaw (first 600):\n{raw[:600]}")


def _fix_pipeline_limits(pipeline: list) -> list:
    if not pipeline:
        return pipeline

    def op(s):  return next(iter(s), None) if isinstance(s, dict) else None
    def lim(s): return s.get("$limit") if isinstance(s, dict) else None

    ops = [op(s) for s in pipeline]

    if "$count" in ops:
        # $count returns exactly one document — $limit is meaningless
        return [s for s in pipeline if op(s) != "$limit"]

    if "$group" in ops:
        gi  = ops.index("$group")
        pre = [i for i, o in enumerate(ops) if o == "$limit" and i < gi]
        # Remove any $limit placed before $group (wrong — truncates input)
        return [s for i, s in enumerate(pipeline) if i not in pre]

    return list(pipeline)


def _fix_query_obj(query_obj: dict) -> dict:
    if query_obj.get("queryType") == "single":
        if query_obj.get("operation", "aggregate") == "aggregate" and "pipeline" in query_obj:
            query_obj["pipeline"] = _fix_pipeline_limits(query_obj["pipeline"])
    elif query_obj.get("queryType") == "dual":
        for q in query_obj.get("queries", []):
            if "pipeline" in q:
                q["pipeline"] = _fix_pipeline_limits(q["pipeline"])
    return query_obj


_VALID_OPERATIONS = frozenset({"aggregate", "countDocuments", "find", "distinct"})
_VALID_DATABASES  = frozenset({"stream-datastore", "appConfigs"})


def _validate_structure(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")

    qt = obj.get("queryType")

    if qt == "single":
        if obj.get("database") not in _VALID_DATABASES:
            raise ValueError(f"Invalid database '{obj.get('database')}'. Must be one of: {sorted(_VALID_DATABASES)}")
        if "collection" not in obj:
            raise ValueError(f"Missing 'collection'. Keys: {list(obj.keys())}")
        operation = obj.get("operation", "aggregate")
        if operation not in _VALID_OPERATIONS:
            print(f"[validator] Unknown operation '{operation}' → defaulting to 'aggregate'")
            operation = "aggregate"
        obj["operation"] = operation
        if operation == "aggregate":
            if not isinstance(obj.get("pipeline"), list):
                raise ValueError(f"Missing or invalid 'pipeline'. Keys: {list(obj.keys())}")
        elif operation == "distinct":
            if "field" not in obj:
                raise ValueError(f"Missing 'field' for distinct. Keys: {list(obj.keys())}")
            obj.setdefault("query", {})
        else:
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
                raise ValueError(f"Dual query[{i}] targets appConfigs but has no 'collection'.")
        return obj

    raise ValueError(f"Unknown queryType '{qt}'. Must be 'single' or 'dual'. Keys: {list(obj.keys())}")


def _extract_field_references(obj, depth: int = 0) -> set[str]:
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
    targets = []
    if query_obj.get("queryType") == "single":
        op = query_obj.get("operation", "aggregate")
        targets.append(query_obj.get("pipeline", []) if op == "aggregate" else query_obj.get("query", {}))
    elif query_obj.get("queryType") == "dual":
        targets.extend(q.get("pipeline", []) for q in query_obj.get("queries", []))

    known      = _get_known_fields()
    suspicious = []
    for target in targets:
        for ref in _extract_field_references(target):
            if "." in ref and known and ref not in known:
                suspicious.append(ref)
    return suspicious


# field names the model keeps getting wrong, mapped to the real ones
_FIELD_ALIASES: dict[str, str] = {
    "clientinfo.country_code":          "clientInfo.country_name",
    "clientinfo.countrycode":           "clientInfo.country_name",
    "clientinfo.country":               "clientInfo.country_name",
    "appinfo.appname":                  "appInfo.appName",
    "appinfo.app_name":                 "appInfo.appName",
    "userdeviceinfo.os":                "userDeviceInfo.os.name",
    "webrtcstatsdata.bitrate":          "webRtcStatsData.avgBitrate",
    "webrtcstatsdata.rtt":              "webRtcStatsData.avgRoundTripTime",
    "starttimestamp":                   "VideoStreamStartedAt_Timestamp",
    "videostreamstartedat_timestamp":   "VideoStreamStartedAt_Timestamp",
    "sessionstart":                     "VideoStreamStartedAt_Timestamp",
    "session_start":                    "VideoStreamStartedAt_Timestamp",
}


def _find_closest_field(bad: str) -> str | None:
    lower = bad.lower()
    if lower in _FIELD_ALIASES:
        return _FIELD_ALIASES[lower]
    known = _get_known_fields()
    for f in known:
        if f.lower() == lower:
            return f
    parts = lower.split(".")
    if len(parts) >= 2:
        leaf       = parts[-1]
        candidates = [f for f in known if f.lower().endswith(f".{leaf}")]
        if candidates:
            return candidates[0]
    return None


def _build_correction_prompt(
    original_question: str,
    collection:        str,
    failed_output:     str,
    error_message:     str,
    suspicious_fields: list[str],
    attempt:           int,
) -> str:
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


async def generate_query(
    question:         str,
    collection:       str = "Apr_2025",
    conversation_ctx: str = "",
) -> dict:
    classification = await classify_with_relationships(question)
    needs_stream     = classification.db_hint in ("stream", "both")
    needs_appconfigs = classification.db_hint in ("appconfigs", "both")
    db_hint          = classification.db_hint

    schema_ctx    = await retrieve_schema_context(question, needs_stream, needs_appconfigs, top_k=20)
    system_prompt = build_system_prompt(
        include_stream     = needs_stream,
        include_appconfigs = needs_appconfigs,
        schema_context     = schema_ctx,
        relationship_block = render_relationship_block(),
    )

    similar = await find_similar_examples_vector(question, db_hint, top_n=2)
    if similar is None:
        similar = find_similar_examples(question, db_hint, top_n=2)
    # remember which examples we showed the model — main.py bumps their
    # success/failure counters once we know how the query went
    retrieved_ids = [s.get("id") for s in (similar or []) if s.get("id")]
    examples_text = format_examples_for_prompt(similar)

    live_ctx = await get_live_context(collection, question)
    live_ctx_block = f"{live_ctx}\n\n" if live_ctx else ""

    digest_text = get_digest_text()
    digest_block = f"{digest_text}\n\n" if digest_text else ""

    classification_block = _render_classification_hints(classification)

    now_unix = int(time.time())
    now_iso  = datetime.now(timezone.utc).isoformat()

    initial_message = (
        f"{examples_text}"
        f"{conversation_ctx}"
        f'Default stream collection: "{collection}"\n'
        f"Current UTC time: {now_iso}\n"
        f"Current Unix timestamp: {now_unix}\n\n"
        f"{digest_block}"
        f"{live_ctx_block}"
        f"{classification_block}"
        f"Question: {question}"
    )

    print(
        f"[generator] '{question[:60]}' | "
        f"type={classification.query_type} db_hint={db_hint} "
        f"owner={classification.owner_hint or '—'} "
        f"merge_key={classification.merge_key or '—'} | "
        f"examples={len(similar)} live_ctx={'yes' if live_ctx else 'no'} digest={'yes' if digest_text else 'no'}"
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
            raise

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

        query_obj = _fix_query_obj(query_obj)

        suspicious = _validate_field_names(query_obj)
        if suspicious:
            last_suspicious = suspicious
            last_error      = f"Unknown field references: {suspicious[:5]}"
            print(f"[generator] Attempt {attempt} — Suspicious fields: {suspicious}")
            if attempt < MAX_ATTEMPTS:
                continue

        if suspicious:
            print(f"[generator] Warning: unresolved fields after {MAX_ATTEMPTS} attempts: {suspicious}")
        else:
            print(f"[generator] Validated on attempt {attempt}")

        # main.py pops this off before the response goes out
        query_obj["_retrieved_example_ids"] = retrieved_ids
        return query_obj

    raise ValueError(
        f"Failed to generate a valid query after {MAX_ATTEMPTS} attempts.\n"
        f"Last error: {last_error}\n"
        "Try rephrasing the question, or check if Ollama is running."
    )


def save_successful_query(
    question:       str,
    query_obj:      dict,
    result_count:   int,
    accuracy_score: int | None = None,
) -> None:
    from lib.query_examples import add_example
    needs_stream, needs_appconfigs = detect_relevant_databases(question)
    db_hint = "both" if needs_stream and needs_appconfigs else "appconfigs" if needs_appconfigs else "stream"
    try:
        add_example(question, query_obj, result_count, db_hint, accuracy_score=accuracy_score)
    except Exception as e:
        print(f"[generator] Failed to save example (non-fatal): {e}")
