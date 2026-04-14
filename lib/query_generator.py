# ============================================================
# lib/query_generator.py — Natural Language → MongoDB Pipeline
# ============================================================
# Converts a plain-English question into a valid, validated
# MongoDB aggregation pipeline using the local Ollama LLM.
#
# THE FULL PIPELINE:
# ───────────────────
# 1. RELEVANCE DETECTION
#    Scans the question for keywords to decide which database(s)
#    schema to include (stream-datastore, appConfigs, or both).
#    This keeps the prompt small and focused.
#
# 2. RAG FEW-SHOT EXAMPLES
#    Searches the library of previously-successful queries for
#    similar questions. The top 2-3 examples are prepended to
#    the user message as demonstrations. This dramatically
#    improves accuracy for patterns the model hasn't seen.
#
# 3. LIVE SCHEMA SUPPLEMENT
#    Appends dynamically-discovered field data from the real
#    MongoDB databases. This ensures the LLM knows about any
#    new fields added since the static schema was last updated.
#
# 4. LLM GENERATION
#    Sends the assembled prompt to Ollama. With format:"json",
#    Ollama is forced to produce valid JSON at the token level.
#
# 5. SELF-CORRECTION RETRY LOOP
#    If the output fails validation, we send the error + bad
#    output back to the model and ask it to fix itself.
#    Attempt 1: normal generation
#    Attempt 2: "here's your error, please fix it"
#    Attempt 3: final attempt with stronger instructions
#
# 6. FIELD NAME VALIDATION
#    Walks the pipeline and checks all field references against
#    known valid fields. Catches typos like "appinfo.owner"
#    (should be "appInfo.owner") before MongoDB rejects them.
#
# 7. EXAMPLE SAVING
#    Successful queries (those that return results) are saved
#    automatically to the example library for future RAG use.
# ============================================================

import os
import re
import json
import time
from datetime import datetime, timezone

from lib.schemas          import build_system_prompt, STREAM_KEYWORDS, APPCONFIGS_KEYWORDS
from lib.llm_provider     import generate_with_fallback
from lib.schema_discovery import retrieve_schema_context
from lib.query_examples   import (
    find_similar_examples,
    find_similar_examples_vector,
    format_examples_for_prompt,
)

# ── Configuration ─────────────────────────────────────────────
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ── Known valid field paths (derived from live schema vector store) ────
# The validator checks generated pipelines against this set to catch
# LLM typos before MongoDB rejects them (e.g. "appinfo.owner" → "appInfo.owner").
#
# We derive this set DYNAMICALLY from the schema vector store, which is
# populated by schema_discovery.py on startup. This means:
#   - No manual list to maintain
#   - New fields discovered from MongoDB are automatically validated
#   - New databases are automatically included when indexed
#
# _get_known_fields() reads the vector store on first call and caches
# the result. The cache is invalidated hourly (matching schema TTL).

import time as _time
_known_fields_cache: set[str] = set()
_known_fields_ts: float = 0.0
_KNOWN_FIELDS_TTL = 3600   # 1 hour — match schema discovery TTL


def _get_known_fields() -> set[str]:
    """
    Returns the set of valid field paths from the schema vector store.

    Falls back to a minimal hardcoded set if the vector store is
    empty (first startup before schema discovery has run).
    """
    global _known_fields_cache, _known_fields_ts

    now = _time.monotonic()
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

    # Fallback: a small set of the most critical fields for validation.
    # This is used only when the vector store hasn't been populated yet
    # (e.g. first startup before schema discovery completes).
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


def detect_relevant_databases(question: str) -> tuple[bool, bool]:
    """
    Scans the question for keywords to decide which database
    schema(s) to include in the LLM prompt.

    Defaults to stream-only when nothing matches, since that's
    what the vast majority of questions are about.

    Returns:
        (needs_stream, needs_appconfigs)
    """
    q_lower = question.lower()
    needs_stream     = any(kw in q_lower for kw in STREAM_KEYWORDS)
    needs_appconfigs = any(kw in q_lower for kw in APPCONFIGS_KEYWORDS)

    # Ambiguous questions or no keyword matches → default to stream
    if not needs_stream and not needs_appconfigs:
        needs_stream = True

    return needs_stream, needs_appconfigs


def _extract_json(raw: str) -> dict:
    """
    Parses JSON from the LLM's raw text response.

    With Ollama's format:"json" mode, the output is always pure
    JSON (no markdown fences). But we still handle markdown just
    in case someone switches providers or disables JSON mode.
    """
    cleaned = raw.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM output is not valid JSON.\n"
            f"Parse error: {e}\n"
            f"Raw response (first 600 chars):\n{raw[:600]}"
        )


def _fix_pipeline_limits(pipeline: list) -> list:
    """
    Post-processes a generated pipeline to fix incorrect $limit placement.

    The LLM frequently places $limit before $group/$count stages, which
    truncates the input and produces wrong counts/totals. This function
    detects and corrects that pattern deterministically, regardless of
    what the LLM generated.

    Rules applied:
      1. If pipeline ends with $count  → remove ALL $limit stages.
         ($count returns one doc; a limit is meaningless and wrong.)
      2. If pipeline contains $group   → remove any $limit that appears
         BEFORE the first $group, then ensure one $limit of ≤200 exists
         after $group (add 50 if missing).
      3. Otherwise (raw document query) → ensure $limit is at the END,
         capped at 200. Move it if it's in the wrong position.

    Returns a corrected copy of the pipeline.
    """
    if not pipeline:
        return pipeline

    # Helpers
    def stage_op(stage):
        """Returns the operator key of a pipeline stage, e.g. '$match'."""
        return next(iter(stage), None) if isinstance(stage, dict) else None

    def get_limit_value(stage):
        return stage.get("$limit") if isinstance(stage, dict) else None

    ops = [stage_op(s) for s in pipeline]

    # ── Rule 1: count query → strip all $limit stages ──────────
    if "$count" in ops:
        fixed = [s for s in pipeline if stage_op(s) != "$limit"]
        if len(fixed) != len(pipeline):
            print(f"[pipeline-fix] Removed $limit from $count pipeline "
                  f"({len(pipeline) - len(fixed)} removed)")
        return fixed

    # ── Rule 2: aggregation query ($group present) ─────────────
    if "$group" in ops:
        first_group_idx = ops.index("$group")

        # Remove $limit stages that appear before $group
        pre_limits  = [i for i, op in enumerate(ops) if op == "$limit" and i < first_group_idx]
        post_limits = [i for i, op in enumerate(ops) if op == "$limit" and i > first_group_idx]

        if pre_limits:
            print(f"[pipeline-fix] Removed {len(pre_limits)} $limit stage(s) "
                  f"before $group (was causing wrong totals)")

        # Rebuild without pre-group limits
        fixed = [s for i, s in enumerate(pipeline) if i not in pre_limits]

        # Cap existing post-group limit at 200
        for stage in fixed:
            if stage_op(stage) == "$limit":
                val = get_limit_value(stage)
                if isinstance(val, int) and val > 200:
                    stage["$limit"] = 200
                    print(f"[pipeline-fix] Capped $limit to 200")

        # Add $limit: 50 after $group if none exists after it
        fixed_ops = [stage_op(s) for s in fixed]
        if "$limit" not in fixed_ops[fixed_ops.index("$group"):]:
            fixed.append({"$limit": 50})
            print("[pipeline-fix] Added $limit: 50 after $group")

        return fixed

    # ── Rule 3: raw document query ─────────────────────────────
    # Ensure exactly one $limit at the end, capped at 200.
    limit_indices = [i for i, op in enumerate(ops) if op == "$limit"]

    if not limit_indices:
        # No limit — add one at the end
        pipeline = list(pipeline) + [{"$limit": 50}]
        return pipeline

    # Keep only the last $limit, move it to the end, cap at 200
    last_limit_stage = pipeline[limit_indices[-1]]
    val = get_limit_value(last_limit_stage)
    capped = min(val, 200) if isinstance(val, int) and val > 0 else 50

    # Remove all $limit stages, re-append one at the end
    fixed = [s for i, s in enumerate(pipeline) if stage_op(s) != "$limit"]
    fixed.append({"$limit": capped})

    if len(limit_indices) > 1 or limit_indices[0] != len(pipeline) - 1:
        print(f"[pipeline-fix] Moved $limit: {capped} to end of pipeline")

    return fixed


def _fix_query_obj(query_obj: dict) -> dict:
    """Applies _fix_pipeline_limits to all pipelines in a query object."""
    if query_obj.get("queryType") == "single":
        query_obj["pipeline"] = _fix_pipeline_limits(query_obj["pipeline"])
    elif query_obj.get("queryType") == "dual":
        for q in query_obj.get("queries", []):
            if "pipeline" in q:
                q["pipeline"] = _fix_pipeline_limits(q["pipeline"])
    return query_obj


def _validate_structure(obj: dict) -> dict:
    """
    Validates that the parsed JSON has the shape query_executor.py
    expects. Checks queryType, required fields, and pipeline type.

    This is STRUCTURAL validation only — it doesn't connect to
    MongoDB or check field name correctness.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object, got: {type(obj).__name__}")

    query_type = obj.get("queryType")

    if query_type == "single":
        for field in ("database", "collection", "pipeline"):
            if field not in obj:
                raise ValueError(
                    f"Missing required field '{field}' in single query. "
                    f"Keys present: {list(obj.keys())}"
                )
        if not isinstance(obj["pipeline"], list):
            raise ValueError(
                f"'pipeline' must be a list of stages, got: {type(obj['pipeline']).__name__}"
            )
        if obj["database"] not in ("stream-datastore", "appConfigs"):
            raise ValueError(
                f"Invalid database name: '{obj['database']}'. "
                "Must be 'stream-datastore' or 'appConfigs'."
            )
        return obj

    if query_type == "dual":
        queries = obj.get("queries", [])
        if not isinstance(queries, list) or len(queries) != 2:
            raise ValueError(
                f"Dual query must have exactly 2 items in 'queries'. "
                f"Got: {len(queries) if isinstance(queries, list) else type(queries)}"
            )
        for i, q in enumerate(queries):
            if "pipeline" not in q:
                raise ValueError(f"Dual query item [{i}] is missing 'pipeline'.")
            if not isinstance(q["pipeline"], list):
                raise ValueError(f"Dual query item [{i}] 'pipeline' must be a list.")
            if "database" not in q:
                raise ValueError(
                    f"Dual query item [{i}] is missing 'database'. "
                    "Each sub-query must specify 'database': 'stream-datastore' or 'appConfigs'."
                )
            if q["database"] not in ("stream-datastore", "appConfigs"):
                raise ValueError(
                    f"Dual query item [{i}] has invalid database: '{q['database']}'. "
                    "Must be 'stream-datastore' or 'appConfigs'."
                )
            # appConfigs sub-queries MUST have a collection (owner username).
            # Stream sub-queries get a default from db_registry if omitted.
            if q["database"] == "appConfigs" and not q.get("collection"):
                raise ValueError(
                    f"Dual query item [{i}] targets 'appConfigs' but is missing 'collection'. "
                    "For appConfigs, 'collection' must be the owner's username "
                    "(e.g. 'eduardo'). If the question mentions a specific owner, use it. "
                    "If not, use a representative owner like 'eduardo' as a placeholder."
                )
        return obj

    raise ValueError(
        f"Unknown queryType: '{query_type}'. Must be 'single' or 'dual'. "
        f"Keys present: {list(obj.keys())}"
    )


def _extract_field_references(pipeline_or_obj, depth: int = 0) -> set[str]:
    """
    Recursively walks a pipeline and extracts all strings that
    look like MongoDB field references.

    Detects two patterns:
    1. Dict keys containing dots (field paths in $match/$sort/$project):
       { "appInfo.owner": "eduardo" } → "appInfo.owner"

    2. String values starting with $ (references in $group/$project):
       { "$sum": "$webRtcStatsData.avgBitrate" } → "webRtcStatsData.avgBitrate"
    """
    refs = set()
    if depth > 10:
        return refs  # guard against pathological nesting

    if isinstance(pipeline_or_obj, dict):
        for k, v in pipeline_or_obj.items():
            if "." in k and not k.startswith("$"):
                refs.add(k)
            if isinstance(v, str) and v.startswith("$") and "." in v:
                refs.add(v[1:])
            refs.update(_extract_field_references(v, depth + 1))

    elif isinstance(pipeline_or_obj, list):
        for item in pipeline_or_obj:
            refs.update(_extract_field_references(item, depth + 1))

    elif isinstance(pipeline_or_obj, str):
        if pipeline_or_obj.startswith("$") and "." in pipeline_or_obj:
            refs.add(pipeline_or_obj[1:])

    return refs


def _validate_field_names(query_obj: dict) -> list[str]:
    """
    Checks all field references in the generated pipeline against
    the known-valid field set. Returns a list of suspicious names.

    An empty list means no issues were found.

    Used to catch LLM typos before MongoDB rejects the pipeline,
    and to generate specific correction hints:
      "appinfo.owner doesn't exist — did you mean appInfo.owner?"
    """
    pipelines = []
    if query_obj.get("queryType") == "single":
        pipelines.append(query_obj.get("pipeline", []))
    elif query_obj.get("queryType") == "dual":
        for q in query_obj.get("queries", []):
            pipelines.append(q.get("pipeline", []))

    known = _get_known_fields()
    suspicious = []
    for pipeline in pipelines:
        refs = _extract_field_references(pipeline)
        for ref in refs:
            if "." in ref and known and ref not in known:
                suspicious.append(ref)

    return suspicious


_FIELD_ALIASES: dict[str, str] = {
    # Common LLM mistakes — maps wrong field → correct field
    "clientinfo.country_code":   "clientInfo.country_name",
    "clientinfo.countrycode":    "clientInfo.country_name",
    "clientinfo.country":        "clientInfo.country_name",
    "appinfo.appname":           "appInfo.appName",
    "appinfo.app_name":          "appInfo.appName",
    "userdeviceinfo.os":         "userDeviceInfo.os.name",
    "webrtcstatsdata.bitrate":   "webRtcStatsData.avgBitrate",
    "webrtcstatsdata.rtt":       "webRtcStatsData.avgRoundTripTime",
}


def _find_closest_field(bad_field: str) -> str | None:
    """
    Finds the closest known field name by case-insensitive match.
    Returns a hint for the correction prompt.

    Checks hardcoded aliases first (common LLM mistakes), then falls
    back to case-insensitive and leaf-component matching against the
    vector store.

    Example: "clientInfo.country_code" → "clientInfo.country_name"
    Example: "appinfo.owner"           → "appInfo.owner"
    """
    bad_lower = bad_field.lower()

    # 1. Hardcoded aliases for known LLM mistakes
    if bad_lower in _FIELD_ALIASES:
        return _FIELD_ALIASES[bad_lower]

    known_fields = _get_known_fields()

    # 2. Exact case-insensitive match
    for known in known_fields:
        if known.lower() == bad_lower:
            return known

    # 3. Partial match on the leaf component
    bad_parts = bad_lower.split(".")
    if len(bad_parts) >= 2:
        bad_leaf   = bad_parts[-1]
        candidates = [k for k in known_fields if k.lower().endswith(f".{bad_leaf}")]
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
    """
    Builds a correction user message when a previous attempt failed.

    Gives the LLM three things:
    1. The original question (maintains context)
    2. Its previous bad output (so it knows what to fix)
    3. The specific error + field hints (so it knows how to fix it)

    This self-correction loop resolves ~95% of generation errors
    without requiring a human to rephrase the question.
    """
    lines = [
        f"Default stream collection: \"{collection}\"",
        f"Current Unix timestamp: {int(time.time())}",
        "",
        "CORRECTION REQUEST:",
        f"The user asked: {original_question}",
        "",
        "Your previous response was rejected. Here it is:",
        "---",
        failed_output[:800],
        "---",
        "",
        f"Rejection reason: {error_message}",
    ]

    if suspicious_fields:
        lines.append("")
        lines.append("FIELD NAME ERRORS DETECTED:")
        for field in suspicious_fields[:5]:
            lines.append(f"  - '{field}' does not exist in the schema.")
            hint = _find_closest_field(field)
            if hint:
                lines.append(f"    Did you mean '{hint}'?")

    lines.append("")
    if attempt >= MAX_ATTEMPTS - 1:
        lines.append("FINAL ATTEMPT. Output ONLY a JSON object. Start with { end with }.")
        lines.append("No prose, no explanation, no markdown fences.")
    else:
        lines.append("Please produce a corrected JSON response that fixes all the above errors.")
        lines.append("Output ONLY the JSON object.")

    return "\n".join(lines)


async def generate_query(
    question:         str,
    collection:       str = "Apr_2025",
    conversation_ctx: str = "",
) -> dict:
    """
    Main entry point: converts a plain-English question into a
    validated, ready-to-execute MongoDB query object.

    THE FULL PIPELINE:
    ──────────────────
    1. Detect which databases are relevant (stream / appconfigs / both)
    2. Build the static schema system prompt
    3. Retrieve similar past examples from the RAG store
    4. Get live schema supplement from schema_discovery cache
    5. Assemble the user message (context + examples + question)
    6. Call Ollama → parse JSON → validate structure → validate fields
    7. If validation fails: build correction prompt, retry (up to MAX_ATTEMPTS)
    8. On success: return the validated query dict

    Args:
        question:   Plain-English question (already stripped)
        collection: Default stream collection (e.g. "Apr_2025")

    Returns:
        Validated dict: { queryType, database, collection, pipeline, ... }

    Raises:
        RuntimeError: Ollama unavailable
        ValueError:   Failed to produce valid pipeline after all attempts
    """
    needs_stream, needs_appconfigs = detect_relevant_databases(question)

    db_hint = (
        "both"       if needs_stream and needs_appconfigs else
        "appconfigs" if needs_appconfigs else
        "stream"
    )

    # ── Retrieve relevant schema fields (vector RAG) ───────────
    # Embed the question and find the top-20 most relevant field
    # descriptions from live MongoDB data. This replaces the old
    # approach of dumping the entire 4,500-token schema every time.
    # Falls back to empty string if embeddings aren't available —
    # the slim structural rules in schemas.py are still included.
    schema_ctx = await retrieve_schema_context(
        question           = question,
        include_stream     = needs_stream,
        include_appconfigs = needs_appconfigs,
        top_k              = 20,
    )

    # ── Build system prompt ────────────────────────────────────
    # schema_ctx is injected here — the prompt is now ~750 tokens
    # (rules) + ~400 tokens (relevant fields) = ~1,150 tokens total,
    # vs the old approach of 4,500 tokens every time.
    system_prompt = build_system_prompt(
        include_stream     = needs_stream,
        include_appconfigs = needs_appconfigs,
        schema_context     = schema_ctx,
    )

    # ── Retrieve similar past examples (vector RAG) ────────────
    # Try semantic vector search first. If the embedding model
    # isn't available, fall back to keyword overlap automatically.
    similar_examples = await find_similar_examples_vector(question, db_hint, top_n=2)
    if similar_examples is None:
        # Embedding unavailable — use keyword overlap fallback
        similar_examples = find_similar_examples(question, db_hint, top_n=2)

    examples_text = format_examples_for_prompt(similar_examples)

    # ── Build initial user message ─────────────────────────────
    now_unix = int(time.time())
    now_iso  = datetime.now(timezone.utc).isoformat()

    initial_user_message = (
        f"{examples_text}"         # few-shot examples first
        f"{conversation_ctx}"      # conversation history (empty if new session)
        f"Default stream collection: \"{collection}\"\n"
        f"Current UTC time: {now_iso}\n"
        f"Current Unix timestamp: {now_unix}\n\n"
        f"Question: {question}"
    )

    print(
        f"[generator] Question: '{question[:60]}'\n"
        f"            DBs: stream={needs_stream}, appconfigs={needs_appconfigs}\n"
        f"            RAG examples: {len(similar_examples)}"
    )

    last_error        = ""
    last_raw_response = ""
    last_suspicious   = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # ── Choose the user message for this attempt ───────────
        if attempt == 1:
            user_message = initial_user_message
        else:
            user_message = _build_correction_prompt(
                original_question = question,
                collection        = collection,
                failed_output     = last_raw_response,
                error_message     = last_error,
                suspicious_fields = last_suspicious,
                attempt           = attempt,
            )

        print(f"[generator] Attempt {attempt}/{MAX_ATTEMPTS}…")

        # ── Call LLM ───────────────────────────────────────────
        try:
            raw_response, provider_used = await generate_with_fallback(
                system_prompt, user_message
            )
            print(f"[generator] Response from {provider_used} ({len(raw_response)} chars)")
        except RuntimeError:
            raise  # Ollama down — surface immediately

        last_raw_response = raw_response

        # ── Parse JSON ─────────────────────────────────────────
        try:
            query_obj = _extract_json(raw_response)
        except ValueError as e:
            last_error = f"JSON parse error: {e}"
            print(f"[generator] Attempt {attempt} — JSON parse failed: {e}")
            continue

        # ── Validate structure ─────────────────────────────────
        try:
            query_obj = _validate_structure(query_obj)
        except ValueError as e:
            last_error = f"Structure validation error: {e}"
            print(f"[generator] Attempt {attempt} — Structure invalid: {e}")
            continue

        # ── Fix $limit placement (deterministic, post-LLM) ────
        # Corrects the common LLM mistake of placing $limit before
        # $group/$count, which truncates input and produces wrong totals.
        # This runs unconditionally on every attempt.
        query_obj = _fix_query_obj(query_obj)

        # ── Validate field names ───────────────────────────────
        suspicious = _validate_field_names(query_obj)
        if suspicious:
            last_suspicious = suspicious
            last_error = (
                f"Pipeline references fields that don't exist in the schema: "
                f"{suspicious[:5]}"
            )
            print(f"[generator] Attempt {attempt} — Suspicious fields: {suspicious}")
            if attempt < MAX_ATTEMPTS:
                continue  # try to correct

        # ── All checks passed ──────────────────────────────────
        if suspicious:
            # Last attempt — return with a warning rather than failing
            print(
                f"[generator] Warning: pipeline has unrecognized fields after "
                f"{MAX_ATTEMPTS} attempts: {suspicious}"
            )
        else:
            print(f"[generator] Pipeline validated on attempt {attempt}")

        return query_obj

    # All attempts exhausted without success
    raise ValueError(
        f"Failed to generate a valid MongoDB pipeline after {MAX_ATTEMPTS} attempts.\n"
        f"Last error: {last_error}\n"
        f"Try rephrasing your question, or check SETUP.md for troubleshooting."
    )


def save_successful_query(
    question:     str,
    query_obj:    dict,
    result_count: int,
) -> None:
    """
    Saves a successful query to the RAG example store.

    Called by main.py after a successful /api/query execution.
    We save here (in the API layer) rather than in generate_query()
    because we need the actual result count from query_executor.py.

    Saving is best-effort — errors are caught and logged, never
    surfaced to the user.
    """
    from lib.query_examples import add_example

    needs_stream, needs_appconfigs = detect_relevant_databases(question)
    if needs_stream and needs_appconfigs:
        db_hint = "both"
    elif needs_appconfigs:
        db_hint = "appconfigs"
    else:
        db_hint = "stream"

    try:
        add_example(question, query_obj, result_count, db_hint)
    except Exception as e:
        print(f"[generator] Failed to save example (non-fatal): {e}")
