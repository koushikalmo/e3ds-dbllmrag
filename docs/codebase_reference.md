# Codebase Reference

Verbose descriptions, function contracts, and architectural notes moved out of source files.
Source files contain only short WHY comments where the constraint is non-obvious.

---

## lib/mongodb.py

Async MongoDB connection manager using Motor 3 (asyncio wrapper over PyMongo).
Manages two singleton Motor clients — one per MongoDB cluster.

- `get_stream_db()` — returns the `stream-datastore` Motor database handle; lazy-initialises the client on first call
- `get_appconfigs_db()` — same for `appConfigs` cluster
- `close_connections()` — closes both clients; called during FastAPI shutdown
- `ping_databases()` — sends `ping` command to both clusters; returns `{"stream": "ok", "appconfigs": "ok"}` or error strings

---

## lib/db_registry.py

Config-driven multi-database registry loaded from `data/db_registry.json`.
Allows adding a new database without changing Python code: add URI to `.env`, add entry to JSON, restart.

- `get_db(db_name)` — returns Motor database handle for a registered DB name; creates client on first use
- `get_default_collection(db_name)` — returns the configured default collection for a DB
- `close_all()` — closes all Motor clients
- `ping_all()` — pings every registered database
- `get_all_descriptions()` — returns formatted list of all registered DBs and their descriptions

---

## lib/schema_discovery.py

Live schema extraction: samples real MongoDB documents on startup and every hour (TTL = 3600s).
Extracted field paths are embedded into the vector store so the LLM prompt always reflects current data.

- `refresh_schema_cache(stream_collection, force)` — re-samples both databases; no-op if cache is still fresh
- `build_dynamic_supplement(include_stream, include_appconfigs, max_fields)` — builds a concise text block of live field names + types for the LLM prompt
- `retrieve_schema_context(question, include_stream, include_appconfigs, top_k)` — semantic retrieval: returns the top_k most relevant field descriptions for this question using the vector store
- `get_cache_status()` — returns `{populated, sampled_at, elapsed_sec, stream_docs, owners_sampled}` for the health endpoint
- `_index_fields_async(cache)` — background task that embeds newly discovered fields into the schema vector store

### How schema discovery works
1. On startup, `refresh_schema_cache()` is called as an async background task.
2. It samples 10 real documents from the active stream collection and 5 owner collections from appConfigs.
3. `_extract_paths()` recursively walks each document and builds `{field_path: {type, example}}`.
4. `_merge_path_sets()` merges across documents, keeping the first non-null example per path.
5. The result is stored in `_schema_cache` and a fingerprint is computed to detect schema changes between refreshes.
6. `_index_fields_async()` embeds all new field paths into `data/vectors/schema.json`.

---

## lib/live_data_context.py

Caches real MongoDB values for LLM prompt injection.
Prevents the LLM from guessing field values ("brasil" vs "Brazil") or inventing owner names.

Four data categories cached per collection:
- **Documents** — 3 slim sample documents (TTL: 30 min)
- **Categorical values** — top countries, cities, OS names, browsers, owners, app names (TTL: 60 min)
- **Global lists** — all month collections + all appConfigs owner names (TTL: 60 min)

- `warm_all_caches(collection)` — pre-populates all caches at startup
- `get_live_context(collection, question)` — returns a formatted context block for the LLM prompt; cold-starts if needed, otherwise serves cached + triggers background refresh of stale entries
- `get_cache_status()` — returns per-collection cache state for the health endpoint

### Document slimming (`_slim_doc`)
Raw session documents contain many fields irrelevant to query generation (e.g. `webRtcStatsData`, `timeRecords`, `iceConnectionStateChanges`). `_slim_doc` strips these and retains only navigation-relevant fields: client location, OS/browser, app owner, server info.

---

## lib/session_memory.py

Short-term in-memory conversation history per session UUID.
Enables follow-up questions: "filter those to Europe" resolves correctly when the prior question's result is known.

- `add_turn(session_id, question, answer)` — records one completed exchange
- `get_context_text(session_id)` — returns conversation history as a formatted block for the LLM prompt, or `""` if no history
- `clear_session(session_id)` — removes one session (called on "New Chat")
- `active_session_count()` — returns number of live sessions (after GC)

Sessions are garbage-collected after `SESSION_TTL = 7200` seconds (2 hours) of inactivity.
No persistence — cleared on server restart.

---

## lib/embeddings.py

Ollama embedding client using `nomic-embed-text` (768-dim vectors).
Used for semantic RAG search across both schema fields and past query examples.

- `embed(text)` — returns a 768-dim float vector, or `None` if unavailable (callers fall back to keyword search)
- `embed_batch(texts)` — embeds a list sequentially; individual failures are `None`
- `is_available()` — returns `True` if Ollama embedding model responds

---

## lib/vector_store.py

Local vector store backed by a single JSON file (`data/vectors/{name}.json`).
Two instances: `VectorStore("schema")` for field descriptions, `VectorStore("examples")` for past queries.

- `upsert(id, text, embedding, metadata)` — insert or update an item; saves to disk immediately
- `search(query_embedding, top_k, filter_fn, min_score)` — returns top_k items by cosine similarity descending
- `remove(id)` — delete one item
- `trim_to(max_items)` — drop oldest items (index 0 first) until store is at `max_items`
- `ids()` — returns set of all stored IDs (used to skip already-indexed items)

Cosine similarity is computed in pure Python. ~5ms for 500×768-dim items (acceptable for RAG with small stores).

---

## lib/query_examples.py

RAG few-shot example store — saves successful queries and retrieves similar ones to prepend to the LLM prompt.

**Weight system:**
- `1.0` — auto-saved from a successful query
- `2.0` — user confirmed correct (👍 feedback)
- `2.5` — user provided corrected pipeline (👎 + correction)

Re-ranking: candidates are fetched at `top_n × 4`, then re-sorted by `cosine_score × weight` before top_n are returned. This prevents many auto-saved examples from drowning out user-verified ones.

- `add_example(question, query_obj, result_count, db_hint)` — saves to JSON file + indexes into vector store (weight 1.0)
- `add_verified_example(question, query_obj, result_count, db_hint)` — updates existing example to weight 2.0 + re-indexes
- `add_corrected_example(question, corrected_obj, result_count, db_hint)` — replaces existing example with corrected pipeline, weight 2.5
- `find_similar_examples_vector(question, db_hint, top_n)` — semantic search; returns `None` if vector store is cold
- `find_similar_examples(question, db_hint, top_n)` — keyword-overlap fallback
- `format_examples_for_prompt(examples)` — formats retrieved examples as a "PAST EXAMPLES" block for the LLM
- `index_all_examples_async()` — on startup, embeds all JSON examples into the vector store (preserving weights)
- `get_example_count()` — returns number of stored examples

---

## lib/collection_resolver.py

Parses month/year references from a question and returns the corresponding MongoDB collection name.
Runs before the LLM so the correct collection is baked into the prompt before it is built.

Patterns matched:
- `"April 2025"` → `Apr_2025`
- `"2025 April"` → `Apr_2025`
- `"Apr '25"` → `Apr_2025`
- `"april"` alone → uses year from `default_collection`
- `"last month"` / `"this month"` → computed from current UTC date

- `resolve_collection(question, default_collection)` — returns the detected collection name, or `default_collection` if none found
- `resolve_and_log(question, default_collection)` — same + prints what was detected (useful in server logs)

---

## lib/query_generator.py

Converts natural language questions into validated MongoDB query objects.

**Pipeline (per attempt):**
1. `detect_relevant_databases()` — keyword scan → `(needs_stream, needs_appconfigs)`
2. `retrieve_schema_context()` — vector search for relevant field descriptions
3. `build_system_prompt()` — assembles static schema + live schema supplement
4. `find_similar_examples_vector()` → `find_similar_examples()` fallback — RAG retrieval
5. `get_live_context()` — real field values (countries, owners, cities)
6. `generate_with_fallback()` — Ollama LLM call (JSON mode)
7. `_extract_json()` — parse JSON; strip markdown fences if present
8. `_validate_structure()` — structural validation; raises `ValueError` if unfixable
9. `_fix_query_obj()` — deterministically fix `$limit` placement in pipelines
10. `_validate_field_names()` — check dotted field references against schema vector store
11. If suspicious fields found and attempts remain → `_build_correction_prompt()` → retry

- `generate_query(question, collection, conversation_ctx)` — main entry point; retries up to `MAX_ATTEMPTS` (default 3)
- `save_successful_query(question, query_obj, result_count)` — adds result to RAG store after a successful query

### Field name correction
`_FIELD_ALIASES` maps common LLM mistakes (e.g. `clientinfo.country_code` → `clientInfo.country_name`).
`_find_closest_field()` tries: alias lookup → case-insensitive match → leaf-name match in known fields.

---

## lib/query_executor.py

Executes validated query objects against MongoDB.
Handles four operation types: `aggregate`, `countDocuments`, `find`, `distinct`.

- `execute_query(query_obj)` — dispatches by `queryType` and `operation`; returns standardised result dict
- `_sanitize_pipeline(pipeline)` — strips `$out` and `$merge` (read-only safety)
- `_enforce_limit(pipeline)` — ensures `$limit ≤ 200` always present
- `_normalize_match_query(query)` — strips problematic fields; converts string values in `_PARTIAL_MATCH_FIELDS` to diacritic-aware regex
- `_expand_diacritics(text)` — builds char-class regex so `"Bogota"` matches `"Bogotá"`
- `_make_serializable(docs)` — converts `ObjectId` and `Decimal128` to JSON-safe Python types
- `_raise_friendly(err, database, coll_name)` — translates MongoDB errors into human-readable messages

### Dual query merging
When `queryType = "dual"`, both queries run concurrently via `asyncio.gather()`.
The Python-side in-memory join uses `mergeKey` to match `appInfo.owner` from stream results to the corresponding appConfigs document.
The merged result is stored as `_configData` on each stream document.

---

## lib/llm_provider.py

Ollama local LLM interface. All LLM calls go through this module.

- `OllamaProvider.generate(system_prompt, user_message, json_mode)` — sends a chat completion to Ollama; `json_mode=True` sets `format: "json"` to force valid JSON at token level
- `OllamaProvider.is_available()` — checks if Ollama is running and model is downloaded
- `warmup_model()` — sends a dummy request at startup to pre-load the model into GPU VRAM
- `generate_with_ollama(system_prompt, user_message, json_mode)` — thin wrapper returning `(text, provider_name)`
- `generate_text(system_prompt, user_message)` — free-text (non-JSON) variant used by `result_summarizer.py`

**Temperature:** 0.1 in JSON mode (deterministic), 0.4 for free-text summaries.

---

## lib/result_summarizer.py

Chunked map-reduce summarizer for large result sets.

- Small results (total serialised chars ≤ 16K) → single LLM call
- Large results → split into 25-doc chunks, summarize each chunk, then synthesize all chunk summaries

- `summarize_results(results, question)` — main entry point; returns `{summary, method, chunksUsed, docsAnalyzed}`
- `_summarize_chunk(docs, question, chunk_index, total_chunks)` — LLM call for one chunk
- `_synthesize_summaries(chunk_summaries, question, total_docs)` — final synthesis LLM call

Sensitive fields (`apiKeys`, `streamingApiKeys`, `timeRecords`, etc.) are stripped before sending to the LLM.

---

## lib/schemas.py

Static LLM system prompt: structural rules, field names, operation types, and safety constraints.
Intentionally compact (~750 tokens). Per-query field details come from the vector RAG retrieval.

- `build_system_prompt(include_stream, include_appconfigs, schema_context)` — assembles the full system prompt for one query
- `STREAM_KEYWORDS` / `APPCONFIGS_KEYWORDS` — keyword sets for database routing in `query_generator.py`

---

## lib/response_validator.py

Automated correctness checks on LLM-generated queries and their results.
Results are rendered in the frontend VALIDATION tab.

Seven checks (in order):
1. **RESULT_COUNT** — zero results warning
2. **NEAR_LIMIT** — result set hit the 200-doc cap
3. **EMPLOYEE_FILTER** — `e3ds_employee` filter present/absent in stream queries
4. **LIMIT_BEFORE_GROUP** — `$limit` before `$group` (produces wrong totals)
5. **NO_MATCH_FILTER** — full collection scan (no `$match`)
6. **RTT_CONVERSION** — `avgRoundTripTime` used without `$toDouble` conversion
7. **DUAL_QUERY** — informational note about cross-database merge being Python-side

Each check returns `{level: "pass"|"info"|"warning"|"error", code: str, message: str}`.

---

## lib/feedback_store.py

Stores user feedback (👍/👎) to MongoDB and feeds it back into the RAG vector store.

- `save_feedback(...)` — persists to `_QUERY_FEEDBACK_` collection, fires `_process_feedback_for_rag()` as a background task; returns inserted document ID
- `_process_feedback_for_rag(...)` — embeds into vector store: good → weight 2.0, bad + corrected pipeline → weight 2.5; marks document as `used_in_rag: True`
- `get_feedback_stats()` — returns `{total, good, bad}` counts for the `/api/health` endpoint

**Weight semantics:**
- `2.0` — user confirmed the AI answer was correct; example is promoted in future RAG searches
- `2.5` — user provided a corrected pipeline; this is a gold ground-truth example, overrides auto-saved ones

---

## lib/chat_history.py

Persistent query history stored in MongoDB (`_QUERY_HISTORY_` collection in stream-datastore).

- `save_query(...)` — inserts a history entry; trims oldest entries if count exceeds `MAX_HISTORY = 500`
- `get_history(limit)` — returns entries newest-first; ObjectId serialised to string `id`
- `delete_entry(entry_id)` — deletes one entry by ObjectId string
- `clear_all()` — deletes all history entries

`asyncio.shield()` is used on `insert_one` so the DB write survives if the HTTP client disconnects mid-response.

---

## lib/chat_sharing.py

Creates and retrieves shareable chat snapshots stored in MongoDB.

- `create_share(turns, title)` — saves a snapshot and returns a short share ID
- `get_share(share_id)` — retrieves the snapshot; returns `None` if not found

`asyncio.shield()` is used on `insert_one` for the same reason as in `chat_history.py`.

---

## main.py

FastAPI app entry point. All HTTP routes and startup/shutdown lifecycle.

**Startup tasks (all concurrent):**
1. `warmup_model()` — pre-loads LLM into GPU VRAM
2. `warm_all_caches(default_collection)` — pre-populates live data context cache
3. `refresh_schema_cache(stream_collection)` — samples live MongoDB schema
4. `index_all_examples_async()` — embeds all RAG examples into vector store

**Routes:**
| Method | Path | Handler |
|---|---|---|
| GET | `/` | serves `static/index.html` |
| GET | `/api/health` | DB ping + cache status + feedback stats |
| GET | `/api/status` | Ollama availability + RAG count |
| POST | `/api/query` | NL → pipeline → results + AI summary |
| POST | `/api/analyze` | AI analysis of an existing result set |
| GET | `/api/history` | query history |
| DELETE | `/api/history/{id}` | delete one history entry |
| DELETE | `/api/history` | clear all history |
| POST | `/api/schema/refresh` | force-refresh schema + live caches |
| POST | `/api/feedback` | save 👍/👎 rating + optional correction |
| POST | `/api/share` | create shareable chat snapshot |
| GET | `/api/share/{id}` | retrieve shared chat data |
| GET | `/share/{id}` | serves `index.html` for shared chat page |
