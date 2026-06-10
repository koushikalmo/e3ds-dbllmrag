# Working Principle

How a plain-English question becomes a verified MongoDB result, and how the
system trains itself to get better at it. Read `README.md` first if you just
want to run the thing.

## TLDR

You type a question in the browser. A local Ollama model turns it into a
MongoDB aggregation pipeline. The backend sanitizes that pipeline (read-only,
capped, time-limited), runs it against one or both databases, checks the
result with deterministic validators, scores it 0-100, and sends everything
back with a plain-English summary. Every query that works gets saved as a
RAG example, so the next similar question comes with a working demonstration
already in the prompt. Thumbs up/down and a password-gated trainer mode tune
the weights of those examples over time. No cloud APIs anywhere.

## The two databases

**stream-datastore** — one collection per month (`Apr_2026`, `Mar_2026`, ...
back to `Dec_2023`). One document = one streaming session heartbeat. The
timestamps that matter (`VideoStreamStartedAt_Timestamp`,
`VideoStreamContinuedAt_Timestamp`, `DataChannelHeartBeatReceivedAt_Timestamp`,
`DisconnectTime_Timestamp`) are Unix **milliseconds**; `startTimeStamp` (no
suffix) is **seconds** — a different field, do not mix them. Session duration
= video-continued (or data heartbeat) minus video-started.

**appConfigs** — there is no `users` collection. Each of the ~5,500
collections IS one owner account, named after the owner's username
(`eduardo`, `imerza`, ...). Inside, the `_id` field discriminates document
types: `usersinfo` (billing/subscription), `InfoToConstructUrls` (URL
config), `default` or `<appName>` (app config). The cross-DB join is
`stream.appInfo.owner` ↔ appConfigs collection name. Most historical bugs
came from the LLM treating appConfigs like a normal collection, which is why
the relationship graph (`lib/relationships.py`) exists.

## Execution order 1 — server startup

`main.py` → `lifespan()` kicks these off concurrently; the API accepts
requests immediately while they warm in the background:

1. `migrate_phase4_metadata()` (`lib/query_examples.py`) — stamps timestamps
   onto old RAG entries so age-decay and pruning don't eat them.
2. `warmup_model()` (`lib/llm_provider.py`) — trivial prompt to Ollama so the
   model is loaded in VRAM before the first real request (keep_alive 30m).
3. `warm_all_caches()` (`lib/live_data_context.py`) — samples real documents
   and top values (countries, owners, apps...) from MongoDB.
4. `refresh_schema_cache()` (`lib/schema_discovery.py`) — samples live docs,
   extracts field paths + types, embeds them into the schema vector store.
   Re-runs hourly.
5. `index_all_examples_async()` (`lib/query_examples.py`) — embeds any RAG
   example that doesn't have a vector yet.
6. `start_digest_scheduler()` (`lib/data_digest.py`) — loads/refreshes the
   on-disk field digest, re-samples every 3 days.
7. `_prune_scheduler()` (`lib/query_examples.py`) — prunes bad RAG examples
   every 6 hours (first run delayed so it doesn't fight cache warming).

## Execution order 2 — one query, end to end

`POST /api/query` lands in `run_query()` in `main.py`:

1. **Session memory** — `get_context_text()` (`lib/session_memory.py`) pulls
   the last few turns so "now filter those to India" resolves. In-memory,
   2h TTL, 10 turns max.
2. **Collection resolution** — `resolve_and_log()` →
   `resolve_collection()` (`lib/collection_resolver.py`) parses dates out of
   the question ("last month", "Q1 2026", "16th April 2026", ISO dates...)
   and picks the month collection before the LLM ever sees the question.
3. **Query generation** — `generate_query()` (`lib/query_generator.py`):
   - `classify_with_relationships()` (`lib/relationships.py`) routes the
     question: stream, appconfigs, or both (dual). Also extracts an owner
     name if one is mentioned and picks the right `_id` discriminator.
   - `retrieve_schema_context()` (`lib/schema_discovery.py`) does a vector
     search (`VectorStore.search()` in `lib/vector_store.py`, embeddings via
     `embed()` in `lib/embeddings.py`) for the ~20 most relevant fields.
   - `find_similar_examples_vector()` (`lib/query_examples.py`) retrieves the
     top 2 past examples ranked by cosine similarity × effective weight;
     `format_examples_for_prompt()` turns them into few-shot demos. Falls
     back to keyword matching (`find_similar_examples()`) if embeddings are
     cold.
   - `get_live_context()` (`lib/live_data_context.py`) and
     `get_digest_text()` (`lib/data_digest.py`) inject real sample documents
     and real field values so the model copies exact spellings.
   - `build_system_prompt()` (`lib/schemas.py`) assembles the rules +
     `render_relationship_block()` + schema context into the system prompt.
   - `generate_with_fallback()` → `OllamaProvider.generate()`
     (`lib/llm_provider.py`) calls Ollama with `format:"json"`, which forces
     valid JSON at the token level.
   - Parse and check: `_extract_json()` → `_validate_structure()` →
     `_fix_query_obj()` (removes `$limit` before `$group`, etc.) →
     `_validate_field_names()` against the known-fields set. On failure,
     `_build_correction_prompt()` shows the model its own mistake and the
     loop retries up to `LLM_MAX_RETRIES` times.
4. **Year expansion** — if `resolve_year()` says the question targets a whole
   year, `get_existing_year_collections()` + `build_year_pipeline()`
   (`lib/query_executor.py`) rewrite the pipeline with `$unionWith` across
   every existing month of that year.
5. **Execution** — `execute_query()` (`lib/query_executor.py`):
   - `_sanitize_pipeline()` strips `$out`/`$merge` (read-only guarantee) and
     splits stages the model accidentally merged into one object.
   - `_normalize_pipeline()` rewrites string equality on city/country/etc.
     into accent-insensitive regex ("Bogota" matches "Bogotá").
   - `_enforce_limit()` caps results: raw pipelines at `HARD_LIMIT_RAW`
     (200), pipelines with a reduce stage (`$group`/`$bucket`/`$facet`...) at
     `HARD_LIMIT_AGG` (1000).
   - Runs with `maxTimeMS=15000` and `allowDiskUse=True` via one of
     `_run_aggregate()`, `_run_find()`, `_run_count_documents()`,
     `_run_distinct()`. Dual queries fan out with `asyncio.gather()` and get
     merged in Python by the merge key. `_make_serializable()` converts BSON
     types for JSON.
6. **Summary** — `summarize_results()` (`lib/result_summarizer.py`). Small
   result sets (≤16k chars) go in one LLM call; bigger ones get chunked,
   summarized per chunk (`_summarize_chunk()`), then synthesized
   (`_synthesize_summaries()`).
7. **Validation** — `validate_query_and_result()`
   (`lib/response_validator.py`) plus `check_relationships()`
   (`lib/relationships.py`). All deterministic, no LLM. Emits coded findings
   like `REL_APPCONFIGS_NO_DISCRIMINATOR`, `LIMIT_BEFORE_GROUP`,
   `NO_EMPLOYEE_FILTER`, `ZERO_RESULTS`. Errors show up in the UI's
   VALIDATION tab.
8. **Scoring** — `score_query()` (`lib/accuracy_scorer.py`) combines five
   deterministic signals (syntax, schema, relationship, validator, execution)
   into a 0-100 score with a tier (high/medium/low). A sixth LLM-critique
   signal exists behind `ACCURACY_USE_LLM_CRITIQUE=true`.
9. **Learning + bookkeeping** (only when rows came back):
   - `add_turn()` updates session memory.
   - `save_successful_query()` → `add_example()` (`lib/query_examples.py`)
     stores the query as a new RAG example.
   - `bump_example()` rewards the examples that were shown in the prompt.
   - `save_query()` (`lib/chat_history.py`) persists to `_QUERY_HISTORY_` —
     fired with `asyncio.create_task`, never awaited, because the aux Mongo
     cluster sometimes stalls writes for 15-45 seconds.
10. **Response** — results, aiSummary, explanation, queryPlan, assumptions,
    confidence, validationWarnings, accuracy, queryMeta, meta.

## Execution order 3 — the learning loop

**Thumbs up/down** — `POST /api/feedback` → `save_feedback()`
(`lib/feedback_store.py`) returns instantly with a pre-generated id; the
insert and the RAG follow-up (`_process_feedback_for_rag()`) run in the
background. Good feedback calls `add_verified_example()` (weight 2.0); bad
feedback with a corrected pipeline calls `add_corrected_example()` (2.5).
Both also `bump_example()` the entries that fed the original prompt.

**Trainer mode** — gated by `TRAINER_PASSWORD` in `.env`:

- `POST /api/train/login` → `verify_password()` + `issue_token()`
  (`lib/trainer.py`), bearer token, in-memory, 2h TTL.
- `POST /api/train/run` → `build_corrected_query()` swaps the hand-edited
  pipeline in, executes and scores it without saving anything.
- `POST /api/train/save` → `save_as_gold()` does the three-store write:
  1. `_append_eval_case()` → locked case in `data/eval_set.json`
  2. `add_trainer_gold_example()` → RAG entry, weight 3.0, never pruned
  3. audit row in `_QUERY_FEEDBACK_` (fire-and-forget)
  If the RAG write fails, the eval case is rolled back.

**Example weighting** — `effective_weight()` in `lib/query_examples.py`:

```
rate     = (successes + 1) / (successes + failures + 2)      # Laplace
acc_mean = (accuracy_sum + 50) / (accuracy_n + 1) / 100
decay    = 1.0 / 0.9 / 0.75 / 0.5   for <30 / <90 / <180 / older days
weight   = min(base * (0.5 + rate) * (0.5 + acc_mean) * decay, 5.0)
```

Retrieval ranks by cosine similarity × this weight, so confirmed and
trainer-gold examples outrank fresh automatic ones.

**Pruning** — `prune_examples()`: trainer_gold never goes; user_verified
only after 365 days unused; auto entries go after 3 failures with no
success, 180 days unused with no success, or a mean accuracy under 40
across 5+ samples.

**Regression eval** — `POST /api/eval/run` (or `python3 scripts/evaluate.py`)
runs every case in `data/eval_set.json` through the live `/api/query`,
compares locked cases against their gold pipeline (`_pipelines_equal()`) and
all cases against their expected result shape (`_shape_ok()`), then writes
`data/eval_report.json` with a delta vs. the previous baseline. That's how
you answer "did my change help or hurt" with one command.
`scripts/propose_eval_set.py` drafts new eval cases from the best RAG
entries for human review.

## File map

| File | Owns |
|---|---|
| `main.py` | FastAPI app, every route, startup/shutdown |
| `lib/mongodb.py` | Motor clients for both DBs |
| `lib/db_registry.py` | Config-driven DB lookup (`data/db_registry.json`) |
| `lib/schemas.py` | The big static system prompt + keyword routing sets |
| `lib/relationships.py` | Join graph, classifier, relationship validators |
| `lib/schema_discovery.py` | Hourly live schema sampling → vector store |
| `lib/live_data_context.py` | Real sample docs + top values for the prompt |
| `lib/data_digest.py` | 3-day field/value digest on disk |
| `lib/collection_resolver.py` | Date phrases → month collection name |
| `lib/session_memory.py` | Follow-up context per browser session |
| `lib/llm_provider.py` | Ollama chat client + warmup |
| `lib/embeddings.py` | Ollama embedding client (`nomic-embed-text`) |
| `lib/vector_store.py` | JSON-file cosine-similarity store |
| `lib/query_examples.py` | RAG store, weights, pruning, bootstrap examples |
| `lib/query_generator.py` | Question → validated query object (retry loop) |
| `lib/query_executor.py` | Sanitized, capped, timed execution |
| `lib/response_validator.py` | Deterministic post-checks |
| `lib/accuracy_scorer.py` | Weighted 0-100 score |
| `lib/result_summarizer.py` | Plain-English summary (map-reduce for big sets) |
| `lib/feedback_store.py` | Thumbs up/down → `_QUERY_FEEDBACK_` + RAG updates |
| `lib/chat_history.py` | `_QUERY_HISTORY_` persistence |
| `lib/chat_sharing.py` | `_SHARED_CHATS_` snapshot links |
| `lib/trainer.py` | Trainer auth + three-store gold save |
| `scripts/evaluate.py` | Regression harness |
| `scripts/propose_eval_set.py` | Draft eval cases from RAG |
| `static/index.html` | The whole frontend in one file |

## Safety rules (non-negotiable)

- Read-only: `$out` and `$merge` are stripped before execution.
- Result caps: 200 raw docs / 1000 aggregated rows, 15s `maxTimeMS`.
- `apiKeys` / `streamingApiKeys` never leave the server.
- All aux writes (history, feedback, shares, RAG indexing) are
  fire-and-forget — a stalled aux cluster must never delay a user response.
- Everything runs locally; the only network calls are to your own MongoDB
  and your own Ollama.

## Extending it

- New month in the UI dropdown → add an `<option>` in `static/index.html`.
- Model keeps botching a field → add it to `lib/schemas.py`, or wait an hour
  for schema discovery to pick it up.
- Model keeps botching a query pattern → fix it once in trainer mode
  (preferred: also becomes an eval case), or hand-add to
  `BOOTSTRAP_EXAMPLES` in `lib/query_examples.py`.
- New cross-DB relationship → `JOINS` / `DISCRIMINATORS` / `ALIASES` in
  `lib/relationships.py`, plus a validator code if it can go wrong silently.
- New model → `ollama pull`, set `OLLAMA_MODEL` in `.env`, run the eval
  suite, compare deltas before trusting it.
