# Detail System Overview

## Part 1 — Start the system, step by step

You need three things running: MongoDB (already hosted), Ollama, and the
FastAPI server. In order:

**Step 1 — Start Ollama and get the models** (first time only for the pulls)

```bash
ollama serve                      # leave this running in its own terminal
# We can pull any larger model based on our vram size( Qwen, Gemma etc...)
ollama pull qwen2.5-coder:7b      # the model that writes the queries
ollama pull nomic-embed-text      # the model that powers RAG similarity search
```

**Step 2 — Set up Python** (first time only)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Step 3 — Configure**

```bash
cp .env.example .env
# then we need to add all environment variables.
```

**Step 4 — Run the server**

```bash
python main.py                    # dev mode, auto-reloads on file changes
```

**Step 5 — Verify it's actually up**

```bash
curl http://localhost:8000/api/health    # should say "ok" for both databases
curl http://localhost:8000/api/status    # should show your Ollama model as available
```


## Part 2 — Each Files Working Principle

**Entry point**

- `main.py` — the FastAPI app. Every HTTP route lives here, plus the startup
  sequence that kicks off all the background warmup tasks.

**Talking to MongoDB**

- `lib/mongodb.py` — opens the two async Motor connections.
- `lib/db_registry.py` — maps database names to connections via
  `data/db_registry.json`, so the executor doesn't hardcode anything.



- `lib/collection_resolver.py` — reads dates out of the question ("last
  month", "Q1 2026", "16th April 2026") and picks the right monthly
  collection.
- `lib/session_memory.py` — remembers the last few turns per browser
  session so follow-ups like "now only India" make sense.
- `lib/relationships.py` — knows how the two databases join (owner name =
  appConfigs collection name), routes the question to the right database(s),
  and later double-checks the query against those rules.

- `lib/schemas.py` — the big static rulebook the LLM reads: output format,
  field names, timestamp units, safety rules.
- `lib/schema_discovery.py` — samples real documents hourly so the prompt
  always reflects the actual current schema, not a stale description.
- `lib/schema_watcher.py` — sits on top of a MongoDB change stream and
  notices new/removed fields as they land, then triggers the same refresh
  (rate-limited to once per hour). Turns the hourly poll into "hourly OR
  when something actually changes." Requires a replica set; degrades
  cleanly to the plain hourly sample on standalone clusters.
- `lib/live_data_context.py` — feeds the prompt real values (actual country
  names, owner names, app names) so the model copies exact spellings.
- `lib/data_digest.py` — slower-moving version of the same idea: a field +
  example-value digest refreshed every 3 days and kept on disk.
- `lib/query_examples.py` — the RAG store. Every successful query is saved
  here; the most similar past examples get pasted into the prompt as
  demonstrations. Also handles example weighting and pruning.
- `lib/embeddings.py` / `lib/vector_store.py` — turn text into vectors via
  Ollama and search them by cosine similarity. Used by schema retrieval and
  example retrieval.

- `lib/llm_provider.py` — the Ollama HTTP client (JSON mode, warmup).
- `lib/query_generator.py` — orchestrates everything above into one prompt,
  calls the model, validates the JSON it returns, and retries with an error
  explanation when the model gets it wrong.
- `lib/query_executor.py` — actually runs the query, safely: strips write
  stages, caps result sizes, 15-second timeout, accent-insensitive text
  matching, runs dual-database queries in parallel and merges them.

- `lib/response_validator.py` — deterministic sanity checks on the query and
  result (no LLM involved). Findings show up in the UI's VALIDATION tab.
- `lib/accuracy_scorer.py` — folds five signals into a 0-100 confidence
  score shown next to every answer.
- `lib/result_summarizer.py` — writes the plain-English summary of the
  results; chunks large result sets map-reduce style.

- `lib/feedback_store.py` — handles thumbs up/down; good answers get
  promoted in the RAG store, corrected answers get saved with extra weight.
- `lib/trainer.py` — password-gated mode where a human fixes a bad pipeline
  and saves it as gold: it becomes a locked eval case, a top-weight RAG
  example, and an audit record in one save.
- `lib/chat_history.py` / `lib/chat_sharing.py` — query history and
  shareable chat links, both written fire-and-forget.

- `scripts/evaluate.py` — replays the locked eval set against the running
  server and reports pass/fail with a delta vs. last run.
- `scripts/propose_eval_set.py` — drafts new eval cases from the best RAG
  entries for human review.
- `static/index.html` — the entire frontend: dark UI, voice input, result
  tabs, feedback buttons, trainer panel. One self-contained file.

## Part 3 — Practical system working principle (From user input to the UI Output)

Say you type: **"Top 5 countries by session count in April"**.

1. **The browser** (`static/index.html`) POSTs it to `/api/query` with your
   session id and selected collection.

2. **Context is gathered** (`main.py`). Session memory is checked for
   earlier turns, and the collection resolver reads "April" and picks
   `Apr_2026` — the LLM never has to guess the collection.

3. **The question is routed** (`lib/relationships.py`). "session count" is
   a stream-datastore signal, no billing words, no owner mentioned — so this
   is a single-database query. Routing hints get added to the prompt.

4. **The prompt is assembled** (`lib/query_generator.py`). Into one prompt
   go: the static rulebook, the relationship rules, the ~20 schema fields
   most relevant to the question (vector search), real value samples
   ("clientInfo.country_name: Brazil, India, ..."), and the 2 most similar
   past successful queries as worked examples.

5. **Ollama writes the query.** JSON mode forces valid JSON. Expected shape:

   ```json
   {
     "queryType": "single",
     "database": "stream-datastore",
     "collection": "Apr_2026",
     "operation": "aggregate",
     "pipeline": [
       { "$match": { "e3ds_employee": { "$ne": true } } },
       { "$group": { "_id": "$clientInfo.country_name", "sessions": { "$sum": 1 } } },
       { "$sort": { "sessions": -1 } },
       { "$limit": 5 }
     ],
     "explanation": "Top 5 countries by session count in April 2026",
     "resultLabel": "Top Countries",
     "confidence": "high"
   }
   ```

6. **The output is checked before running** (`lib/query_generator.py`).
   JSON parses? Structure valid? Every field name actually exists in the
   schema? If anything fails, the model gets its own output back with the
   error spelled out ("'clientinfo.country' does not exist. Did you mean
   'clientInfo.country_name'?") and tries again, up to the retry limit.

7. **The query runs** (`lib/query_executor.py`). Write stages are stripped,
   the result cap is enforced, the 15s timeout is set, and the aggregation
   executes. If the question had been about a whole year, the pipeline
   would first get expanded across all 12 monthly collections with
   `$unionWith`. Dual-database questions run both halves in parallel and
   merge by owner in Python.

8. **The result is checked after running** (`lib/response_validator.py` +
   `lib/accuracy_scorer.py`). Deterministic checks (employee filter present?
   zero results? limit placed correctly?) plus a 0-100 accuracy score.

9. **The answer is explained** (`lib/result_summarizer.py`). The rows go
   back to the LLM once more for a short plain-English summary: "Brazil led
   April with 1,204 sessions, followed by India..."

10. **The system learns** (background, never blocks the response). The
    successful query is embedded into the RAG store, the examples that were
    shown in the prompt get a success bump, and the query lands in history.

11. **The browser renders it**: the summary up top, then tabs for the
    ready-to-run **mongosh command** (QUERY), the raw table, the JSON, the
    generated pipeline, the validation findings, the accuracy breakdown,
    and thumbs up/down — which feeds step 10 the next time. The QUERY tab
    is the fastest way to sanity-check what the LLM produced: copy it, run
    it in Compass or the shell, compare row counts against what the UI
    showed.

If the model fails all its retries, you get a readable error and an "Ask
Again" button instead of a stack trace.

## Part 4 — TLDR; How the system works

The core trick is that the LLM is never trusted and never alone. It's
**boxed in before** generation (the collection, the database routing, the
exact field names and real values are all decided or supplied by
deterministic code), **checked after** generation (structure validation,
field validation, retry-with-error loop), **sandboxed during** execution
(read-only, capped, time-limited), and **audited after** execution
(validators, accuracy score).

Around that box sits a feedback loop: every success becomes a few-shot
example, user feedback re-weights those examples, a human trainer can inject
gold-standard corrections, and a regression suite replays locked cases so
you can tell whether any change — new model, new prompt, new code — made
things better or worse. The longer it runs, the better its prompts get,
because the prompts are built from its own verified history.

Everything — generation, embeddings, summaries — runs on local Ollama. The
only things it talks to are your MongoDB and your GPU.
