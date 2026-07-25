# Improvement Plan — Accuracy, Self-Learning, Trainer, Relationships

Status: DRAFT — for review before any code is written.
Owner: Claude / Motta
Scope: Four co-ordinated upgrades to the NL→MongoDB query system.

---

## 0. Executive Summary

The system already has an auto-learning RAG, user feedback capture, a rule-based
validator, and live schema discovery. This plan closes four remaining gaps:

| # | Feature | Problem today | Outcome |
|---|---|---|---|
| 1 | Cross-DB **Relationship Graph** | The LLM has to *infer* joins from free-form prose in `schemas.py`. Owner↔collection-name and `_id` discriminators are described in English, not data. | A machine-readable relationship declaration file that is injected into the prompt AND used by the classifier and validator, so the LLM no longer has to guess joins. |
| 2 | **Accuracy Score + Eval Harness** | No numeric score; no regression suite. A change that makes 5 queries better and 10 worse is invisible. | Every query gets a 0–100 score with breakdown. A JSON eval set + `evaluate.py` runs all questions on every change and reports delta vs. baseline. |
| 3 | **Trainer Mode** | Feedback is a 👍/👎. There is no way for a human expert to iterate on a bad pipeline, correct it, and lock the correction as a test case. | A Train button that opens an editable pipeline view, runs the edit, and commits the corrected pipeline as both (a) a weight-3.0 RAG example and (b) a locked eval case. |
| 4 | **Robust Self-Learning** | All auto-saved examples get weight 1.0 forever. No usage counters, no decay, no pruning. Stale and wrong examples poison retrieval. | Per-example success/failure counters, a blended effective-weight formula, age decay, and automatic pruning of low-performers. |

**Order of implementation:** 1 → 2 → 3 → 4. Each builds on the previous.

**Estimated size:** ~1700 lines of new Python + ~400 lines of HTML/JS.
Breakdown per feature in §7.

---

## 1. Feature — Cross-Database Relationship Graph

### 1.1 Problem

The two databases have an unusual join shape:

- `stream-datastore.<month>.appInfo.owner` **equals the collection name** in
  `appConfigs`. That is, owner "eduardo" means the collection
  `appConfigs.eduardo` — the collection *name itself* is the foreign key.
- Inside each `appConfigs.<owner>` collection, the `_id` field is a
  **discriminator**: values like `"usersinfo"`, `"InfoToConstructUrls"`,
  `"default"` identify document types. Querying `appConfigs.eduardo` without
  `_id: "usersinfo"` returns unrelated docs.
- `appInfo.owner` (account operator) ≠ `loggedInUserData.name` (end user).
  The LLM conflates these ~20% of the time.

Today all of this lives as prose in `lib/schemas.py` and in a few bootstrap
examples. The LLM has to infer the joins every single call.

### 1.2 Design

New module `lib/relationships.py` exporting three data structures plus a
prompt-rendering function. Everything is declarative Python — no code paths.

```python
# lib/relationships.py — excerpt

JOINS = [
    {
        "id":       "owner-across-dbs",
        "kind":     "collection-name-fk",
        "from":     {"db": "stream-datastore", "any_collection": True,
                     "field": "appInfo.owner"},
        "to":       {"db": "appConfigs", "collection_is_key": True},
        "merge_key": "owner",
        "description":
            "Stream sessions reference an owner by name in appInfo.owner. "
            "In appConfigs, that same name IS the collection name. "
            "To join: run stream aggregate grouped by appInfo.owner, then "
            "for each owner run a second query against appConfigs.<owner>.",
    },
]

DISCRIMINATORS = [
    {
        "db":         "appConfigs",
        "collection": "*",                 # applies to every owner collection
        "field":      "_id",
        "values":     {
            "usersinfo":          "billing, subscription, maxUserLimit, paidMinutes",
            "InfoToConstructUrls":"streaming URL templates and config",
            "default":            "default streaming app configuration",
        },
        "required":   True,
        "notes":      "Always include {_id: <value>} in $match when querying appConfigs.",
    },
]

ALIASES = [
    {"term": "owner", "meaning":
        "appInfo.owner (stream-datastore) OR collection name (appConfigs). "
        "NOT the logged-in user."},
    {"term": "user", "meaning":
        "loggedInUserData.name — the end user who logged in to the session. "
        "NOT the owner."},
    {"term": "current month", "meaning":
        "Defaults to the DEFAULT_STREAM_COLLECTION env var (e.g. Apr_2026)."},
]
```

### 1.3 Working Principle — end-to-end

```
User question: "show subscription info and session count for eduardo"
                         │
                         ▼
        query_generator.classify(question)
                         │
    ┌────────────────────┼────────────────────┐
    │   uses JOINS       │   matches ALIAS    │
    │   mentions session │   "owner"="eduardo"│
    │   AND subscription │                    │
    └────────────────────┬────────────────────┘
                         │
                         ▼
               queryType = "dual"
               merge_key = "owner"   (from JOINS[0])
               hint: "appConfigs query MUST filter _id:'usersinfo'"
                         │
                         ▼
      render_relationship_block()  → prompt section
                         │
                         ▼
                  LLM call (Ollama)
                         │
                         ▼
          response_validator.check_relationships(q)
            - dual query missing mergeKey? → error
            - appConfigs query missing _id discriminator? → error
            - stream query filtering on "username" field? → error
              (correct field is loggedInUserData.name)
                         │
                         ▼
                 execute + return
```

### 1.4 Input / Output Contract

**`render_relationship_block() → str`**
- Input: none (reads JOINS, DISCRIMINATORS, ALIASES)
- Output: a ~500-char text block with bullet points, injected into the LLM
  system prompt *before* the few-shot examples.

**`classify_with_relationships(question: str) → dict`**
- Input: raw question string
- Output:
  ```python
  {
    "query_type":     "single" | "dual",
    "db_hint":        "stream" | "appconfigs" | "both",
    "owner_hint":     str | None,   # extracted if alias-matched to a known owner
    "merge_key":      str | None,
    "required_filters": list[dict], # e.g. [{"field": "_id", "value": "usersinfo"}]
    "reasoning":      str,          # for debug/log
  }
  ```

**`check_relationships(query_obj: dict) → list[dict]`**
- Input: the LLM-generated query object
- Output: validator checks (same shape as `response_validator`):
  `{"level": "error"|"warning"|"info", "code": "...", "message": "..."}`

### 1.5 Detail Process — classifier upgrade

Current `query_generator.py` has a ~20-line keyword scan. Replace with:

1. **Alias pass** — tokenise question, match against ALIASES terms. Each match
   contributes a *signal* (db-hint, required field, etc.).
2. **Join-trigger pass** — for each JOIN, check if question mentions words from
   *both* sides (e.g. "session" matches stream side, "subscription" matches
   appConfigs side → owner-across-dbs triggers).
3. **Owner-name extraction** — scan for quoted strings, `owner <name>`,
   `for <name>`. Validate the candidate against the known list of owner
   collection names (sampled from `appConfigs` by `schema_discovery`).
4. **Synthesise classification** — combine signals. If dual is triggered,
   write `merge_key` and `required_filters` for downstream.

### 1.6 Files Touched

- **NEW** `lib/relationships.py` (~220 lines) — data + render + check + classify.
- **EDIT** `lib/query_generator.py` (~40 lines changed) — call classifier,
  prepend relationship block to prompt, pass `required_filters` to validator.
- **EDIT** `lib/response_validator.py` (~50 lines added) — consume
  `check_relationships` output alongside existing checks.
- **EDIT** `lib/schemas.py` (~20 lines removed) — delete the prose that is now
  replaced by structured JOINS/DISCRIMINATORS.

### 1.7 Risks / Open Questions

- The owner-name list for appConfigs has ~5500 entries. Loading that list into
  memory is fine (it's ~80 KB), but re-scanning on every classification call
  would be slow. Cache it at startup with a 1-hour TTL.
- What if a user legitimately writes "owner" to mean the logged-in user? The
  alias disambiguation will be wrong. Mitigation: the validator produces a
  `NOTE` (not an error) explaining the interpretation; the UI surfaces it.

---

## 2. Feature — Accuracy Score + Evaluation Harness

### 2.1 Problem

- Validator returns a list of pass/warn/error flags but no single number.
- There is no regression test set. A change to `query_generator.py` could
  silently break queries that used to work.
- The RAG weight formula uses a fixed `weight` field (1.0/2.0/2.5) with no
  objective accuracy signal.

### 2.2 Design — Two-Part

**Part A — per-query accuracy scoring** (runs inline, every query)
**Part B — offline evaluation harness** (runs on-demand or in CI)

Both use the same `AccuracyScorer` class.

### 2.3 Per-Query Scoring (Part A)

Scoring is a weighted sum of six signals. All signals normalise to 0–100.

| Signal | Weight | What it measures |
|---|---|---|
| `syntax` | 10% | Pipeline parses; no banned stages; aggregate stages in valid order. |
| `schema` | 25% | Every referenced field exists in the static + live-discovered schema. Unknown field = −15 each, capped at 0. |
| `relationship` | 15% | Cross-DB merge key correct; discriminator present where required (from Feature 1). |
| `validator` | 20% | Sum of validator checks: +10 per pass, −10 per warn, −30 per error, capped 0–100. |
| `execution` | 20% | `0` if timeout or exception; `70` if runs but 0 rows; `100` if runs with rows. |
| `llm_critique` | 10% | OPTIONAL. A second Ollama call: "score 0–100 how well this pipeline answers the question." Cached; skipped when latency matters. |

Final score: `Σ(signal × weight)`. Rounded to int.

**Confidence tier** (derived):
- `high` ≥ 80
- `medium` 50–79
- `low` < 50

**Score breakdown object** returned with every query:

```python
{
  "score": 87,
  "tier":  "high",
  "signals": {
    "syntax":       {"score": 100, "notes": []},
    "schema":       {"score":  95, "notes": ["field 'clientInfo.region' not in static schema — present in live discovery"]},
    "relationship": {"score": 100, "notes": []},
    "validator":    {"score":  80, "notes": ["1 warning: no $match stage on large collection"]},
    "execution":    {"score": 100, "notes": ["42 rows in 340ms"]},
    "llm_critique": {"score":  70, "notes": ["pipeline answers the question but includes extra fields"]},
  },
}
```

### 2.4 Input / Output Contract — AccuracyScorer

```python
class AccuracyScorer:
    async def score(
        self,
        question:      str,
        query_obj:     dict,
        result:        dict,            # execution output, or {"error": "..."}
        result_count:  int,
        validator_checks: list[dict],
        *,
        skip_llm_critique: bool = False,
    ) -> ScoreBreakdown: ...
```

Non-blocking: the scorer adds ~50 ms (without LLM critique) or ~800 ms (with).
The `/api/query` handler computes the score *after* returning results to the
client — it is attached to the response but does not delay it. Done via
`asyncio.gather` over (return-response, compute-score).

Actually: we want the score visible to the user, so: score is computed
synchronously without `llm_critique` (fast path, ~50 ms) and returned in the
response. `llm_critique` is computed asynchronously and stored alongside the
RAG entry so it influences *future* weight, not this response.

### 2.5 Evaluation Harness (Part B)

**`data/eval_set.json`** — hand-curated test cases:

```json
[
  {
    "id":           "eval-001",
    "question":     "how many sessions per country this month?",
    "db_hint":      "stream",
    "gold_pipeline": [...],
    "gold_result_shape": {"rows": ">10", "columns": ["_id", "sessions"]},
    "locked":       true,
    "tags":         ["aggregation", "country"]
  },
  ...
]
```

Two kinds of eval entries:
- `locked: true` — gold_pipeline must match structurally (after normalisation).
  Added by the trainer flow (Feature 3).
- `locked: false` — only gold_result_shape is checked; the LLM may produce any
  pipeline that yields equivalent output.

**`scripts/evaluate.py`** — CLI runner:

```
$ python scripts/evaluate.py
Running 47 eval cases against http://localhost:8000 ...

  eval-001 [PASS] score=95 (schema:100 validator:90)
  eval-002 [FAIL] score=42 — wrong field: 'country' (should be 'clientInfo.country_name')
  ...

Summary:
  Passed:            39 / 47   (83%)
  Avg score:         78.2
  Regression vs last run (saved in .eval_baseline.json):
    +3 passing, -1 passing
    Δ avg score:     +1.4
  Failures saved to: eval_failures_2026-04-24.json
```

Output is also JSON (`eval_report.json`) for CI integration.

### 2.6 Working Principle

```
┌──────────── per query ────────────┐
│  POST /api/query                   │
│        │                           │
│        ▼                           │
│  generate → execute → validate     │
│        │                           │
│        ▼                           │
│  AccuracyScorer.score(...)         │
│   (without llm_critique, fast)     │
│        │                           │
│        ▼                           │
│  response includes score + breakdown
│        │                           │
│        └── async: critique + store
└────────────────────────────────────┘

┌──────────── batch eval ────────────┐
│  scripts/evaluate.py               │
│    for each case in eval_set.json: │
│      run through full /api/query   │
│      compare to gold               │
│    aggregate → report              │
└────────────────────────────────────┘
```

### 2.7 Files Touched

- **NEW** `lib/accuracy_scorer.py` (~300 lines).
- **NEW** `scripts/evaluate.py` (~180 lines).
- **NEW** `data/eval_set.json` — seeded with ~30 cases extracted from
  `BOOTSTRAP_EXAMPLES` + good queries in `query_examples.json`.
- **EDIT** `main.py` — include score in `/api/query` response; new
  `GET /api/eval/baseline` (returns last report) and
  `POST /api/eval/run` (triggers background run).
- **EDIT** `static/index.html` — accuracy badge next to results
  (colour: green ≥80, amber 50–79, red <50), click to expand breakdown.
- **EDIT** `lib/query_examples.py` — include score in metadata, include in
  `effective_weight` formula (Feature 4).

### 2.8 Risks / Open Questions

- `llm_critique` needs a separate prompt and adds load. First release: off by
  default, opt-in via env var `ACCURACY_USE_LLM_CRITIQUE=true`.
- Gold pipeline equality is tricky: `{$match: {a:1, b:2}}` vs `{$match: {b:2, a:1}}`
  are equivalent. Normaliser: sort keys, canonicalise field order inside
  operators, stringify. Document in `lib/accuracy_scorer.py`.
- Eval set curation is manual. First pass: me proposing 30 cases from real
  query history, you approving/editing before commit.

---

## 3. Feature — Trainer Mode

### 3.1 Problem

Today the only feedback loop is 👍/👎. If the LLM produces a wrong pipeline,
the expert has no way to:
- See the failure alongside their intent
- Edit the pipeline and re-run it
- Commit the correction as a *locked test case* so the system cannot
  regress on it silently

### 3.2 Design

**Entry point:** a new **Train** button in the UI, next to Ask Again.
Visible only when `TRAINER_MODE=true` in `.env` (server-side gate) — prevents
random users from polluting the eval set.

**Editor panel (new UI section):**

```
┌─ Train: "how many sessions per owner last week?" ────────────────┐
│                                                                  │
│ Original pipeline (what LLM generated):                          │
│ ┌────────────────────────────────────────────────────────────┐  │
│ │ [{"$match": {"e3ds_employee": false}}, ...]                │  │
│ │                                           (read-only)       │  │
│ └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│ Corrected pipeline (edit me):                                    │
│ ┌────────────────────────────────────────────────────────────┐  │
│ │ [{"$match": {...}}, ...]                                    │  │
│ │                                           (JSON editor)     │  │
│ └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│ [ Run Corrected ]   [ Diff Original↔Corrected ]                 │
│                                                                  │
│ Result preview:    42 rows in 380ms                             │
│                                                                  │
│ Accuracy score:    92 (was 54)                                  │
│                                                                  │
│ [ Save as Gold ]   [ Discard ]                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Save as Gold** performs three writes atomically (or rolls back):

1. Append to `data/eval_set.json` as `{locked: true, gold_pipeline: <corrected>}`.
2. Upsert to RAG `examples` store with `weight: 3.0, source: "trainer_gold"`.
3. Insert into MongoDB `_QUERY_FEEDBACK_` with `rating: "corrected", source: "trainer"`.

### 3.3 API

```
POST /api/train/run
  body: { session_id, question, corrected_pipeline, database, collection }
  auth: checks TRAINER_MODE env + optional TRAINER_TOKEN header
  returns: { result, result_count, score, score_breakdown }

POST /api/train/save
  body: { session_id, question, original_query_obj, corrected_pipeline,
          database, collection, eval_tags: [] }
  returns: { eval_id, rag_id, feedback_id }

GET /api/train/list
  returns: { total_gold_cases, recent_10: [...] }
```

### 3.4 Working Principle — end-to-end

```
User sees bad result
        │
        ▼
Clicks "Train" button (visible if TRAINER_MODE=true)
        │
        ▼
UI fetches current pipeline → opens editor panel
        │
        ▼
User edits JSON, clicks "Run Corrected"
        │
        ▼
POST /api/train/run
        │
        ├─► query_executor runs corrected pipeline
        ├─► response_validator checks it
        ├─► AccuracyScorer scores it
        │
        ▼
UI shows: results + score + diff
        │
        ▼
User clicks "Save as Gold"
        │
        ▼
POST /api/train/save
  │
  ├─► eval_set.json   (locked=true)
  ├─► RAG examples    (weight=3.0, source=trainer_gold)
  └─► _QUERY_FEEDBACK_ (rating=corrected, source=trainer)
        │
        ▼
UI confirms: "Saved as gold case eval-047"
```

### 3.5 Input / Output Contract

**`POST /api/train/save`**
- Input:
  ```json
  {
    "session_id":          "...",
    "question":            "...",
    "original_query_obj":  { "queryType": "single", "pipeline": [...], ... },
    "corrected_pipeline":  [ { "$match": ... }, ... ],
    "database":            "stream-datastore",
    "collection":          "Apr_2026",
    "eval_tags":           ["aggregation", "owner"]
  }
  ```
- Output:
  ```json
  {
    "eval_id":     "eval-047",
    "rag_id":      "sha1(question)",
    "feedback_id": "ObjectId(...)",
    "score":       92,
    "saved_at":    "2026-04-24T14:00:00Z"
  }
  ```

### 3.6 Files Touched

- **NEW** `lib/trainer.py` (~200 lines) — coordinates the three writes with
  rollback on partial failure.
- **EDIT** `main.py` (~90 lines added) — three endpoints + auth gate.
- **EDIT** `static/index.html` (~350 lines added) — trainer panel UI, diff
  viewer, JSON editor (no external lib — plain `<textarea>` + validation).
- **EDIT** `lib/query_examples.py` (~15 lines) — accept `weight=3.0` and new
  `source="trainer_gold"`.

### 3.7 Risks / Open Questions

- JSON editing in a `<textarea>` is crude. First release: that plus
  real-time parse-error hint. If needed later we can add Monaco/CodeMirror
  (adds ~300 KB).
- Concurrent trainers could race on `eval_set.json`. Mitigation: file lock
  via `fcntl.flock` in `lib/trainer.py`.
- Rollback: if RAG write succeeds but eval_set write fails, we need to
  remove the RAG entry. Use a staged write (write temp → validate → commit)
  with explicit `try/except + compensating delete`.

---

## 4. Feature — Robust Self-Learning

### 4.1 Problem

Current `add_example` behaviour:
- Every successful query (result_count > 0) is saved, weight 1.0, forever.
- No tracking of whether an example has actually been useful when retrieved.
- No decay: a 2-year-old example retrieved for the current question
  competes with last week's on raw cosine similarity.
- No pruning: `MAX_EXAMPLES=500` is a hard cap but eviction is chronological
  (oldest out), not quality-based.
- Auto-save treats "query returned rows" as "query was correct" — this is
  false when the pipeline partially matched but missed the intent.

### 4.2 Design — New per-example metadata

Extend the vector-store metadata schema:

```python
{
  "query":          {...},       # existing
  "result_count":   int,          # existing
  "db_hint":        str,          # existing
  "weight":         float,        # existing: 1.0 / 2.0 / 2.5 / 3.0
  "source":         str,          # existing

  # NEW fields
  "success_count":  int,          # times retrieved → downstream query succeeded
  "failure_count":  int,          # times retrieved → downstream query failed
                                   # (LLM produced bad output OR user marked bad)
  "accuracy_sum":   float,        # sum of accuracy scores (Feature 2) of the
                                   # queries it helped produce
  "accuracy_n":     int,
  "last_used":      ISO timestamp, # last time retrieved for a query
  "last_good_at":   ISO timestamp, # last time it helped produce a good query
  "created_at":     ISO timestamp, # existing-ish (was "timestamp")
}
```

### 4.3 Effective Weight Formula

```python
def effective_weight(m: dict) -> float:
    base      = m.get("weight", 1.0)

    # success rate — Laplace smoothed
    s         = m.get("success_count", 0)
    f         = m.get("failure_count", 0)
    rate      = (s + 1) / (s + f + 2)           # ∈ (0, 1)

    # accuracy mean — Laplace smoothed to 50
    a_sum     = m.get("accuracy_sum", 0.0)
    a_n       = m.get("accuracy_n", 0)
    acc_mean  = (a_sum + 50) / (a_n + 1) / 100   # ∈ (0, 1)

    # age decay
    age_days  = _days_since(m.get("last_used") or m.get("created_at"))
    if   age_days <  30: decay = 1.0
    elif age_days <  90: decay = 0.9
    elif age_days < 180: decay = 0.75
    else:                decay = 0.5

    return base * (0.5 + rate) * (0.5 + acc_mean) * decay
```

- At zero evidence: `rate = 0.5, acc_mean = 0.5, decay = 1.0 → effective = base * 1.0`.
- High-performing example: rate=1.0, acc=0.95 → multiplier ≈ 2.18.
- Chronically bad example: rate=0.1, acc=0.3 → multiplier ≈ 0.38.

### 4.4 Counter Updates

Two hook points:

**A. On retrieval + successful query** (in `main.py` after `/api/query`):
```python
for ex_id in retrieved_example_ids:
    store.bump(ex_id, success=True, accuracy=score)
```

**B. On user feedback** (in `feedback_store.py`):
```python
if rating == "good":    bump(success=True,  accuracy=score)
if rating == "bad":     bump(success=False, accuracy=score)
```

New method on `VectorStore`:
```python
def bump(self, id: str, *, success: bool, accuracy: int) -> None: ...
```

### 4.5 Pruning

New function `prune_examples(dry_run=False) → dict`:

```python
# Rules, evaluated in order. First match wins.
REMOVE_RULES = [
    # 1. Trainer-gold is NEVER pruned.
    ("keep", lambda m: m.get("source") == "trainer_gold"),
    # 2. User-verified is almost never pruned — only if truly dead.
    ("remove_if", lambda m: m.get("source") == "user_verified"
                  and _days_since(m.get("last_used")) > 365),
    # 3. Auto examples with repeated failures.
    ("remove_if", lambda m: (m.get("failure_count", 0) >= 3
                             and m.get("success_count", 0) == 0)),
    # 4. Ancient unused.
    ("remove_if", lambda m: (_days_since(m.get("last_used")) > 180
                             and m.get("success_count", 0) == 0)),
    # 5. Low accuracy after sufficient evidence.
    ("remove_if", lambda m: (m.get("accuracy_n", 0) >= 5
                             and (m.get("accuracy_sum", 0) / m.get("accuracy_n", 1)) < 40)),
]
```

Invoked:
- On a schedule (every 6 hours) — background task started in `main.py` lifespan.
- Manually via `POST /api/rag/prune`.
- Returns `{"removed": N, "kept": M, "details": [...]}`.

### 4.6 Stats Endpoint

```
GET /api/rag/stats
  returns: {
    "total":          478,
    "by_source":      { "auto": 420, "user_verified": 40, "user_corrected": 15, "trainer_gold": 3 },
    "avg_weight":     1.23,
    "avg_accuracy":   76.4,
    "top_10":         [ { question, effective_weight, success_count, ... } ],
    "bottom_10":      [ ... ],
    "stale":          12,   # count of last_used > 90 days
  }
```

Used by a new admin sidebar panel in the UI.

### 4.7 Files Touched

- **EDIT** `lib/vector_store.py` (~40 lines added) — `bump`, metadata
  update helper.
- **EDIT** `lib/query_examples.py` (~80 lines added) — `effective_weight`,
  `prune_examples`, `rag_stats`. Replace the existing weighted sort in
  `find_similar_examples_vector` with `effective_weight`.
- **EDIT** `main.py` (~60 lines) — `/api/rag/stats`, `/api/rag/prune`,
  background prune task, bump hooks on `/api/query` and feedback path.
- **EDIT** `lib/feedback_store.py` (~15 lines) — call `bump` in
  `_process_feedback_for_rag`.
- **EDIT** `static/index.html` — small stats card (optional for v1).

### 4.8 Risks / Open Questions

- **Accuracy-score feedback loop.** Using the current query's accuracy to
  bump the example that helped produce it creates a loop: a good example
  helps produce a high score, which further boosts the example. This is
  intended but needs a cap to prevent runaway weights. Mitigation: cap
  `base * multiplier` at 5.0.
- **Storage.** Five new fields × 500 examples ≈ 40 KB extra JSON. Fine.
- **Migration.** Existing examples lack the new fields. On load, fill with
  defaults (`success_count=0`, `last_used=created_at`, etc.). No separate
  migration script.

---

## 5. Cross-Cutting Concerns

### 5.1 Environment Variables — New

| Var | Default | Purpose |
|---|---|---|
| `TRAINER_MODE` | `false` | Gate trainer endpoints + UI button. |
| `TRAINER_TOKEN` | *(unset)* | Optional bearer token for trainer endpoints. |
| `ACCURACY_USE_LLM_CRITIQUE` | `false` | Turn on the 6th scoring signal. |
| `RAG_PRUNE_INTERVAL_HOURS` | `6` | Background prune cadence. |
| `EVAL_BASELINE_PATH` | `data/.eval_baseline.json` | Where to persist last eval run. |

### 5.2 Directory Additions

```
mongodb-llm-rag/
├── data/
│   ├── eval_set.json              NEW — gold test cases
│   ├── eval_report.json           NEW — last eval run (gitignored)
│   └── .eval_baseline.json        NEW — previous eval for delta
│
├── docs/
│   └── IMPROVEMENT_PLAN.md        THIS FILE
│
├── lib/
│   ├── relationships.py           NEW — Feature 1
│   ├── accuracy_scorer.py         NEW — Feature 2
│   └── trainer.py                 NEW — Feature 3
│
└── scripts/
    └── evaluate.py                NEW — Feature 2 CLI
```

### 5.3 .gitignore Additions

```
data/eval_report.json
data/.eval_baseline.json
```

(`eval_set.json` IS committed — it is the ground truth.)

### 5.4 Backwards Compatibility

- All new metadata fields on RAG examples have default fallback values on
  read. Old examples continue to work.
- `feedback_store.py` keeps its existing schema; the trainer writes
  additional rows, not modifies existing ones.
- `/api/query` response adds a top-level `accuracy` field but keeps all
  existing fields. The UI tolerates its absence.

---

## 6. Phased Rollout

| Phase | Features | LOC | Ship criteria |
|---|---|---|---|
| **P1 — Relationships** | Feature 1 | ~330 | `evaluate.py` run on current example set shows no regression; 10 manual cross-DB questions produce correct dual queries. |
| **P2 — Scoring + Eval** | Feature 2 | ~650 | Every query in UI shows an accuracy badge; `evaluate.py` runs end-to-end and emits `eval_report.json`; baseline saved. |
| **P3 — Trainer Mode** | Feature 3 | ~650 | With `TRAINER_MODE=true`, correcting a bad pipeline persists to all three stores; `eval_set.json` grows; next eval run incorporates the new case. |
| **P4 — Self-Learning Polish** | Feature 4 | ~200 | Prune dry-run on current examples shows sensible removals; stats endpoint returns expected shape; effective-weight ranking changes vector results in a measurable way. |

Each phase ends with a commit + eval run + short sanity test plan in the PR
description. No merge to `main` until accuracy score on the eval set is
**≥** baseline.

---

## 7. File-by-File Change Summary

### NEW FILES

| File | Lines | Phase |
|---|---|---|
| `lib/relationships.py` | ~220 | P1 |
| `lib/accuracy_scorer.py` | ~300 | P2 |
| `scripts/evaluate.py` | ~180 | P2 |
| `data/eval_set.json` | (data) | P2 |
| `lib/trainer.py` | ~200 | P3 |
| `docs/IMPROVEMENT_PLAN.md` | (this) | P0 |

### EDITED FILES

| File | ± LOC | Feature(s) |
|---|---|---|
| `lib/query_generator.py` | +60 / -30 | F1, F2 |
| `lib/response_validator.py` | +50 | F1 |
| `lib/schemas.py` | -20 | F1 |
| `lib/query_examples.py` | +95 | F1, F2, F4 |
| `lib/vector_store.py` | +40 | F4 |
| `lib/feedback_store.py` | +15 | F4 |
| `main.py` | +180 | F2, F3, F4 |
| `static/index.html` | +650 | F2, F3, F4 |
| `.gitignore` | +2 | F2 |
| `CLAUDE.md` | +30 | all (docs) |

**Grand total: ~1700 Python LOC + ~650 UI LOC.**

---

## 8. Decisions (finalised)

| # | Question | Decision |
|---|---|---|
| 1 | Trainer auth | **UI password prompt.** A password is checked server-side against `TRAINER_PASSWORD` env var. Session token returned on success, stored in `sessionStorage`, required on `/api/train/*` calls. |
| 2 | LLM critique (sixth scoring signal) | **Disabled by default.** Enable via `ACCURACY_USE_LLM_CRITIQUE=true`. Not used for per-query latency path; only for async post-hoc weighting. |
| 3 | Eval set seeding | **Auto-propose.** A one-shot script `scripts/propose_eval_set.py` scans the existing `data/query_examples.json` + `_QUERY_FEEDBACK_` collection for "good" rated queries, dedupes by question tokens, and emits `data/eval_set.proposed.json` for human review. |
| 4 | Train button placement | **Under Raw Data, right side of Validation.** Keeps the Train action spatially grouped with the inspection tools, separate from user-facing Ask Again. |
| 5 | Relationship graph format | **Python module** (`lib/relationships.py`). Comments, typed constants, IDE support. |

Phase 1 starts now.
