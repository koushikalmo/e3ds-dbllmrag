# Code Comments Archive

All verbose block comments and docstrings removed from lib/ and main.py.
Preserved here for reference.

---

## lib/mongodb.py

**Why Motor instead of PyMongo?**
FastAPI runs on asyncio. PyMongo is synchronous — it blocks the entire thread
during a query. Motor is the async wrapper that integrates with asyncio via `await`.

**Why module-level singletons?**
Opening a connection pool is expensive (DNS, TLS, auth). Creating one on every
request would waste thousands of ms per minute. Singletons created at first call
reuse the same pool until shutdown.

**stream-datastore document example:**
```json
{
  "_id": "userInfo_000156c0-...",
  "appInfo": { "owner": "eduardo", "appName": "FluidFluxProject" },
  "clientInfo": { "city": "Itajaí", "country_name": "Brazil" },
  "loadTime": 8.514,
  "startTimeStamp": 1745851775,
  "DisconnectTime_Timestamp": 1745851874,
  "webRtcStatsData": {
    "avgBitrate": 50226,
    "packetsLost": 399,
    "avgRoundTripTime": "0.177"
  },
  "elInfo": { "computerName": "E3DS-S23", "systemInfo": { "cpu": {...} } }
}
```
Note: `avgRoundTripTime` is a STRING. `startTimeStamp` is a FLOAT.

**appConfigs document example:**
```json
{
  "_id": "eduardo",
  "maxUserLimit": "5",
  "SubscriptionStartDate": { "_seconds": 1666027056 },
  "SubscriptionEndDate":   { "_seconds": 1714435200 },
  "apiKeys": [{ "isActive": true }],
  "streamingApiKeys": [{ "isActive": true }]
}
```
Note: `maxUserLimit` is a STRING (use `$toInt`). Dates are Firebase Firestore format `{ "_seconds": int }`.
`_id` = owner username = collection name in appConfigs.

---

## lib/llm_provider.py

**Why Ollama?**
- Runs entirely on local GPU — no cloud API calls, no cost
- JSON mode (`"format": "json"`) forces valid JSON at the token level
- Auto-manages VRAM, keeps models cached between requests
- Simple REST API alongside FastAPI

**JSON mode note:**
`format:"json"` guarantees VALID JSON, not CORRECT JSON. The retry loop in
`query_generator.py` handles structural correctness.

**Model recommendations:**
- 8GB GPU: `qwen2.5-coder:7b` (~4.7GB VRAM), `mistral:7b` (~4.1GB)
- 16GB GPU: `qwen2.5-coder:14b` (~9GB), `codestral:22b` (~13GB Q4)

**Timeout rationale:**
300 seconds: system prompt ~4,500 tokens + RAG examples + output can reach
7,000+ tokens. On 8GB GPU at 15 tok/s, a 2,048-token response = ~140 seconds.
Cold start adds 15–30s. 300s gives comfortable headroom.

---

## lib/schemas.py

**Why the system prompt is small (~750 tokens):**
Old approach: dump 4,500 tokens of schema on every query.
New approach: 750 tokens of structural rules + ~400 tokens of relevant fields
retrieved by semantic search = ~1,150 tokens total. Fits in 8192 num_ctx.

---

## lib/query_examples.py

**RAG few-shot approach:**
When a new question arrives, search the library of previous successful queries
for similar ones. The top 2–3 are prepended to the LLM prompt as demonstrations.
Seeing concrete examples of the exact pattern dramatically improves accuracy.

**Where examples come from:**
1. Automatic: every successful query (result_count > 0) is saved.
2. Manual: edit `data/query_examples.json` directly.
3. Bootstrap: `BOOTSTRAP_EXAMPLES` in the file — hand-crafted for common patterns.

**Similarity search:**
Primary: semantic vector search via `nomic-embed-text` embeddings.
Fallback: keyword overlap (Jaccard similarity) when embedding model is unavailable.

**Storage format (`data/query_examples.json`):**
```json
{
  "question": "How many sessions per country?",
  "query": { "queryType": "single", "pipeline": [...] },
  "result_count": 42,
  "timestamp": "2026-04-12T10:30:00Z",
  "db_hint": "stream"
}
```

---

## lib/result_summarizer.py

**Map-reduce approach:**
- Small results (< 16K chars total) → single LLM call.
- Large results → split into chunks of 25 docs.
- Each chunk summarized separately, then all summaries synthesized.
- Works for any size with an 8K context window.

**Why sequential chunks (not parallel)?**
Local GPU can only process one request at a time.

**Sensitive fields stripped before sending to LLM:**
`apiKeys`, `streamingApiKeys`, `elInfo.availableApps`, `elInfo.xirysobj`,
`timeRecords`, `iceConnectionStateChanges`, `candidates_selected`

---

## lib/schema_discovery.py

**Why live schema discovery?**
Static schema in `schemas.py` gets stale when new fields are added.
This module samples 10 real docs on startup and every hour, extracts all
field paths with types and example values, and injects them into prompts.

**What it detects:**
- All field paths that actually exist (with real types)
- Example values so the LLM understands the data
- New fields added to MongoDB that aren't in the static schema

**Detection flow:**
1. Sample 10 docs from stream-datastore (most recent month)
2. List appConfigs collections, sample `usersinfo` + `default` from 5 owners
3. Cache for 1 hour
4. On schema change: log added/removed fields as warnings

---

## lib/live_data_context.py

**Why this module exists:**
Without it, the LLM guesses field values ("brasil" instead of "Brazil"),
queries months that don't exist, and invents owner names.
This gives the model ground truth by caching real values from MongoDB.

**Four things cached per collection:**
1. Document samples — 3 real stripped sessions (exact field names + types)
2. Categorical values — top countries, OS, browsers, owners, apps by frequency
3. Collection list — which Month_Year collections actually exist
4. Active owners — owners with real sessions (for appConfigs cross-queries)

**TTLs:** docs=30min, values=60min, collections/owners=60min

---

## lib/embeddings.py

**Why nomic-embed-text?**
- 274 MB download, fast CPU inference
- 768-dimensional output vectors
- Runs on CPU while qwen2.5-coder:7b uses the GPU simultaneously
- Pull once: `ollama pull nomic-embed-text`

**Why embeddings beat keyword overlap:**
"Which countries had the most viewers" and "top geographic regions by user count"
share ZERO words. Keyword search returns nothing. Semantic similarity finds
the right schema fields and examples for both questions.

**Graceful fallback:**
All functions return `None` when unavailable. The app is fully functional
without the embedding model — just less accurate.

---

## lib/vector_store.py

**Two stores used:**
- `data/vectors/schema.json` → one item per schema field description
- `data/vectors/examples.json` → one item per past successful query

**Performance:**
Pure Python cosine search over 500 items with 768-dim vectors: ~5ms.
Negligible compared to the LLM call (5–30 seconds).

**Cosine similarity formula:**
`dot(a,b) / (|a| * |b|)` — range [-1, 1], 1 = identical direction.

---

## lib/db_registry.py

**Why this exists:**
Original code hardcoded two databases in `query_executor.py`.
Adding a third required editing Python files, review, and redeploy.
Now: add URI to `.env` + add entry to `data/db_registry.json` + restart.

**data/db_registry.json format:**
```json
[
  {
    "name": "stream-datastore",
    "env_uri": "MONGODB_URI_STREAM",
    "env_db_name": "STREAM_DB_NAME",
    "default_db_name": "stream-datastore",
    "description": "Monthly streaming sessions",
    "default_collection": "Apr_2026"
  }
]
```

---

## lib/session_memory.py

**Why this exists:**
Without it: User asks "filter those to Europe" and the LLM has no idea what
"those" refers to.
With it: Last MAX_TURNS (10) exchanges are included in the next prompt.
Each turn ≈ 200–500 chars → 10 turns ≈ 500 tokens. Fits in 8K context.

**No persistence:**
Memory is cleared on server restart or new session (new tab = new UUID).

---

## lib/chat_history.py

**Storage location:**
- Database: stream-datastore
- Collection: `_QUERY_HISTORY_` (underscore prefix separates it from month collections)

**Retention:**
Max 500 entries. When exceeded, oldest entries are deleted automatically.
Individual entries can be removed via the UI (DELETE /api/history/{id}).

---

## lib/collection_resolver.py

**Why this runs before the LLM:**
If the LLM guesses the collection from the question, it sometimes picks wrong months.
Running regex detection first sets the correct collection before the prompt is built.

**Supported patterns:**
- `"October 2025"` → `Oct_2025`
- `"last month"` → previous calendar month
- `"this month"` → current month
- `"april"` (no year) → April of the default collection's year
- `"oct '25"` → `Oct_2025` (2-digit year shorthand)
