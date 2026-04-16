# CLAUDE.md — Eagle 3D Streaming Query System

This file is read by Claude Code to understand the project.
Whenever you ask Claude a question in this workspace, it reads this file first.

---

## Project Summary

A Python/FastAPI backend that lets users query two MongoDB databases
using **plain English or voice**. A local Ollama LLM converts the question
into a MongoDB aggregation pipeline, runs it safely, and returns results
in a dark-themed browser UI. No cloud API keys required.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web server | FastAPI + uvicorn |
| Async MongoDB driver | Motor 3 (wraps PyMongo with asyncio) |
| LLM for query generation | Ollama (local) — `qwen2.5-coder:7b` default |
| HTTP client | httpx (async) |
| Config | python-dotenv + `.env` file |
| Frontend | Vanilla HTML/CSS/JS served by FastAPI |

---

## Project Structure

```
mongodb-llm-rag/
├── main.py                    FastAPI app, all routes, startup/shutdown
├── requirements.txt           All Python dependencies
├── .env                       Secrets (never commit — copy from .env.example)
├── .env.example               Template for .env (Ollama-only, no cloud keys)
├── CLAUDE.md                  This file
│
├── lib/
│   ├── __init__.py
│   ├── mongodb.py             Async Motor connection manager for both DBs
│   ├── schemas.py             Schema text descriptions fed to the LLM
│   ├── llm_provider.py        Ollama client (local LLM, JSON mode)
│   ├── query_generator.py     NL → MongoDB pipeline (RAG + retry loop)
│   ├── query_executor.py      Safe async pipeline executor
│   ├── result_summarizer.py   Chunked map-reduce summarization for /api/analyze
│   ├── schema_discovery.py    Live schema extraction from real MongoDB data
│   └── query_examples.py      RAG few-shot example store
│
├── data/
│   └── query_examples.json    Persistent RAG example library (auto-managed)
│
└── static/
    └── index.html             Complete self-contained frontend (HTML+CSS+JS)
```

---

## The Two Databases

### `stream-datastore`
- Collections named by month: `Apr_2026`, `Mar_2026`, ..., `Dec_2023`
- Each document = one streaming session heartbeat
- Key fields: `appInfo.owner`, `appInfo.appName`, `clientInfo.city`,
  `clientInfo.country_name`, `userDeviceInfo.os.name`, `loadTime`,
  `webRtcStatsData.*`, `elInfo.systemInfo.*`
- **Timestamps are FLOATS** (not ints): `startTimeStamp`, `DisconnectTime_Timestamp`
- Session duration: `DisconnectTime_Timestamp - startTimeStamp` (seconds)

### `appConfigs`
- **CRITICAL: NOT a `users` collection.** Each collection = one owner account.
- Collection name IS the owner's username (e.g. `"eduardo"`, `"imerza"`)
- ~5,500 collections total, one per owner
- Each owner collection has up to 4 document types (by `_id`):
  - `"usersinfo"` — billing/subscription (`maxUserLimit` INT, `SubscriptionEndDate._seconds`, etc.)
  - `"InfoToConstructUrls"` — streaming URL config
  - `"default"` or `"<appName>"` — streaming app configuration
- **Cross-DB join**: `appInfo.owner` (stream) ↔ collection name (appConfigs)

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serves `static/index.html` |
| GET | `/api/health` | Pings both MongoDB databases + schema cache status |
| GET | `/api/status` | Shows Ollama LLM status + RAG example count |
| POST | `/api/query` | Main query endpoint (NL → pipeline → results) |
| POST | `/api/analyze` | AI analysis of query results (chunked map-reduce) |
| POST | `/api/schema/refresh` | Force-refresh the live schema discovery cache |

---

## How the Query Pipeline Works

```
User question (text or voice)
        │
        ▼
query_generator.py
  1. Keyword scan → which DB(s) does this question need?
  2. RAG lookup → find 2-3 similar past examples from data/query_examples.json
  3. Build system prompt: static schema + live schema supplement
  4. Send to Ollama (format:"json" forces valid JSON at token level)
  5. Parse + validate JSON query object
  6. Field name validation → correction retry loop (up to 3 attempts)
        │
        ▼
query_executor.py
  1. Strip $out / $merge stages (read-only safety)
  2. Enforce $limit ≤ 200
  3. Run aggregation with maxTimeMS=15000 + allowDiskUse=True
  4. For dual queries: asyncio.gather() runs both concurrently
  5. Optionally merge results in Python by owner key
  6. Convert BSON types (ObjectId, Decimal128) to JSON-safe types
        │
        ▼
main.py /api/query route
  → Saves successful query to RAG store (auto-learning)
  → Returns JSON response to the browser
        │
        ▼
static/index.html
  → Renders as table / JSON / pipeline view
```

### Schema Discovery (live database sampling)
`lib/schema_discovery.py` samples real MongoDB documents on startup and
every hour. The discovered fields are injected into the LLM prompt as a
live supplement, ensuring the model knows about new fields even if
`lib/schemas.py` hasn't been updated yet.

### RAG Few-Shot Examples
`lib/query_examples.py` stores every successful query in
`data/query_examples.json`. When a new question comes in, similar past
examples are retrieved and prepended to the LLM prompt as demonstrations.
This "few-shot" approach dramatically improves accuracy for complex patterns.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MONGODB_URI_STREAM` | ✅ | MongoDB connection for stream-datastore |
| `MONGODB_URI_APPCONFIGS` | ✅ | MongoDB connection for appConfigs |
| `STREAM_DB_NAME` | — | Database name (default: `stream-datastore`) |
| `APPCONFIGS_DB_NAME` | — | Database name (default: `appConfigs`) |
| `DEFAULT_STREAM_COLLECTION` | — | Default month (default: `Apr_2026`) |
| `OLLAMA_BASE_URL` | — | Ollama URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | — | Model name (default: `qwen2.5-coder:7b`) |
| `OLLAMA_NUM_CTX` | — | Context window tokens (default: `8192`) |
| `LLM_MAX_RETRIES` | — | Correction attempts on bad output (default: `3`) |
| `HOST` | — | Bind address (default: `0.0.0.0`) |
| `PORT` | — | Port (default: `8000`) |

---

## Common Tasks

### Add a new monthly collection to the UI dropdown
Edit `static/index.html`, find `<select id="collectionSelect">`,
add `<option value="May_2026">May 2026</option>` at the top.

### Add a new field to help the LLM query it correctly
Edit `lib/schemas.py` — find the right schema string, add the field with
its real type and description. Or wait — `schema_discovery.py` will
detect it automatically from live data within 1 hour.

### The LLM is generating wrong field names
1. Run the query in the UI → click PIPELINE tab → copy it
2. Paste into MongoDB Compass to reproduce
3. Check `lib/schemas.py` against a real document
4. Add a corrected example to `data/query_examples.json` for RAG

### The LLM struggles with a specific query pattern
Add a hand-crafted example to `BOOTSTRAP_EXAMPLES` in `lib/query_examples.py`.
This becomes a permanent few-shot demonstration.

### Force schema cache refresh (after database structure changes)
```
POST /api/schema/refresh
```
Or restart the server — it re-samples on every startup.

### Change the Ollama model
Edit `OLLAMA_MODEL=qwen2.5-coder:14b` in `.env` (after pulling it).
Upgrade from 7b → 14b after GPU upgrade to 16GB.

### Run the server
```bash
# Development (auto-reloads on file changes)
python main.py

# Or explicitly with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production (4 workers)
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

---

## Safety Constraints

- All queries are **read-only** — `$out` and `$merge` stages are stripped
- Hard cap: **200 documents** per result set
- **15-second timeout** per query (MongoDB maxTimeMS)
- API keys in `appConfigs` documents are never returned in results
- `.env` is never committed to git
- No cloud API calls — all LLM inference is local via Ollama

---

## Coding Conventions

- All functions that touch MongoDB must be `async def`
- Use type hints on all function signatures
- Keep Pydantic models in `main.py` (they're small)
- Error messages to the client should be human-readable — no stack traces
- Full stack traces go to server stdout only
- All new Python files go in `lib/` with a module docstring explaining purpose
