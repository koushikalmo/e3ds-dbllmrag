# Eagle 3D Streaming Query System — Setup & Code Guide

A plain-English query interface for two MongoDB databases (`stream-datastore` and `appConfigs`). Type or speak a question, and the system uses Google Gemini to generate a MongoDB aggregation pipeline, run it safely, and display the results in a browser UI.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.10 works but 3.11 is recommended for performance |
| MongoDB Atlas (or local) | Any | Two separate connection strings required |
| Google Gemini API Key | — | Free at [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| Browser | Chrome or Edge | For voice input; any modern browser works for text |

---

## Project Structure

```
mongodb-llm-rag/
│
├── main.py                  FastAPI app — routes, middleware, startup/shutdown
├── requirements.txt         All Python dependencies
├── .env                     Your secrets (NEVER commit this)
├── .env.example             Template — copy this to create .env
├── CLAUDE.md                Project guide for Claude Code AI assistant
├── SETUP.md                 This file
│
├── lib/                     All backend logic lives here
│   ├── __init__.py          Makes lib/ a Python package (empty, required)
│   ├── mongodb.py           MongoDB connection manager (Motor async client)
│   ├── schemas.py           Database field descriptions fed to the LLM
│   ├── query_generator.py   Natural language → MongoDB pipeline (Gemini API)
│   └── query_executor.py    Safe pipeline runner + result serializer
│
└── static/
    └── index.html           Complete self-contained frontend (HTML + CSS + JS)
```

---

## Step 1 — Clone / Download the Code

If you're working from a git repo:
```bash
git clone <repo-url>
cd mongodb-llm-rag
```

Or just copy the folder to your machine.

---

## Step 2 — Create a Python Virtual Environment

A virtual environment keeps these dependencies isolated from your system Python.

```bash
# Create the virtual environment
python -m venv venv

# Activate it:
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows CMD
.\venv\Scripts\Activate.ps1       # Windows PowerShell
```

You should see `(venv)` in your terminal prompt. **Always activate the venv before running the server.**

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, Motor (async MongoDB driver), httpx, python-dotenv, and Pydantic. Takes about 30 seconds on a fresh venv.

---

## Step 4 — Configure Environment Variables

The server reads secrets from a `.env` file. Copy the template and fill it in:

```bash
cp .env.example .env
```

Now open `.env` in a text editor and set these values:

```env
# ── MongoDB ────────────────────────────────────────────────────────
# Your connection strings from MongoDB Atlas:
# Atlas Dashboard → Your Cluster → Connect → Drivers → Python
MONGODB_URI_STREAM=mongodb+srv://username:password@yourcluster.mongodb.net/
MONGODB_URI_APPCONFIGS=mongodb+srv://username:password@yourcluster.mongodb.net/

# Database names (leave defaults unless your DBs are named differently)
STREAM_DB_NAME=stream-datastore
APPCONFIGS_DB_NAME=appConfigs

# Default collection to query in stream-datastore
DEFAULT_STREAM_COLLECTION=Apr_2025

# Collection name inside appConfigs DB
APPCONFIGS_COLLECTION=users

# ── Gemini API ─────────────────────────────────────────────────────
# Get a free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=AIza...your_key_here

# ── Server ─────────────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
```

> **Important:** Never commit `.env` to git. It's already in `.gitignore`. The `.env.example` file (which has no real values) is safe to commit.

---

## Step 5 — Run the Server

```bash
# Development mode (auto-restarts when you edit any .py file)
python main.py

# Or directly with uvicorn:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
══════════════════════════════════════════════════════
  Eagle 3D Streaming Query System — Starting
  Listening on http://0.0.0.0:8000
  Stream DB   : stream-datastore
  AppConfigs  : appConfigs
  Gemini key  : set
══════════════════════════════════════════════════════
INFO:     Application startup complete.
```

Open your browser at **http://localhost:8000**

---

## Step 6 — Verify Everything Works

**Check the health endpoint:**
```bash
curl http://localhost:8000/api/health
```
Expected response:
```json
{
  "status": "ok",
  "databases": { "stream": "ok", "appconfigs": "ok" },
  "time": "2025-04-28T14:49:34Z"
}
```

If you see `"error: ..."` for a database, check your connection string in `.env`.

**Try a query from the UI:**
1. Open http://localhost:8000
2. Click one of the example queries on the left sidebar
3. Press ENTER or click **RUN QUERY →**
4. Results should appear in a few seconds

---

## Production Deployment

For production, use multiple worker processes:
```bash
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

Each worker runs its own Motor connection pool. With 4 workers and `maxPoolSize=10`, you have up to 40 concurrent MongoDB connections. Tune based on your Atlas tier.

For Docker deployment, a minimal `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## How Each File Works

### `main.py` — The Application Entry Point

This file creates the FastAPI app and defines the HTTP routes. It's intentionally thin — no business logic here.

**Key decisions:**
- `load_dotenv()` is called **before** any other imports. This is critical because `lib/mongodb.py` reads environment variables at import time. If you import it before loading `.env`, the MongoDB URIs will be `None`.
- The `lifespan` context manager handles startup and shutdown. Code before `yield` runs when the server starts. Code after runs when it shuts down (calls `close_connections()` to cleanly close MongoDB sockets).
- Routes delegate immediately to `lib/` functions — the route handlers contain almost no logic themselves.

**Routes:**
- `GET /` → serves `static/index.html`
- `GET /api/health` → pings both databases, returns status
- `POST /api/query` → main query flow: generate pipeline → execute → return results

---

### `lib/mongodb.py` — Database Connections

Manages two Motor (async MongoDB) client objects — one per database.

**Why Motor instead of PyMongo?**
FastAPI is async. PyMongo is synchronous — calling it inside an async route handler blocks the entire event loop, freezing all other requests during the query. Motor is the official async wrapper that integrates with asyncio via `await`.

**Why module-level singletons?**
Creating a new MongoDB client on every request is expensive (DNS lookup, TLS handshake, auth). We create one client per database on the first call and reuse it for the lifetime of the server. This is the standard pattern for async MongoDB in Python.

**Key functions:**
- `get_stream_db()` → returns the Motor database for `stream-datastore`
- `get_appconfigs_db()` → returns the Motor database for `appConfigs`
- `close_connections()` → called at shutdown to release sockets
- `ping_databases()` → health check, called by `/api/health`

---

### `lib/schemas.py` — LLM Knowledge Base

This is the most important file for query quality. It contains text descriptions of every database field, including types, examples, and query tips. This text is injected into the Gemini system prompt so the model knows what fields exist and how to use them.

**The two databases described:**

`STREAM_SCHEMA` — describes `stream-datastore`:
- Session identifiers, timing fields, geographic data
- Device information (OS, browser, device type)
- WebRTC video quality stats (bitrate, FPS, RTT, packet loss)
- Streaming server hardware info (GPU, CPU, RAM)
- Special notes: `avgRoundTripTime` is a **string** (not float), session duration must be computed by subtracting timestamps

`APPCONFIGS_SCHEMA` — describes `appConfigs`:
- Owner username (_id), subscription dates, max user limits
- Notes: `maxUserLimit` is stored as a **string** (needs `$toInt`)
- Subscription dates are Firebase Firestore timestamps: `{ "_seconds": int }` format
- API keys should never be returned in results

`CROSS_DB_HINT` — explains how to write dual queries since MongoDB can't `$lookup` across two cluster URIs.

`build_system_prompt()` assembles only the relevant schema sections based on which databases the question needs. This keeps prompts lean and focused.

`STREAM_KEYWORDS` and `APPCONFIGS_KEYWORDS` — word sets used to detect which database(s) a question is about.

**How to update the schema:**
If the LLM generates pipelines with wrong field names, the schema is out of date. Check a real document in MongoDB Compass, find the correct field name, and update it here.

---

### `lib/query_generator.py` — Natural Language to Pipeline

This module calls the Gemini API to convert a plain-English question into a MongoDB aggregation pipeline.

**The full flow:**

1. **`detect_relevant_databases(question)`**
   Scans the question for keywords from `STREAM_KEYWORDS` and `APPCONFIGS_KEYWORDS`. Returns `(needs_stream, needs_appconfigs)`. Defaults to stream-only if nothing matches.

2. **`build_system_prompt()`**
   Assembles the system prompt with only the relevant schema(s). The fewer tokens in the prompt, the faster and more focused the response.

3. **`_call_gemini(system_prompt, user_message)`**
   Makes an `async` HTTP POST to the Gemini REST API using `httpx.AsyncClient`. Uses `temperature=0.1` for near-deterministic JSON output. The `user_message` includes the current time (so "this month" works) and the selected collection name.

4. **`_extract_json(raw)`**
   Strips markdown code fences if Gemini wrapped the JSON. Parses the text as JSON.

5. **`_validate_query_object(obj)`**
   Checks the parsed JSON has the required fields (`database`, `collection`, `pipeline` for single queries; `queries` array for dual queries).

**Changing the Gemini model:**
Edit `GEMINI_MODEL = "gemini-1.5-flash"` to `"gemini-1.5-pro"` for better accuracy on complex multi-stage pipelines. Flash is faster and free; Pro is slower but smarter.

---

### `lib/query_executor.py` — Safe Query Runner

Takes the validated query dict and runs it against the real MongoDB database(s).

**Safety features:**

- `_sanitize_pipeline()` — strips `$out` and `$merge` stages (write protection)
- `_enforce_limit()` — ensures a `$limit` ≤ 200 always exists
- `allowDiskUse=True` — prevents large sort failures on big collections
- `maxTimeMS=15000` — MongoDB kills the query server-side after 15 seconds

**Type conversion:**
Motor returns documents with MongoDB-specific types (`ObjectId`, `Decimal128`) that Python's `json` module can't serialize. `_make_serializable()` recursively converts these to strings and floats.

**Dual query execution:**
Both pipelines are launched simultaneously with `asyncio.gather()`. This cuts total wait time from `t1 + t2` to `max(t1, t2)`. After both finish, results are optionally merged in Python by the owner key (`appInfo.owner` ↔ `_id`).

---

### `static/index.html` — The Complete Frontend

A single self-contained HTML file with no build step, no npm, no frameworks. Plain HTML + CSS + JavaScript.

**UI sections:**
- **Header** — logo, month collection dropdown, live status dot
- **Sidebar** — example queries (click to fill textarea), recent history
- **Input area** — textarea, voice button, run button
- **Loading state** — three bouncing dots while Gemini + MongoDB work
- **Error state** — red box with the error message
- **Results panel** — three tabs: TABLE, JSON, PIPELINE

**Table rendering:**
Documents from MongoDB are deeply nested. The `flattenDoc()` function converts nested objects to dot-notation keys (e.g. `clientInfo.city`). Priority columns are shown first; the rest follow.

**Voice input:**
Uses the browser's built-in `SpeechRecognition` API. Runs locally — no external service, no cost. Works in Chrome and Edge. After the user stops speaking, the transcribed text is automatically submitted.

**Tab views:**
- **TABLE** — flattened key-value table, best for scanning many records
- **JSON** — raw document structure, useful for seeing all fields
- **PIPELINE** — the exact MongoDB aggregation stages that ran, with a copy button (useful for pasting into MongoDB Compass)

---

## Common Tasks

### Add a new month to the dropdown

Open `static/index.html`, find the `<select id="collectionSelect">` element, add:
```html
<option value="May_2025">May 2025</option>
```

### Fix the LLM generating wrong field names

The schema in `lib/schemas.py` is out of date. Find the actual field name in a real document using MongoDB Compass or:
```bash
# In mongo shell:
db.Apr_2025.findOne({}, { "fieldName": 1 })
```
Then update the relevant section in `STREAM_SCHEMA` or `APPCONFIGS_SCHEMA`.

### Add a new field the LLM should know about

In `lib/schemas.py`, find the correct NESTED section and add a line:
```
fieldName.subField  (type)  Description of what this contains.
```
If the field needs special handling (stored as string, needs conversion, etc.), add a QUERY TIP.

### Change from Gemini Flash to Gemini Pro (better accuracy)

In `lib/query_generator.py`, change:
```python
GEMINI_MODEL = "gemini-1.5-flash"
```
to:
```python
GEMINI_MODEL = "gemini-1.5-pro"
```
Pro is slower (2–5x) but handles complex multi-stage pipelines better.

### Debug a bad pipeline

1. Run the query and click the **PIPELINE** tab
2. Click **COPY PIPELINE**
3. Paste into MongoDB Compass → Aggregation view
4. Run it there to see the raw error
5. Once you understand the problem, add a tip to the relevant section in `lib/schemas.py`

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `MONGODB_URI_STREAM is not set` | `.env` file missing or not loaded | Copy `.env.example` to `.env` and fill in values |
| `HTTP 503` on `/api/health` | MongoDB unreachable | Check URI in `.env`, verify Atlas IP whitelist includes your IP |
| `Gemini API returned HTTP 400` | Invalid API key | Check `GEMINI_API_KEY` in `.env` |
| `Gemini returned invalid JSON` | Model hallucinated non-JSON | Usually self-correcting; retry. If persistent, add examples to `build_system_prompt()` |
| Voice button doesn't work | Wrong browser | Use Chrome or Edge (Firefox doesn't support `SpeechRecognition`) |
| Voice button says "permission denied" | Microphone blocked | Click the camera icon in the browser address bar and allow microphone |
| Table shows `[1 items]` for a column | That field is an array | Use the JSON tab to see the full array content |
| Query times out | Pipeline scanning too much data | Add a `$match` filter at the start of the question (e.g., add an owner name or date) |
| `NamespaceNotFound` error | Collection doesn't exist | Select the correct month in the dropdown |

---

## Architecture Overview

```
Browser (index.html)
      │  POST /api/query { question, collection }
      ▼
FastAPI (main.py)
      │  generate_query(question, collection)
      ▼
query_generator.py
  1. detect_relevant_databases(question) → which DB(s)?
  2. build_system_prompt(schema) → system instructions
  3. POST to Gemini API → raw JSON text
  4. _extract_json() → parse JSON
  5. _validate_query_object() → check structure
      │  returns validated query dict
      ▼
query_executor.py
  1. _sanitize_pipeline() → strip $out, $merge
  2. _enforce_limit() → cap at 200 docs
  3. Motor aggregate() → MongoDB query
     (dual queries: asyncio.gather for parallel execution)
  4. _make_serializable() → fix ObjectId/Decimal128
  5. _summarize() → count + preview
      │  returns results dict
      ▼
FastAPI (main.py)
  → JSONResponse with results + meta
      │
      ▼
Browser (index.html)
  → Renders TABLE / JSON / PIPELINE view
```

---

## How the Two Databases Connect

The join key between the databases is the **owner username**:

```
stream-datastore.Apr_2025
  └─ document.appInfo.owner = "eduardo"
                                    │
                                    └─── matches ───┐
                                                     ▼
                                    appConfigs.users
                                      └─ document._id = "eduardo"
```

Because these are on different MongoDB connections (or clusters), a standard `$lookup` won't work. Instead, when a question needs both, the system runs two separate pipelines concurrently and merges the results in Python by matching the owner strings.
