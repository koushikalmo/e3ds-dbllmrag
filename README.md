# Eagle 3D Streaming Query System

Ask questions about two MongoDB databases in plain English (or voice) and get
real query results back. A local Ollama model writes the aggregation
pipeline, the backend runs it safely, and a dark-themed web UI shows the
results with an AI summary. Nothing leaves your machine — no cloud API keys.

## TLDR — run it

```bash
# 1. Python deps
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Local LLM
ollama serve                      # install from https://ollama.com if missing
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text      # used for RAG embeddings

# 3. Config
cp .env.example .env              # then fill in your two MongoDB URIs

# 4. Go
python main.py
```

Open http://localhost:8000 and ask something like *"top 5 countries by
session count in April"*.

## What you need

- Python 3.10+
- MongoDB with read access to `stream-datastore` and `appConfigs`
- Ollama running locally (a GPU with 8 GB VRAM is enough for the 7b model;
  CPU works but expect each query to take a minute or more)

## Config

Everything lives in `.env` (copy from `.env.example`). The two that are
actually required:

| Variable | What it is |
|---|---|
| `MONGODB_URI_STREAM` | Connection string for stream-datastore |
| `MONGODB_URI_APPCONFIGS` | Connection string for appConfigs |

Useful optional ones: `DEFAULT_STREAM_COLLECTION` (which month to query when
the question doesn't say), `OLLAMA_MODEL` (swap models after `ollama pull`),
`TRAINER_PASSWORD` (unlocks trainer mode in the UI), `PORT` (default 8000).

## Running it other ways

```bash
# dev with auto-reload (same as python main.py)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# production
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

Quick health check once it's up:

```bash
curl http://localhost:8000/api/health    # both DBs + cache status
curl http://localhost:8000/api/status    # is Ollama up, which model
```

First startup is slower than usual — the server warms the model, samples the
databases for live schema, and embeds the RAG examples in the background.
The API works during warmup; queries just get smarter once it finishes.

## Checking that changes didn't break anything

```bash
python3 scripts/evaluate.py        # replays data/eval_set.json against the running server
```

Writes `data/eval_report.json` with a pass/fail per case and a delta vs. the
last run.

## More docs

- `system_overview.md` — start here: run steps, what each file does, and a
  worked example of a query going through the system
- `working_principle.md` — how the whole pipeline works, file by file,
  function by function, in execution order