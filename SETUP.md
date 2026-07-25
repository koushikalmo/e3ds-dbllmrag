# Setup, Run & Deploy — Eagle 3D Streaming Query System

## 1. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10+ (3.11 recommended) | Async + typing features |
| MongoDB | 5.0+ | Atlas or self-hosted. Read access to `stream-datastore` and `appConfigs` |
| Ollama | latest | Runs the local LLM. https://ollama.com |
| GPU (optional) | 8 GB VRAM | CPU works but slow; 8 GB → `qwen2.5-coder:7b`, 16 GB → `qwen2.5-coder:14b` |
| Disk | ~15 GB | Models + embeddings cache |
| Browser | Chrome / Edge | Voice input only — any modern browser works for text |

---

## 2. One-time setup

### 2.1 Clone and enter the repo

```bash
git clone <repo-url> mongodb-llm-rag
cd mongodb-llm-rag
```

### 2.2 Create and activate a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows (cmd)
venv\Scripts\activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```


### 2.3 Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2.4 Install and start Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama daemon
ollama serve
```

### 2.5 Pull the models

```bash
#We can use any local model (Gemma, Qwen etc)
ollama pull qwen2.5-coder:7b      # query generator
ollama pull nomic-embed-text      # embeddings for RAG
```

Confirm:

```bash
ollama list
curl http://localhost:11434/api/tags    # JSON with both models listed
```

### 2.6 Create `.env`

```bash
cp .env.example .env
```

```env
# For MongoDB 
MONGODB_URI_STREAM=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGODB_URI_APPCONFIGS=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority
STREAM_DB_NAME=stream-datastore
APPCONFIGS_DB_NAME=appConfigs
DEFAULT_STREAM_COLLECTION=

# For Ollama 
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_NUM_CTX=8192
LLM_MAX_RETRIES=3

# For Server 

HOST=0.0.0.0
PORT=8000

# For Trainer mode (optional/ We can skip this) 
# If set, POST /api/train/login with this password unlocks
# the trainer endpoints (which can mutate RAG + eval_set).
TRAINER_PASSWORD=

# For Schema change-stream watcher 
# Watches both databases with MongoDB change streams and
# refreshes the schema cache whenever a new/removed field
# shows up. Requires a replica set or sharded cluster —
# change streams don't work on a standalone. If unsupported,
# the watcher logs a warning and exits; hourly TTL sampling
# in schema_discovery still runs.
#
# SCHEMA_WATCHER_ENABLED — 1 to run, 0 to disable entirely.
# SCHEMA_WATCHER_WAKE_SEC — how often the refresher loop wakes
#     up and checks the dirty flag. Keep small (seconds).
# SCHEMA_WATCHER_COOLDOWN_SEC — minimum gap between two forced
#     refreshes. 3600 = at most one refresh per hour, no matter
#     how many change events arrive.
SCHEMA_WATCHER_ENABLED=1
SCHEMA_WATCHER_WAKE_SEC=60
SCHEMA_WATCHER_COOLDOWN_SEC=3600
```


### 2.7 Verify MongoDB access

- Required read: every collection in `stream-datastore` and `appConfigs`
- Required write: `_QUERY_HISTORY_`, `_QUERY_FEEDBACK_`, `_SHARED_CHATS_`
  (created on first write). These live in `stream-datastore` by default.


## 3. Run in development

```bash
source venv/bin/activate      # if not already
python main.py
# or:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On a clean boot we can see:

```
  Eagle 3D Streaming Query System — Starting
  Listening on http://0.0.0.0:8000
  Stream DB   : stream-datastore
  AppConfigs  : appConfigs
  #We can add any local model(Qwen, Gemma, etc)
  Ollama      : http://localhost:11434 · qwen2.5-coder:7b

INFO:     Application startup complete.
```


### 3.1 Test the server

```bash
# DB + schema cache health
curl -s http://localhost:8000/api/health | jq

# Ollama + RAG status
curl -s http://localhost:8000/api/status | jq

# Real query
curl -s -X POST http://localhost:8000/api/query \
  -H 'content-type: application/json' \
  -d '{"question":"How many sessions in Apr_2026?","collection":"Apr_2026","session_id":"smoke"}' | jq '.queryMeta, .results | length'
```

## 4. Production deployment


### 4.1 systemd (single host, baremetal / VM)



**`/etc/systemd/system/eagle-api.service`**

```ini
[Unit]
Description=Eagle 3D Streaming Query System (FastAPI)
After=network-online.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=eagle
WorkingDirectory=/opt/eagle/mongodb-llm-rag
EnvironmentFile=/opt/eagle/mongodb-llm-rag/.env
ExecStart=/opt/eagle/mongodb-llm-rag/venv/bin/uvicorn main:app \
  --host 0.0.0.0 --port 8000 --workers 4 \
  --log-level info --access-log
Restart=on-failure
RestartSec=3

# Resource guardrails
MemoryMax=4G
CPUQuota=400%

[Install]
WantedBy=multi-user.target
```

**`/etc/systemd/system/ollama.service`** (if not installed by the Ollama installer)

```ini
[Unit]
Description=Ollama local LLM
After=network-online.target

[Service]
Type=simple
User=ollama
Environment=OLLAMA_HOST=127.0.0.1:11434
Environment=OLLAMA_KEEP_ALIVE=24h
ExecStart=/usr/local/bin/ollama serve
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ollama eagle-api
sudo systemctl status eagle-api
journalctl -u eagle-api -f
```

### 4.2 Docker / Docker Compose

**`Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**`docker-compose.yml`**

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    restart: unless-stopped
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "127.0.0.1:11434:11434"
    # If you have an NVIDIA GPU, uncomment:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - capabilities: [gpu]

  api:
    build: .
    restart: unless-stopped
    env_file: .env
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    depends_on:
      - ollama
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data     # persist RAG store + vectors across restarts

volumes:
  ollama_models:
```

One-time model pull inside the container:

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5-coder:7b
docker compose exec ollama ollama pull nomic-embed-text
docker compose up -d api
```

### 4.3 Reverse proxy (nginx)

Put the API behind nginx for TLS and HTTP/2.

```nginx
server {
    listen 443 ssl http2;
    server_name query.example.com;

    ssl_certificate     /etc/letsencrypt/live/query.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/query.example.com/privkey.pem;

    client_max_body_size 25M;         # voice uploads

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        # Streaming + long LLM waits
        proxy_read_timeout   180s;
        proxy_send_timeout   180s;
        proxy_buffering      off;
    }
}
```

## 5. API'S

### 5.1 Health & observability

| Signal | Where |
|---|---|
| DB + schema cache | `GET /api/health` |
| Ollama up/down + model + RAG size | `GET /api/status` |
| Change-stream watcher (running? dirty? last refresh?) | `GET /api/schema/watcher` |
| RAG weight distribution / top-N | `GET /api/rag/stats` |
| Last regression result | `GET /api/eval/baseline` |
| Full stack traces | server stdout (`journalctl -u eagle-api` / `docker logs`) |

### 6.2 Keeping the schema fresh

Two mechanisms keep the LLM's field list in sync with the real database:

1. **Hourly sampling** — `schema_discovery` re-samples both databases every
   hour on a timer. This runs regardless of cluster topology and is the
   baseline safety net.
2. **Change-stream watcher** — `lib/schema_watcher.py` opens a MongoDB
   change stream on each database at startup. When a document is inserted,
   updated or replaced with a field the current schema cache doesn't know
   about (or when a collection is dropped/renamed), it flips a dirty flag.
   A separate refresher loop wakes every `SCHEMA_WATCHER_WAKE_SEC` and, if
   the flag is set and the cooldown has expired, calls the same refresh
   routine — capped at one refresh per `SCHEMA_WATCHER_COOLDOWN_SEC` no
   matter how many events arrive in between.

The watcher needs a replica set or sharded cluster (change streams don't
exist on a standalone). On an unsupported cluster it logs a warning and
quietly disables itself; the hourly sampling in (1) still runs, so the
schema stays reasonable — just up to an hour behind reality instead of a
few minutes.

Check the watcher's state at any time:

```bash
curl -s http://localhost:8000/api/schema/watcher | jq
# { "watcher": { "running": true, "dirty": false, "last_refresh_ago": 812.4, ... },
#   "schema_cache": { "populated": true, "sampled_at": "...", ... } }
```

Force a refresh manually (useful right after a schema migration or bulk
import, when you don't want to wait for the timer):

```bash
curl -X POST http://localhost:8000/api/schema/refresh
```

Also regenerate the digest (top-K owner/country/app values) after
large data imports:

```bash
curl -X POST http://localhost:8000/api/digest/refresh
```

### 6.3 Managing the RAG store

```bash
curl -s http://localhost:8000/api/rag/stats | jq
curl -s -X POST http://localhost:8000/api/rag/prune -H 'content-type: application/json' \
  -d '{"max_age_days":90, "min_weight":0.2}' | jq
```

Pruning removes entries that are old and never used. Trainer-sourced
"gold" entries are protected.

### 6.4 Trainer workflow

```bash
# Unlock
curl -X POST http://localhost:8000/api/train/login \
  -H 'content-type: application/json' \
  -d '{"password":"<TRAINER_PASSWORD>"}'

# Re-run a question under trainer mode (returns accuracy breakdown)
curl -X POST http://localhost:8000/api/train/run \
  -H 'content-type: application/json' \
  -d '{"question":"Top 5 countries by session count in Apr_2026","collection":"Apr_2026"}'

# Save a correction as gold (appends to eval_set + bumps RAG + records feedback)
curl -X POST http://localhost:8000/api/train/save \
  -H 'content-type: application/json' \
  -d '{"question":"…","original_query":{…},"corrected_query":{…},"notes":"…"}'
```

### 6.5 Backups

Back up `data/` nightly. The critical files:

- `data/query_examples.json` — learned memory (most valuable)
- `data/eval_set.json` — regression suite
- `data/eval_report.json` — last baseline
- `data/vectors/` — serialised embeddings (regenerable but expensive)

MongoDB auxiliary collections (`_QUERY_HISTORY_`, `_QUERY_FEEDBACK_`,
`_SHARED_CHATS_`) live in your Atlas backups.