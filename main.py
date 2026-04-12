# ============================================================
# main.py — FastAPI Application Entry Point
# ============================================================
# Wires everything together: routes, middleware, startup/shutdown.
#
# STARTUP SEQUENCE:
#   1. Connect to MongoDB (lazy — first request triggers it)
#   2. Run live schema discovery (samples both databases to
#      extract real field paths and cache them for 1 hour)
#   3. Start serving requests
#
# KEY ROUTES:
#   GET  /              → serves the browser UI
#   GET  /api/health    → pings both MongoDB databases
#   GET  /api/status    → shows Ollama LLM status
#   POST /api/query     → main query endpoint (NL → MongoDB → results)
#   POST /api/analyze   → AI analysis of query results with chunking
# ============================================================

import os
import time
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
# CRITICAL: load .env BEFORE importing lib modules — they read
# env vars at import time (e.g. OLLAMA_BASE_URL = os.getenv(...)).
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lib.mongodb           import close_connections, ping_databases
from lib.query_generator   import generate_query, save_successful_query
from lib.query_executor    import execute_query
from lib.result_summarizer import summarize_results
from lib.llm_provider      import OllamaProvider, OLLAMA_MODEL
from lib.schema_discovery  import refresh_schema_cache, get_cache_status
from lib.query_examples    import get_example_count
from lib.chat_history      import save_query, get_history, delete_entry, clear_all
from lib.query_examples    import index_all_examples_async
from lib.session_memory    import add_turn, get_context_text, clear_session, active_session_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────
    default_collection = os.getenv("DEFAULT_STREAM_COLLECTION", "Apr_2025")

    print("═" * 56)
    print("  Eagle 3D Streaming Query System v3")
    print(f"  URL: http://{os.getenv('HOST','0.0.0.0')}:{os.getenv('PORT','8000')}")
    print(f"  LLM: LOCAL Ollama ({OLLAMA_MODEL})")
    print(f"  Stream DB : {os.getenv('STREAM_DB_NAME', 'stream-datastore')}")
    print(f"  AppConfigs: {os.getenv('APPCONFIGS_DB_NAME', 'appConfigs')}")
    print(f"  RAG examples: {get_example_count()} in store")
    print("═" * 56)

    # Kick off live schema discovery in the background.
    # This also triggers embedding of discovered fields into the
    # vector store (if nomic-embed-text is available in Ollama).
    asyncio.create_task(
        refresh_schema_cache(stream_collection=default_collection)
    )

    # Index all stored examples (bootstrap + past queries) into the
    # vector store so semantic search is ready from the first query.
    # No-op if embeddings are unavailable — keyword search is used instead.
    asyncio.create_task(index_all_examples_async())

    yield

    # ── Shutdown ──────────────────────────────────────────────
    await close_connections()
    print("[shutdown] MongoDB connections closed.")


app = FastAPI(
    title       = "Eagle 3D Streaming Query API v3",
    description = "Natural language analytics with local Ollama LLM",
    version     = "3.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["GET", "POST"],
    allow_headers = ["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request models ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str = Field(..., min_length=1, max_length=500)
    collection: str = Field("Apr_2025")
    session_id: str = Field("", description="Optional session ID for conversation context. Generate a UUID in the frontend and send it with every request in the same chat session. Omit (or send empty string) for stateless one-off queries.")


class AnalyzeRequest(BaseModel):
    """
    Body for POST /api/analyze.

    results:  The array of MongoDB documents to analyze.
              Same 'results' array from a /api/query response.
    question: The original question that produced these results.
              Gives the LLM context for what to focus on.
    """
    results:  list = Field(..., description="Query results from /api/query")
    question: str  = Field(..., min_length=1, max_length=500,
                           description="The question these results answer")


# ── Routes ─────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health_check():
    """
    Pings both MongoDB databases.
    Returns HTTP 200 if both are reachable, 503 if either is not.
    Also reports schema discovery cache status.
    """
    db_status = await ping_databases()
    all_ok    = all(v == "ok" for v in db_status.values())

    return JSONResponse(
        content={
            "status":    "ok" if all_ok else "degraded",
            "databases": db_status,
            "schema_cache": get_cache_status(),
            "rag_examples": get_example_count(),
            "time":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        status_code=200 if all_ok else 503,
    )


@app.get("/api/status")
async def llm_status():
    """
    Returns information about the Ollama LLM provider:
    whether it's running, which model is loaded, and RAG/schema stats.

    Used by the frontend to show the LOCAL / OFFLINE indicator.

    Response example:
        {
          "provider":  "ollama",
          "model":     "qwen2.5-coder:7b",
          "available": true,
          "schema_cache": { "populated": true, "sampled_at": "..." },
          "rag_examples": 42
        }
    """
    ollama    = OllamaProvider()
    available = await ollama.is_available()

    return JSONResponse(content={
        "provider":     "ollama",
        "model":        OLLAMA_MODEL,
        "available":    available,
        "schema_cache": get_cache_status(),
        "rag_examples": get_example_count(),
    })


@app.post("/api/query")
async def run_query(body: QueryRequest):
    """
    Main query endpoint. Converts a plain-English question into
    a MongoDB aggregation pipeline and returns the results.

    Flow:
      1. generate_query()  → Ollama generates the pipeline
                             (with RAG examples + live schema + retry loop)
      2. execute_query()   → runs it safely against MongoDB
      3. save_successful_query() → saves to RAG store for future use
      4. Returns results as JSON

    POST body:  { "question": "...", "collection": "Apr_2025" }

    Success: { "success": true, "data": { results, summary, meta, ... } }
    Error:   { "success": false, "error": "human-readable message" }
    """
    start = time.perf_counter()

    try:
        # Retrieve conversation context from this session (if any).
        # Empty string for first query in a session or stateless mode.
        conversation_ctx = get_context_text(body.session_id)

        query_obj = await generate_query(
            question         = body.question.strip(),
            collection       = body.collection,
            conversation_ctx = conversation_ctx,
        )

        result  = await execute_query(query_obj)
        elapsed = round(time.perf_counter() - start, 2)

        # ── Auto-save to RAG example store ─────────────────────
        # Save the successful query so future similar questions
        # can use it as a few-shot example. We do this after
        # getting the result so we know how many docs were returned.
        result_count = 0
        if result.get("queryType") == "single":
            result_count = len(result.get("results", []))
        elif result.get("queryType") == "dual":
            primary = result.get("results", {})
            if isinstance(primary, dict):
                result_count = len(primary.get("primary", []))

        if result_count > 0:
            # Record this exchange in session memory so follow-up questions
            # can refer to "that", "those results", "now filter by...", etc.
            if body.session_id:
                answer_summary = (
                    f"Returned {result_count} results — "
                    f"'{query_obj.get('resultLabel', 'Results')}'"
                )
                add_turn(body.session_id, body.question.strip(), answer_summary)

            # Save to RAG example store (improves future query generation)
            save_successful_query(
                question     = body.question.strip(),
                query_obj    = query_obj,
                result_count = result_count,
            )
            # Save to MongoDB chat history (user-facing history panel)
            try:
                await save_query(
                    question        = body.question.strip(),
                    collection      = body.collection,
                    result_count    = result_count,
                    result_label    = query_obj.get("resultLabel", ""),
                    explanation     = query_obj.get("explanation", ""),
                    elapsed_seconds = round(time.perf_counter() - start, 2),
                )
            except Exception as e:
                # History save failure is non-fatal — the query result still returns
                print(f"[/api/query] History save failed (non-fatal): {e}")

        return JSONResponse(content={
            "success": True,
            "data": {
                **result,
                "meta": {
                    "question":       body.question.strip(),
                    "collectionUsed": body.collection,
                    "elapsedSeconds": elapsed,
                    "generatedAt":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            },
        })

    except TimeoutError as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=504,
        )
    except ValueError as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=400,
        )
    except Exception as e:
        import traceback
        print(f"[/api/query] Unhandled exception:\n{traceback.format_exc()}")
        return JSONResponse(
            content={"success": False, "error": f"Server error: {str(e)[:200]}"},
            status_code=500,
        )


@app.post("/api/analyze")
async def analyze_results(body: AnalyzeRequest):
    """
    AI-powered analysis of query results.

    Accepts the results from a previous /api/query call and asks
    Ollama to analyze / summarize them. Uses a chunked map-reduce
    approach so it works correctly even when the result set exceeds
    the LLM's context window.

    HOW CHUNKING HANDLES LARGE RESULTS:
    ─────────────────────────────────────
    Small results (< 16K chars total) → single LLM call.
    Large results → split into chunks of 25 docs.
      Each chunk summarized separately, then all summaries
      synthesized into a final answer. Works for any size.

    POST body:
      {
        "results":  [ ...the results array from /api/query... ],
        "question": "Which cities had the most sessions?"
      }

    Success:
      {
        "success": true,
        "data": {
          "summary":        "Brazil dominated with 45%...",
          "method":         "chunked",
          "chunksUsed":     6,
          "docsAnalyzed":   150,
          "elapsedSeconds": 8.3
        }
      }
    """
    start = time.perf_counter()

    if not body.results:
        return JSONResponse(
            content={"success": False, "error": "No results provided to analyze."},
            status_code=400,
        )

    if not isinstance(body.results, list):
        return JSONResponse(
            content={"success": False, "error": "'results' must be a list of documents."},
            status_code=400,
        )

    try:
        analysis = await summarize_results(
            results  = body.results,
            question = body.question.strip(),
        )

        elapsed = round(time.perf_counter() - start, 2)

        return JSONResponse(content={
            "success": True,
            "data": {
                **analysis,
                "elapsedSeconds": elapsed,
            },
        })

    except Exception as e:
        import traceback
        print(f"[/api/analyze] Unhandled exception:\n{traceback.format_exc()}")
        return JSONResponse(
            content={"success": False, "error": f"Analysis failed: {str(e)[:200]}"},
            status_code=500,
        )


@app.get("/api/history")
async def get_query_history(limit: int = 100):
    """
    Returns the user's query history from MongoDB, newest first.

    Each entry contains the question, which collection was queried,
    how many results came back, and when it ran.

    Query param:
        limit  — how many entries to return (default 100, max 500)

    Response:
        { "success": true, "history": [ { id, question, collection,
          result_count, result_label, explanation, timestamp }, ... ] }
    """
    try:
        entries = await get_history(limit=limit)
        return JSONResponse(content={"success": True, "history": entries})
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500,
        )


@app.delete("/api/history/{entry_id}")
async def delete_history_entry(entry_id: str):
    """
    Deletes a single history entry by its MongoDB _id.

    Path param:
        entry_id — the hex string ObjectId of the history document
    """
    try:
        deleted = await delete_entry(entry_id)
        return JSONResponse(content={"success": deleted})
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500,
        )


@app.delete("/api/history")
async def clear_history():
    """
    Deletes all history entries from MongoDB.

    Used by the "CLEAR" button in the history panel.
    """
    try:
        count = await clear_all()
        return JSONResponse(content={"success": True, "deleted": count})
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500,
        )


@app.post("/api/schema/refresh")
async def refresh_schema():
    """
    Manually triggers a live schema re-discovery from MongoDB.

    Useful after deploying database schema changes — call this
    to force an immediate refresh without waiting for the TTL.

    This is an admin/debug endpoint. In production you might
    want to add authentication to it.
    """
    collection = os.getenv("DEFAULT_STREAM_COLLECTION", "Apr_2025")
    await refresh_schema_cache(stream_collection=collection, force=True)
    return JSONResponse(content={
        "success": True,
        "cache":   get_cache_status(),
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host        = os.getenv("HOST", "0.0.0.0"),
        port        = int(os.getenv("PORT", 8000)),
        reload      = True,
        reload_dirs = ["."],
    )
