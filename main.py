# main.py — FastAPI app entry point
# Routes: GET / | GET /api/health | GET /api/status | POST /api/query | POST /api/analyze

import os
import time
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lib.mongodb            import close_connections, ping_databases
from lib.query_generator    import generate_query, save_successful_query
from lib.query_executor     import execute_query
from lib.result_summarizer  import summarize_results
from lib.llm_provider       import OllamaProvider, OLLAMA_MODEL, warmup_model
from lib.schema_discovery   import refresh_schema_cache, get_cache_status
from lib.live_data_context  import warm_all_caches, get_cache_status as get_live_cache_status
from lib.query_examples     import get_example_count
from lib.chat_history       import save_query, get_history, delete_entry, clear_all
from lib.chat_sharing       import create_share, get_share
from lib.query_examples     import index_all_examples_async
from lib.session_memory     import add_turn, get_context_text, clear_session, active_session_count
from lib.collection_resolver import resolve_and_log


@asynccontextmanager
async def lifespan(app: FastAPI):
    default_collection = os.getenv("DEFAULT_STREAM_COLLECTION", "Apr_2025")

    print("═" * 56)
    print("  Eagle 3D Streaming Query System v3")
    print(f"  URL: http://{os.getenv('HOST','0.0.0.0')}:{os.getenv('PORT','8000')}")
    print(f"  LLM: LOCAL Ollama ({OLLAMA_MODEL})")
    print(f"  Stream DB : {os.getenv('STREAM_DB_NAME', 'stream-datastore')}")
    print(f"  AppConfigs: {os.getenv('APPCONFIGS_DB_NAME', 'appConfigs')}")
    print(f"  RAG examples: {get_example_count()} in store")
    print("═" * 56)

    # All four startup tasks run concurrently in the background
    asyncio.create_task(warmup_model())
    asyncio.create_task(warm_all_caches(default_collection))
    asyncio.create_task(refresh_schema_cache(stream_collection=default_collection))
    asyncio.create_task(index_all_examples_async())

    yield

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
    collection: str = Field("Apr_2026")
    session_id: str = Field("", description="UUID from frontend — enables follow-up questions")


class ShareRequest(BaseModel):
    turns: list = Field(..., description="Chat turns to share")
    title: str  = Field("", description="Human-readable title for the shared chat")


class AnalyzeRequest(BaseModel):
    results:  list = Field(..., description="Query results from /api/query")
    question: str  = Field(..., min_length=1, max_length=500)


# ── Routes ─────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health_check():
    db_status = await ping_databases()
    all_ok    = all(v == "ok" for v in db_status.values())
    return JSONResponse(
        content={
            "status":       "ok" if all_ok else "degraded",
            "databases":    db_status,
            "schema_cache": get_cache_status(),
            "live_context": get_live_cache_status(),
            "rag_examples": get_example_count(),
            "time":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        status_code=200 if all_ok else 503,
    )


@app.get("/api/status")
async def llm_status():
    ollama    = OllamaProvider()
    available = await ollama.is_available()
    return JSONResponse(content={
        "provider":     "ollama",
        "model":        OLLAMA_MODEL,
        "available":    available,
        "schema_cache": get_cache_status(),
        "live_context": get_live_cache_status(),
        "rag_examples": get_example_count(),
    })


@app.post("/api/query")
async def run_query(body: QueryRequest):
    """NL question → MongoDB pipeline → results + AI summary."""
    start = time.perf_counter()

    try:
        conversation_ctx = get_context_text(body.session_id)
        default_col      = body.collection or os.getenv("DEFAULT_STREAM_COLLECTION", "Apr_2026")
        collection       = resolve_and_log(body.question.strip(), default_col)

        query_obj = await generate_query(
            question         = body.question.strip(),
            collection       = collection,
            conversation_ctx = conversation_ctx,
        )
        result = await execute_query(query_obj)

        # Auto-analyze results immediately
        rows_for_analysis = result.get("results", [])
        if isinstance(rows_for_analysis, dict):
            rows_for_analysis = rows_for_analysis.get("merged") or rows_for_analysis.get("primary") or []

        ai_summary = None
        if rows_for_analysis:
            try:
                analysis   = await summarize_results(results=rows_for_analysis, question=body.question.strip())
                ai_summary = analysis.get("summary", "")
            except Exception as e:
                print(f"[/api/query] Auto-analysis failed (non-fatal): {e}")

        elapsed = round(time.perf_counter() - start, 2)

        # Build query plan summary for frontend PIPELINE tab
        query_plan = None
        qt = query_obj.get("queryType")
        if qt == "single":
            query_plan = {
                "databases":   [query_obj.get("database", "")],
                "collections": [query_obj.get("collection", "")],
                "queryType":   "single",
                "stageCount":  len(query_obj.get("pipeline", [])),
            }
        elif qt == "dual":
            queries = query_obj.get("queries", [])
            query_plan = {
                "databases":   [q.get("database", "")  for q in queries],
                "collections": [q.get("collection", "") for q in queries],
                "queryType":   "dual",
                "stageCount":  sum(len(q.get("pipeline", [])) for q in queries),
                "dualQueries": [
                    {"database": q.get("database", ""), "collection": q.get("collection", ""), "stageCount": len(q.get("pipeline", []))}
                    for q in queries
                ],
            }

        result_count = 0
        if result.get("queryType") == "single":
            result_count = len(result.get("results", []))
        elif result.get("queryType") == "dual":
            primary = result.get("results", {})
            if isinstance(primary, dict):
                result_count = len(primary.get("primary", []))

        if result_count > 0:
            if body.session_id:
                add_turn(
                    body.session_id,
                    body.question.strip(),
                    f"Returned {result_count} results — '{query_obj.get('resultLabel', 'Results')}'",
                )
            save_successful_query(body.question.strip(), query_obj, result_count)
            # Save to chat history in background — don't await (would block the response)
            asyncio.create_task(save_query(
                question        = body.question.strip(),
                collection      = collection,
                result_count    = result_count,
                result_label    = query_obj.get("resultLabel", ""),
                explanation     = query_obj.get("explanation", ""),
                elapsed_seconds = round(time.perf_counter() - start, 2),
            ))

        return JSONResponse(content={
            "success": True,
            "data": {
                **result,
                "aiSummary":   ai_summary,
                "explanation": query_obj.get("explanation", ""),
                "queryPlan":   query_plan,
                "meta": {
                    "question":       body.question.strip(),
                    "collectionUsed": collection,
                    "elapsedSeconds": elapsed,
                    "generatedAt":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            },
        })

    except TimeoutError as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=504)
    except ValueError as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=400)
    except Exception as e:
        import traceback
        print(f"[/api/query] Unhandled exception:\n{traceback.format_exc()}")
        return JSONResponse(content={"success": False, "error": f"Server error: {str(e)[:200]}"}, status_code=500)


@app.post("/api/analyze")
async def analyze_results(body: AnalyzeRequest):
    """AI analysis of an existing result set. Uses chunked map-reduce for large sets."""
    start = time.perf_counter()

    if not body.results:
        return JSONResponse(content={"success": False, "error": "No results provided."}, status_code=400)
    if not isinstance(body.results, list):
        return JSONResponse(content={"success": False, "error": "'results' must be a list."}, status_code=400)

    try:
        analysis = await summarize_results(results=body.results, question=body.question.strip())
        return JSONResponse(content={"success": True, "data": {**analysis, "elapsedSeconds": round(time.perf_counter() - start, 2)}})
    except Exception as e:
        import traceback
        print(f"[/api/analyze] Unhandled exception:\n{traceback.format_exc()}")
        return JSONResponse(content={"success": False, "error": f"Analysis failed: {str(e)[:200]}"}, status_code=500)


@app.get("/api/history")
async def get_query_history(limit: int = 100):
    try:
        entries = await get_history(limit=limit)
        return JSONResponse(content={"success": True, "history": entries})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.delete("/api/history/{entry_id}")
async def delete_history_entry(entry_id: str):
    try:
        deleted = await delete_entry(entry_id)
        return JSONResponse(content={"success": deleted})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.delete("/api/history")
async def clear_history():
    try:
        count = await clear_all()
        return JSONResponse(content={"success": True, "deleted": count})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.post("/api/schema/refresh")
async def refresh_schema():
    """Force-refresh all data caches (schema discovery + live context)."""
    collection = os.getenv("DEFAULT_STREAM_COLLECTION", "Apr_2026")
    await asyncio.gather(
        refresh_schema_cache(stream_collection=collection, force=True),
        warm_all_caches(collection),
    )
    return JSONResponse(content={
        "success":      True,
        "schema_cache": get_cache_status(),
        "live_context": get_live_cache_status(),
    })


@app.post("/api/share")
async def share_chat(body: ShareRequest):
    """Save a chat snapshot and return a shareable ID."""
    try:
        share_id = await create_share(body.turns, body.title)
        return JSONResponse(content={"success": True, "share_id": share_id})
    except Exception as e:
        import traceback
        print(f"[/api/share] Error:\n{traceback.format_exc()}")
        return JSONResponse(content={"success": False, "error": f"Failed to create share: {str(e)[:200]}"}, status_code=500)


@app.get("/api/share/{share_id}")
async def get_shared_chat(share_id: str):
    try:
        data = await get_share(share_id)
        if not data:
            return JSONResponse(content={"success": False, "error": "Shared chat not found."}, status_code=404)
        return JSONResponse(content={"success": True, "data": data})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.get("/share/{share_id}", include_in_schema=False)
async def serve_shared_chat_page(share_id: str):
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host        = os.getenv("HOST", "0.0.0.0"),
        port        = int(os.getenv("PORT", 8000)),
        reload      = True,
        reload_dirs = ["."],
    )
