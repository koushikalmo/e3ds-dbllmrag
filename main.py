import os
import time
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
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
from lib.collection_resolver import resolve_and_log, resolve_year
from lib.query_executor     import get_existing_year_collections, build_year_pipeline
from lib.response_validator import validate_query_and_result
from lib.feedback_store     import save_feedback, get_feedback_stats
from lib.data_digest        import start_digest_scheduler, get_digest_status, refresh_digest
from lib.vector_store       import _get_client as _get_chroma_client


def _chroma_status() -> dict:
    try:
        client = _get_chroma_client()
        if client is None:
            return {"status": "unavailable", "collections": []}
        cols = client.list_collections()
        return {
            "status":      "ok",
            "collections": [
                {"name": c.name, "count": c.count()}
                for c in cols
            ],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


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

    asyncio.create_task(warmup_model())
    asyncio.create_task(warm_all_caches(default_collection))
    asyncio.create_task(refresh_schema_cache(stream_collection=default_collection))
    asyncio.create_task(index_all_examples_async())
    asyncio.create_task(start_digest_scheduler())

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
    allow_methods = ["GET", "POST", "DELETE"],
    allow_headers = ["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


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


class FeedbackRequest(BaseModel):
    session_id:         str  = Field("",  description="Frontend session UUID")
    question:           str  = Field(..., min_length=1, max_length=500)
    query_meta:         dict = Field({},  description="Query metadata returned by /api/query")
    result_count:       int  = Field(0,   ge=0)
    rating:             str  = Field(..., description="'good' or 'bad'")
    correction_note:    str  = Field("",  max_length=2000)
    corrected_pipeline: list = Field([],  description="User-supplied corrected pipeline JSON")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health_check():
    db_status      = await ping_databases()
    all_ok         = all(v == "ok" for v in db_status.values())
    feedback_stats = await get_feedback_stats()
    return JSONResponse(
        content={
            "status":        "ok" if all_ok else "degraded",
            "databases":     db_status,
            "schema_cache":  get_cache_status(),
            "live_context":  get_live_cache_status(),
            "rag_examples":  get_example_count(),
            "feedback":      feedback_stats,
            "data_digest":   get_digest_status(),
            "vector_db":     _chroma_status(),
            "time":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        "vector_db":    _chroma_status(),
    })


@app.post("/api/query")
async def run_query(body: QueryRequest):
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

        # Year query: expand single-collection pipeline across all months via $unionWith
        year = resolve_year(body.question.strip())
        if (
            year
            and query_obj.get("queryType") == "single"
            and query_obj.get("database") == "stream-datastore"
            and isinstance(query_obj.get("pipeline"), list)
        ):
            year_colls = await get_existing_year_collections(query_obj["database"], year)
            primary    = query_obj.get("collection", "")
            extras     = [c for c in year_colls if c != primary]
            if extras:
                query_obj["pipeline"] = build_year_pipeline(query_obj["pipeline"], extras)
                print(f"[year_query] Expanded across {len(year_colls)} collections for {year}: {year_colls}")

        result = await execute_query(query_obj)

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

        validation_warnings = validate_query_and_result(query_obj, result, result_count)

        # Extract LLM transparency fields (may be absent on old Ollama builds)
        assumptions = query_obj.get("assumptions", [])
        if not isinstance(assumptions, list):
            assumptions = []
        confidence = query_obj.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        # Build a lightweight query-metadata object the frontend sends back for feedback
        query_meta = {
            "queryType":  query_obj.get("queryType"),
            "database":   query_obj.get("database"),
            "collection": query_obj.get("collection"),
            "operation":  query_obj.get("operation", "aggregate"),
            "pipeline":   query_obj.get("pipeline"),
            "queries":    query_obj.get("queries"),
        }

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
                "aiSummary":         ai_summary,
                "explanation":       query_obj.get("explanation", ""),
                "queryPlan":         query_plan,
                "assumptions":       assumptions,
                "confidence":        confidence,
                "validationWarnings": validation_warnings,
                "queryMeta":         query_meta,
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


@app.post("/api/digest/refresh")
async def refresh_digest_endpoint():
    await refresh_digest(force=True)
    return JSONResponse(content={"success": True, "data_digest": get_digest_status()})


@app.post("/api/schema/refresh")
async def refresh_schema():
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


@app.post("/api/feedback")
async def submit_feedback(body: FeedbackRequest):
    if body.rating not in ("good", "bad"):
        return JSONResponse(
            content={"success": False, "error": "rating must be 'good' or 'bad'"},
            status_code=400,
        )
    try:
        doc_id = await save_feedback(
            session_id         = body.session_id,
            question           = body.question,
            query_obj          = body.query_meta,
            result_count       = body.result_count,
            rating             = body.rating,
            correction_note    = body.correction_note,
            corrected_pipeline = body.corrected_pipeline,
        )
        return JSONResponse(content={"success": True, "id": doc_id})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        import traceback
        print(f"[/api/feedback] Error:\n{traceback.format_exc()}")
        return JSONResponse(
            content={"success": False, "error": f"Failed to save feedback: {str(e)[:200]}"},
            status_code=500,
        )


@app.post("/api/share")
async def share_chat(body: ShareRequest):
    try:
        share_id = await create_share(body.turns, body.title)
        return JSONResponse(content={"success": True, "share_id": share_id})
    except asyncio.CancelledError:
        raise
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


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    import tempfile, os
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return JSONResponse({"error": "faster-whisper not installed. Run: pip install faster-whisper"}, status_code=501)
    try:
        if not hasattr(app.state, "whisper_model"):
            app.state.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        data = await audio.read()
        ext = ".webm" if "webm" in (audio.content_type or "") else ".wav" if "wav" in (audio.content_type or "") else ".ogg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(data)
            tmp_path = f.name
        try:
            segments, _ = app.state.whisper_model.transcribe(tmp_path, language="en")
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            os.unlink(tmp_path)
        return {"transcript": transcript}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host        = os.getenv("HOST", "0.0.0.0"),
        port        = int(os.getenv("PORT", 8000)),
        reload      = True,
        reload_dirs = ["."],
    )
