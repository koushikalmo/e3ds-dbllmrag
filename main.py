from __future__ import annotations
import os
import time
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Request
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
from lib.schema_watcher     import start_schema_watcher, get_watcher_status
from lib.live_data_context  import warm_all_caches, get_cache_status as get_live_cache_status
from lib.query_examples     import (
    get_example_count, index_all_examples_async,
    bump_example, rag_stats, prune_examples, _prune_scheduler,
    migrate_phase4_metadata,
)
from lib.chat_history       import save_query, get_history, delete_entry, clear_all
from lib.chat_sharing       import create_share, get_share
from lib.session_memory     import add_turn, get_context_text, clear_session, active_session_count
from lib.collection_resolver import resolve_and_log, resolve_year
from lib.query_executor     import get_existing_year_collections, build_year_pipeline
from lib.response_validator import validate_query_and_result
from lib.accuracy_scorer    import score_query
from lib.feedback_store     import save_feedback, get_feedback_stats
from lib.data_digest        import start_digest_scheduler, get_digest_status, refresh_digest
from lib                    import trainer


# Populate the schema cache before starting the watcher — otherwise the
# very first change event has nothing to diff against and would trigger a
# redundant refresh.
async def _boot_schema_pipeline(default_collection: str) -> None:
    try:
        await refresh_schema_cache(stream_collection=default_collection)
    except Exception as e:
        print(f"[startup] initial schema refresh failed (non-fatal): {e}")
    try:
        await start_schema_watcher(default_collection)
    except Exception as e:
        print(f"[startup] schema watcher failed to start (non-fatal): {e}")


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

    # old RAG entries have no timestamps, which makes the age-decay treat them
    # as ancient and the pruner delete them. Stamp them once here (idempotent).
    migrate_phase4_metadata()

    asyncio.create_task(warmup_model())
    asyncio.create_task(warm_all_caches(default_collection))
    asyncio.create_task(_boot_schema_pipeline(default_collection))
    asyncio.create_task(index_all_examples_async())
    asyncio.create_task(start_digest_scheduler())
    asyncio.create_task(_prune_scheduler())

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


class TrainLoginRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=200)


class TrainRunRequest(BaseModel):
    session_id:         str  = Field("",   description="Frontend session UUID")
    question:           str  = Field(...,  min_length=1, max_length=500)
    original_query:     dict = Field(...,  description="The query_obj the LLM first produced")
    corrected_pipeline: list = Field(...,  description="Pipeline JSON the trainer edited")
    collection:         str  = Field("",   description="Override collection (stream-datastore)")


class TrainSaveRequest(BaseModel):
    session_id:         str  = Field("",   description="Frontend session UUID")
    question:           str  = Field(...,  min_length=1, max_length=500)
    original_query:     dict = Field(...,  description="Original query_obj from LLM (kept for audit)")
    corrected_pipeline: list = Field(...,  description="Pipeline JSON the trainer edited")
    collection:         str  = Field("",   description="Override collection")
    result_count:       int  = Field(0,    ge=0)
    eval_tags:          list = Field([],   description="Optional free-form tags")


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
        # internal bookkeeping field — pop it before it can leak into the response
        retrieved_example_ids = query_obj.pop("_retrieved_example_ids", []) or []

        # whole-year question? union the pipeline across every month collection
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

        # summary for the PIPELINE tab in the UI
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

        try:
            accuracy_breakdown = await score_query(
                question         = body.question.strip(),
                query_obj        = query_obj,
                result           = result,
                result_count     = result_count,
                validator_checks = validation_warnings,
            )
            accuracy = accuracy_breakdown.to_dict()
        except Exception as e:
            print(f"[/api/query] Accuracy scoring failed (non-fatal): {e}")
            accuracy = {"score": None, "tier": "unknown", "signals": {}}

        # the model doesn't always fill these in, so sanitize
        assumptions = query_obj.get("assumptions", [])
        if not isinstance(assumptions, list):
            assumptions = []
        confidence = query_obj.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        # the frontend echoes this back with thumbs up/down feedback
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
            save_successful_query(
                body.question.strip(),
                query_obj,
                result_count,
                accuracy_score = accuracy.get("score"),
            )
            # reward the examples that were shown in the prompt. Failures get
            # recorded on the bad-feedback path in feedback_store, not here.
            acc_score = accuracy.get("score") if isinstance(accuracy, dict) else None
            for ex_id in retrieved_example_ids:
                try:
                    bump_example(ex_id, success=True, accuracy=acc_score)
                except Exception as e:
                    print(f"[/api/query] bump_example failed for {ex_id[:8]}: {e}")
            # history write goes to the aux cluster which can stall — never await it
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
                "accuracy":          accuracy,
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


@app.post("/api/eval/run")
async def trigger_eval_run():
    # runs scripts/evaluate.py as a subprocess against this server, in the background
    import asyncio, json
    from pathlib import Path

    target = f"http://127.0.0.1:{os.getenv('PORT', '8000')}"
    script = Path(__file__).parent / "scripts" / "evaluate.py"

    async def _runner():
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", str(script), "--url", target,
                cwd    = str(Path(__file__).parent),
                stdout = asyncio.subprocess.PIPE,
                stderr = asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
            print(f"[/api/eval/run] finished rc={proc.returncode}")
            if err:
                print(f"[/api/eval/run] stderr: {err.decode()[:500]}")
        except Exception as e:
            print(f"[/api/eval/run] failed: {e}")

    asyncio.create_task(_runner())
    return JSONResponse(content={"success": True, "message": "Eval run started in background."})


@app.get("/api/eval/baseline")
async def get_eval_baseline():
    # last finished eval report, 404 if none yet
    import json
    from pathlib import Path
    report_path = Path(__file__).parent / "data" / "eval_report.json"
    if not report_path.exists():
        return JSONResponse(
            content={"success": False, "error": "No eval report found. Run POST /api/eval/run first."},
            status_code=404,
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return JSONResponse(content={"success": True, "data": report})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": f"Failed to read report: {e}"}, status_code=500)


# trainer mode: password-gated pipeline editor + gold-example saver
def _trainer_auth(request) -> tuple[bool, JSONResponse | None]:
    """Returns (ok, error_response) — error_response is ready to send as-is."""
    if not trainer.is_enabled():
        return False, JSONResponse(
            content={"success": False, "error": "Trainer disabled. Set TRAINER_PASSWORD in .env to enable."},
            status_code=403,
        )
    token = trainer.extract_bearer(request.headers.get("authorization"))
    if not trainer.verify_token(token):
        return False, JSONResponse(
            content={"success": False, "error": "Invalid or expired trainer token. Re-login."},
            status_code=401,
        )
    return True, None


@app.get("/api/train/status")
async def trainer_status():
    return JSONResponse(content={
        "success": True,
        "enabled": trainer.is_enabled(),
    })


@app.post("/api/train/login")
async def trainer_login(body: TrainLoginRequest):
    if not trainer.is_enabled():
        return JSONResponse(
            content={"success": False, "error": "Trainer disabled. Set TRAINER_PASSWORD in .env."},
            status_code=403,
        )
    if not trainer.verify_password(body.password):
        return JSONResponse(
            content={"success": False, "error": "Incorrect password."},
            status_code=401,
        )
    token = trainer.issue_token()
    return JSONResponse(content={
        "success":    True,
        "token":      token,
        "expires_in": trainer.TOKEN_TTL_SEC,
    })


@app.post("/api/train/logout")
async def trainer_logout(request: Request):
    token = trainer.extract_bearer(request.headers.get("authorization"))
    trainer.revoke_token(token)
    return JSONResponse(content={"success": True})


@app.post("/api/train/run")
async def trainer_run(body: TrainRunRequest, request: Request):
    from lib.query_executor    import execute_query
    from lib.response_validator import validate_query_and_result

    ok, err = _trainer_auth(request)
    if not ok:
        return err

    try:
        corrected = trainer.build_corrected_query(
            body.original_query,
            body.corrected_pipeline,
            collection=body.collection or None,
        )
    except ValueError as e:
        return JSONResponse(
            content={"success": False, "error": f"invalid pipeline: {e}"},
            status_code=400,
        )

    try:
        result = await execute_query(corrected)
    except TimeoutError as e:
        return JSONResponse(content={"success": False, "error": f"timeout: {e}"}, status_code=504)
    except Exception as e:
        import traceback
        print(f"[/api/train/run] execution failed:\n{traceback.format_exc()}")
        return JSONResponse(content={"success": False, "error": f"execution failed: {str(e)[:300]}"}, status_code=400)

    rows = result.get("results", [])
    if isinstance(rows, dict):
        rows = rows.get("merged") or rows.get("primary") or []
    result_count = len(rows) if isinstance(rows, list) else 0

    checks = validate_query_and_result(corrected, result, result_count)

    try:
        breakdown = await score_query(
            question         = body.question.strip(),
            query_obj        = corrected,
            result           = result,
            result_count     = result_count,
            validator_checks = checks,
        )
        accuracy = breakdown.to_dict()
    except Exception as e:
        print(f"[/api/train/run] scoring failed (non-fatal): {e}")
        accuracy = {"score": None, "tier": "unknown", "signals": {}}

    return JSONResponse(content={
        "success": True,
        "data": {
            "correctedQuery":     corrected,
            "result":             result,
            "resultCount":        result_count,
            "validationWarnings": checks,
            "accuracy":           accuracy,
        },
    })


@app.post("/api/train/save")
async def trainer_save(body: TrainSaveRequest, request: Request):
    ok, err = _trainer_auth(request)
    if not ok:
        return err

    try:
        corrected = trainer.build_corrected_query(
            body.original_query,
            body.corrected_pipeline,
            collection=body.collection or None,
        )
    except ValueError as e:
        return JSONResponse(
            content={"success": False, "error": f"invalid pipeline: {e}"},
            status_code=400,
        )

    try:
        ids = await trainer.save_as_gold(
            session_id      = body.session_id,
            question        = body.question.strip(),
            corrected_query = corrected,
            result_count    = body.result_count,
            eval_tags       = body.eval_tags,
        )
    except Exception as e:
        import traceback
        print(f"[/api/train/save] save failed:\n{traceback.format_exc()}")
        return JSONResponse(content={"success": False, "error": f"save failed: {str(e)[:300]}"}, status_code=500)

    return JSONResponse(content={"success": True, "data": ids})


@app.get("/api/train/list")
async def trainer_list(request: Request):
    ok, err = _trainer_auth(request)
    if not ok:
        return err
    return JSONResponse(content={"success": True, "data": trainer.list_gold()})


@app.get("/api/rag/stats")
async def rag_stats_endpoint():
    try:
        return JSONResponse(content={"success": True, "data": rag_stats()})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": f"stats failed: {e}"}, status_code=500)


class RagPruneRequest(BaseModel):
    dry_run: bool = Field(True, description="When true, report what would be removed without deleting")


@app.post("/api/rag/prune")
async def rag_prune_endpoint(body: RagPruneRequest):
    try:
        return JSONResponse(content={"success": True, "data": prune_examples(dry_run=body.dry_run)})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": f"prune failed: {e}"}, status_code=500)


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


@app.get("/api/schema/watcher")
async def schema_watcher_status():
    return JSONResponse(content={
        "success": True,
        "watcher": get_watcher_status(),
        "schema_cache": get_cache_status(),
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
