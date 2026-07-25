"""Trainer mode: a human fixes a bad pipeline and saves it as gold.

One save writes three places — data/eval_set.json (locked case), the RAG
store (weight 3.0, source trainer_gold), and _QUERY_FEEDBACK_ in Mongo for
audit. If the RAG write fails the eval case is rolled back so the harness
and the store never disagree.

Auth is just TRAINER_PASSWORD from .env (unset = trainer off). Login hands
out a bearer token that lives in process memory for 2 hours.
"""
from __future__ import annotations
import os
import json
import hmac
import time
import secrets
import hashlib
import logging
import fcntl
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = ROOT / "data" / "eval_set.json"
TOKEN_TTL_SEC = int(os.getenv("TRAINER_TOKEN_TTL_SEC", "7200"))

_TOKENS: dict[str, float] = {}


def is_enabled() -> bool:
    # no password configured = trainer endpoints answer 403
    return bool(os.getenv("TRAINER_PASSWORD", "").strip())


def verify_password(password: str) -> bool:
    expected = os.getenv("TRAINER_PASSWORD", "")
    if not expected or not password:
        return False
    return hmac.compare_digest(password.encode("utf-8"), expected.encode("utf-8"))


def issue_token() -> str:
    _prune_tokens()
    token = secrets.token_urlsafe(24)
    _TOKENS[token] = time.time() + TOKEN_TTL_SEC
    return token


def verify_token(token: str) -> bool:
    _prune_tokens()
    exp = _TOKENS.get(token or "")
    return bool(exp and exp > time.time())


def revoke_token(token: str) -> bool:
    return _TOKENS.pop(token or "", None) is not None


def _prune_tokens() -> None:
    now = time.time()
    dead = [t for t, e in _TOKENS.items() if e <= now]
    for t in dead:
        _TOKENS.pop(t, None)


def extract_bearer(header: str | None) -> str:
    if not header:
        return ""
    parts = header.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""


def _next_eval_id(cases: list[dict]) -> str:
    # eval-gold-NNN, first number not already taken
    used = {c.get("id", "") for c in cases}
    for i in range(1, 10_000):
        candidate = f"eval-gold-{i:03d}"
        if candidate not in used:
            return candidate
    return f"eval-gold-{int(time.time())}"


def _infer_db_hint(query_obj: dict) -> str:
    if query_obj.get("queryType") == "dual":
        return "both"
    db = query_obj.get("database", "stream-datastore")
    return "appconfigs" if db == "appConfigs" else "stream"


def _required_columns(query_obj: dict) -> list[str]:
    # best effort: which top-level keys should each result row have
    qt = query_obj.get("queryType", "single")
    if qt == "dual":
        return ["_id"]
    pipe = query_obj.get("pipeline", []) if query_obj.get("operation", "aggregate") == "aggregate" else []
    for stage in reversed(pipe):
        if not isinstance(stage, dict):
            continue
        if "$group" in stage and isinstance(stage["$group"], dict):
            return list(stage["$group"].keys())[:6]
        if "$project" in stage and isinstance(stage["$project"], dict):
            return [k for k, v in stage["$project"].items() if v not in (0, False)][:6]
    return []


def _append_eval_case(corrected_query: dict, question: str, result_count: int, tags: list[str]) -> str:
    # append a locked case; file lock in case two trainers save at once
    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    if EVAL_SET_PATH.exists():
        try:
            cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"[trainer] eval_set.json unreadable ({e}); starting fresh")
            cases = []

    eval_id = _next_eval_id(cases)
    db_hint = _infer_db_hint(corrected_query)

    case = {
        "id":            eval_id,
        "question":      question,
        "db_hint":       db_hint,
        "collection":    corrected_query.get("collection")
                         or (corrected_query.get("queries", [{}])[0] or {}).get("collection", ""),
        "gold_pipeline": corrected_query,
        "gold_result_shape": {
            "min_rows":         1 if result_count > 0 else 0,
            "max_rows":         max(200, result_count * 2),
            "required_columns": _required_columns(corrected_query),
        },
        "locked":        True,
        "tags":          ["trainer-gold", db_hint, *tags],
        "trainer_meta": {
            "saved_at":     datetime.now(timezone.utc).isoformat(),
            "result_count": result_count,
        },
    }
    cases.append(case)

    # write to a temp file first, then atomic rename
    tmp = EVAL_SET_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(cases, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, EVAL_SET_PATH)
    return eval_id


def _rollback_eval_case(eval_id: str) -> None:
    if not EVAL_SET_PATH.exists():
        return
    try:
        cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
        cases = [c for c in cases if c.get("id") != eval_id]
        EVAL_SET_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"[trainer] rolled back eval case {eval_id}")
    except Exception as e:
        logger.error(f"[trainer] rollback for {eval_id} failed: {e}")


async def save_as_gold(
    session_id:      str,
    question:        str,
    corrected_query: dict,
    result_count:    int,
    eval_tags:       list[str] | None = None,
) -> dict:
    """The three-store save. Eval case first, then RAG; a RAG failure rolls
    the eval case back. Returns {eval_id, rag_id, feedback_id, saved_at}.
    """
    from lib.query_examples import add_trainer_gold_example
    from lib.feedback_store  import save_feedback

    import asyncio
    eval_tags = eval_tags or []
    saved_at  = datetime.now(timezone.utc).isoformat()

    eval_id = _append_eval_case(corrected_query, question, result_count, eval_tags)
    logger.info(f"[trainer] eval case written: {eval_id}")

    try:
        rag_id = await add_trainer_gold_example(
            question     = question,
            query_obj    = corrected_query,
            result_count = result_count,
            db_hint      = _infer_db_hint(corrected_query),
        )
        logger.info(f"[trainer] RAG example upserted: {rag_id}")
    except Exception as e:
        _rollback_eval_case(eval_id)
        raise RuntimeError(f"RAG write failed, eval case rolled back: {e}") from e

    # feedback row is just an audit log (eval_set + RAG are the source of
    # truth), and this Mongo cluster can stall writes for ages — fire and forget
    asyncio.create_task(_fire_and_forget_feedback(
        session_id      = session_id or "trainer",
        question        = question,
        corrected_query = corrected_query,
        result_count    = result_count,
        eval_id         = eval_id,
    ))

    return {
        "eval_id":     eval_id,
        "rag_id":      rag_id,
        "feedback_id": None,   # written asynchronously
        "saved_at":    saved_at,
    }


async def _fire_and_forget_feedback(
    session_id:      str,
    question:        str,
    corrected_query: dict,
    result_count:    int,
    eval_id:         str,
) -> None:
    from lib.feedback_store import save_feedback
    try:
        fid = await save_feedback(
            session_id         = session_id,
            question           = question,
            query_obj          = corrected_query,
            result_count       = result_count,
            rating             = "good",
            correction_note    = f"TRAINER_GOLD: saved as {eval_id}",
            corrected_pipeline = (
                corrected_query.get("pipeline", [])
                if corrected_query.get("queryType") == "single"
                else []
            ),
        )
        logger.info(f"[trainer] feedback doc written (bg): {fid}")
    except Exception as e:
        logger.error(f"[trainer] feedback bg write failed (non-fatal): {e}")


def list_gold(limit: int = 10) -> dict:
    if not EVAL_SET_PATH.exists():
        return {"total_gold": 0, "recent": []}
    try:
        cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"total_gold": 0, "recent": []}

    gold = [c for c in cases if "trainer-gold" in (c.get("tags") or [])]
    gold_sorted = sorted(
        gold,
        key=lambda c: c.get("trainer_meta", {}).get("saved_at", ""),
        reverse=True,
    )
    return {
        "total_gold": len(gold),
        "recent": [
            {
                "id":           c.get("id"),
                "question":     (c.get("question") or "")[:120],
                "db_hint":      c.get("db_hint"),
                "result_count": c.get("trainer_meta", {}).get("result_count"),
                "saved_at":     c.get("trainer_meta", {}).get("saved_at"),
                "tags":         [t for t in (c.get("tags") or []) if t != "trainer-gold"],
            }
            for c in gold_sorted[:limit]
        ],
    }


def build_corrected_query(original: dict, corrected_pipeline: list, *, collection: str | None = None) -> dict:
    """Copy of the original query_obj with the edited pipeline swapped in.
    For dual queries only the primary branch is editable for now.
    """
    if not isinstance(corrected_pipeline, list):
        raise ValueError("corrected_pipeline must be a JSON array of stages")
    if not all(isinstance(s, dict) and len(s) > 0 for s in corrected_pipeline):
        raise ValueError("each stage must be a non-empty object, e.g. {'$match': {...}}")

    out = json.loads(json.dumps(original))  # deep copy, BSON-safe
    qt  = out.get("queryType", "single")

    if qt == "single":
        out["pipeline"]  = corrected_pipeline
        out["operation"] = out.get("operation", "aggregate")
        if collection:
            out["collection"] = collection
    elif qt == "dual":
        queries = out.get("queries") or []
        if not queries:
            raise ValueError("dual query has no queries[] to edit")
        queries[0]["pipeline"] = corrected_pipeline
        if collection:
            queries[0]["collection"] = collection
        out["queries"] = queries
    else:
        raise ValueError(f"unknown queryType: {qt}")

    return out
