"""Auto-propose an evaluation set from existing query history.

Scans ``data/query_examples.json`` (and, when reachable, the ``_QUERY_FEEDBACK_``
MongoDB collection) for queries that previously worked, deduplicates by
question tokens, scores each candidate, and writes the top N to
``data/eval_set.proposed.json`` for human review.

Usage:
    python3 scripts/propose_eval_set.py               # default 30 cases
    python3 scripts/propose_eval_set.py --limit 50    # more cases
    python3 scripts/propose_eval_set.py --include-feedback  # also pull MongoDB
"""
from __future__ import annotations
import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


EXAMPLES_FILE      = ROOT / "data" / "query_examples.json"
PROPOSED_FILE_PATH = ROOT / "data" / "eval_set.proposed.json"


_STOP = {
    "the", "a", "an", "is", "are", "was", "of", "in", "for", "to", "and", "or", "by", "with",
    "this", "that", "what", "which", "how", "many", "show", "me", "get", "list", "find", "from",
    "all", "any", "give", "tell",
}


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
                     if len(w) > 2 and w not in _STOP)


def _score_candidate(ex: dict) -> float:
    weight = float(ex.get("weight") or 1.0)
    source = ex.get("source", "auto")

    src_bonus = {"user_corrected": 5.0, "user_verified": 3.0, "trainer_gold": 6.0}.get(source, 0.0)

    rc = int(ex.get("result_count") or 0)
    rc_bonus = min(10.0, rc / 10.0)

    ts = ex.get("timestamp") or ex.get("created_at") or ""
    fresh_bonus = 0.0
    try:
        if ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - dt).days
            if age_days < 30:    fresh_bonus = 3.0
            elif age_days < 90:  fresh_bonus = 1.0
    except Exception:
        pass

    acc = ex.get("accuracy_score")
    acc_bonus = 0.0
    if isinstance(acc, (int, float)):
        acc_bonus = (float(acc) - 50.0) / 10.0

    return weight + src_bonus + rc_bonus + fresh_bonus + acc_bonus


def _required_columns_from_pipeline(query: dict) -> list[str]:
    """Best-effort extract of expected top-level keys in each result row."""
    qt = query.get("queryType", "single")
    cols: list[str] = []

    if qt == "single":
        op   = query.get("operation", "aggregate")
        pipe = query.get("pipeline", []) if op == "aggregate" else []
        for stage in reversed(pipe):
            if not isinstance(stage, dict):
                continue
            if "$group" in stage:
                g = stage["$group"]
                if isinstance(g, dict):
                    cols = list(g.keys())
                    break
            if "$project" in stage:
                p = stage["$project"]
                if isinstance(p, dict):
                    cols = [k for k, v in p.items() if v not in (0, False)]
                    break
        if op == "countDocuments":
            cols = ["count"]
        if op == "distinct":
            cols = ["value"]

    if qt == "dual":
        cols = ["_id"]

    return cols


def _build_eval_entry(ex: dict, idx: int) -> dict:
    question     = ex.get("question", "")
    query        = ex.get("query") or ex.get("query_obj") or {}
    result_count = int(ex.get("result_count") or 0)
    db_hint      = ex.get("db_hint", "stream")

    required_cols = _required_columns_from_pipeline(query)

    return {
        "id":            f"eval-auto-{idx:03d}",
        "question":      question,
        "db_hint":       db_hint,
        "gold_pipeline": query,
        "gold_result_shape": {
            "min_rows":         1 if result_count > 0 else 0,
            "max_rows":         max(200, result_count * 2),
            "required_columns": required_cols[:6],
        },
        "locked":        False,
        "tags":          ["auto-proposed", db_hint, ex.get("source", "auto")],
        "source_example": {
            "weight":         float(ex.get("weight") or 1.0),
            "source":         ex.get("source", "auto"),
            "result_count":   result_count,
            "timestamp":      ex.get("timestamp") or ex.get("created_at", ""),
            "accuracy_score": ex.get("accuracy_score"),
        },
    }


def _load_local() -> list[dict]:
    if not EXAMPLES_FILE.exists():
        print(f"[propose] Local examples file not found: {EXAMPLES_FILE}", file=sys.stderr)
        return []
    try:
        return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[propose] Failed to read {EXAMPLES_FILE}: {e}", file=sys.stderr)
        return []


def _load_feedback_mongo() -> list[dict]:
    try:
        import asyncio
        from lib.mongodb import get_stream_db, close_connections
    except Exception as e:
        print(f"[propose] MongoDB helpers unavailable: {e}", file=sys.stderr)
        return []

    async def _pull() -> list[dict]:
        try:
            db = get_stream_db()
            cursor = db["_QUERY_FEEDBACK_"].find({"rating": "good"}).sort("created_at", -1).limit(500)
            items = []
            async for d in cursor:
                items.append({
                    "question":     d.get("question", ""),
                    "query":        d.get("query_obj", {}),
                    "result_count": int(d.get("result_count") or 0),
                    "db_hint":      _infer_db_hint(d.get("query_obj", {})),
                    "weight":       2.0,
                    "source":       "user_verified",
                    "timestamp":    (d.get("created_at").isoformat()
                                      if d.get("created_at") else ""),
                })
            return items
        except Exception as e:
            print(f"[propose] MongoDB fetch failed: {e}", file=sys.stderr)
            return []
        finally:
            await close_connections()

    return asyncio.run(_pull())


def _infer_db_hint(query_obj: dict) -> str:
    if not isinstance(query_obj, dict):
        return "stream"
    if query_obj.get("queryType") == "dual":
        return "both"
    return "appconfigs" if query_obj.get("database") == "appConfigs" else "stream"


def propose(limit: int, include_feedback: bool) -> list[dict]:
    pool = _load_local()
    if include_feedback:
        pool.extend(_load_feedback_mongo())

    if not pool:
        print("[propose] No candidates found.", file=sys.stderr)
        return []

    pool = [ex for ex in pool
            if ex.get("question") and ex.get("query") and (ex.get("result_count") or 0) > 0]

    seen: dict[frozenset, dict] = {}
    for ex in pool:
        toks = _tokenize(ex["question"])
        if not toks:
            continue
        picked = False
        for key in list(seen.keys()):
            overlap = len(toks & key) / max(1, len(toks | key))
            if overlap > 0.85:
                if _score_candidate(ex) > _score_candidate(seen[key]):
                    del seen[key]
                    seen[toks] = ex
                picked = True
                break
        if not picked:
            seen[toks] = ex

    ranked = sorted(seen.values(), key=_score_candidate, reverse=True)[:limit]
    return [_build_eval_entry(ex, i + 1) for i, ex in enumerate(ranked)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--include-feedback", action="store_true",
                    help="Also pull rating=good entries from MongoDB _QUERY_FEEDBACK_")
    ap.add_argument("--out", type=str, default=str(PROPOSED_FILE_PATH))
    args = ap.parse_args()

    proposed = propose(limit=args.limit, include_feedback=args.include_feedback)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(proposed, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[propose] Wrote {len(proposed)} eval cases to {out_path}")
    by_hint: dict[str, int] = {}
    for e in proposed:
        by_hint[e["db_hint"]] = by_hint.get(e["db_hint"], 0) + 1
    print(f"[propose] Split by db_hint: {by_hint}")
    print(f"[propose] All entries default to locked=false. Review, optionally "
          f"set locked=true on cases you want to freeze as gold, then copy to data/eval_set.json.")


if __name__ == "__main__":
    main()
