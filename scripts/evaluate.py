"""Regression evaluator for the NL→MongoDB query pipeline.

Runs every case in ``data/eval_set.json`` through the running ``/api/query``
endpoint, scores each response with the accuracy scorer, compares it to the
gold_pipeline (when ``locked``) or gold_result_shape (when unlocked), and
writes a report to ``data/eval_report.json``. Also saves a baseline file
so the next run can show deltas.

Usage:
    python3 scripts/evaluate.py                        # uses http://localhost:8000
    python3 scripts/evaluate.py --url http://host:8000 # custom target
    python3 scripts/evaluate.py --timeout 60           # per-case timeout (s)
    python3 scripts/evaluate.py --case eval-auto-001   # just one case
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import httpx

EVAL_SET_PATH = ROOT / "data" / "eval_set.json"
REPORT_PATH   = ROOT / "data" / "eval_report.json"
BASELINE_PATH = ROOT / "data" / ".eval_baseline.json"


# Pipeline normaliser — makes two equivalent pipelines compare equal.
# Canonicalises dict key order and ignores transient $limit value differences.
def _canon(obj):
    if isinstance(obj, dict):
        return {k: _canon(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canon(x) for x in obj]
    return obj


def _structural_equal(a: dict, b: dict) -> bool:
    return json.dumps(_canon(a)) == json.dumps(_canon(b))


def _pipelines_equal(gold: dict, got: dict) -> tuple[bool, list[str]]:
    diffs: list[str] = []
    if gold.get("queryType") != got.get("queryType"):
        diffs.append(f"queryType: gold={gold.get('queryType')} got={got.get('queryType')}")
        return False, diffs

    if gold.get("queryType") == "single":
        if gold.get("database") != got.get("database"):
            diffs.append(f"database: gold={gold.get('database')} got={got.get('database')}")
        if gold.get("operation", "aggregate") != got.get("operation", "aggregate"):
            diffs.append(f"operation: gold={gold.get('operation')} got={got.get('operation')}")
        gp = gold.get("pipeline") or []
        ap = got.get("pipeline") or []
        if gp or ap:
            if not _structural_equal(gp, ap):
                diffs.append(f"pipeline stages differ (gold={len(gp)} got={len(ap)})")

    elif gold.get("queryType") == "dual":
        gq = gold.get("queries", [])
        aq = got.get("queries", [])
        if len(gq) != len(aq):
            diffs.append(f"dual query count: gold={len(gq)} got={len(aq)}")
        else:
            for i, (g, a) in enumerate(zip(gq, aq)):
                if g.get("database") != a.get("database"):
                    diffs.append(f"queries[{i}].database: gold={g.get('database')} got={a.get('database')}")
                if not _structural_equal(g.get("pipeline", []), a.get("pipeline", [])):
                    diffs.append(f"queries[{i}].pipeline differs")

    return (len(diffs) == 0), diffs


def _shape_ok(gold_shape: dict, got_rows: list, got_cols: set[str]) -> tuple[bool, list[str]]:
    diffs: list[str] = []
    min_rows = int(gold_shape.get("min_rows", 0))
    max_rows = int(gold_shape.get("max_rows", 10**6))
    req_cols = set(gold_shape.get("required_columns") or [])

    n = len(got_rows)
    if n < min_rows:
        diffs.append(f"rows {n} < min_rows {min_rows}")
    if n > max_rows:
        diffs.append(f"rows {n} > max_rows {max_rows}")
    if req_cols and n > 0:
        missing = req_cols - got_cols
        if missing:
            diffs.append(f"missing required columns: {sorted(missing)}")

    return (len(diffs) == 0), diffs


def _extract_rows_and_cols(response_data: dict) -> tuple[list, set[str]]:
    rows: list = []
    results = response_data.get("results")
    if isinstance(results, list):
        rows = results
    elif isinstance(results, dict):
        rows = results.get("merged") or results.get("primary") or []

    cols: set[str] = set()
    if rows and isinstance(rows[0], dict):
        cols = set(rows[0].keys())
    return rows, cols


# single-case runner
async def _run_one(client: httpx.AsyncClient, url: str, case: dict, timeout: float) -> dict:
    case_id    = case["id"]
    question   = case["question"]
    collection = case.get("collection") or os.getenv("DEFAULT_STREAM_COLLECTION", "Apr_2026")
    locked     = bool(case.get("locked"))

    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/api/query",
            json    = {"question": question, "collection": collection, "session_id": "eval"},
            timeout = timeout,
        )
        elapsed = round(time.perf_counter() - t0, 2)
    except Exception as e:
        return {
            "id": case_id, "question": question, "passed": False, "elapsed": None,
            "error": f"request failed: {e}", "score": 0, "tier": "low",
        }

    if resp.status_code != 200:
        return {
            "id": case_id, "question": question, "passed": False, "elapsed": elapsed,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}", "score": 0, "tier": "low",
        }

    payload = resp.json().get("data", {}) or {}
    accuracy = payload.get("accuracy") or {}
    score    = int(accuracy.get("score") or 0)
    tier     = accuracy.get("tier", "low")

    got_query = payload.get("queryMeta") or {}
    rows, cols = _extract_rows_and_cols(payload)

    diffs: list[str] = []
    passed = True

    if locked:
        ok, d = _pipelines_equal(case.get("gold_pipeline", {}), got_query)
        passed = passed and ok
        diffs.extend(f"pipeline: {s}" for s in d)

    ok, d = _shape_ok(case.get("gold_result_shape", {}), rows, cols)
    passed = passed and ok
    diffs.extend(f"shape: {s}" for s in d)

    return {
        "id":       case_id,
        "question": question[:120],
        "locked":   locked,
        "passed":   passed,
        "score":    score,
        "tier":     tier,
        "elapsed":  elapsed,
        "rows":     len(rows),
        "diffs":    diffs[:6],
    }


# reporting
def _summary(report: dict) -> dict:
    cases = report["cases"]
    n     = len(cases)
    passed = sum(1 for c in cases if c["passed"])
    scores = [c["score"] for c in cases if isinstance(c.get("score"), (int, float))]
    avg    = round(sum(scores) / len(scores), 1) if scores else 0.0
    return {"total": n, "passed": passed, "failed": n - passed, "avg_score": avg}


def _compute_delta(summary: dict, baseline_path: Path) -> dict | None:
    if not baseline_path.exists():
        return None
    try:
        prev = json.loads(baseline_path.read_text(encoding="utf-8"))
        return {
            "passed_delta":    summary["passed"]    - prev.get("passed", 0),
            "avg_score_delta": round(summary["avg_score"] - prev.get("avg_score", 0.0), 1),
        }
    except Exception:
        return None


def _print_report(cases: list[dict], summary: dict, delta: dict | None) -> None:
    for c in cases:
        mark = "PASS" if c["passed"] else "FAIL"
        err  = f" — {c['error']}" if c.get("error") else (f" ({'; '.join(c['diffs'])})" if c.get("diffs") else "")
        print(f"  {c['id']:<18s} [{mark}] score={c['score']:3d} rows={c.get('rows', '—'):>3} {c['question'][:60]}{err}")

    print()
    print(f"Summary: {summary['passed']} / {summary['total']} passed ({100*summary['passed']//max(1,summary['total'])}%)")
    print(f"Average score: {summary['avg_score']}")
    if delta is not None:
        sign = lambda x: f"+{x}" if x > 0 else str(x)
        print(f"Δ vs baseline: passed {sign(delta['passed_delta'])}  "
              f"avg_score {sign(delta['avg_score_delta'])}")
    else:
        print("(no baseline found — this run will become the baseline)")


# entry point
async def run(url: str, timeout: float, only_case: str | None) -> dict:
    if not EVAL_SET_PATH.exists():
        print(f"[eval] {EVAL_SET_PATH} not found — run scripts/propose_eval_set.py first",
              file=sys.stderr)
        sys.exit(2)

    cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    if only_case:
        cases = [c for c in cases if c["id"] == only_case]
        if not cases:
            print(f"[eval] No case with id '{only_case}'", file=sys.stderr)
            sys.exit(2)

    print(f"[eval] Running {len(cases)} case(s) against {url} ...")

    async with httpx.AsyncClient() as client:
        results = []
        for i, case in enumerate(cases, 1):
            print(f"  ({i}/{len(cases)}) {case['id']} ...", flush=True)
            r = await _run_one(client, url, case, timeout)
            results.append(r)

    report = {
        "ran_at":    datetime.now(timezone.utc).isoformat(),
        "target":    url,
        "total":     len(results),
        "cases":     results,
    }
    summary   = _summary(report)
    delta     = _compute_delta(summary, BASELINE_PATH)
    report.update(summary=summary, delta=delta)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    BASELINE_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    _print_report(results, summary, delta)
    print()
    print(f"Full report: {REPORT_PATH}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url",     default=os.getenv("EVAL_TARGET_URL", "http://localhost:8000"))
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--case",    default=None, help="Run only the case with this id")
    args = ap.parse_args()

    import asyncio
    asyncio.run(run(url=args.url, timeout=args.timeout, only_case=args.case))


if __name__ == "__main__":
    main()
