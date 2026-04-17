from __future__ import annotations
import json


def validate_query_and_result(
    query_obj:    dict,
    result:       dict,
    result_count: int,
) -> list[dict]:
    checks:   list[dict] = []
    qt        = query_obj.get("queryType", "single")
    operation = query_obj.get("operation", "aggregate")
    database  = query_obj.get("database", "")

    if operation == "countDocuments":
        checks.append(_pass("RESULT_COUNT", "countDocuments operation — exact count returned."))
    elif result_count == 0:
        checks.append(_warn(
            "ZERO_RESULTS",
            "No records returned. Possible causes: wrong collection month, "
            "field value is case-sensitive and does not match, query filter is too specific, "
            "or the field name is incorrect.",
        ))
    else:
        checks.append(_pass("RESULT_COUNT", f"{result_count} record{'s' if result_count != 1 else ''} returned."))

    if result_count >= 190:
        checks.append(_warn(
            "NEAR_LIMIT",
            f"Result set hit the 200-document cap ({result_count} returned). "
            "The true total is likely larger — use a $group + $count pipeline to get the real number.",
        ))

    if database == "stream-datastore":
        if _has_employee_filter(query_obj):
            checks.append(_pass(
                "EMPLOYEE_FILTER",
                "Internal employee traffic correctly excluded (e3ds_employee filter present).",
            ))
        else:
            checks.append(_warn(
                "NO_EMPLOYEE_FILTER",
                "e3ds_employee filter not detected. Internal test/employee sessions may be "
                "included in results, which inflates session counts and skews metrics.",
            ))

    for pipe in _get_pipelines(query_obj):
        if _limit_before_group(pipe):
            checks.append(_error(
                "LIMIT_BEFORE_GROUP",
                "$limit appears before $group in the pipeline. "
                "This truncates input data before aggregating, producing wrong totals. "
                "The $limit stage must always come AFTER $group and $sort.",
            ))
            break

    if operation == "aggregate" and database == "stream-datastore":
        for pipe in _get_pipelines(query_obj):
            if pipe and not any("$match" in stage for stage in pipe):
                checks.append(_info(
                    "NO_MATCH_FILTER",
                    "No $match stage found — the entire collection was scanned. "
                    "Adding a $match filter improves performance on large monthly collections.",
                ))
                break

    pipeline_text = json.dumps(_get_pipelines(query_obj))
    if "avgRoundTripTime" in pipeline_text:
        if "$toDouble" in pipeline_text or "$toDecimal" in pipeline_text or "$toFloat" in pipeline_text:
            checks.append(_pass(
                "RTT_CONVERSION",
                "webRtcStatsData.avgRoundTripTime correctly converted to a number "
                "before sorting or comparison.",
            ))
        else:
            checks.append(_warn(
                "RTT_NO_CONVERSION",
                "webRtcStatsData.avgRoundTripTime is stored as a STRING in MongoDB. "
                "Sorting or filtering without $toDouble will give wrong results. "
                "Wrap with: { \"$toDouble\": \"$webRtcStatsData.avgRoundTripTime\" }",
            ))

    if qt == "dual":
        checks.append(_info(
            "DUAL_QUERY",
            "Cross-database query: results from stream-datastore and appConfigs are "
            "fetched in parallel and merged in Python by owner key — "
            "this is not a native MongoDB $lookup.",
        ))

    return checks


def _pass(code: str, message: str)    -> dict: return {"level": "pass",    "code": code, "message": message}
def _info(code: str, message: str)    -> dict: return {"level": "info",    "code": code, "message": message}
def _warn(code: str, message: str)    -> dict: return {"level": "warning", "code": code, "message": message}
def _error(code: str, message: str)   -> dict: return {"level": "error",   "code": code, "message": message}


def _has_employee_filter(query_obj: dict) -> bool:
    return "e3ds_employee" in json.dumps(query_obj)


def _get_pipelines(query_obj: dict) -> list[list]:
    qt = query_obj.get("queryType", "single")
    if qt == "single":
        return [query_obj.get("pipeline", [])]
    if qt == "dual":
        return [q.get("pipeline", []) for q in query_obj.get("queries", [])]
    return []


def _limit_before_group(pipeline: list) -> bool:
    seen_limit = False
    for stage in pipeline:
        if not isinstance(stage, dict):
            continue
        key = next(iter(stage), None)
        if key == "$limit":
            seen_limit = True
        elif key == "$group" and seen_limit:
            return True
    return False
