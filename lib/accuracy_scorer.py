"""Scores each query 0-100 from weighted signals. Five of them are
deterministic and cheap (a few ms). The sixth asks the LLM for a second
opinion and is off by default (ACCURACY_USE_LLM_CRITIQUE=true) because it
adds a whole model round-trip per query.
"""
from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

USE_LLM_CRITIQUE_DEFAULT = os.getenv("ACCURACY_USE_LLM_CRITIQUE", "false").lower() == "true"


SIGNAL_WEIGHTS: dict[str, int] = {
    "syntax":       10,
    "schema":       25,
    "relationship": 15,
    "validator":    20,
    "execution":    20,
    "llm_critique": 10,
}


@dataclass
class SignalScore:
    score: int                       # 0–100
    notes: list[str] = field(default_factory=list)
    weight: int      = 0
    enabled: bool    = True


@dataclass
class ScoreBreakdown:
    score:   int
    tier:    str                     # "high" | "medium" | "low"
    signals: dict[str, SignalScore]

    def to_dict(self) -> dict:
        return {
            "score":   self.score,
            "tier":    self.tier,
            "signals": {k: asdict(v) for k, v in self.signals.items()},
        }


_BANNED_STAGES  = frozenset({"$out", "$merge"})
_VALID_STAGES   = frozenset({
    "$match", "$project", "$group", "$sort", "$limit", "$skip", "$unwind",
    "$addFields", "$set", "$lookup", "$facet", "$bucket", "$count", "$replaceRoot",
    "$replaceWith", "$sample", "$geoNear", "$graphLookup", "$redact", "$fill",
    "$unionWith", "$densify", "$documents", "$search", "$out", "$merge",
})


def _clamp(n: int) -> int:
    return max(0, min(100, int(n)))


def _tier(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _extract_pipelines(query_obj: dict) -> list[list]:
    qt = query_obj.get("queryType", "single")
    if qt == "single":
        if query_obj.get("operation", "aggregate") == "aggregate":
            pipe = query_obj.get("pipeline")
            if isinstance(pipe, list):
                return [pipe]
        return []
    if qt == "dual":
        out = []
        for q in query_obj.get("queries", []):
            if isinstance(q.get("pipeline"), list):
                out.append(q["pipeline"])
        return out
    return []


# signal 1: pipeline shape — valid stages, no $limit before $group
def _score_syntax(query_obj: dict) -> SignalScore:
    notes: list[str]   = []
    penalty            = 0

    qt = query_obj.get("queryType", "single")
    if qt not in ("single", "dual"):
        return SignalScore(score=0, notes=[f"unknown queryType: '{qt}'"])

    pipelines = _extract_pipelines(query_obj)
    for pipe in pipelines:
        for i, stage in enumerate(pipe):
            if not isinstance(stage, dict) or len(stage) == 0:
                penalty += 30
                notes.append(f"malformed stage at index {i}")
                continue
            op = next(iter(stage))
            if op in _BANNED_STAGES:
                penalty += 30
                notes.append(f"banned write stage '{op}'")
            elif not op.startswith("$"):
                penalty += 20
                notes.append(f"stage '{op}' does not start with '$'")

        ops = [next(iter(s), None) for s in pipe if isinstance(s, dict)]
        if "$group" in ops:
            gi = ops.index("$group")
            if "$limit" in ops[:gi]:
                penalty += 25
                notes.append("$limit appears before $group (truncates input)")

    if qt == "dual":
        if len(query_obj.get("queries", [])) != 2:
            penalty += 30
            notes.append(f"dual query must have exactly 2 queries (got {len(query_obj.get('queries', []))})")
        if not query_obj.get("mergeKey"):
            penalty += 15
            notes.append("dual query missing 'mergeKey'")

    score = _clamp(100 - penalty)
    if not notes:
        notes.append("pipeline structure valid")
    return SignalScore(score=score, notes=notes)


# signal 2: every referenced field exists in the known-fields set
def _score_schema(query_obj: dict, known_fields: set[str] | None = None) -> SignalScore:
    notes: list[str] = []

    if known_fields is None:
        try:
            from lib.query_generator import _get_known_fields
            known_fields = _get_known_fields()
        except Exception as e:
            return SignalScore(score=80, notes=[f"known-fields lookup failed — ungraded: {e}"])

    refs = _collect_field_refs(query_obj)
    if not refs:
        return SignalScore(score=100, notes=["no dotted field references to check"])

    bad = []
    for ref in refs:
        if "." not in ref:
            continue
        if ref not in known_fields:
            bad.append(ref)

    if not bad:
        return SignalScore(score=100, notes=[f"all {len(refs)} field refs valid"])

    penalty = min(100, 15 * len(bad))
    score = _clamp(100 - penalty)
    notes.append(f"{len(bad)} unknown field reference(s): {bad[:5]}")
    return SignalScore(score=score, notes=notes)


def _collect_field_refs(obj, depth: int = 0) -> set[str]:
    refs: set[str] = set()
    if depth > 12:
        return refs
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "." in k and not k.startswith("$"):
                refs.add(k)
            if isinstance(v, str) and v.startswith("$") and "." in v:
                refs.add(v[1:])
            refs.update(_collect_field_refs(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            refs.update(_collect_field_refs(item, depth + 1))
    elif isinstance(obj, str) and obj.startswith("$") and "." in obj:
        refs.add(obj[1:])
    return refs


def _is_relationship_check(c: dict) -> bool:
    return str(c.get("code", "")).startswith("REL_")


# signal 3: cross-DB join / discriminator correctness. Takes only the REL_*
# checks so signal 4 doesn't count the same findings twice.
def _score_relationship(validator_checks: list[dict]) -> SignalScore:
    rel = [c for c in validator_checks if _is_relationship_check(c)]
    if not rel:
        return SignalScore(score=100, notes=["no relationship checks fired"])

    errors = [c for c in rel if c["level"] == "error"]
    warns  = [c for c in rel if c["level"] == "warning"]
    passes = [c for c in rel if c["level"] == "pass"]

    penalty = 40 * len(errors) + 15 * len(warns)
    bonus   = 5  * len(passes)
    score   = _clamp(100 - penalty + bonus)

    notes: list[str] = []
    for c in errors[:3]:  notes.append(f"ERROR {c['code']}: {c['message'][:80]}")
    for c in warns [:2]:  notes.append(f"WARN  {c['code']}: {c['message'][:80]}")
    if not errors and not warns:
        notes.append(f"{len(passes)} relationship check(s) passed")
    return SignalScore(score=score, notes=notes)


# signal 4: the remaining rule-based validator checks (non-REL_*)
def _score_validator(validator_checks: list[dict]) -> SignalScore:
    rule_checks = [c for c in validator_checks if not _is_relationship_check(c)]
    if not rule_checks:
        return SignalScore(score=60, notes=["no rule-based validator checks fired"])

    errs  = sum(1 for c in rule_checks if c.get("level") == "error")
    warns = sum(1 for c in rule_checks if c.get("level") == "warning")
    passes= sum(1 for c in rule_checks if c.get("level") == "pass")
    infos = sum(1 for c in rule_checks if c.get("level") == "info")

    score = _clamp(50 + 10 * passes - 10 * warns - 30 * errs + 2 * infos)
    notes = [f"pass={passes} warn={warns} err={errs} info={infos}"]
    return SignalScore(score=score, notes=notes)


# signal 5: did it actually run, and did rows come back
def _score_execution(result: dict | None, result_count: int) -> SignalScore:
    if not isinstance(result, dict):
        return SignalScore(score=0, notes=["no result object"])

    if result.get("error") or result.get("timeout"):
        return SignalScore(score=0, notes=[f"execution failed: {result.get('error') or 'timeout'}"])

    if result_count == 0:
        return SignalScore(score=70, notes=["query ran but returned 0 rows"])

    return SignalScore(score=100, notes=[f"{result_count} row(s) returned"])


# signal 6 (optional): ask the LLM to grade its own query
async def _score_llm_critique(question: str, query_obj: dict) -> SignalScore:
    try:
        from lib.llm_provider import generate_with_fallback
    except Exception as e:
        return SignalScore(score=50, notes=[f"llm_provider unavailable: {e}"])

    pipeline_text = json.dumps(query_obj, indent=None)[:1500]
    sys_prompt = (
        "You are a MongoDB query reviewer. Given a user question and a generated "
        "query, reply with ONE raw JSON object: {\"score\": <0-100>, \"reason\": \"<one sentence>\"}. "
        "Score 0-100 how well the query answers the question. Nothing else."
    )
    user_msg = (
        f"Question: {question}\n"
        f"Generated query: {pipeline_text}\n"
        f"Reply with ONLY the JSON object."
    )

    try:
        raw, _ = await generate_with_fallback(sys_prompt, user_msg)
    except Exception as e:
        return SignalScore(score=50, notes=[f"critique call failed: {e}"])

    try:
        text = raw.strip()
        if text.startswith("```"):
            nl = text.find("\n")
            text = text[nl + 1:] if nl != -1 else text[3:]
            if text.endswith("```"): text = text[:-3].rstrip()
        obj = json.loads(text)
        s = _clamp(int(obj.get("score", 50)))
        r = str(obj.get("reason", ""))[:120]
        return SignalScore(score=s, notes=[f"LLM: {r}"] if r else ["LLM critique ok"])
    except Exception as e:
        return SignalScore(score=50, notes=[f"critique parse failed: {e}; raw='{raw[:80]}...'"])


class AccuracyScorer:
    """Runs all signals and folds them into one weighted 0-100 score."""

    def __init__(self, use_llm_critique: bool = USE_LLM_CRITIQUE_DEFAULT):
        self.use_llm_critique = use_llm_critique

    async def score(
        self,
        question:         str,
        query_obj:        dict,
        result:           dict | None,
        result_count:     int,
        validator_checks: list[dict],
        *,
        known_fields:     set[str] | None = None,
    ) -> ScoreBreakdown:
        signals: dict[str, SignalScore] = {}

        signals["syntax"]       = _score_syntax(query_obj)
        signals["schema"]       = _score_schema(query_obj, known_fields=known_fields)
        signals["relationship"] = _score_relationship(validator_checks)
        signals["validator"]    = _score_validator(validator_checks)
        signals["execution"]    = _score_execution(result, result_count)

        if self.use_llm_critique:
            signals["llm_critique"] = await _score_llm_critique(question, query_obj)
        else:
            signals["llm_critique"] = SignalScore(
                score   = 0,
                notes   = ["disabled — set ACCURACY_USE_LLM_CRITIQUE=true to enable"],
                enabled = False,
            )

        for name, sig in signals.items():
            sig.weight = SIGNAL_WEIGHTS[name]

        # weighted average over the enabled signals only, so disabling the
        # LLM critique doesn't silently drag the score down
        total_w = sum(sig.weight for sig in signals.values() if sig.enabled)
        if total_w == 0:
            final = 0
        else:
            final = round(sum(sig.score * sig.weight for sig in signals.values() if sig.enabled) / total_w)
        final = _clamp(final)

        return ScoreBreakdown(score=final, tier=_tier(final), signals=signals)


async def score_query(
    question:         str,
    query_obj:        dict,
    result:           dict | None,
    result_count:     int,
    validator_checks: list[dict],
    *,
    known_fields:     set[str] | None = None,
    use_llm_critique: bool | None     = None,
) -> ScoreBreakdown:
    # one-call wrapper used by /api/query
    scorer = AccuracyScorer(
        use_llm_critique = USE_LLM_CRITIQUE_DEFAULT if use_llm_critique is None else use_llm_critique
    )
    return await scorer.score(
        question         = question,
        query_obj        = query_obj,
        result           = result,
        result_count     = result_count,
        validator_checks = validator_checks,
        known_fields     = known_fields,
    )
