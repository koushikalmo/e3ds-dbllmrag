import re
import os
from datetime import datetime, timezone, timedelta

_MONTH_MAP: dict[str, str] = {
    "january": "Jan",   "jan": "Jan",
    "february": "Feb",  "feb": "Feb",
    "march": "Mar",     "mar": "Mar",
    "april": "Apr",     "apr": "Apr",
    "may": "May",
    "june": "Jun",      "jun": "Jun",
    "july": "Jul",      "jul": "Jul",
    "august": "Aug",    "aug": "Aug",
    "september": "Sep", "sept": "Sep", "sep": "Sep",
    "october": "Oct",   "oct": "Oct",
    "november": "Nov",  "nov": "Nov",
    "december": "Dec",  "dec": "Dec",
}

# Longer names first so "september" matches before "sep"
_MONTH_ALT = "|".join(sorted(_MONTH_MAP.keys(), key=len, reverse=True))

_PATTERN_MONTH_YEAR  = re.compile(rf"\b({_MONTH_ALT})\b[,\s]+(\d{{4}})\b",       re.IGNORECASE)
_PATTERN_YEAR_MONTH  = re.compile(rf"\b(\d{{4}})\b[,\s]+({_MONTH_ALT})\b",       re.IGNORECASE)
_PATTERN_MONTH_YEAR2 = re.compile(rf"\b({_MONTH_ALT})\b[,\s]+'?(\d{{2}})\b",     re.IGNORECASE)
_PATTERN_MONTH_ONLY  = re.compile(rf"\b({_MONTH_ALT})\b",                         re.IGNORECASE)


def _expand_year(two_digit: str) -> int:
    y = int(two_digit)
    return 2000 + y if y <= 50 else 1900 + y


def _default_year(default_collection: str) -> int:
    try:
        return int(default_collection.split("_")[1])
    except (IndexError, ValueError):
        return datetime.now(timezone.utc).year


def resolve_collection(question: str, default_collection: str) -> str:
    q     = question.strip()
    now   = datetime.now(timezone.utc)
    lower = q.lower()

    if "last month" in lower:
        prev = now.replace(day=1) - timedelta(days=1)
        return f"{prev.strftime('%b')}_{prev.year}"

    if "this month" in lower or "current month" in lower:
        return f"{now.strftime('%b')}_{now.year}"

    m = _PATTERN_MONTH_YEAR.search(q)
    if m:
        return f"{_MONTH_MAP[m.group(1).lower()]}_{m.group(2)}"

    m = _PATTERN_YEAR_MONTH.search(q)
    if m:
        return f"{_MONTH_MAP[m.group(2).lower()]}_{m.group(1)}"

    m = _PATTERN_MONTH_YEAR2.search(q)
    if m:
        return f"{_MONTH_MAP[m.group(1).lower()]}_{_expand_year(m.group(2))}"

    m = _PATTERN_MONTH_ONLY.search(q)
    if m:
        abbr      = _MONTH_MAP[m.group(1).lower()]
        candidate = f"{abbr}_{_default_year(default_collection)}"
        if candidate != default_collection:
            return candidate

    return default_collection


def resolve_and_log(question: str, default_collection: str) -> str:
    resolved = resolve_collection(question, default_collection)
    if resolved != default_collection:
        print(f"[resolver] '{default_collection}' → '{resolved}' (detected in question)")
    else:
        print(f"[resolver] Using default collection: '{resolved}'")
    return resolved


_PATTERN_YEAR_ONLY = re.compile(r'\b(20\d{2})\b')
_MONTH_ABBRS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def resolve_year(question: str) -> int | None:
    """Return the year if the question targets a full year with no specific month mentioned."""
    if _PATTERN_MONTH_ONLY.search(question):
        return None  # specific month takes precedence
    m = _PATTERN_YEAR_ONLY.search(question)
    return int(m.group(1)) if m else None


def all_collections_for_year(year: int) -> list[str]:
    return [f"{m}_{year}" for m in _MONTH_ABBRS]
