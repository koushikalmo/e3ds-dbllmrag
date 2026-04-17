import re
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

# numeric month → abbreviation
_NUM_TO_MONTH = {
    1: "Jan", 2: "Feb",  3: "Mar", 4: "Apr",
    5: "May", 6: "Jun",  7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# first month of each quarter
_QUARTER_START = {1: "Jan", 2: "Apr", 3: "Jul", 4: "Oct"}

_MONTH_ABBRS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# longer names first so "september" is tried before "sep"
_MONTH_ALT = "|".join(sorted(_MONTH_MAP.keys(), key=len, reverse=True))

# ── compiled patterns (ordered from most-specific to least) ──────────────────

# "16th April 2026", "16 April 2026", "16th of April 2026"
_PAT_ORDINAL_DMY = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+({_MONTH_ALT})\b[,\s]*(\d{{4}})\b",
    re.IGNORECASE,
)
# "April 16th, 2026", "April 16 2026"
_PAT_ORDINAL_MDY = re.compile(
    rf"\b({_MONTH_ALT})\b[,\s]+(\d{{1,2}})(?:st|nd|rd|th)?[,\s]+(\d{{4}})\b",
    re.IGNORECASE,
)
# ISO: 2026-04-16 or 2026/04/16
_PAT_ISO = re.compile(r'\b(20\d{2})[-/](\d{1,2})[-/]\d{1,2}\b')
# US numeric: 04/16/2026 or 04-16-2026
_PAT_US_DATE = re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b')
# Quarter: Q1 2026, Q2 2026
_PAT_QUARTER = re.compile(r'\bQ([1-4])\s*(20\d{2})\b', re.IGNORECASE)
# "first/second/third/fourth quarter of 2026"
_PAT_QUARTER_WORD = re.compile(
    r'\b(first|second|third|fourth)\s+quarter(?:\s+of)?\s+(20\d{2})\b',
    re.IGNORECASE,
)
# Month + 4-digit year: "April 2026", "apr 2026"
_PAT_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALT})\b[,.\s]+(\d{{4}})\b", re.IGNORECASE)
# Year + Month: "2026 April"
_PAT_YEAR_MONTH = re.compile(rf"\b(\d{{4}})\b[,.\s]+({_MONTH_ALT})\b", re.IGNORECASE)
# Month + 2-digit year: "Apr '26", "April 26"
_PAT_MONTH_YEAR2 = re.compile(rf"\b({_MONTH_ALT})\b[,.\s]+'?(\d{{2}})\b", re.IGNORECASE)
# Month only (lowest priority)
_PAT_MONTH_ONLY = re.compile(rf"\b({_MONTH_ALT})\b", re.IGNORECASE)
# Full year only
_PAT_YEAR_ONLY = re.compile(r'\b(20\d{2})\b')

_QUARTER_WORD_MAP = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def _expand_year(two_digit: str) -> int:
    y = int(two_digit)
    return 2000 + y if y <= 50 else 1900 + y


def _default_year(default_collection: str) -> int:
    try:
        return int(default_collection.split("_")[1])
    except (IndexError, ValueError):
        return datetime.now(timezone.utc).year


def _col(abbr: str, year) -> str:
    return f"{abbr}_{year}"


def resolve_collection(question: str, default_collection: str) -> str:
    q     = question.strip()
    now   = datetime.now(timezone.utc)
    lower = q.lower()

    # ── relative day references ───────────────────────────────────────────────
    if "yesterday" in lower:
        d = now - timedelta(days=1)
        return _col(d.strftime("%b"), d.year)

    if "today" in lower:
        return _col(now.strftime("%b"), now.year)

    # "last N days" / "past N days" → find the month N days ago
    m = re.search(r'\b(?:last|past)\s+(\d+)\s+days?\b', lower)
    if m:
        d = now - timedelta(days=int(m.group(1)))
        return _col(d.strftime("%b"), d.year)

    # "last N weeks"
    m = re.search(r'\b(?:last|past)\s+(\d+)\s+weeks?\b', lower)
    if m:
        d = now - timedelta(weeks=int(m.group(1)))
        return _col(d.strftime("%b"), d.year)

    if "last month" in lower:
        prev = now.replace(day=1) - timedelta(days=1)
        return _col(prev.strftime("%b"), prev.year)

    if "this month" in lower or "current month" in lower:
        return _col(now.strftime("%b"), now.year)

    if "last year" in lower:
        return _col("Jan", now.year - 1)  # year query — $unionWith expansion covers all months

    if "this year" in lower or "current year" in lower:
        return _col("Jan", now.year)

    # ── specific date formats ─────────────────────────────────────────────────

    # ordinal DMY: "16th April 2026"
    m = _PAT_ORDINAL_DMY.search(q)
    if m:
        return _col(_MONTH_MAP[m.group(2).lower()], m.group(3))

    # ordinal MDY: "April 16th 2026"
    m = _PAT_ORDINAL_MDY.search(q)
    if m:
        return _col(_MONTH_MAP[m.group(1).lower()], m.group(3))

    # ISO date: 2026-04-16
    m = _PAT_ISO.search(q)
    if m:
        month_num = int(m.group(2))
        if 1 <= month_num <= 12:
            return _col(_NUM_TO_MONTH[month_num], m.group(1))

    # US date: 04/16/2026
    m = _PAT_US_DATE.search(q)
    if m:
        month_num = int(m.group(1))
        if 1 <= month_num <= 12:
            return _col(_NUM_TO_MONTH[month_num], m.group(3))

    # quarter word: "first quarter of 2026"
    m = _PAT_QUARTER_WORD.search(q)
    if m:
        q_num = _QUARTER_WORD_MAP[m.group(1).lower()]
        return _col(_QUARTER_START[q_num], m.group(2))

    # quarter: Q1 2026
    m = _PAT_QUARTER.search(q)
    if m:
        return _col(_QUARTER_START[int(m.group(1))], m.group(2))

    # ── month + year ──────────────────────────────────────────────────────────
    m = _PAT_MONTH_YEAR.search(q)
    if m:
        return _col(_MONTH_MAP[m.group(1).lower()], m.group(2))

    m = _PAT_YEAR_MONTH.search(q)
    if m:
        return _col(_MONTH_MAP[m.group(2).lower()], m.group(1))

    m = _PAT_MONTH_YEAR2.search(q)
    if m:
        return _col(_MONTH_MAP[m.group(1).lower()], _expand_year(m.group(2)))

    # ── month only — assume year from default collection ──────────────────────
    m = _PAT_MONTH_ONLY.search(q)
    if m:
        abbr      = _MONTH_MAP[m.group(1).lower()]
        candidate = _col(abbr, _default_year(default_collection))
        if candidate != default_collection:
            return candidate

    return default_collection


def resolve_and_log(question: str, default_collection: str) -> str:
    resolved = resolve_collection(question, default_collection)
    if resolved != default_collection:
        print(f"[resolver] '{default_collection}' → '{resolved}' (extracted from question)")
    else:
        print(f"[resolver] No date detected — using default: '{resolved}'")
    return resolved


def resolve_year(question: str) -> int | None:
    """Return the year if the question targets a full year with no specific month mentioned."""
    if _PAT_MONTH_ONLY.search(question):
        return None  # specific month takes precedence
    m = _PAT_YEAR_ONLY.search(question)
    return int(m.group(1)) if m else None


def all_collections_for_year(year: int) -> list[str]:
    return [f"{m}_{year}" for m in _MONTH_ABBRS]
