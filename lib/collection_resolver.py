# ============================================================
# lib/collection_resolver.py — Auto-detect Collection from Question
# ============================================================
# Parses month/year references from a natural-language question
# and maps them to MongoDB collection names (e.g. "Oct_2025").
#
# This runs BEFORE the LLM so the correct collection is already
# set when the prompt is assembled. The LLM is then told which
# collection to use — it doesn't have to guess.
#
# COLLECTION NAME FORMAT:
#   3-letter month abbreviation + underscore + 4-digit year
#   Examples: Apr_2026, Oct_2025, Dec_2024, Jan_2025
#
# SUPPORTED PATTERNS:
#   "October 2025"          → Oct_2025
#   "in march 2026"         → Mar_2026
#   "2025 november"         → Nov_2025
#   "oct 25" / "oct '25"    → Oct_2025  (2-digit year)
#   "last month"            → previous calendar month
#   "this month"            → current calendar month
#   "april" (no year)       → April of the default collection's year
#   no date mention         → default_collection unchanged
# ============================================================

import re
import os
from datetime import datetime, timezone, timedelta


# ── Month name → 3-letter abbreviation ────────────────────────
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

# Regex alternation — longer names first so "september" beats "sep"
_MONTH_ALT = "|".join(sorted(_MONTH_MAP.keys(), key=len, reverse=True))

# Compiled patterns (fastest order: most specific first)
_PATTERN_MONTH_YEAR = re.compile(
    rf"\b({_MONTH_ALT})\b[,\s]+(\d{{4}})\b",
    re.IGNORECASE,
)
_PATTERN_YEAR_MONTH = re.compile(
    rf"\b(\d{{4}})\b[,\s]+({_MONTH_ALT})\b",
    re.IGNORECASE,
)
# "oct 25" or "oct '25" — 2-digit year shorthand
_PATTERN_MONTH_YEAR2 = re.compile(
    rf"\b({_MONTH_ALT})\b[,\s]+'?(\d{{2}})\b",
    re.IGNORECASE,
)
# Month name alone, no year (e.g. "in april")
_PATTERN_MONTH_ONLY = re.compile(
    rf"\b({_MONTH_ALT})\b",
    re.IGNORECASE,
)


def _expand_year(two_digit: str) -> int:
    """Converts a 2-digit year to 4-digit: '25' → 2025, '99' → 1999."""
    y = int(two_digit)
    return 2000 + y if y <= 50 else 1900 + y


def _default_year(default_collection: str) -> int:
    """
    Extracts the year from the default collection name.
    Falls back to the current UTC year if parsing fails.

    Example: "Apr_2026" → 2026
    """
    try:
        return int(default_collection.split("_")[1])
    except (IndexError, ValueError):
        return datetime.now(timezone.utc).year


def resolve_collection(question: str, default_collection: str) -> str:
    """
    Detects a month/year reference in the question and returns the
    corresponding collection name. Returns ``default_collection`` if
    no date reference is found.

    Args:
        question:           The user's natural-language question.
        default_collection: Fallback collection (e.g. "Apr_2026").

    Returns:
        Collection name string, e.g. "Oct_2025" or the unchanged default.
    """
    q = question.strip()
    now = datetime.now(timezone.utc)

    # ── Relative references ────────────────────────────────────
    q_lower = q.lower()

    if "last month" in q_lower:
        # Go back one calendar month
        first_of_this = now.replace(day=1)
        last_month_dt = first_of_this - timedelta(days=1)
        return f"{last_month_dt.strftime('%b')}_{last_month_dt.year}"

    if "this month" in q_lower or "current month" in q_lower:
        return f"{now.strftime('%b')}_{now.year}"

    # ── "Month YYYY" — most common explicit form ───────────────
    m = _PATTERN_MONTH_YEAR.search(q)
    if m:
        abbr = _MONTH_MAP[m.group(1).lower()]
        return f"{abbr}_{m.group(2)}"

    # ── "YYYY Month" ───────────────────────────────────────────
    m = _PATTERN_YEAR_MONTH.search(q)
    if m:
        abbr = _MONTH_MAP[m.group(2).lower()]
        return f"{abbr}_{m.group(1)}"

    # ── "Month '25" / "Month 25" (2-digit year) ───────────────
    m = _PATTERN_MONTH_YEAR2.search(q)
    if m:
        abbr = _MONTH_MAP[m.group(1).lower()]
        year = _expand_year(m.group(2))
        return f"{abbr}_{year}"

    # ── Month name only (no year) ──────────────────────────────
    # Use the year from the default collection so "april" stays
    # within the same year context the user is working in.
    m = _PATTERN_MONTH_ONLY.search(q)
    if m:
        abbr  = _MONTH_MAP[m.group(1).lower()]
        year  = _default_year(default_collection)
        candidate = f"{abbr}_{year}"
        # Only override if different from the default — avoids a no-op
        if candidate != default_collection:
            return candidate

    return default_collection


def resolve_and_log(question: str, default_collection: str) -> str:
    """
    Resolves the collection and prints a log line if the collection changed.
    Drop-in for resolve_collection when you want server-side visibility.
    """
    resolved = resolve_collection(question, default_collection)
    if resolved != default_collection:
        print(f"[resolver] '{default_collection}' → '{resolved}' (detected in question)")
    else:
        print(f"[resolver] Using default collection: '{resolved}'")
    return resolved
