# ============================================================
# lib/result_summarizer.py — Chunked Result Summarizer
# ============================================================
# When MongoDB returns a large result set, we want to let
# users ask the LLM to analyze or summarize it. But LLMs have
# a finite context window — you can't just paste 200 documents
# into the prompt and expect good results.
#
# THIS MODULE SOLVES THAT WITH A MAP-REDUCE APPROACH:
# ──────────────────────────────────────────────────────
#
#   INPUT: 150 MongoDB result documents + user's question
#      │
#      ├── Chunk 1 (docs 1–25)   → LLM → "Chunk 1 summary"
#      ├── Chunk 2 (docs 26–50)  → LLM → "Chunk 2 summary"
#      ├── ...
#      └── Chunk 6 (docs 126–150)→ LLM → "Chunk 6 summary"
#                 │
#                 ▼
#         All chunk summaries combined
#                 │
#                 ▼
#      LLM → "Final synthesis / overall analysis"
#
# WHY THIS WORKS REGARDLESS OF RESULT SIZE:
# ───────────────────────────────────────────
#   Each chunk is small enough to always fit in context (25 docs
#   of typical stream data ≈ 3000 tokens).
#   The final synthesis only receives the summaries (not the raw
#   docs), so it also always fits in context.
#   This lets you analyze 200 documents with a model that has
#   an 8K context window — or 2000 documents with 3 extra rounds.
#
# WHEN IT RUNS PARALLEL vs SEQUENTIAL:
# ─────────────────────────────────────
#   Chunk summaries run sequentially by default because local
#   models can only process one request at a time (GPU constraint).
#   If using Gemini (cloud), we could parallelize them, but the
#   sequential approach is safer and works for both.
# ============================================================

import json
import asyncio
from typing import Any

from lib.llm_provider import generate_with_fallback

# ── Configuration ─────────────────────────────────────────────
# How many documents per chunk. Adjust if you hit context limits.
# 25 is conservative (fits in 4K context easily).
# 50 is fine for models with 8K+ context.
CHUNK_SIZE = 25

# Maximum total documents to feed into summarization.
# Beyond this, we still chunk but only process the first N docs.
# This prevents runaway API costs or extremely slow local inference.
MAX_DOCS_TO_SUMMARIZE = 200

# Fields to strip from documents before sending to LLM summarization.
# These are security-sensitive or too bulky to be useful.
_STRIP_FIELDS = {
    "apiKeys", "streamingApiKeys",
    "elInfo.availableApps",   # huge nested object, not useful for analysis
    "elInfo.xirysobj",        # TURN server credentials
    "timeRecords",            # 40+ timestamp fields, rarely useful
    "iceConnectionStateChanges",
    "candidates_selected",
}


def _strip_sensitive(doc: dict, fields: set = _STRIP_FIELDS) -> dict:
    """
    Removes bulky or sensitive fields from a document before
    sending it to the LLM for analysis.

    We don't want to:
    1. Send API keys or credentials to any LLM
    2. Waste context window on huge objects that don't add analysis value
    3. Confuse the model with 40+ timestamp fields

    This also significantly reduces the token count per document,
    letting us fit more documents per chunk.
    """
    cleaned = {}
    for k, v in doc.items():
        if k in fields:
            continue
        if isinstance(v, dict):
            # Recursively strip nested fields
            # Check full path like "elInfo.availableApps"
            nested = _strip_sensitive(v, {
                f.split(".", 1)[1] for f in fields
                if f.startswith(k + ".") and "." in f
            })
            if nested:
                cleaned[k] = nested
        else:
            cleaned[k] = v
    return cleaned


def _docs_to_compact_text(docs: list[dict], max_chars_per_doc: int = 500) -> str:
    """
    Converts a list of documents to a compact text representation
    suitable for LLM input.

    We use JSON with indent=None (compact) and truncate long field
    values. We don't want to paste 100KB of raw MongoDB documents
    into a prompt — we want enough information for meaningful analysis.

    Each document gets at most max_chars_per_doc characters.
    """
    parts = []
    for i, doc in enumerate(docs):
        doc_json = json.dumps(doc, default=str)[:max_chars_per_doc]
        parts.append(f"[{i+1}] {doc_json}")
    return "\n".join(parts)


def chunk_list(items: list, chunk_size: int = CHUNK_SIZE) -> list[list]:
    """
    Splits a list into chunks of at most chunk_size items.

    Example:
        chunk_list([1,2,3,4,5], chunk_size=2)
        → [[1,2], [3,4], [5]]
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def _summarize_chunk(
    docs:          list[dict],
    question:      str,
    chunk_index:   int,
    total_chunks:  int,
) -> str:
    """
    Asks the LLM to summarize a single chunk of documents.

    The prompt is intentionally simple — we're not asking for
    JSON here, just a short analytical paragraph. The model is
    less likely to hallucinate specific numbers, so we ask for
    observations rather than exact statistics.

    Args:
        docs:          The chunk of MongoDB documents (already stripped)
        question:      The user's original question (for context)
        chunk_index:   1-based chunk number
        total_chunks:  Total number of chunks

    Returns:
        A short text summary of this chunk.
    """
    # Strip sensitive/bulky fields before building the prompt
    clean_docs = [_strip_sensitive(doc) for doc in docs]
    docs_text  = _docs_to_compact_text(clean_docs)

    system_prompt = (
        "You are a data analyst summarizing streaming session data from Eagle 3D Streaming. "
        "Your job is to identify patterns, outliers, and key insights from the data. "
        "Be concise, factual, and specific. Use numbers when they're clearly present in the data. "
        "Do not make up data that isn't shown."
    )

    user_message = (
        f"The user asked: \"{question}\"\n\n"
        f"Here is chunk {chunk_index} of {total_chunks} from the query results "
        f"({len(docs)} documents):\n\n"
        f"{docs_text}\n\n"
        f"Summarize the key insights from this chunk relevant to the question. "
        f"Be brief (3-5 sentences). Focus on patterns, not individual records."
    )

    text, provider = await generate_with_fallback(system_prompt, user_message)
    print(f"[summarizer] Chunk {chunk_index}/{total_chunks} summarized via {provider}")
    return text.strip()


async def _synthesize_summaries(
    chunk_summaries: list[str],
    question:        str,
    total_docs:      int,
) -> str:
    """
    Takes all chunk summaries and asks the LLM for a final
    synthesized analysis.

    This is the "reduce" step of the map-reduce approach.
    We're feeding only the summaries (not raw docs) so this
    always fits comfortably in context.

    Returns:
        A final analytical text response.
    """
    summaries_text = "\n\n".join(
        f"=== Chunk {i+1} ===\n{s}"
        for i, s in enumerate(chunk_summaries)
    )

    system_prompt = (
        "You are a senior data analyst for Eagle 3D Streaming. "
        "Synthesize multiple chunk-level summaries into one clear, concise analysis. "
        "Highlight the most important patterns, answer the user's question directly, "
        "and call out any notable outliers or anomalies. "
        "Be specific with numbers when available. Do not repeat information unnecessarily."
    )

    user_message = (
        f"The user asked: \"{question}\"\n\n"
        f"I analyzed {total_docs} records by splitting them into {len(chunk_summaries)} chunks.\n"
        f"Here are the summaries from each chunk:\n\n"
        f"{summaries_text}\n\n"
        f"Please synthesize these into a final answer to the user's question. "
        f"Start with a direct answer, then provide supporting details."
    )

    text, provider = await generate_with_fallback(system_prompt, user_message)
    print(f"[summarizer] Final synthesis via {provider}")
    return text.strip()


async def summarize_results(
    results:  list[dict],
    question: str,
) -> dict:
    """
    Main entry point: summarizes a list of MongoDB result documents.

    Chooses between:
    1. Direct analysis (results are small enough for one LLM call)
    2. Chunked map-reduce (results are large)

    The threshold is whether the compact document text fits within
    about half the expected context window. We use a character-based
    estimate: ~4 chars per token, 8192 tokens * 4 = ~32768 chars.
    We aim for 50% of context for the data, leaving room for
    the system prompt and response.

    Args:
        results:  List of dicts from execute_query()
        question: The original user question (for context)

    Returns:
        {
            "summary":    "The final analysis text...",
            "method":     "direct" | "chunked",
            "chunksUsed": N,
            "docsAnalyzed": N,
        }
    """
    # Cap at MAX_DOCS_TO_SUMMARIZE to prevent runaway processing
    docs_to_analyze = results[:MAX_DOCS_TO_SUMMARIZE]
    total = len(docs_to_analyze)

    if total == 0:
        return {
            "summary":      "No results to analyze.",
            "method":       "direct",
            "chunksUsed":   0,
            "docsAnalyzed": 0,
        }

    print(f"[summarizer] Analyzing {total} documents for: '{question[:60]}'")

    # Estimate token count: strip sensitive fields first, then measure
    clean_docs  = [_strip_sensitive(doc) for doc in docs_to_analyze]
    total_chars = sum(len(json.dumps(d, default=str)) for d in clean_docs)

    # ~16000 chars ≈ ~4000 tokens — safe for direct analysis in 8K context
    # (leaves ~4K tokens for system prompt + response)
    DIRECT_THRESHOLD = 16_000

    if total_chars <= DIRECT_THRESHOLD:
        # ── Direct analysis (small result set) ──────────────────
        print(f"[summarizer] Using direct analysis ({total_chars} chars)")
        docs_text = _docs_to_compact_text(clean_docs)

        system_prompt = (
            "You are a data analyst for Eagle 3D Streaming. "
            "Analyze the provided query results and answer the user's question. "
            "Be specific, factual, and concise. Use numbers from the data. "
            "Do not make up information that isn't in the data."
        )
        user_message = (
            f"The user asked: \"{question}\"\n\n"
            f"Here are the query results ({total} documents):\n\n"
            f"{docs_text}\n\n"
            f"Please analyze these results and provide a clear, direct answer."
        )

        text, provider = await generate_with_fallback(system_prompt, user_message)
        print(f"[summarizer] Direct analysis completed via {provider}")

        return {
            "summary":      text.strip(),
            "method":       "direct",
            "chunksUsed":   1,
            "docsAnalyzed": total,
        }

    else:
        # ── Chunked map-reduce (large result set) ────────────────
        chunks = chunk_list(docs_to_analyze, CHUNK_SIZE)
        print(
            f"[summarizer] Using chunked analysis: "
            f"{len(chunks)} chunks of {CHUNK_SIZE} docs each "
            f"({total_chars} chars total)"
        )

        # MAP: summarize each chunk sequentially
        # (sequential because local GPU can only handle one request at a time)
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await _summarize_chunk(
                docs         = chunk,
                question     = question,
                chunk_index  = i + 1,
                total_chunks = len(chunks),
            )
            chunk_summaries.append(summary)

        # REDUCE: synthesize all chunk summaries into one final answer
        final_summary = await _synthesize_summaries(
            chunk_summaries = chunk_summaries,
            question        = question,
            total_docs      = total,
        )

        return {
            "summary":      final_summary,
            "method":       "chunked",
            "chunksUsed":   len(chunks),
            "docsAnalyzed": total,
        }
