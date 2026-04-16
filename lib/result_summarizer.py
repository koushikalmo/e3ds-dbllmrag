# lib/result_summarizer.py — Chunked map-reduce summarizer for large result sets
# Small results (<16K chars) → single LLM call.
# Large results → split into 25-doc chunks, summarize each, then synthesize.

import json
import asyncio
from typing import Any

from lib.llm_provider import generate_text

CHUNK_SIZE           = 25
MAX_DOCS_TO_SUMMARIZE = 200

# Strip these fields before sending to LLM — sensitive or too bulky
_STRIP_FIELDS = {
    "apiKeys", "streamingApiKeys",
    "elInfo.availableApps",
    "elInfo.xirysobj",
    "timeRecords",
    "iceConnectionStateChanges",
    "candidates_selected",
}


def _strip_sensitive(doc: dict, fields: set = _STRIP_FIELDS) -> dict:
    cleaned = {}
    for k, v in doc.items():
        if k in fields:
            continue
        if isinstance(v, dict):
            nested = _strip_sensitive(v, {f.split(".", 1)[1] for f in fields if f.startswith(k + ".") and "." in f})
            if nested:
                cleaned[k] = nested
        else:
            cleaned[k] = v
    return cleaned


def _docs_to_compact_text(docs: list[dict], max_chars_per_doc: int = 500) -> str:
    return "\n".join(f"[{i+1}] {json.dumps(doc, default=str)[:max_chars_per_doc]}" for i, doc in enumerate(docs))


def chunk_list(items: list, chunk_size: int = CHUNK_SIZE) -> list[list]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def _summarize_chunk(docs: list[dict], question: str, chunk_index: int, total_chunks: int) -> str:
    clean_docs = [_strip_sensitive(doc) for doc in docs]
    system_prompt = (
        "You are a data analyst summarizing streaming session data from Eagle 3D Streaming. "
        "Be concise, factual, and specific. Use numbers from the data. Do not make up data."
    )
    user_message = (
        f'The user asked: "{question}"\n\n'
        f"Chunk {chunk_index} of {total_chunks} ({len(docs)} documents):\n\n"
        f"{_docs_to_compact_text(clean_docs)}\n\n"
        f"Summarize key insights in 3–5 sentences. Focus on patterns, not individual records."
    )
    text, provider = await generate_text(system_prompt, user_message)
    print(f"[summarizer] Chunk {chunk_index}/{total_chunks} via {provider}")
    return text.strip()


async def _synthesize_summaries(chunk_summaries: list[str], question: str, total_docs: int) -> str:
    summaries_text = "\n\n".join(f"=== Chunk {i+1} ===\n{s}" for i, s in enumerate(chunk_summaries))
    system_prompt = (
        "You are a senior data analyst for Eagle 3D Streaming. "
        "Synthesize multiple summaries into one clear analysis. "
        "Answer the user's question directly. Use numbers where available."
    )
    user_message = (
        f'The user asked: "{question}"\n\n'
        f"Analyzed {total_docs} records across {len(chunk_summaries)} chunks:\n\n"
        f"{summaries_text}\n\n"
        f"Provide a final answer. Start with a direct answer, then supporting details."
    )
    text, provider = await generate_text(system_prompt, user_message)
    print(f"[summarizer] Final synthesis via {provider}")
    return text.strip()


async def summarize_results(results: list[dict], question: str) -> dict:
    """Summarize a list of MongoDB result documents. Chooses direct or chunked approach."""
    docs_to_analyze = results[:MAX_DOCS_TO_SUMMARIZE]
    total = len(docs_to_analyze)

    if total == 0:
        return {"summary": "No results to analyze.", "method": "direct", "chunksUsed": 0, "docsAnalyzed": 0}

    print(f"[summarizer] Analyzing {total} documents for: '{question[:60]}'")

    clean_docs  = [_strip_sensitive(doc) for doc in docs_to_analyze]
    total_chars = sum(len(json.dumps(d, default=str)) for d in clean_docs)

    if total_chars <= 16_000:
        # Small enough for a single LLM call
        system_prompt = (
            "You are a data analyst for Eagle 3D Streaming. "
            "Analyze the results and answer the user's question. "
            "Be specific, factual, and concise. Do not make up data."
        )
        user_message = (
            f'The user asked: "{question}"\n\n'
            f"Query results ({total} documents):\n\n"
            f"{_docs_to_compact_text(clean_docs)}\n\n"
            f"Provide a clear, direct answer."
        )
        text, provider = await generate_text(system_prompt, user_message)
        print(f"[summarizer] Direct analysis via {provider}")
        return {"summary": text.strip(), "method": "direct", "chunksUsed": 1, "docsAnalyzed": total}

    else:
        # Too large for a single call — map-reduce across chunks
        chunks = chunk_list(docs_to_analyze, CHUNK_SIZE)
        print(f"[summarizer] Chunked: {len(chunks)} chunks × {CHUNK_SIZE} docs ({total_chars} chars)")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await _summarize_chunk(chunk, question, i + 1, len(chunks))
            chunk_summaries.append(summary)

        final_summary = await _synthesize_summaries(chunk_summaries, question, total)
        return {"summary": final_summary, "method": "chunked", "chunksUsed": len(chunks), "docsAnalyzed": total}
