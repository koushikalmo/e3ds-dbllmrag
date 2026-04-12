# ============================================================
# lib/embeddings.py — Ollama Semantic Embedding Client
# ============================================================
# Generates dense vector embeddings by calling the local Ollama
# instance. These vectors power the semantic RAG search used for:
#
#   1. SCHEMA FIELD RETRIEVAL
#      "Which fields are relevant to this question?"
#      Instead of dumping all 200+ field descriptions into every
#      prompt (4,500 tokens), we embed the question and retrieve
#      only the 15-20 most similar field descriptions (~400 tokens).
#
#   2. QUERY EXAMPLE RETRIEVAL
#      "Which past queries are most similar to this one?"
#      Replaces the old keyword-overlap (Jaccard) approach with
#      true semantic similarity — finds examples by meaning rather
#      than shared words.
#
# WHY THIS BEATS KEYWORD OVERLAP:
# ─────────────────────────────────
#   "Which countries had the most viewers" and
#   "top geographic regions by user count" share ZERO words.
#   Keyword search returns nothing useful for the second question.
#   Embedding similarity correctly finds the country/city field
#   descriptions and the country-grouping example for both.
#
# MODEL: nomic-embed-text
# ────────────────────────
#   - 274 MB download, fast inference on CPU
#   - 768-dimensional output vectors
#   - Runs on CPU while qwen2.5-coder:7b runs on GPU
#   - Pull once with: ollama pull nomic-embed-text
#
# GRACEFUL FALLBACK:
# ───────────────────
#   All functions return None when the embedding model is
#   unavailable. query_generator.py falls back to keyword search
#   and the full static schema if embeddings fail. The app is
#   fully functional without the embedding model — it's just
#   more accurate with it.
# ============================================================

import os
import logging
import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


async def embed(text: str) -> list[float] | None:
    """
    Returns a 768-dimensional vector for the given text, or None
    on any failure (model not pulled, Ollama down, network error).

    Timeout is 30 seconds — generous for cold-start model loading
    (first embed after Ollama restart). Normal latency is ~50ms.

    Args:
        text: Any text to embed. Gets stripped before sending.

    Returns:
        list[float] of length 768, or None if unavailable.
    """
    if not text or not text.strip():
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text.strip()},
            )

        if r.status_code == 200:
            data = r.json()
            emb  = data.get("embedding")
            if emb and isinstance(emb, list) and len(emb) > 0:
                return emb
            logger.warning(
                f"[embeddings] Unexpected response shape — "
                f"keys: {list(data.keys())}"
            )
            return None

        # 404 usually means the model isn't pulled yet
        if r.status_code == 404:
            logger.warning(
                f"[embeddings] Model '{EMBED_MODEL}' not found in Ollama. "
                f"Pull it with: ollama pull {EMBED_MODEL}"
            )
        else:
            logger.warning(f"[embeddings] HTTP {r.status_code}: {r.text[:200]}")

    except httpx.TimeoutException:
        logger.debug(f"[embeddings] Timeout calling Ollama for embed")
    except Exception as e:
        logger.debug(f"[embeddings] embed() failed: {e}")

    return None


async def embed_batch(texts: list[str]) -> list[list[float] | None]:
    """
    Embeds a list of texts sequentially.

    Ollama doesn't have a native batch endpoint for embeddings,
    so we call embed() for each text. Used during schema indexing
    on startup — called once, results cached in vector_store.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embeddings (same length as input).
        Individual failures are None — caller should filter these out.
    """
    results = []
    for text in texts:
        emb = await embed(text)
        results.append(emb)
    return results


async def is_available() -> bool:
    """
    Returns True if the embedding model is reachable and working.
    Used by the startup log and health endpoint.
    """
    result = await embed("connection test")
    return result is not None
