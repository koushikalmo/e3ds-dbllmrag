# lib/embeddings.py — Ollama embedding client (nomic-embed-text)
# Generates 768-dim vectors used for semantic RAG search (schema fields + query examples).
# Returns None on any failure — callers fall back to keyword search.

import os
import logging
import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


async def embed(text: str) -> list[float] | None:
    """Returns a 768-dim float vector, or None if unavailable."""
    if not text or not text.strip():
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text.strip()},
            )

        if r.status_code == 200:
            emb = r.json().get("embedding")
            if emb and isinstance(emb, list) and len(emb) > 0:
                return emb
            logger.warning(f"[embeddings] Unexpected response shape — keys: {list(r.json().keys())}")
            return None

        if r.status_code == 404:
            logger.warning(f"[embeddings] Model '{EMBED_MODEL}' not found. Run: ollama pull {EMBED_MODEL}")
        else:
            logger.warning(f"[embeddings] HTTP {r.status_code}: {r.text[:200]}")

    except httpx.TimeoutException:
        logger.debug("[embeddings] Timeout calling Ollama")
    except Exception as e:
        logger.debug(f"[embeddings] embed() failed: {e}")

    return None


async def embed_batch(texts: list[str]) -> list[list[float] | None]:
    """Embeds a list sequentially. Individual failures are None."""
    return [await embed(t) for t in texts]


async def is_available() -> bool:
    return await embed("connection test") is not None
