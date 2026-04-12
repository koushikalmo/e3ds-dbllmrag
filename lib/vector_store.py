# ============================================================
# lib/vector_store.py — Pure Python Local Vector Store
# ============================================================
# Stores and searches vector embeddings using cosine similarity.
# No external database, index library, or numpy required.
# Data persists to JSON files in data/vectors/.
#
# TWO STORES ARE USED BY THIS APPLICATION:
#
#   data/vectors/schema.json
#     Each item = one schema field description.
#     id:       "stream::clientInfo.country_name"
#     text:     "clientInfo.country_name (string): user's country"
#     metadata: { "db": "stream-datastore", "field": "...", "type": "..." }
#
#   data/vectors/examples.json
#     Each item = one past successful query.
#     id:       SHA-1 of the question text
#     text:     The user's question
#     metadata: { "query": {...pipeline...}, "db_hint": "stream", ... }
#
# PERFORMANCE:
#   Pure Python cosine search over 500 items with 768-dim vectors:
#   ~5ms. Negligible vs the LLM call (5–30 seconds).
#
# SCALABILITY:
#   New database? schema_discovery.py samples it and indexes fields
#   here automatically. No code changes needed.
# ============================================================

import json
import math
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_STORE_DIR = Path(__file__).parent.parent / "data" / "vectors"


def _cosine(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two float vectors.
    Returns a value in [-1, 1] where 1 = identical direction.

    No numpy needed — pure Python runs ~5ms for 500 × 768-dim items.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class VectorStore:
    """
    A simple local vector store backed by a single JSON file.

    Thread safety: single-process asyncio use only. All writes
    go through _save() which flushes to disk immediately.
    """

    def __init__(self, name: str):
        """
        Args:
            name: Logical name for this store.
                  File will be at data/vectors/{name}.json
        """
        self._path  = _STORE_DIR / f"{name}.json"
        self._items: list[dict] = []
        self._load()

    # ── Persistence ─────────────────────────────────────────────

    def _load(self) -> None:
        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            self._items = json.loads(raw) if raw.strip() else []
            logger.debug(
                f"[vector_store:{self._path.stem}] "
                f"Loaded {len(self._items)} items"
            )
        except Exception as e:
            logger.warning(
                f"[vector_store:{self._path.stem}] "
                f"Load error: {e} — starting fresh"
            )
            self._items = []

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._items, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(
                f"[vector_store:{self._path.stem}] Save error: {e}"
            )

    # ── Inspection ───────────────────────────────────────────────

    def count(self) -> int:
        """Number of items in the store."""
        return len(self._items)

    def ids(self) -> set[str]:
        """Set of all item IDs currently stored."""
        return {i["id"] for i in self._items}

    # ── Write ────────────────────────────────────────────────────

    def upsert(
        self,
        id:        str,
        text:      str,
        embedding: list[float],
        metadata:  dict,
    ) -> None:
        """
        Insert or update an item. Matching is by id.

        Args:
            id:        Unique identifier (e.g. "stream::appInfo.owner").
            text:      The original text that was embedded.
            embedding: Float vector from embeddings.embed().
            metadata:  Arbitrary dict. Store whatever context you need
                       for the search filter or result rendering.
        """
        for item in self._items:
            if item["id"] == id:
                item["text"]      = text
                item["embedding"] = embedding
                item["metadata"]  = metadata
                self._save()
                return

        self._items.append({
            "id":        id,
            "text":      text,
            "embedding": embedding,
            "metadata":  metadata,
        })
        self._save()

    def remove(self, id: str) -> bool:
        """Delete an item by id. Returns True if found and removed."""
        before = len(self._items)
        self._items = [i for i in self._items if i["id"] != id]
        if len(self._items) < before:
            self._save()
            return True
        return False

    def trim_to(self, max_items: int) -> int:
        """
        Drop oldest items so the store stays at most max_items long.
        Items are stored in insertion order — oldest = index 0.
        Returns the count of removed items.
        """
        if len(self._items) <= max_items:
            return 0
        removed = len(self._items) - max_items
        self._items = self._items[removed:]   # drop oldest from front
        self._save()
        return removed

    # ── Search ───────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k:           int  = 5,
        filter_fn        = None,
        min_score:       float = 0.0,
    ) -> list[dict]:
        """
        Cosine similarity search. Returns the top_k most similar items.

        Args:
            query_embedding: The vector to search for (from embed()).
            top_k:           How many results to return.
            filter_fn:       Optional callable(item) -> bool.
                             Pre-filters items before scoring.
                             Use this to restrict by db_name, db_hint, etc.
            min_score:       Minimum cosine similarity to include.
                             0.0 = return everything (no floor).

        Returns:
            List of result dicts, each containing:
              { "id", "text", "metadata", "score" }
            Sorted by score descending (most similar first).
        """
        candidates = [
            i for i in self._items
            if i.get("embedding")
            and (filter_fn is None or filter_fn(i))
        ]
        if not candidates:
            return []

        scored = [
            (_cosine(query_embedding, c["embedding"]), c)
            for c in candidates
        ]
        # Sort descending by score, then by insertion order for ties
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "id":       c["id"],
                "text":     c["text"],
                "metadata": c["metadata"],
                "score":    round(score, 4),
            }
            for score, c in scored[:top_k]
            if score >= min_score
        ]
