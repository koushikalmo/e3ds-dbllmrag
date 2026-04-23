"""
Vector store backed by ChromaDB (preferred) with automatic fallback to the
legacy JSON-file store when chromadb is not yet installed.

Install ChromaDB on your server:
    pip install chromadb>=0.5.0

ChromaDB persists to data/chroma/ and survives restarts without re-indexing.
The JSON fallback persists to data/vectors/<name>.json (same as before).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Try to import ChromaDB ────────────────────────────────────────────────────

try:
    import chromadb as _chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _chromadb         = None  # type: ignore[assignment]
    _CHROMA_AVAILABLE = False
    logger.warning(
        "[vector_store] chromadb not installed — falling back to JSON file store. "
        "Run: pip install 'chromadb>=0.5.0'  to enable persistent ChromaDB."
    )

# ── Shared paths ──────────────────────────────────────────────────────────────

_CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
_JSON_DIR   = Path(__file__).parent.parent / "data" / "vectors"

# ── ChromaDB singleton client ─────────────────────────────────────────────────

_chroma_client = None


def _get_client():
    global _chroma_client
    if not _CHROMA_AVAILABLE:
        return None
    if _chroma_client is not None:
        return _chroma_client
    try:
        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = _chromadb.PersistentClient(path=str(_CHROMA_DIR))
        logger.info(f"[vector_store] ChromaDB ready at {_CHROMA_DIR}")
    except Exception as e:
        logger.error(f"[vector_store] ChromaDB init failed: {e}")
        _chroma_client = None
    return _chroma_client


# ── Metadata serialisation for ChromaDB (only accepts primitives) ─────────────

def _serialize_meta(metadata: dict) -> dict:
    out: dict = {}
    for k, v in metadata.items():
        if v is None:
            out[k] = "__null__"
        elif isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _deserialize_meta(metadata: dict) -> dict:
    out: dict = {}
    for k, v in metadata.items():
        if v == "__null__":
            out[k] = None
        elif isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, (dict, list)):
                    out[k] = parsed
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            out[k] = v
        else:
            out[k] = v
    return out


# ── Legacy JSON cosine helper ─────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ═════════════════════════════════════════════════════════════════════════════
# VectorStore — ChromaDB-backed when available, JSON file-backed as fallback
# ═════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Unified vector store interface.
    Uses ChromaDB (persistent, fast, production-ready) when installed;
    falls back to the lightweight JSON-file store otherwise.
    """

    def __init__(self, name: str):
        self._name = name
        self._col  = None          # ChromaDB collection (None when using JSON fallback)
        self._use_chroma = False

        client = _get_client()
        if client is not None:
            try:
                self._col = client.get_or_create_collection(
                    name     = name,
                    metadata = {"hnsw:space": "cosine"},
                )
                self._use_chroma = True
            except Exception as e:
                logger.error(f"[vector_store:{name}] ChromaDB collection error: {e}")

        if not self._use_chroma:
            # JSON fallback
            self._path:  Path       = _JSON_DIR / f"{name}.json"
            self._items: list[dict] = []
            self._json_load()

    # ── JSON helpers ──────────────────────────────────────────────

    def _json_load(self) -> None:
        _JSON_DIR.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            self._items = json.loads(raw) if raw.strip() else []
        except Exception as e:
            logger.warning(f"[vector_store:{self._name}] JSON load error: {e}")
            self._items = []

    def _json_save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._items, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error(f"[vector_store:{self._name}] JSON save error: {e}")

    # ── Public API ────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return "chromadb" if self._use_chroma else "json"

    def count(self) -> int:
        if self._use_chroma:
            try:
                return self._col.count()
            except Exception:
                return 0
        return len(self._items)

    def ids(self) -> set[str]:
        if self._use_chroma:
            try:
                return set(self._col.get(include=[])["ids"])
            except Exception:
                return set()
        return {i["id"] for i in self._items}

    def all_items(self) -> list[dict]:
        if self._use_chroma:
            try:
                r = self._col.get(include=["documents", "metadatas", "embeddings"])
                return [
                    {
                        "id":        r["ids"][i],
                        "text":      (r.get("documents") or [""])[i],
                        "embedding": (r.get("embeddings") or [[]])[i],
                        "metadata":  _deserialize_meta((r.get("metadatas") or [{}])[i]),
                    }
                    for i in range(len(r["ids"]))
                ]
            except Exception as e:
                logger.error(f"[vector_store:{self._name}] all_items error: {e}")
                return []
        return self._items

    def upsert(self, id: str, text: str, embedding: list[float], metadata: dict) -> None:
        if self._use_chroma:
            try:
                self._col.upsert(
                    ids        = [id],
                    documents  = [text],
                    embeddings = [embedding],
                    metadatas  = [_serialize_meta(metadata)],
                )
            except Exception as e:
                logger.error(f"[vector_store:{self._name}] upsert error: {e}")
            return

        for item in self._items:
            if item["id"] == id:
                item.update({"text": text, "embedding": embedding, "metadata": metadata})
                self._json_save()
                return
        self._items.append({"id": id, "text": text, "embedding": embedding, "metadata": metadata})
        self._json_save()

    def remove(self, id: str) -> bool:
        if self._use_chroma:
            try:
                self._col.delete(ids=[id])
                return True
            except Exception:
                return False
        before = len(self._items)
        self._items = [i for i in self._items if i["id"] != id]
        if len(self._items) < before:
            self._json_save()
            return True
        return False

    def trim_to(self, max_items: int) -> int:
        if self._use_chroma:
            try:
                current = self._col.count()
                if current <= max_items:
                    return 0
                ids_all = self._col.get(include=[])["ids"]
                to_del  = ids_all[: current - max_items]
                if to_del:
                    self._col.delete(ids=to_del)
                return len(to_del)
            except Exception as e:
                logger.error(f"[vector_store:{self._name}] trim_to error: {e}")
                return 0

        if len(self._items) <= max_items:
            return 0
        removed      = len(self._items) - max_items
        self._items  = self._items[removed:]
        self._json_save()
        return removed

    def search(
        self,
        query_embedding: list[float],
        top_k:           int   = 5,
        filter_fn               = None,
        min_score:       float = 0.0,
    ) -> list[dict]:
        if self._use_chroma:
            n = self._col.count()
            if n == 0:
                return []
            try:
                fetch  = min(top_k * 4 if filter_fn else top_k, n)
                result = self._col.query(
                    query_embeddings = [query_embedding],
                    n_results        = fetch,
                    include          = ["documents", "metadatas", "distances"],
                )
            except Exception as e:
                logger.error(f"[vector_store:{self._name}] search error: {e}")
                return []

            ids       = result["ids"][0]
            docs      = (result.get("documents") or [[]])[0]
            metas     = (result.get("metadatas")  or [[]])[0]
            distances = (result.get("distances")  or [[]])[0]

            out = []
            for id_, doc, meta, dist in zip(ids, docs, metas, distances):
                # ChromaDB cosine distance = 1 - similarity  → score = 1 - distance
                score = max(0.0, round(1.0 - float(dist), 4))
                if score < min_score:
                    continue
                item = {"id": id_, "text": doc, "metadata": _deserialize_meta(meta), "score": score}
                if filter_fn is not None and not filter_fn(item):
                    continue
                out.append(item)
                if len(out) >= top_k:
                    break
            return out

        # JSON fallback
        candidates = [
            i for i in self._items
            if i.get("embedding") and (filter_fn is None or filter_fn(i))
        ]
        if not candidates:
            return []
        scored = sorted(
            [(_cosine(query_embedding, c["embedding"]), c) for c in candidates],
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            {"id": c["id"], "text": c["text"], "metadata": c["metadata"], "score": round(s, 4)}
            for s, c in scored[:top_k]
            if s >= min_score
        ]
