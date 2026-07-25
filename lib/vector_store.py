from __future__ import annotations
import json
import math
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_STORE_DIR = Path(__file__).parent.parent / "data" / "vectors"


def _cosine(a: list[float], b: list[float]) -> float:
    # pure Python — ~5ms for 500×768-dim items, acceptable for small RAG stores
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class VectorStore:
    def __init__(self, name: str):
        self._path  = _STORE_DIR / f"{name}.json"
        self._items: list[dict] = []
        self._load()

    def _load(self) -> None:
        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            self._items = json.loads(raw) if raw.strip() else []
        except Exception as e:
            logger.warning(f"[vector_store:{self._path.stem}] Load error: {e} — starting fresh")
            self._items = []

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._items, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error(f"[vector_store:{self._path.stem}] Save error: {e}")

    def count(self) -> int:
        return len(self._items)

    def ids(self) -> set[str]:
        return {i["id"] for i in self._items}

    def all_items(self) -> list[dict]:
        return self._items

    def upsert(self, id: str, text: str, embedding: list[float], metadata: dict) -> None:
        for item in self._items:
            if item["id"] == id:
                item["text"]      = text
                item["embedding"] = embedding
                item["metadata"]  = metadata
                self._save()
                return
        self._items.append({"id": id, "text": text, "embedding": embedding, "metadata": metadata})
        self._save()

    def remove(self, id: str) -> bool:
        before = len(self._items)
        self._items = [i for i in self._items if i["id"] != id]
        if len(self._items) < before:
            self._save()
            return True
        return False

    def update_metadata(self, id: str, updates: dict) -> bool:
        for item in self._items:
            if item["id"] == id:
                item.setdefault("metadata", {}).update(updates)
                self._save()
                return True
        return False

    def bump(self, id: str, *, success: bool, accuracy: int | None = None) -> bool:
        """Bump success/failure counters and fold in an accuracy score.
        Missing counters count as 0, so old entries migrate lazily.
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for item in self._items:
            if item["id"] != id:
                continue
            m = item.setdefault("metadata", {})
            if success:
                m["success_count"] = int(m.get("success_count", 0)) + 1
                m["last_good_at"]  = now
            else:
                m["failure_count"] = int(m.get("failure_count", 0)) + 1
            if accuracy is not None:
                try:
                    score = float(accuracy)
                except (TypeError, ValueError):
                    score = 0.0
                m["accuracy_sum"] = float(m.get("accuracy_sum", 0.0)) + score
                m["accuracy_n"]   = int(m.get("accuracy_n", 0)) + 1
            m["last_used"] = now
            self._save()
            return True
        return False

    def trim_to(self, max_items: int) -> int:
        if len(self._items) <= max_items:
            return 0
        removed = len(self._items) - max_items
        self._items = self._items[removed:]
        self._save()
        return removed

    def search(
        self,
        query_embedding: list[float],
        top_k:           int   = 5,
        filter_fn               = None,
        min_score:       float = 0.0,
    ) -> list[dict]:
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
            {"id": c["id"], "text": c["text"], "metadata": c["metadata"], "score": round(score, 4)}
            for score, c in scored[:top_k]
            if score >= min_score
        ]
