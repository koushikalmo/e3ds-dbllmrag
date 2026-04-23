#!/usr/bin/env python3
"""
Migrate existing JSON vector files  →  ChromaDB persistent store.

Source files:   data/vectors/examples.json   (RAG query examples)
                data/vectors/schema.json      (live schema field embeddings)

Target:         data/chroma/                  (ChromaDB PersistentClient)

Usage:
    python scripts/migrate_to_chroma.py          # migrate both collections
    python scripts/migrate_to_chroma.py --verify # migrate then verify counts
    python scripts/migrate_to_chroma.py --reset  # wipe chroma first, then migrate

Run from the project root directory.
"""

import sys
import json
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────

EXAMPLES_JSON = ROOT / "data" / "vectors" / "examples.json"
SCHEMA_JSON   = ROOT / "data" / "vectors" / "schema.json"
CHROMA_DIR    = ROOT / "data" / "chroma"

# ── Helpers ────────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'─' * 56}")
    print(f"  {msg}")
    print(f"{'─' * 56}")


def _serialize_meta(metadata: dict) -> dict:
    """Flatten nested dicts/lists to JSON strings (ChromaDB only accepts primitives)."""
    out = {}
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


def load_json_store(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [SKIP] {path.name} not found — nothing to migrate.")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            print(f"  [WARN] {path.name} is not a JSON array — skipping.")
            return []
        print(f"  [OK]   Loaded {len(data):,} items from {path.name}")
        return data
    except Exception as e:
        print(f"  [ERR]  Could not read {path.name}: {e}")
        return []


def migrate_collection(
    client,
    name:    str,
    items:   list[dict],
    *,
    reset:   bool = False,
    batch:   int  = 100,
) -> int:
    """Upsert all items into a named ChromaDB collection. Returns count inserted."""
    if not items:
        return 0

    if reset:
        try:
            client.delete_collection(name)
            print(f"  [RESET] Deleted existing '{name}' collection.")
        except Exception:
            pass

    col = client.get_or_create_collection(
        name     = name,
        metadata = {"hnsw:space": "cosine"},
    )

    existing_ids = set(col.get(include=[])["ids"])
    to_add = [i for i in items if i.get("id") and i.get("embedding") and i["id"] not in existing_ids]

    if not to_add:
        print(f"  [SKIP] All {len(items)} items already in '{name}' — nothing to do.")
        return 0

    print(f"  [INFO] Inserting {len(to_add):,} new items into '{name}'  "
          f"({len(items) - len(to_add)} already present)…")

    inserted = 0
    for start in range(0, len(to_add), batch):
        chunk = to_add[start : start + batch]
        try:
            col.upsert(
                ids        = [i["id"] for i in chunk],
                documents  = [str(i.get("text", "")) for i in chunk],
                embeddings = [i["embedding"] for i in chunk],
                metadatas  = [_serialize_meta(i.get("metadata", {})) for i in chunk],
            )
            inserted += len(chunk)
            pct = 100 * inserted / len(to_add)
            print(f"  [....] {inserted:>4}/{len(to_add)}  ({pct:.0f}%)", end="\r")
        except Exception as e:
            print(f"\n  [ERR]  Batch {start}–{start+len(chunk)} failed: {e}")

    print(f"  [DONE] Inserted {inserted:,} items into '{name}' collection.    ")
    return inserted


def verify(client) -> None:
    banner("Verification")
    cols = client.list_collections()
    if not cols:
        print("  No collections found.")
        return
    total = 0
    for c in cols:
        n = c.count()
        total += n
        print(f"  Collection '{c.name}': {n:,} vectors")
    print(f"\n  Total vectors in ChromaDB: {total:,}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate JSON vector files to ChromaDB.")
    parser.add_argument("--verify", action="store_true",  help="Print collection stats after migration.")
    parser.add_argument("--reset",  action="store_true",  help="Delete existing ChromaDB data before migrating.")
    parser.add_argument("--only",   choices=["examples", "schema"], help="Migrate only one collection.")
    args = parser.parse_args()

    # ── Import check ────────────────────────────────────────────────────────────
    try:
        import chromadb
    except ImportError:
        print("\n❌  chromadb is not installed.\n")
        print("    Install it first:")
        print("      pip install 'chromadb>=0.5.0'\n")
        sys.exit(1)

    banner(f"ChromaDB Migration  (v{chromadb.__version__})")
    print(f"  Source dir : {ROOT / 'data' / 'vectors'}")
    print(f"  Target dir : {CHROMA_DIR}")
    print(f"  Reset mode : {'YES — existing data will be deleted' if args.reset else 'no'}")

    # ── Connect ─────────────────────────────────────────────────────────────────
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        print(f"\n  [OK]   ChromaDB PersistentClient ready at {CHROMA_DIR}")
    except Exception as e:
        print(f"\n  [ERR]  Could not open ChromaDB: {e}")
        sys.exit(1)

    t0 = time.perf_counter()
    total_inserted = 0

    # ── Migrate examples ────────────────────────────────────────────────────────
    if args.only in (None, "examples"):
        banner("Collection: examples  (RAG query examples)")
        items = load_json_store(EXAMPLES_JSON)
        total_inserted += migrate_collection(client, "examples", items, reset=args.reset)

    # ── Migrate schema ──────────────────────────────────────────────────────────
    if args.only in (None, "schema"):
        banner("Collection: schema  (live schema field embeddings)")
        items = load_json_store(SCHEMA_JSON)
        total_inserted += migrate_collection(client, "schema", items, reset=args.reset)

    # ── Summary ─────────────────────────────────────────────────────────────────
    elapsed = round(time.perf_counter() - t0, 2)
    banner(f"Migration complete  ({elapsed}s)")
    print(f"  Total inserted this run: {total_inserted:,} vectors")

    if args.verify or True:  # always show final counts
        verify(client)

    print("\n  Next step: restart the server.\n"
          "  ChromaDB will be used automatically from now on.\n")


if __name__ == "__main__":
    main()
