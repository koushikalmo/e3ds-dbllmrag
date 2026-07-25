# AI-Generated Comments Archive

Removed from source files. Kept here for reference.

---

## lib/schemas.py
```python
# lib/schemas.py — LLM system prompt: structural rules + field gotchas
# Intentionally small (~750 tokens). Field details come from vector RAG per query.
```

## lib/query_examples.py
```python
# lib/query_examples.py — RAG few-shot example store
# Saves successful queries and retrieves similar ones to prepend to LLM prompts.
# Primary search: semantic vector similarity. Fallback: keyword overlap.
```
```python
def find_similar_examples_vector(...):
    """
    Semantic search via embeddings with weight-boosted re-ranking.
    Retrieves top_n*4 candidates by cosine similarity, then re-ranks by
    (cosine_score * weight) so user-verified and user-corrected examples
    rank above auto-saved ones when semantically similar.
    Returns None if embedding model is unavailable.
    """
```
```python
def find_similar_examples(...):
    """Keyword overlap fallback when vector search is unavailable."""
```
```python
def add_example(...):
    """Save a successful query as a future few-shot example. Skips 0-result queries and near-duplicates."""
```
```python
async def _index_example_async(...):
    """
    Background task: embed question and upsert into vector store.
    weight values:
      1.0 — auto-saved from a successful query (default)
      2.0 — user confirmed correct (good)
      2.5 — user provided a correction (bad + pipeline)
    """
```
```python
def format_examples_for_prompt(...):
    """Format retrieved examples as a few-shot block for the LLM prompt."""
```
```python
async def add_verified_example(...):
    """
    Called when a user marks a response as correct.
    Upserts the example into the vector store with weight=2.0.
    Also updates the weight in the JSON file if the example already exists.
    """
```
```python
async def add_corrected_example(...):
    """
    Called when a user provides a corrected pipeline.
    Saves with weight=2.5 — the highest priority (ground truth).
    """
```
```python
def _update_example_weight_in_file(...):
    """Update the weight of an existing example in the JSON file."""
```
```python
async def index_all_examples_async():
    """Index all examples into the vector store on startup."""
```
```python
    # Re-rank: multiply cosine score by the item's weight (default 1.0)
```

## lib/query_executor.py
```python
def _make_serializable(...):
    """Convert ObjectId and Decimal128 to JSON-safe Python types."""
```
```python
        # In-memory join by mergeKey if provided
```

## lib/query_generator.py
```python
# Field name cache — populated from the schema vector store, refreshed hourly
```
```python
    # Raw doc query — move $limit to end, cap at 200
```

## lib/mongodb.py
```python
# Singletons — created on first use, reused until shutdown
```

## lib/schema_discovery.py
```python
    # Detect schema changes between refreshes
```
```python
    # If appConfigs was needed but not found in the vector store, append the static fallback
```

## lib/feedback_store.py
```python
        # Replace first sub-query pipeline with the corrected one
```
```python
        # Mark document as processed in MongoDB
```

## lib/live_data_context.py
```python
        # Serve cached data, refresh stale entries in background
```

## lib/response_validator.py
```python
    # ── 1. Result count ──────────────────────────────────────────
    # ── 2. Near the hard 200-document cap ───────────────────────
    # ── 3. e3ds_employee filter (internal traffic exclusion) ─────
    # ── 4. $limit placed before $group (wrong aggregation order) ─
    # ── 5. No $match on a stream collection (full scan) ──────────
    # ── 6. RTT string-to-number conversion ───────────────────────
    # ── 7. Dual query info ───────────────────────────────────────
    # ── Helpers ───────────────────────────────────────────────────────
```

## main.py
```python
        # Auto-analyze results immediately
```
```python
        # Run automated validation checks
```
