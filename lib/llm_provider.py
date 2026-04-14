# ============================================================
# lib/llm_provider.py — Ollama Local LLM Interface
# ============================================================
# All LLM calls in this application go through this module.
# We use Ollama exclusively — a local LLM runner that keeps
# your data private, costs nothing per query, and works
# offline once the model is downloaded.
#
# WHY OLLAMA?
# ────────────
#   - Runs entirely on your GPU — no cloud API calls, no cost
#   - Supports JSON mode ("format": "json") which forces valid
#     JSON output at the token level — critical for generating
#     MongoDB pipelines reliably
#   - Auto-manages VRAM — loads models, handles quantization,
#     keeps models cached between requests
#   - Simple REST API, easy to run alongside FastAPI
#
# HOW TO USE:
# ────────────
#   1. Install Ollama: https://ollama.com
#   2. Start it:   ollama serve
#   3. Pull model: ollama pull qwen2.5-coder:7b
#   4. Set OLLAMA_MODEL in your .env
#
# MODEL RECOMMENDATIONS:
# ───────────────────────
#   8GB GPU (current hardware):
#     qwen2.5-coder:7b  — best for JSON/query generation (~4.7GB VRAM)
#     mistral:7b        — good general model (~4.1GB VRAM)
#     llama3.1:8b       — strong general purpose (~4.7GB VRAM)
#
#   16GB GPU (planned upgrade):
#     qwen2.5-coder:14b — significantly better quality (~9GB VRAM)
#     codestral:22b     — state-of-the-art code model (~13GB Q4 quant)
#
#   All models above handle structured JSON well. qwen2.5-coder
#   is specifically fine-tuned for code and query generation.
#
# JSON MODE (format:"json"):
# ───────────────────────────
#   This is the most important Ollama feature for our use case.
#   When "format": "json" is set, the model is FORCED to output
#   valid JSON at the token level — it literally cannot produce
#   malformed JSON. This eliminates ~99% of JSON parse failures
#   that plague cloud LLMs (which sometimes wrap output in
#   markdown code fences or add explanatory prose).
#
#   Note: format:"json" guarantees VALID JSON, not CORRECT JSON.
#   The model might produce { "foo": "bar" } instead of our expected
#   query object. That's what the retry loop in query_generator.py
#   handles — it checks structure and field names, then sends
#   specific correction feedback back to the model.
# ============================================================

import os
import httpx

# ── Configuration (all read from .env) ────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5-coder:7b")

# Context window size in tokens.
# With vector RAG, the full prompt (rules + relevant fields + examples +
# question) stays under 2,000 tokens. 8192 gives 6,000 tokens of
# generation headroom — more than enough for any pipeline.
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

LLM_PROVIDER = "ollama"


class OllamaProvider:
    """
    Sends prompts to a local Ollama instance and returns responses.

    Ollama must be running before the server starts:
        ollama serve
        ollama pull qwen2.5-coder:7b

    KEY FEATURE — JSON MODE:
    We pass "format": "json" in every request. This activates
    constrained decoding: the model is forced to produce valid
    JSON at the token sampling level. It cannot generate invalid
    JSON when this flag is set. For a system that depends on
    parsing the model's output as a MongoDB pipeline, this is
    the single most important reliability feature available.

    STREAMING vs. NON-STREAMING:
    We use "stream": false so the full response arrives in one
    HTTP response body rather than as a stream of SSE events.
    This simplifies our code — we just read the response JSON
    once and parse it.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model:    str = OLLAMA_MODEL,
        num_ctx:  int = OLLAMA_NUM_CTX,
    ):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.num_ctx  = num_ctx

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    async def generate(
        self,
        system_prompt: str,
        user_message:  str,
        json_mode:     bool = True,
    ) -> str:
        """
        Calls the Ollama /api/chat endpoint and returns the model's text.

        Args:
            system_prompt: Full system instruction (schema + rules)
            user_message:  The user's question + collection context
            json_mode:     When True, forces valid JSON output at the token
                           level (used for query generation). Set False for
                           free-text responses like result summarization.

        Returns:
            Raw text from the model.

        Raises:
            RuntimeError: If Ollama is not running or the request fails.
        """
        payload = {
            "model":    self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1 if json_mode else 0.4,
                "num_ctx":     self.num_ctx,
                "num_predict": 2048,
            },
        }
        if json_mode:
            payload["format"] = "json"   # forces valid JSON at token level

        # 300 second timeout — our system prompt is ~4,500 tokens and with
        # RAG examples + output the total can reach 7,000+ tokens. On an
        # 8GB GPU at 15 tok/s, a 2,048-token response takes ~140 seconds.
        # Cold-start (model loading from disk) adds another 15-30 seconds.
        # 300 seconds gives comfortable headroom for all of that.
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}.\n"
                f"Is Ollama running? Start it with: ollama serve\n"
                f"Then pull the model: ollama pull {self.model}\n"
                f"Response body: {response.text[:400]}"
            )

        data = response.json()

        # Ollama chat response shape:
        # { "message": { "role": "assistant", "content": "..." }, "done": true, ... }
        text = data.get("message", {}).get("content", "")

        if not text:
            raise RuntimeError(
                f"Ollama returned an empty response from model '{self.model}'.\n"
                "This usually means the model ran out of context tokens.\n"
                f"Try increasing OLLAMA_NUM_CTX (currently {self.num_ctx}) "
                "or using a model with a larger context window."
            )

        return text

    async def is_available(self) -> bool:
        """
        Quick health check — returns True if Ollama is running and
        our configured model is downloaded and available.

        Called by the /api/status endpoint so the frontend can show
        the LOCAL / OFFLINE indicator in the header.

        We check both:
        1. Is Ollama's HTTP server responsive?
        2. Is our specific model in the list of downloaded models?

        Returns False (not raises) on any failure — this is a
        UI status check, not a hard dependency.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")

            if r.status_code != 200:
                return False

            # Check if our model is in the list
            # Models are listed with full tags: "qwen2.5-coder:7b"
            # We match on the base name to handle tag variants
            models     = [m["name"] for m in r.json().get("models", [])]
            base_name  = self.model.split(":")[0]
            return any(base_name in m for m in models)

        except Exception:
            return False


# ── Module-level singleton ─────────────────────────────────────
# query_generator.py and other callers use this function so
# they don't need to know about the class directly.

def get_ollama() -> OllamaProvider:
    """Returns a configured OllamaProvider instance."""
    return OllamaProvider()


async def generate_with_ollama(
    system_prompt: str,
    user_message:  str,
    json_mode:     bool = True,
) -> tuple[str, str]:
    """
    Generates a response using the local Ollama LLM.

    Returns:
        (response_text, provider_name)

    Raises:
        RuntimeError: If Ollama is not running or the call fails.
    """
    provider = OllamaProvider()
    text = await provider.generate(system_prompt, user_message, json_mode=json_mode)
    return text, provider.name


async def warmup_model() -> None:
    """
    Loads the query generation model into Ollama's memory.

    Called once at server startup so the first real user query
    doesn't have to wait 60-90s for the 4GB model to load from disk.
    Uses a minimal prompt — the goal is just to trigger model loading,
    not to produce useful output.

    Failure is non-fatal: if Ollama is offline or the model isn't
    pulled, this silently does nothing. The query endpoint will show
    a proper error when the user actually submits a query.
    """
    print(f"[llm] Warming up '{OLLAMA_MODEL}' (loading into GPU VRAM)…")
    try:
        # Use /api/chat with format:json — the same endpoint real queries use —
        # so Ollama caches the exact model state needed for pipeline generation.
        async with httpx.AsyncClient(timeout=240.0) as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model":   OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": "respond with {\"ok\":true}"}],
                    "format":  "json",
                    "stream":  False,
                    "options": {"temperature": 0, "num_predict": 10},
                    "keep_alive": "30m",
                },
            )
        if r.status_code == 200:
            print(f"[llm] '{OLLAMA_MODEL}' is warm and ready.")
        else:
            print(f"[llm] Warmup HTTP {r.status_code} — first query may be slow.")
    except Exception as e:
        print(f"[llm] Warmup skipped ({e}) — first query will load the model.")


# Backward-compatible alias — query_generator.py calls this name
generate_with_fallback = generate_with_ollama


async def generate_text(
    system_prompt: str,
    user_message:  str,
) -> tuple[str, str]:
    """
    Generates a free-text (non-JSON) response using Ollama.
    Used by result_summarizer.py for natural language analysis.

    Returns:
        (response_text, provider_name)
    """
    return await generate_with_ollama(system_prompt, user_message, json_mode=False)
