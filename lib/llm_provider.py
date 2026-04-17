import os
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5-coder:7b")
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

LLM_PROVIDER = "ollama"


class OllamaProvider:
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

    async def generate(self, system_prompt: str, user_message: str, json_mode: bool = True) -> str:
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
            payload["format"] = "json"

        # 300s timeout — large prompts on 8GB GPU can take ~140s + 30s cold start
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}. "
                f"Start it with: ollama serve && ollama pull {self.model}\n"
                f"Body: {response.text[:400]}"
            )

        text = response.json().get("message", {}).get("content", "")
        if not text:
            raise RuntimeError(
                f"Ollama returned empty response from '{self.model}'. "
                f"Try increasing OLLAMA_NUM_CTX (currently {self.num_ctx})."
            )
        return text

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
            if r.status_code != 200:
                return False
            models    = [m["name"] for m in r.json().get("models", [])]
            base_name = self.model.split(":")[0]
            return any(base_name in m for m in models)
        except Exception:
            return False


def get_ollama() -> OllamaProvider:
    return OllamaProvider()


async def generate_with_ollama(
    system_prompt: str,
    user_message:  str,
    json_mode:     bool = True,
) -> tuple[str, str]:
    provider = OllamaProvider()
    text = await provider.generate(system_prompt, user_message, json_mode=json_mode)
    return text, provider.name


async def warmup_model() -> None:
    print(f"[llm] Warming up '{OLLAMA_MODEL}'…")
    try:
        async with httpx.AsyncClient(timeout=240.0) as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model":    OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": 'respond with {"ok":true}'}],
                    "format":   "json",
                    "stream":   False,
                    "options":  {"temperature": 0, "num_predict": 10},
                    "keep_alive": "30m",
                },
            )
        if r.status_code == 200:
            print(f"[llm] '{OLLAMA_MODEL}' is warm and ready.")
        else:
            print(f"[llm] Warmup HTTP {r.status_code} — first query may be slow.")
    except Exception as e:
        print(f"[llm] Warmup skipped ({e}) — first query will load the model.")


# Alias kept for backward compatibility
generate_with_fallback = generate_with_ollama


async def generate_text(system_prompt: str, user_message: str) -> tuple[str, str]:
    return await generate_with_ollama(system_prompt, user_message, json_mode=False)
