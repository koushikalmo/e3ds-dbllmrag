import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

# 10 exchanges ≈ 500 tokens — keeps follow-up context without blowing the 8K context window
MAX_TURNS   = 10
SESSION_TTL = 7200  # 2 hours — sessions idle past this are GC'd

# { session_id: { "messages": deque, "last_active": float } }
_sessions: dict[str, dict] = {}


def _gc() -> None:
    cutoff  = time.monotonic() - SESSION_TTL
    expired = [sid for sid, s in _sessions.items() if s["last_active"] < cutoff]
    for sid in expired:
        del _sessions[sid]


def add_turn(session_id: str, question: str, answer: str) -> None:
    if not session_id:
        return
    _gc()
    if session_id not in _sessions:
        _sessions[session_id] = {
            "messages":    deque(maxlen=MAX_TURNS * 2),
            "last_active": time.monotonic(),
        }
    session = _sessions[session_id]
    session["messages"].append({"role": "user",      "content": question})
    session["messages"].append({"role": "assistant",  "content": answer})
    session["last_active"] = time.monotonic()


def get_context_text(session_id: str) -> str:
    if not session_id or session_id not in _sessions:
        return ""
    messages = list(_sessions[session_id]["messages"])
    if not messages:
        return ""

    lines = [
        "─────────────────────────────────────────────────────────────",
        "CONVERSATION CONTEXT (refer to this if the question contains",
        "'that', 'those', 'same', 'also', 'filter', 'compare', etc.):",
        "─────────────────────────────────────────────────────────────",
    ]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "AI"
        lines.append(f"{role}: {msg['content'][:300]}")
    lines.append("─────────────────────────────────────────────────────────────")
    lines.append("")
    return "\n".join(lines)


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def active_session_count() -> int:
    _gc()
    return len(_sessions)
