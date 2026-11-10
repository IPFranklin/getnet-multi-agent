"""Short-term conversation memory, keyed by user_id (and optional session_id).

In-process and bounded (last N turns). Swap for Redis (with TTL) when running more than one
replica — the interface stays the same.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque


class ConversationMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._store: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_turns * 2))
        self._lock = threading.Lock()

    @staticmethod
    def key(user_id: str, session_id: str | None) -> str:
        return f"{user_id}:{session_id or 'default'}"

    def history(self, key: str) -> list[dict]:
        with self._lock:
            return list(self._store[key])

    def append(self, key: str, user_message: str, assistant_message: str, agent: str) -> None:
        with self._lock:
            self._store[key].append({"role": "user", "content": user_message})
            self._store[key].append({"role": "assistant", "content": assistant_message, "agent": agent})

    def clear(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)
