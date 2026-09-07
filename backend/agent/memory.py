"""Bounded, expiring short-term conversation memory for active Agent sessions."""

from collections import deque
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Callable, Iterable


@dataclass
class _Session:
    messages: deque[dict[str, str]]
    touched_at: float


class ShortTermMemory:
    def __init__(self, max_messages: int = 12, ttl_seconds: float = 1800,
                 clock: Callable[[], float] = monotonic):
        if max_messages < 2 or ttl_seconds <= 0:
            raise ValueError("短期记忆容量至少为 2，TTL 必须大于 0")
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._sessions: dict[str, _Session] = {}
        self._lock = RLock()

    def has(self, session_id: str) -> bool:
        with self._lock:
            self._prune()
            return session_id in self._sessions

    def seed(self, session_id: str, messages: Iterable[dict[str, str]]) -> None:
        values = deque(({"role": item["role"], "content": item["content"]} for item in messages),
                       maxlen=self.max_messages)
        with self._lock:
            self._sessions[session_id] = _Session(values, self._clock())

    def remember(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("短期记忆只接受 user/assistant 消息")
        with self._lock:
            self._prune()
            session = self._sessions.setdefault(
                session_id, _Session(deque(maxlen=self.max_messages), self._clock())
            )
            session.messages.append({"role": role, "content": content})
            session.touched_at = self._clock()

    def messages(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            self._prune()
            session = self._sessions.get(session_id)
            if not session:
                return []
            session.touched_at = self._clock()
            return [dict(item) for item in session.messages]

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _prune(self) -> None:
        now = self._clock()
        expired = [key for key, session in self._sessions.items()
                   if now - session.touched_at >= self.ttl_seconds]
        for key in expired:
            self._sessions.pop(key, None)
