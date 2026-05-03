from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class URLTask:
    url: str
    depth: int = 0


class URLQueue:
    def __init__(self) -> None:
        self._queue: deque[URLTask] = deque()
        self._seen: set[str] = set()

    def push(self, url: str, depth: int = 0) -> bool:
        if url in self._seen:
            return False
        self._seen.add(url)
        self._queue.append(URLTask(url=url, depth=depth))
        return True

    def pop(self) -> URLTask:
        return self._queue.popleft()

    def __len__(self) -> int:
        return len(self._queue)
