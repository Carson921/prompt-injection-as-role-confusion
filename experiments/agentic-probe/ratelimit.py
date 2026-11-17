"""Process-local requests-per-minute limiter shared by all threads (OpenRouter caps some models at 300 rpm)."""
from __future__ import annotations

import collections
import threading
import time


class RateLimiter:
    def __init__(self, rpm: float):
        self.rpm, self.window = rpm, collections.deque()
        self.lock = threading.Lock()

    def acquire(self):
        if self.rpm <= 0:
            return
        while True:
            with self.lock:
                now = time.time()
                while self.window and now - self.window[0] > 60:
                    self.window.popleft()
                if len(self.window) < self.rpm:
                    self.window.append(now)
                    return
                wait = 60 - (now - self.window[0]) + 0.05
            time.sleep(max(0.05, wait))


LIMITERS: dict[str, RateLimiter] = {}
_LOCK = threading.Lock()


def limiter(key: str, rpm: float) -> RateLimiter:
    with _LOCK:
        if key not in LIMITERS:
            LIMITERS[key] = RateLimiter(rpm)
        return LIMITERS[key]
