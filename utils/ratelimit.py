"""Tiny in-memory rate limiter (sliding window, per key)."""
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self.hits = defaultdict(deque)

    def is_limited(self, key: str, limit: int) -> bool:
        """Return True if `key` already used up `limit` requests in the window."""
        now = time.time()
        q = self.hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= limit:
            return True
        q.append(now)
        return False
