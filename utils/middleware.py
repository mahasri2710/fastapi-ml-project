"""Security + monitoring middleware: IP blocking, rate limiting, metrics, logging."""
import logging
import os
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from utils import metrics
from utils.ratelimit import RateLimiter

logger = logging.getLogger("ml_api")

RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
LOGIN_LIMIT = int(os.getenv("LOGIN_LIMIT_PER_MIN", "10"))
BLOCKED_IPS = {ip.strip() for ip in os.getenv("BLOCKED_IPS", "").split(",") if ip.strip()}


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.limiter = RateLimiter(window_seconds=60)

    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        blocked = None
        if ip in BLOCKED_IPS:
            blocked = JSONResponse({"detail": "Your IP is blocked"}, status_code=403)
        elif path == "/login" and method == "POST" and self.limiter.is_limited(f"login:{ip}", LOGIN_LIMIT):
            blocked = JSONResponse(
                {"detail": "Too many login attempts. Try again in a minute."},
                status_code=429, headers={"Retry-After": "60"},
            )
        elif self.limiter.is_limited(f"all:{ip}", RATE_LIMIT):
            blocked = JSONResponse(
                {"detail": "Too many requests. Try again in a minute."},
                status_code=429, headers={"Retry-After": "60"},
            )

        if blocked is not None:
            response = blocked
            label = path if path in ("/login", "/docs", "/metrics", "/health", "/") else "other"
            logger.warning("%s %s blocked: %s ip=%s", method, path, response.status_code, ip)
        else:
            response = await call_next(request)
            route = request.scope.get("route")
            label = getattr(route, "path", None) or "other"

        seconds = time.perf_counter() - start
        metrics.record(method, label, response.status_code, seconds)
        if blocked is None:
            logger.info("%s %s -> %s (%.1f ms) ip=%s", method, path, response.status_code, seconds * 1000, ip)
        return response
