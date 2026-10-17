"""Very small Prometheus-style metrics (no extra packages needed)."""
from collections import defaultdict

_requests = defaultdict(int)       # (method, path, status) -> count
_latency_sum = defaultdict(float)  # (method, path) -> seconds
_latency_count = defaultdict(int)  # (method, path) -> count


def record(method: str, path: str, status: int, seconds: float) -> None:
    _requests[(method, path, str(status))] += 1
    _latency_sum[(method, path)] += seconds
    _latency_count[(method, path)] += 1


def render() -> str:
    lines = [
        "# HELP http_requests_total Total HTTP requests",
        "# TYPE http_requests_total counter",
    ]
    for (method, path, status), n in sorted(_requests.items()):
        lines.append(
            f'http_requests_total{{method="{method}",path="{path}",status="{status}"}} {n}'
        )
    lines += [
        "# HELP http_request_duration_seconds Request duration in seconds",
        "# TYPE http_request_duration_seconds summary",
    ]
    for (method, path), total in sorted(_latency_sum.items()):
        labels = f'method="{method}",path="{path}"'
        lines.append(f"http_request_duration_seconds_sum{{{labels}}} {total:.6f}")
        lines.append(
            f"http_request_duration_seconds_count{{{labels}}} {_latency_count[(method, path)]}"
        )
    return "\n".join(lines) + "\n"
