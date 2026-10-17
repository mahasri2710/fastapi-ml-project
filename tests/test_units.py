"""Unit tests: small pieces tested on their own (no server needed)."""
import pytest
from pydantic import ValidationError

from utils import auth, metrics, ml_utils, ratelimit
from utils.schemas import BatchPredictRequest, PredictRequest


# ---------- rate limiter ----------
def test_ratelimiter_blocks_after_limit():
    rl = ratelimit.RateLimiter(window_seconds=60)
    results = [rl.is_limited("ip1", 3) for _ in range(5)]
    assert results == [False, False, False, True, True]


def test_ratelimiter_keys_are_separate():
    rl = ratelimit.RateLimiter(window_seconds=60)
    for _ in range(3):
        rl.is_limited("a", 3)
    assert rl.is_limited("a", 3) is True
    assert rl.is_limited("b", 3) is False


def test_ratelimiter_window_expires(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(ratelimit.time, "time", lambda: clock["t"])
    rl = ratelimit.RateLimiter(window_seconds=60)
    assert rl.is_limited("x", 1) is False
    assert rl.is_limited("x", 1) is True
    clock["t"] += 61
    assert rl.is_limited("x", 1) is False


# ---------- metrics ----------
def test_metrics_render_contains_recorded_request():
    metrics.record("GET", "/unit-test-path", 200, 0.01)
    text = metrics.render()
    assert 'path="/unit-test-path"' in text
    assert "http_request_duration_seconds_count" in text


# ---------- authentication ----------
def test_authenticate_roles():
    assert auth.authenticate(auth.API_USERNAME, auth.API_PASSWORD) == "admin"
    assert auth.authenticate(auth.STUDENT_USERNAME, auth.STUDENT_PASSWORD) == "user"


def test_authenticate_rejects_bad_input():
    assert auth.authenticate(auth.API_USERNAME, "wrong") is None
    assert auth.authenticate("nobody", "whatever") is None
    assert auth.authenticate("", "") is None


# ---------- pydantic validation ----------
def test_predict_request_valid():
    r = PredictRequest(hours_studied=5, attendance=80)
    assert r.hours_studied == 5


@pytest.mark.parametrize(
    "hours, attendance",
    [(-1, 80), (5, -1), (5, 100.5), ("abc", 80), (5, "abc")],
)
def test_predict_request_invalid(hours, attendance):
    with pytest.raises(ValidationError):
        PredictRequest(hours_studied=hours, attendance=attendance)


def test_batch_request_limits():
    with pytest.raises(ValidationError):
        BatchPredictRequest(items=[])
    with pytest.raises(ValidationError):
        BatchPredictRequest(items=[{"hours_studied": 1, "attendance": 50}] * 101)


# ---------- CSV upload validation (bad files are rejected, nothing is saved) ----------
@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"just,some,text\n",
        b"hours_studied,attendance\n1,2\n",
        b"hours_studied,attendance,passed\n1,2,5\n",
        b"hours_studied,attendance,passed\n",
        b"\xff\xfe\x00bad bytes",
    ],
)
def test_upload_rejects_bad_csv(content):
    with pytest.raises(ValueError):
        ml_utils.save_uploaded_dataset(content)


def test_upload_rejects_big_file():
    big = b"hours_studied,attendance,passed\n" + b"1,2,1\n" * 200_000
    with pytest.raises(ValueError):
        ml_utils.save_uploaded_dataset(big)
