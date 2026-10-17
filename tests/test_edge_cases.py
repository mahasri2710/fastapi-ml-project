"""Edge case tests: wrong input, bad tokens, blocked requests."""
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app
from utils import auth, middleware

client = TestClient(app)


def _login(username, password):
    r = client.post("/login", data={"username": username, "password": password})
    assert r.status_code == 200
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def admin_token():
    return _login(auth.API_USERNAME, auth.API_PASSWORD)


@pytest.fixture(scope="module")
def admin(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="module")
def trained(admin):
    assert client.post("/train", headers=admin).status_code == 200


# ---------- wrong input to /predict ----------
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"hours_studied": 5},
        {"attendance": 80},
        {"hours_studied": -1, "attendance": 80},
        {"hours_studied": 5, "attendance": -1},
        {"hours_studied": 5, "attendance": 100.01},
        {"hours_studied": "abc", "attendance": 80},
        {"hours_studied": None, "attendance": 80},
    ],
)
def test_predict_rejects_bad_body(admin, body):
    assert client.post("/predict", json=body, headers=admin).status_code == 422


def test_predict_zero_values(trained, admin):
    r = client.post("/predict", json={"hours_studied": 0, "attendance": 0}, headers=admin)
    assert r.status_code == 200
    assert "passed" in r.json()


def test_predict_very_large_hours(trained, admin):
    r = client.post("/predict", json={"hours_studied": 1000000, "attendance": 100}, headers=admin)
    assert r.status_code == 200


def test_batch_with_one_bad_item(admin):
    body = {"items": [
        {"hours_studied": 5, "attendance": 80},
        {"hours_studied": -5, "attendance": 80},
    ]}
    assert client.post("/predict/batch", json=body, headers=admin).status_code == 422


# ---------- wrong routes and methods ----------
def test_unknown_route_is_404():
    assert client.get("/this-does-not-exist").status_code == 404


def test_wrong_method_is_405(admin):
    assert client.get("/train", headers=admin).status_code == 405


def test_login_missing_fields_is_422():
    assert client.post("/login", data={}).status_code == 422


# ---------- bad tokens ----------
def test_garbage_token_is_401():
    r = client.get("/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_tampered_token_is_401(admin_token):
    tampered = admin_token.rsplit(".", 1)[0] + ".invalidsignature"
    r = client.get("/me", headers={"Authorization": f"Bearer {tampered}"})
    assert r.status_code == 401


def test_expired_token_is_401():
    expired = jwt.encode(
        {"sub": auth.API_USERNAME, "role": "admin",
         "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
        auth.SECRET_KEY, algorithm=auth.ALGORITHM,
    )
    r = client.get("/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401


def test_token_signed_with_wrong_secret_is_401():
    fake = jwt.encode(
        {"sub": auth.API_USERNAME, "role": "admin",
         "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        "some-other-secret-that-is-long-enough-123456", algorithm="HS256",
    )
    r = client.get("/me", headers={"Authorization": f"Bearer {fake}"})
    assert r.status_code == 401


def test_token_for_unknown_user_is_401():
    fake = jwt.encode(
        {"sub": "hacker", "role": "admin",
         "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        auth.SECRET_KEY, algorithm=auth.ALGORITHM,
    )
    r = client.get("/me", headers={"Authorization": f"Bearer {fake}"})
    assert r.status_code == 401


# ---------- bad uploads (as admin) ----------
def test_upload_wrong_extension(admin):
    files = {"file": ("data.txt", b"hello", "text/plain")}
    assert client.post("/upload-dataset", files=files, headers=admin).status_code == 400


def test_upload_empty_csv(admin):
    files = {"file": ("data.csv", b"", "text/csv")}
    assert client.post("/upload-dataset", files=files, headers=admin).status_code == 400


def test_upload_csv_missing_columns(admin):
    files = {"file": ("data.csv", b"a,b\n1,2\n", "text/csv")}
    r = client.post("/upload-dataset", files=files, headers=admin)
    assert r.status_code == 400
    assert "missing" in r.json()["detail"].lower()


# ---------- rate limiting and IP blocking (through the real app) ----------
def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setattr(middleware, "RATE_LIMIT", 3)
    other = TestClient(app, client=("9.9.9.9", 5000))
    codes = [other.get("/health").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_blocked_ip_returns_403(monkeypatch):
    monkeypatch.setattr(middleware, "BLOCKED_IPS", {"8.8.8.8"})
    blocked = TestClient(app, client=("8.8.8.8", 5000))
    assert blocked.get("/health").status_code == 403
    assert client.get("/health").status_code == 200
