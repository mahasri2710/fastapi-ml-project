"""Automated tests. Run from the project folder with:  pytest"""
import pytest
from fastapi.testclient import TestClient

from main import app
from utils import auth

client = TestClient(app)


def _login(username, password):
    r = client.post("/login", data={"username": username, "password": password})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def admin():
    return _login(auth.API_USERNAME, auth.API_PASSWORD)


@pytest.fixture(scope="module")
def user():
    return _login(auth.STUDENT_USERNAME, auth.STUDENT_PASSWORD)


@pytest.fixture(scope="module")
def trained(admin):
    r = client.post("/train", headers=admin)
    assert r.status_code == 200
    return r.json()


# ---------- public ----------
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


# ---------- login and roles ----------
def test_login_wrong_password():
    r = client.post("/login", data={"username": auth.API_USERNAME, "password": "wrong"})
    assert r.status_code == 401


def test_me_needs_token():
    assert client.get("/me").status_code == 401


def test_me_shows_roles(admin, user):
    assert client.get("/me", headers=admin).json()["role"] == "admin"
    assert client.get("/me", headers=user).json()["role"] == "user"


def test_train_needs_token():
    assert client.post("/train").status_code == 401


def test_user_cannot_train(user):
    assert client.post("/train", headers=user).status_code == 403


def test_user_cannot_upload(user):
    files = {"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}
    assert client.post("/upload-dataset", files=files, headers=user).status_code == 403


def test_upload_needs_token():
    files = {"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}
    assert client.post("/upload-dataset", files=files).status_code == 401


# ---------- admin actions ----------
def test_admin_can_train(trained):
    assert trained["message"] == "Model trained successfully"


def test_admin_can_tune(admin):
    r = client.post("/train?tune=true", headers=admin)
    assert r.status_code == 200
    assert r.json()["tuned"] is True


# ---------- prediction (any logged in user) ----------
def test_predict_needs_token(trained):
    r = client.post("/predict", json={"hours_studied": 8, "attendance": 90})
    assert r.status_code == 401


def test_predict_as_user(trained, user):
    r = client.post("/predict", json={"hours_studied": 8, "attendance": 90}, headers=user)
    assert r.status_code == 200
    assert "passed" in r.json()


def test_predict_validation_error(user):
    r = client.post("/predict", json={"hours_studied": 5, "attendance": 150}, headers=user)
    assert r.status_code == 422


def test_batch_predict(trained, user):
    body = {"items": [
        {"hours_studied": 8, "attendance": 90},
        {"hours_studied": 1, "attendance": 40},
    ]}
    r = client.post("/predict/batch", json=body, headers=user)
    assert r.status_code == 200
    assert r.json()["count"] == 2
    assert len(r.json()["results"]) == 2


def test_batch_too_many_items(user):
    body = {"items": [{"hours_studied": 1, "attendance": 50}] * 101}
    assert client.post("/predict/batch", json=body, headers=user).status_code == 422


def test_batch_empty(user):
    assert client.post("/predict/batch", json={"items": []}, headers=user).status_code == 422


def test_model_info(trained, user):
    r = client.get("/model-info", headers=user)
    assert r.status_code == 200
    assert r.json()["features"] == ["hours_studied", "attendance"]


def test_evaluate(trained, user):
    r = client.get("/evaluate", headers=user)
    assert r.status_code == 200
    assert 0 <= r.json()["accuracy"] <= 1
