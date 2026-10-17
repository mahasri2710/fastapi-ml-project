"""Automated tests. Run from the project folder with:  pytest"""
import pytest
from fastapi.testclient import TestClient

from main import app
from utils import auth

client = TestClient(app)


@pytest.fixture(scope="module")
def headers():
    r = client.post("/login", data={"username": auth.API_USERNAME, "password": auth.API_PASSWORD})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_login_wrong_password():
    r = client.post("/login", data={"username": auth.API_USERNAME, "password": "wrong"})
    assert r.status_code == 401


def test_me_needs_token():
    assert client.get("/me").status_code == 401


def test_me_with_token(headers):
    r = client.get("/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["username"] == auth.API_USERNAME


def test_train_needs_token():
    assert client.post("/train").status_code == 401


def test_upload_needs_token():
    files = {"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}
    assert client.post("/upload-dataset", files=files).status_code == 401


def test_train_with_token(headers):
    r = client.post("/train", headers=headers)
    assert r.status_code == 200
    assert r.json()["message"] == "Model trained successfully"


def test_train_with_tuning(headers):
    r = client.post("/train?tune=true", headers=headers)
    assert r.status_code == 200
    assert r.json()["tuned"] is True


def test_predict_after_train(headers):
    client.post("/train", headers=headers)
    r = client.post("/predict", json={"hours_studied": 8, "attendance": 90})
    assert r.status_code == 200
    assert "passed" in r.json()


def test_predict_validation_error():
    r = client.post("/predict", json={"hours_studied": 5, "attendance": 150})
    assert r.status_code == 422


def test_evaluate_after_train(headers):
    client.post("/train", headers=headers)
    r = client.get("/evaluate")
    assert r.status_code == 200
    assert 0 <= r.json()["accuracy"] <= 1


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text
