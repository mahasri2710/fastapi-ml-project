"""Integration tests: talk to a REAL running server using the requests library.

Start the server in one terminal first:   uvicorn main:app --reload
Then run in another terminal:             pytest tests/test_integration.py -v
If the server is not running, these tests are skipped (not failed).
"""
import os

import pytest
import requests
from jsonschema import validate

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
USERNAME = os.getenv("API_USERNAME", "admin")
PASSWORD = os.getenv("API_PASSWORD", "admin123")

PREDICT_SCHEMA = {
    "type": "object",
    "required": ["input", "passed", "pass_probability"],
    "properties": {
        "input": {
            "type": "object",
            "required": ["hours_studied", "attendance"],
            "properties": {
                "hours_studied": {"type": "number"},
                "attendance": {"type": "number"},
            },
        },
        "passed": {"type": "boolean"},
        "pass_probability": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

EVALUATE_SCHEMA = {
    "type": "object",
    "required": ["accuracy", "precision", "recall", "f1_score", "confusion_matrix", "test_samples"],
    "properties": {
        "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
        "precision": {"type": "number", "minimum": 0, "maximum": 1},
        "recall": {"type": "number", "minimum": 0, "maximum": 1},
        "f1_score": {"type": "number", "minimum": 0, "maximum": 1},
        "confusion_matrix": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        },
        "test_samples": {"type": "integer", "minimum": 1},
    },
}

BATCH_SCHEMA = {
    "type": "object",
    "required": ["count", "results"],
    "properties": {
        "count": {"type": "integer"},
        "results": {"type": "array", "items": PREDICT_SCHEMA},
    },
}


@pytest.fixture(scope="module", autouse=True)
def server_is_running():
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.RequestException:
        pytest.skip("Server is not running. Start it with: uvicorn main:app --reload")


@pytest.fixture(scope="module")
def headers():
    r = requests.post(
        f"{BASE_URL}/login", data={"username": USERNAME, "password": PASSWORD}, timeout=10
    )
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_full_flow(headers):
    # 1. train
    r = requests.post(f"{BASE_URL}/train", headers=headers, timeout=60)
    assert r.status_code == 200

    # 2. predict one student, check the JSON format
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"hours_studied": 8, "attendance": 90},
        headers=headers, timeout=10,
    )
    assert r.status_code == 200
    validate(instance=r.json(), schema=PREDICT_SCHEMA)

    # 3. predict many students
    body = {"items": [
        {"hours_studied": 8, "attendance": 90},
        {"hours_studied": 1, "attendance": 40},
    ]}
    r = requests.post(f"{BASE_URL}/predict/batch", json=body, headers=headers, timeout=10)
    assert r.status_code == 200
    validate(instance=r.json(), schema=BATCH_SCHEMA)
    assert r.json()["count"] == 2

    # 4. evaluate
    r = requests.get(f"{BASE_URL}/evaluate", headers=headers, timeout=10)
    assert r.status_code == 200
    validate(instance=r.json(), schema=EVALUATE_SCHEMA)


def test_error_format_is_json():
    r = requests.post(f"{BASE_URL}/train", timeout=10)
    assert r.status_code == 401
    assert "detail" in r.json()


def test_validation_error_format(headers):
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"hours_studied": 5, "attendance": 150},
        headers=headers, timeout=10,
    )
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)
