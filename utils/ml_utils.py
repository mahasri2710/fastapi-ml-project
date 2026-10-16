"""ML helper functions: train, predict, evaluate.

Paths are built from this file's location, so the code works
no matter which folder you start uvicorn from.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger("ml_api")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sample.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

FEATURES = ["hours_studied", "attendance"]
TARGET_CANDIDATES = ["passed", "pass", "result", "label", "target"]


class ModelNotReadyError(Exception):
    """Raised when predict/evaluate is called before a valid /train."""


def _load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset is empty")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df


def _target_column(df: pd.DataFrame) -> str:
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    # fall back to the last column
    return df.columns[-1]


def train_model() -> dict:
    """Train on data/sample.csv, save the model, return a summary."""
    logger.info("Training started")
    df = _load_dataset()
    target = _target_column(df)
    df = df.dropna(subset=FEATURES + [target])

    X = df[FEATURES]
    y = df[target].astype(int)
    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes (pass and fail) to train")

    # keep a small test set so /evaluate gives an honest score
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    if y_train.nunique() < 2:
        X_train, y_train, X_test, y_test = X, y, X, y

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "X_test": X_test, "y_test": y_test}, MODEL_PATH)

    train_acc = float(model.score(X_train, y_train))
    test_acc = float(model.score(X_test, y_test))
    logger.info(
        "Training finished: rows=%d train_acc=%.3f test_acc=%.3f",
        len(df), train_acc, test_acc,
    )
    return {
        "message": "Model trained successfully",
        "rows_used": int(len(df)),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
    }


def _load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise ModelNotReadyError("Model not trained yet. Call POST /train first.")
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ModelNotReadyError(
            "Old model file found. Call POST /train again to create a new one."
        )
    return bundle


def predict(hours_studied: float, attendance: float) -> dict:
    bundle = _load_bundle()
    model = bundle["model"]
    row = pd.DataFrame([[hours_studied, attendance]], columns=FEATURES)
    passed = bool(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][list(model.classes_).index(1)])
    logger.info(
        "Prediction: hours=%s attendance=%s -> passed=%s prob=%.3f",
        hours_studied, attendance, passed, prob,
    )
    return {"passed": passed, "pass_probability": round(prob, 4)}


def evaluate() -> dict:
    bundle = _load_bundle()
    model, X_test, y_test = bundle["model"], bundle["X_test"], bundle["y_test"]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    result = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "test_samples": int(len(y_test)),
    }
    logger.info("Evaluation: %s", result)
    return result
