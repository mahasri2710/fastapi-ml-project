"""ML model code: load data, train, save, load, predict."""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("ml_api")

DATA_PATH = Path("data/sample.csv")
MODEL_PATH = Path("models/model.pkl")
FEATURES = ["hours_studied", "attendance"]


def train_model(data_path: Path = DATA_PATH) -> dict:
    """Train a simple LogisticRegression model and save it."""
    logger.info("Training started. Loading data from %s", data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    missing = [c for c in FEATURES + ["passed"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.dropna()
    X = df[FEATURES].values
    y = df["passed"].values

    model = LogisticRegression()
    model.fit(X, y)
    accuracy = float(model.score(X, y))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Training done. Rows=%d, accuracy=%.2f", len(df), accuracy)

    return {"rows_used": int(len(df)), "train_accuracy": round(accuracy, 3)}


def load_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet. Call POST /train first.")
    return joblib.load(MODEL_PATH)


def predict(hours_studied: float, attendance: float) -> dict:
    """Predict pass/fail for one student."""
    model = load_model()
    x = np.array([[hours_studied, attendance]])
    label = int(model.predict(x)[0])
    prob = float(model.predict_proba(x)[0][1])
    return {"passed": bool(label), "pass_probability": round(prob, 3)}
