import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ml_api")

app = FastAPI(
    title="ML API with FastAPI",
    description="Train a simple ML model and get predictions as JSON.",
    version="2.0",
)


class PredictRequest(BaseModel):
    hours_studied: float = Field(..., ge=0, le=24, examples=[6])
    attendance: float = Field(..., ge=0, le=100, examples=[80])


@app.get("/")
def home():
    return {"message": "My ML API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train():
    """Train the model on data/sample.csv."""
    try:
        result = models.train_model()
        return {"message": "Model trained successfully", **result}
    except (FileNotFoundError, ValueError) as e:
        logger.error("Training failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail=f"Training error: {e}")


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict whether a student passes (needs /train first)."""
    logger.info("Predict request: %s", req.model_dump())
    try:
        result = models.predict(req.hours_studied, req.attendance)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    logger.info("Predict response: %s", result)
    return {"input": req.model_dump(), **result}
