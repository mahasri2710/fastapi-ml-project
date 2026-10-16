"""API routes for training, prediction and evaluation."""
import logging

from fastapi import APIRouter, HTTPException

from utils import ml_utils
from utils.schemas import (
    EvaluateResponse,
    PredictRequest,
    PredictResponse,
    TrainResponse,
)

logger = logging.getLogger("ml_api")
router = APIRouter()


@router.post("/train", response_model=TrainResponse)
def train():
    """Train the model on data/sample.csv."""
    try:
        return ml_utils.train_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during training")
        raise HTTPException(status_code=500, detail="Training failed")


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict pass/fail for one student."""
    try:
        result = ml_utils.predict(request.hours_studied, request.attendance)
        return {"input": request, **result}
    except ml_utils.ModelNotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.get("/evaluate", response_model=EvaluateResponse)
def evaluate():
    """Show accuracy, precision, recall, F1 and confusion matrix."""
    try:
        return ml_utils.evaluate()
    except ml_utils.ModelNotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during evaluation")
        raise HTTPException(status_code=500, detail="Evaluation failed")
