"""API routes for dataset upload, training, prediction and evaluation."""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from utils import ml_utils
from utils.auth import get_current_user
from utils.schemas import (
    DatasetInfoResponse,
    EvaluateResponse,
    PredictRequest,
    PredictResponse,
    TrainResponse,
    UploadResponse,
)

logger = logging.getLogger("ml_api")
router = APIRouter()


@router.post(
    "/upload-dataset",
    response_model=UploadResponse,
    dependencies=[Depends(get_current_user)],
)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV (columns: hours_studied, attendance, passed). Needs login."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")
    try:
        content = await file.read()
        return ml_utils.save_uploaded_dataset(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during upload")
        raise HTTPException(status_code=500, detail="Upload failed")


@router.get("/dataset-info", response_model=DatasetInfoResponse)
def dataset_info():
    """Show which dataset will be used for training."""
    try:
        return ml_utils.dataset_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error reading dataset info")
        raise HTTPException(status_code=500, detail="Could not read dataset")


@router.post(
    "/train",
    response_model=TrainResponse,
    dependencies=[Depends(get_current_user)],
)
def train(tune: bool = False):
    """Train the model. Use tune=true for tuning. Needs login."""
    try:
        return ml_utils.train_model(tune=tune)
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
