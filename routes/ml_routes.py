"""API routes for dataset upload, training, prediction and evaluation.

Access rules:
  admin only   : /upload-dataset, /train
  any logged in: /predict, /predict/batch, /evaluate, /model-info, /dataset-info
"""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from utils import ml_utils
from utils.auth import get_current_user, require_role
from utils.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DatasetInfoResponse,
    EvaluateResponse,
    ModelInfoResponse,
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
    dependencies=[Depends(require_role("admin"))],
)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV (columns: hours_studied, attendance, passed). Admin only."""
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


@router.get(
    "/dataset-info",
    response_model=DatasetInfoResponse,
    dependencies=[Depends(get_current_user)],
)
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
    dependencies=[Depends(require_role("admin"))],
)
def train(tune: bool = False):
    """Train the model. Use tune=true for tuning. Admin only."""
    try:
        return ml_utils.train_model(tune=tune)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during training")
        raise HTTPException(status_code=500, detail="Training failed")


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    dependencies=[Depends(get_current_user)],
)
def model_info():
    """Show details of the model that is currently loaded."""
    try:
        return ml_utils.model_info()
    except ml_utils.ModelNotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error reading model info")
        raise HTTPException(status_code=500, detail="Could not read model info")


@router.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(get_current_user)],
)
def predict(request: PredictRequest):
    """Predict pass/fail for one student. Needs login."""
    try:
        result = ml_utils.predict(request.hours_studied, request.attendance)
        return {"input": request, **result}
    except ml_utils.ModelNotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    dependencies=[Depends(get_current_user)],
)
def predict_batch(request: BatchPredictRequest):
    """Classify up to 100 students in one request. Needs login."""
    try:
        rows = [(i.hours_studied, i.attendance) for i in request.items]
        results = ml_utils.predict_batch(rows)
        return {
            "count": len(results),
            "results": [
                {"input": item, **result}
                for item, result in zip(request.items, results)
            ],
        }
    except ml_utils.ModelNotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during batch prediction")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@router.get(
    "/evaluate",
    response_model=EvaluateResponse,
    dependencies=[Depends(get_current_user)],
)
def evaluate():
    """Show accuracy, precision, recall, F1 and confusion matrix. Needs login."""
    try:
        return ml_utils.evaluate()
    except ml_utils.ModelNotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during evaluation")
        raise HTTPException(status_code=500, detail="Evaluation failed")
