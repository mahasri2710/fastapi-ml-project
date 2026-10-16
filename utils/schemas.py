"""Pydantic models for request and response validation."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    hours_studied: float = Field(..., ge=0, description="Hours studied")
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage (0-100)")


class TrainResponse(BaseModel):
    message: str
    rows_used: int
    train_accuracy: float
    test_accuracy: float
    tuned: bool = False
    best_params: Optional[Dict[str, float]] = None
    cv_score: Optional[float] = None


class UploadResponse(BaseModel):
    message: str
    rows: int
    columns: List[str]


class DatasetInfoResponse(BaseModel):
    source: str
    rows: int
    columns: List[str]


class PredictResponse(BaseModel):
    input: PredictRequest
    passed: bool
    pass_probability: float


class EvaluateResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    test_samples: int
