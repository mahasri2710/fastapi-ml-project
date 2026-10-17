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


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class MeResponse(BaseModel):
    username: str
    role: str


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchItemResult(BaseModel):
    input: PredictRequest
    passed: bool
    pass_probability: float


class BatchPredictResponse(BaseModel):
    count: int
    results: List[BatchItemResult]


class ModelInfoResponse(BaseModel):
    model_type: str
    features: List[str]
    classes: List[int]
    trained_at: str
