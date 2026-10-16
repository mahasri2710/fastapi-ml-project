"""Pydantic models for request and response validation."""
from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    hours_studied: float = Field(..., ge=0, description="Hours studied")
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage (0-100)")


class TrainResponse(BaseModel):
    message: str
    rows_used: int
    train_accuracy: float
    test_accuracy: float


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
