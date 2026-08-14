"""Pydantic response models for the FastAPI app."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class ClassificationTypeInfo(BaseModel):
    key: str
    label: str
    available: bool
    dataset_label: str
    dataset_hint: str
    classes: List[str] = []  # user-friendly class names; empty if not yet available


class ClassificationTypesResponse(BaseModel):
    classification_types: List[ClassificationTypeInfo]


class PredictionResponse(BaseModel):
    classification_type: str
    classification_label: str
    dataset_label: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    gradcam_image: str  # data:image/png;base64,... overlay
    original_image: str  # data:image/png;base64,... resized input, for side-by-side display
    low_confidence: bool
    low_confidence_warning: Optional[str] = None
    disclaimer: str = (
        "This is a Model Prediction generated for research / decision-support purposes. "
        "It is not a clinical diagnosis and does not replace assessment by a qualified "
        "medical professional."
    )
