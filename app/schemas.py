#file : \brain-tumor-backend\app\schemas.py
from typing import Dict

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
