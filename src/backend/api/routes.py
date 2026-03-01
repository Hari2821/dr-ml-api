from fastapi import APIRouter, Response

from src.backend.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse
)
from src.backend.services.predictor import predict_disease

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "API is healthy and running"
    }


# IMPORTANT: Handle CORS preflight for /predict explicitly
# Preflight requests do not contain JSON body, so they must not hit validation.
@router.options("/predict")
def predict_preflight():
    return Response(status_code=200)


@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    disease = request.disease
    features = request.features

    # Optional hardening: normalize input (avoids "Heart Disease" vs "heart_disease" issues)
    if isinstance(disease, str):
        disease = disease.strip().lower().replace(" ", "_")

    result = predict_disease(
        disease=disease,
        input_data=features
    )
    return PredictionResponse(**result)
