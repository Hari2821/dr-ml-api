import logging
from fastapi import APIRouter, Response

from src.backend.schemas.prediction_schema import PredictionRequest, PredictionResponse
from src.backend.services.predictor import predict_disease

logger = logging.getLogger("uvicorn.error")

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok", "message": "API is healthy and running"}


@router.options("/predict")
def predict_preflight():
    return Response(status_code=200)


@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    logger.info(f"Incoming disease value from client: {request.disease!r}")
    result = predict_disease(disease=request.disease, input_data=request.features)
    return PredictionResponse(**result)
