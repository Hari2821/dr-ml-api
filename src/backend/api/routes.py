import logging
import os
from pathlib import Path

from fastapi import APIRouter, Response

from src.backend.schemas.prediction_schema import PredictionRequest, PredictionResponse
from src.backend.services.predictor import predict_disease
from src.backend.services import predictor as predictor_module

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


# NEW: verify exactly what code file is running on Render
@router.get("/version")
def api_version():
    return {
        "render_git_commit": os.environ.get("RENDER_GIT_COMMIT"),
        "routes_py": str(Path(__file__).resolve()),
        "predictor_py": str(Path(predictor_module.__file__).resolve()),
    }
