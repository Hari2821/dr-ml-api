import logging
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
from fastapi import HTTPException

logger = logging.getLogger("uvicorn.error")

# predictor.py path:
# <repo>/src/backend/services/predictor.py
# parents:
# 0 = services
# 1 = backend
# 2 = src
# 3 = repo root (dr-ml-api)
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "model_dir"

HEART_MODEL_PATH = MODEL_DIR / "heart_disease_prediction_pipeline.joblib"
DIABETES_MODEL_PATH = MODEL_DIR / "diabetes_prediction_pipeline.joblib"

_models: Dict[str, Any] = {}


def _normalize_disease(disease) -> str:
    if disease is None:
        raise HTTPException(status_code=422, detail="Missing field: disease")

    if not isinstance(disease, str):
        raise HTTPException(status_code=422, detail="Field 'disease' must be a string")

    d = disease.strip().lower()
    d = d.replace("-", "_").replace(" ", "_")

    aliases = {
        "heart": "heart_disease",
        "heartdisease": "heart_disease",
        "cardio": "heart_disease",
        "cardiac": "heart_disease",
        "diabetic": "diabetes",
        "sugar": "diabetes",
    }

    d = aliases.get(d, d)

    if d not in {"diabetes", "heart_disease"}:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid disease type: '{disease}'. Use 'diabetes' or 'heart_disease'"
        )

    return d


def load_models() -> None:
    global _models
    if _models:
        return

    logger.info("Loading trained models...")

    if not MODEL_DIR.exists():
        raise RuntimeError(f"model_dir not found at: {MODEL_DIR}")

    if not HEART_MODEL_PATH.exists():
        raise RuntimeError(f"Missing model file: {HEART_MODEL_PATH}")

    if not DIABETES_MODEL_PATH.exists():
        raise RuntimeError(f"Missing model file: {DIABETES_MODEL_PATH}")

    _models["heart_disease"] = joblib.load(HEART_MODEL_PATH)
    _models["diabetes"] = joblib.load(DIABETES_MODEL_PATH)

    logger.info(
        f"Models loaded successfully. heart={HEART_MODEL_PATH.name}, diabetes={DIABETES_MODEL_PATH.name}"
    )


def _vector_from_features(disease: str, features: dict) -> np.ndarray:
    if not isinstance(features, dict):
        raise HTTPException(status_code=422, detail="Field 'features' must be an object/dict")

    if disease == "heart_disease":
        keys: List[str] = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
    else:
        # If your diabetes pipeline expects different names, update only these keys
        keys = [
            "pregnancies", "glucose", "bloodpressure", "skinthickness",
            "insulin", "bmi", "diabetespedigreefunction", "age"
        ]

    missing = [k for k in keys if k not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing feature(s) for {disease}: {missing}"
        )

    try:
        x = [float(features[k]) for k in keys]
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"All features for {disease} must be numeric")

    return np.array(x, dtype=float).reshape(1, -1)


def _predict(model: Any, X: np.ndarray) -> Dict[str, Any]:
    pred = int(model.predict(X)[0])

    prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba is not None and len(proba[0]) > 1:
            prob = float(proba[0][1])

    return {"prediction": pred, "probability": prob}


def predict_disease(disease: str, input_data: dict) -> Dict[str, Any]:
    disease = _normalize_disease(disease)
    load_models()

    model = _models.get(disease)
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded for: {disease}")

    X = _vector_from_features(disease, input_data)
    out = _predict(model, X)

    return {"disease": disease, **out}
