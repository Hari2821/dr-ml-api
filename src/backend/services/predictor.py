import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
from fastapi import HTTPException

logger = logging.getLogger("uvicorn.error")

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None


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


def _canonicalize_key(k: str) -> str:
    # Make key matching tolerant: BloodPressure, blood_pressure, bloodpressure all match
    return "".join(ch for ch in k.strip().lower() if ch.isalnum())


def _make_lookup(features: dict) -> Dict[str, Any]:
    if not isinstance(features, dict):
        raise HTTPException(status_code=422, detail="Field 'features' must be an object/dict")
    return {_canonicalize_key(k): v for k, v in features.items()}


def _get_value(lookup: Dict[str, Any], target_key: str) -> Any:
    ck = _canonicalize_key(target_key)
    if ck in lookup:
        return lookup[ck]
    raise KeyError(target_key)


def _default_keys_for_disease(disease: str) -> List[str]:
    if disease == "heart_disease":
        return [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
    # Diabetes (support both common PIMA styles)
    return [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]


def _build_model_input(model: Any, disease: str, features: dict):
    lookup = _make_lookup(features)

    # Best: use the exact training columns if the model exposes them
    cols = None
    if hasattr(model, "feature_names_in_"):
        try:
            cols = list(getattr(model, "feature_names_in_"))
        except Exception:
            cols = None

    if not cols:
        cols = _default_keys_for_disease(disease)

    row = {}
    missing = []
    for c in cols:
        try:
            row[c] = float(_get_value(lookup, c))
        except KeyError:
            missing.append(c)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"Feature '{c}' must be numeric")

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing feature(s) for {disease}: {missing}"
        )

    # Prefer DataFrame if available (many pipelines require it)
    if pd is not None:
        return pd.DataFrame([row], columns=cols)

    # Fallback to numpy array in the same column order
    return np.array([row[c] for c in cols], dtype=float).reshape(1, -1)


def _predict(model: Any, X) -> Dict[str, Any]:
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

    try:
        X = _build_model_input(model, disease, input_data)
        out = _predict(model, X)
        return {"disease": disease, **out}
    except HTTPException:
        raise
    except Exception as e:
        # This will give you a JSON error and a full traceback in Render logs
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
