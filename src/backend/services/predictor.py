import logging
from fastapi import HTTPException

logger = logging.getLogger("uvicorn.error")


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
        "heart_disease_prediction": "heart_disease",
        "cardio": "heart_disease",
        "cardiac": "heart_disease",
        "diabetic": "diabetes",
        "diabetes_prediction": "diabetes",
        "sugar": "diabetes",
    }

    d = aliases.get(d, d)

    allowed = {"diabetes", "heart_disease"}
    if d not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid disease type: '{disease}'. Use 'diabetes' or 'heart_disease'"
        )

    return d


def predict_disease(disease: str, input_data: dict):
    disease = _normalize_disease(disease)

    # IMPORTANT:
    # Keep your existing model inference logic below.
    # I am not changing your model code, only making disease routing safe.

    if disease == "diabetes":
        # your existing diabetes prediction logic here
        # return {"disease": "diabetes", "prediction": ..., "probability": ...}
        raise HTTPException(status_code=500, detail="Diabetes model logic not wired here yet")

    if disease == "heart_disease":
        # your existing heart disease prediction logic here
        # return {"disease": "heart_disease", "prediction": ..., "probability": ...}
        raise HTTPException(status_code=500, detail="Heart disease model logic not wired here yet")
