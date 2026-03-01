import os
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api.routes import router
from src.backend.services.predictor import load_models

app = FastAPI(
    title="Dr. ML Prediction App",
    version="1.0.0",
    description="Multi-disease prediction backend"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://care-predictor-tool.lovable.app",
        "https://lovable.app",
        "https://www.lovable.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_origin_regex=r"^https://.*\.lovable\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_models()

@app.get("/")
def root():
    return {"status": "ok", "service": "dr-ml-api"}

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.get("/health")
def health():
    return {"ok": True}

# NEW: verify exactly what Render is running
@app.get("/version")
def version():
    return {
        "render_git_commit": os.environ.get("RENDER_GIT_COMMIT"),
        "render_service_id": os.environ.get("RENDER_SERVICE_ID"),
        "main_py": str(Path(__file__).resolve()),
    }

app.include_router(router, prefix="/api")
