from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api.routes import router


app = FastAPI(
    title="Dr. ML Prediction App",
    version="1.0.0",
    description="Multi-disease prediction backend"
)

# Allow Lovable frontend domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://care-predictor-tool.lovable.app",
        "https://lovable.app",
        "https://www.lovable.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")
