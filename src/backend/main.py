from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api.routes import router

app = FastAPI(
    title="Dr. ML Prediction App",
    version="1.0.0",
    description="Multi-disease prediction backend"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with your Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
