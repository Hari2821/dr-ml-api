from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api.routes import router

app = FastAPI(
    title="Dr. ML Prediction App",
    version="1.0.0",
    description="Multi-disease prediction backend"
)

# CORS (Lovable + localhost + any lovable subdomain previews)
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

# Health + root endpoints (Render hits / and HEAD /)
@app.get("/")
def root():
    return {"status": "ok", "service": "dr-ml-api"}

@app.get("/health")
def health():
    return {"ok": True}

# Include API routes
app.include_router(router, prefix="/api")
