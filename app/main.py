from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.detection import router as detection_router
from app.api.stats import router as stats_router
from app.api.text import router as text_router
from app.api.live import router as live_router
from app.api.url import router as url_router

app = FastAPI(
    title="AI Deepfake Detection & Digital Authenticity Verification System",
    description="Detect manipulated images, videos, audio and AI generated content",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(detection_router, prefix="/api/detect", tags=["Detection"])
app.include_router(text_router, prefix="/api/text", tags=["Text Analysis"])
app.include_router(live_router, prefix="/api/live", tags=["Live Detection"])
app.include_router(url_router, prefix="/api/url", tags=["URL Verification"])
app.include_router(stats_router, prefix="/api/stats", tags=["Statistics"])


@app.get("/")
def root():
    return {
        "message": "Deepfake Detection API Running",
        "version": "1.0.0"
    }