# app/api/stats.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.services.detection_service import DetectionService
from app.schemas.responses import StatsResponse

router = APIRouter(prefix="/api", tags=["stats"])

@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """
    Get dashboard statistics
    Matches your frontend dashboard stats
    """
    service = DetectionService(db)
    return service.get_stats()

@router.get("/detections/recent")
async def get_recent_detections(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get recent detections for dashboard table
    """
    service = DetectionService(db)
    return service.get_recent_detections(limit)