# app/schemas/responses.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class DetectionResponse(BaseModel):
    id: str
    type: str
    filename: Optional[str]
    result: str
    confidence: float
    processing_time: float
    created_at: datetime
    
    # For image
    faces_detected: Optional[int] = 0
    
    # For text
    word_count: Optional[int] = None
    readability_score: Optional[float] = None
    
    # For URL
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None

class DetailedDetectionResponse(DetectionResponse):
    ai_probability: Optional[float] = None
    human_probability: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    faces_data: Optional[List[Dict]] = None
    suspicious_segments: Optional[List[Dict]] = None
    threat_types: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    heatmap: Optional[Dict[str, Any]] = None

class StatsResponse(BaseModel):
    total_scans: int
    fake_detected: int
    real_detected: int
    suspicious_detected: int
    accuracy: float = 98.3  # Matches your frontend
    avg_response_time: float = 0.8  # Matches your frontend (<1s)
    
    # For charts
    daily_stats: Dict[str, int]
    type_distribution: Dict[str, int]
    recent_detections: List[Dict]

class DetectionRequest(BaseModel):
    generate_heatmap: bool = False

class TextRequest(BaseModel):
    text: str

class URLRequest(BaseModel):
    url: str