# app/db_models/detection.py
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.sql import func
import uuid
from app.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Detection(Base):
    __tablename__ = "detections"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    type = Column(String, nullable=False)  # image, video, audio, text, url
    filename = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    
    # Results - matches your frontend stats
    result = Column(String, nullable=False)  # REAL, FAKE, SUSPICIOUS
    confidence = Column(Float, nullable=False)
    ai_probability = Column(Float, nullable=True)
    human_probability = Column(Float, nullable=True)
    
    # Detailed data
    faces_detected = Column(Integer, default=0)
    faces_data = Column(JSON, nullable=True)
    probabilities = Column(JSON, nullable=True)
    
    # Text specific
    word_count = Column(Integer, nullable=True)
    readability_score = Column(Float, nullable=True)
    perplexity = Column(Float, nullable=True)
    burstiness = Column(Float, nullable=True)
    suspicious_segments = Column(JSON, nullable=True)
    
    # URL specific
    url = Column(String, nullable=True)
    domain = Column(String, nullable=True)
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String, nullable=True)
    threat_types = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=True)
    
    # Metadata
    processing_time = Column(Float)
    ip_address = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Heatmap (base64)
    heatmap_data = Column(Text, nullable=True)
    overlay_data = Column(Text, nullable=True)
    manipulated_regions = Column(JSON, nullable=True)

class SystemMetric(Base):
    __tablename__ = "system_metrics"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    total_scans = Column(Integer, default=0)
    fake_detected = Column(Integer, default=0)
    real_detected = Column(Integer, default=0)