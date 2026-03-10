# app/api/detection.py
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List
import logging

from app.database import get_db
from app.services.detection_service import DetectionService
from app.services.file_service import FileService
from app.schemas.responses import DetectionResponse, DetailedDetectionResponse

router = APIRouter(prefix="/api", tags=["detection"])
logger = logging.getLogger(__name__)

@router.post("/detect/image")
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    generate_heatmap: bool = Form(False),
    db: Session = Depends(get_db)
):
    """
    Detect deepfake in uploaded image
    Matches your frontend ImageUploader component
    """
    try:
        # Read file
        file_data = await file.read()
        
        # Validate
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        # Save temporarily
        file_service = FileService()
        file_path = file_service.save_file(file_data, file.filename, 'images')
        
        # Process
        detection_service = DetectionService(db)
        result = detection_service.process_image(
            file_path, 
            file.filename, 
            generate_heatmap,
            request.client.host
        )
        
        # Clean up
        file_service.delete_file(file_path)
        
        if not result.get('success', True):
            raise HTTPException(500, result.get('error', 'Detection failed'))
        
        return result
        
    except Exception as e:
        logger.error(f"Image detection error: {e}")
        raise HTTPException(500, str(e))

@router.post("/detect/text")
async def detect_text(
    request: Request,
    text: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Detect AI-generated text
    Matches your frontend TextAnalyzer component
    """
    try:
        detection_service = DetectionService(db)
        result = detection_service.process_text(text, request.client.host)
        
        if not result.get('success', True):
            raise HTTPException(500, result.get('error', 'Detection failed'))
        
        return result
        
    except Exception as e:
        logger.error(f"Text detection error: {e}")
        raise HTTPException(500, str(e))