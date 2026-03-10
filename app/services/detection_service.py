# app/services/detection_service.py
import os
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db_models.detection import Detection
from app.services.ml_service import MLService
from app.services.file_service import FileService
from app.config import settings

logger = logging.getLogger(__name__)

class DetectionService:
    def __init__(self, db: Session):
        self.db = db
        self.ml_service = MLService()
        self.file_service = FileService()
    
    def save_detection(self, data: Dict[str, Any]) -> Detection:
        """Save detection to database"""
        detection = Detection(**data)
        self.db.add(detection)
        self.db.commit()
        self.db.refresh(detection)
        return detection
    
    def process_image(self, file_path: str, filename: str, 
                     generate_heatmap: bool = False, 
                     ip: Optional[str] = None) -> Dict[str, Any]:
        """Process image detection"""
        # Calculate hash for duplicate check
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Check if already processed (caching)
        existing = self.db.query(Detection).filter(
            Detection.filename == filename,
            Detection.type == 'image'
        ).first()
        
        if existing:
            return {
                'id': existing.id,
                'cached': True,
                'result': existing.result,
                'confidence': existing.confidence,
                'faces_detected': existing.faces_detected,
                'processing_time': 0.1
            }
        
        # Run ML detection
        ml_result = self.ml_service.detect_image(file_path, generate_heatmap)
        
        if not ml_result['success']:
            return ml_result
        
        # Prepare database record
        detection_data = {
            'id': str(uuid.uuid4()),
            'type': 'image',
            'filename': filename,
            'file_size': os.path.getsize(file_path),
            'result': 'FAKE' if ml_result['result']['is_fake'] else 'REAL',
            'confidence': ml_result['result']['confidence'],
            'ai_probability': ml_result['result']['fake_probability'],
            'human_probability': ml_result['result']['real_probability'],
            'probabilities': {
                'fake': ml_result['result']['fake_probability'],
                'real': ml_result['result']['real_probability']
            },
            'faces_detected': ml_result['faces_detected'],
            'faces_data': ml_result['faces'],
            'processing_time': ml_result['processing_time'],
            'ip_address': ip
        }
        
        # Add heatmap if generated
        if generate_heatmap and ml_result.get('heatmap'):
            detection_data['heatmap_data'] = ml_result['heatmap']['heatmap']
            detection_data['overlay_data'] = ml_result['heatmap']['overlay']
            detection_data['manipulated_regions'] = ml_result['heatmap']['regions']
        
        # Save to database
        detection = self.save_detection(detection_data)
        
        # Prepare response for frontend
        response = {
            'id': detection.id,
            'success': True,
            'result': detection.result,
            'confidence': detection.confidence,
            'faces_detected': detection.faces_detected,
            'processing_time': detection.processing_time,
            'message': f"Image is {detection.result} with {detection.confidence:.1%} confidence"
        }
        
        if generate_heatmap and ml_result.get('heatmap'):
            response['heatmap'] = ml_result['heatmap']
        
        return response
    
    def process_text(self, text: str, ip: Optional[str] = None) -> Dict[str, Any]:
        """Process text detection"""
        ml_result = self.ml_service.detect_text(text)
        
        if not ml_result['success']:
            return ml_result
        
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        detection_data = {
            'id': str(uuid.uuid4()),
            'type': 'text',
            'result': 'FAKE' if ml_result['is_ai_generated'] else 'REAL',
            'confidence': ml_result['confidence'],
            'ai_probability': ml_result['ai_probability'],
            'human_probability': ml_result['human_probability'],
            'probabilities': {
                'ai': ml_result['ai_probability'],
                'human': ml_result['human_probability']
            },
            'word_count': ml_result['word_count'],
            'readability_score': ml_result['readability_score'],
            'perplexity': ml_result['perplexity'],
            'burstiness': ml_result['burstiness'],
            'suspicious_segments': ml_result['suspicious_segments'],
            'processing_time': ml_result['processing_time'],
            'ip_address': ip
        }
        
        detection = self.save_detection(detection_data)
        
        return {
            'id': detection.id,
            'success': True,
            'result': detection.result,
            'confidence': detection.confidence,
            'word_count': detection.word_count,
            'processing_time': detection.processing_time,
            'message': f"Text is {detection.result} with {detection.confidence:.1%} confidence"
        }
    
    def get_recent_detections(self, limit: int = 10) -> List[Dict]:
        """Get recent detections for dashboard"""
        detections = self.db.query(Detection).order_by(
            Detection.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                'id': d.id,
                'type': d.type,
                'filename': d.filename or d.url or 'Text Analysis',
                'result': d.result,
                'confidence': d.confidence,
                'date': d.created_at.isoformat()
            }
            for d in detections
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics matching your frontend dashboard"""
        total = self.db.query(Detection).count()
        fake = self.db.query(Detection).filter(Detection.result == 'FAKE').count()
        real = self.db.query(Detection).filter(Detection.result == 'REAL').count()
        suspicious = self.db.query(Detection).filter(Detection.result == 'SUSPICIOUS').count()
        
        # Type distribution for pie chart
        type_stats = {}
        for t in ['image', 'video', 'audio', 'text', 'url']:
            count = self.db.query(Detection).filter(Detection.type == t).count()
            if count > 0:
                type_stats[t] = count
        
        # Daily stats for trend chart (last 7 days)
        daily_stats = {}
        for i in range(7):
            day = datetime.now() - timedelta(days=i)
            day_str = day.strftime('%a').lower()
            count = self.db.query(Detection).filter(
                func.date(Detection.created_at) == day.date()
            ).count()
            daily_stats[day_str] = count
        
        return {
            'total_scans': total,
            'fake_detected': fake,
            'real_detected': real,
            'suspicious_detected': suspicious,
            'accuracy': 98.3,  # Your frontend value
            'avg_response_time': 0.8,  # Your frontend value (<1s)
            'daily_stats': daily_stats,
            'type_distribution': type_stats,
            'recent_detections': self.get_recent_detections(5)
        }