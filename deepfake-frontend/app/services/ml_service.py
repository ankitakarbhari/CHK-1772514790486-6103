# app/services/ml_service.py
"""
ML Service for DeepShield AI
Simplified version that works immediately
"""

import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MLService:
    """ML Service that works without dependencies"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        logger.info("✅ ML Service initialized (working mode)")
    
    def detect_image(self, image_path: str, generate_heatmap: bool = False) -> Dict[str, Any]:
        """Detect deepfake in image"""
        logger.info(f"🔍 Analyzing image: {image_path}")
        
        # Return realistic mock data
        return {
            'success': True,
            'faces_detected': 1,
            'faces': [{
                'bbox': {'x': 100, 'y': 100, 'width': 200, 'height': 200},
                'confidence': 0.98,
                'is_fake': False,
                'fake_probability': 0.02,
                'real_probability': 0.98
            }],
            'result': {
                'is_fake': False,
                'fake_probability': 0.02,
                'real_probability': 0.98,
                'confidence': 0.98
            },
            'processing_time': 0.45,
            'message': '✅ Image analyzed successfully (mock mode)'
        }
    
    def detect_text(self, text: str) -> Dict[str, Any]:
        """Detect AI-generated text"""
        logger.info(f"🔍 Analyzing text: {text[:50]}...")
        
        word_count = len(text.split())
        
        return {
            'success': True,
            'is_ai_generated': False,
            'ai_probability': 0.12,
            'human_probability': 0.88,
            'confidence': 0.88,
            'word_count': word_count,
            'sentence_count': max(1, word_count // 10),
            'readability_score': 65.5,
            'perplexity': 45.2,
            'burstiness': 0.35,
            'suspicious_segments': [],
            'processing_time': 0.32,
            'message': '✅ Text analyzed successfully (mock mode)'
        }
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze URL for threats"""
        logger.info(f"🔍 Analyzing URL: {url}")
        
        return {
            'success': True,
            'url': url,
            'domain': url.split('/')[2] if '://' in url else url,
            'risk_score': 15.5,
            'risk_level': 'LOW',
            'is_phishing': False,
            'is_malware': False,
            'is_scam': False,
            'threat_types': [],
            'warnings': [],
            'page_title': 'Sample Page',
            'word_count': 250,
            'processing_time': 0.67,
            'message': '✅ URL analyzed successfully (mock mode)'
        }

# Create singleton instance
ml_service = MLService()