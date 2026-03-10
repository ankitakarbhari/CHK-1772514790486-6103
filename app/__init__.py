# app/__init__.py
"""
DeepShield AI - Deepfake Detection System
Main application package initializer
"""

import os
import logging
from pathlib import Path

# Set Keras backend to PyTorch (for Python 3.14 compatibility)
os.environ["KERAS_BACKEND"] = "torch"

# Package metadata
__version__ = "1.0.0"
__author__ = "DeepShield AI"
__description__ = "AI-Powered Deepfake Detection System"

# Configure logging for the entire app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create necessary directories
def create_directories():
    """Create required directories if they don't exist"""
    dirs = [
        Path("uploads"),
        Path("uploads/images"),
        Path("uploads/videos"),
        Path("uploads/audio"),
        Path("logs"),
        Path("static"),
        Path("static/heatmaps"),
        Path("static/reports"),
        Path("static/certificates"),
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Required directories created")

# Run directory creation on import
create_directories()

# What gets exported when someone does "from app import *"
__all__ = [
    '__version__',
    '__author__',
    '__description__',
]