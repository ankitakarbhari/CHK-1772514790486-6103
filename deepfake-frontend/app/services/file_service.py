# app/services/file_service.py
import os
import uuid
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self):
        self.upload_dir = Path("./uploads")
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Create necessary directories"""
        dirs = [
            self.upload_dir / 'images',
            self.upload_dir / 'videos',
            self.upload_dir / 'audio',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_file(self, file_data: bytes, filename: str, file_type: str) -> str:
        """Save uploaded file"""
        file_id = str(uuid.uuid4())
        ext = Path(filename).suffix
        safe_name = f"{file_id}{ext}"
        
        file_path = self.upload_dir / file_type / safe_name
        file_path.write_bytes(file_data)
        
        logger.info(f"✅ File saved: {file_path}")
        return str(file_path)
    
    def delete_file(self, file_path: str):
        """Delete temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")