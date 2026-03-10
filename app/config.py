# app/config.py - Complete Working Version
import os
from typing import List

class Settings:
    # App
    APP_NAME = os.getenv("APP_NAME", "DeepShield AI")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "deepshield-secret-key-2026")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/deepshield_db")
    DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "20"))
    
    # Redis (optional)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # File Upload
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))  # 10MB
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    
    # Models
    MODEL_DIR = os.getenv("MODEL_DIR", "./models")
    ENSEMBLE_MODEL_PATH = os.getenv("ENSEMBLE_MODEL_PATH", "./models/ensemble_model.keras")
    TEXT_MODEL_PATH = os.getenv("TEXT_MODEL_PATH", "./models/text_detection_model.pt")
    
    # CORS
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:8000"]
    
    # 🔴 IMPORTANT: ADD THESE TWO LINES (they were missing)
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "./logs/app.log")

# Create settings instance
settings = Settings()

print("✅ Settings loaded successfully")
print(f"📡 Server: http://{settings.HOST}:{settings.PORT}")
print(f"🗄️  Database: {settings.DATABASE_URL}")
print(f"📊 Pool Size: {settings.DATABASE_POOL_SIZE}")
print(f"📝 Log Level: {settings.LOG_LEVEL}")
print(f"📁 Log File: {settings.LOG_FILE}")
print(f"🌐 CORS: {settings.ALLOWED_ORIGINS}")