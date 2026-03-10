# app/main.py
"""
Enhanced FastAPI Backend for Deepfake Detection System
With CORS support for React frontend and WebSocket for real-time detection
"""

import os
import sys
import time
import json
import uuid
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== FASTAPI IMPORTS ==========
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

# ========== ADDITIONAL IMPORTS ==========
import numpy as np
from PIL import Image
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========== IMPORT PROJECT MODULES ==========
from app.models import (
    EnsembleFactory, 
    TextDetectorFactory, 
    initialize_models,
    get_model_versions
)
from app.utils.face_detection import FaceDetectionEnsemble, FacePreprocessor
from app.utils.heatmap import EnsembleHeatmapGenerator, HeatmapVisualizer
from app.utils.video_processor import VideoProcessor
from app.utils.audio_processor import AudioProcessorFactory
from app.utils.blockchain import BlockchainVerifier, MediaHasher, VerificationCertificate
from app.link_analyzer import LinkAnalyzer
from app.camera import CameraApp, RealTimeDeepfakeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs(PROJECT_ROOT / 'uploads', exist_ok=True)
os.makedirs(PROJECT_ROOT / 'logs', exist_ok=True)


# ============================================
# LIFESPAN MANAGER
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for startup and shutdown events
    """
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 DEEPFAKE DETECTION SYSTEM STARTING UP")
    logger.info("=" * 60)
    
    app.state.start_time = time.time()
    app.state.models_loaded = False
    
    try:
        # Initialize models
        logger.info("📦 Loading AI models...")
        
        # Initialize all models
        app.state.models = initialize_models(verbose=True)
        
        # Face detector
        app.state.face_detector = FaceDetectionEnsemble()
        logger.info("✅ Face detector loaded")
        
        # Video processor
        app.state.video_processor = VideoProcessor(
            face_detector=app.state.face_detector,
            deepfake_detector=app.state.models.get('ensemble')
        )
        logger.info("✅ Video processor loaded")
        
        # Audio detector
        app.state.audio_detector = AudioProcessorFactory.create_detector()
        logger.info("✅ Audio detector loaded")
        
        # URL analyzer
        app.state.link_analyzer = LinkAnalyzer(
            text_detector=app.state.models.get('text_detector'),
            image_detector=app.state.models.get('ensemble')
        )
        logger.info("✅ URL analyzer loaded")
        
        # Blockchain verifier
        app.state.blockchain = BlockchainVerifier()
        app.state.hasher = MediaHasher()
        logger.info("✅ Blockchain module loaded")
        
        # Heatmap generator
        if 'ensemble' in app.state.models:
            app.state.heatmap_generator = EnsembleHeatmapGenerator(
                app.state.models['ensemble'].model
            )
            logger.info("✅ Heatmap generator loaded")
        
        app.state.models_loaded = True
        
        # Print versions
        versions = get_model_versions()
        logger.info(f"📚 Library versions: {json.dumps(versions, indent=2)}")
        
        logger.info("✅ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        app.state.models_loaded = False
    
    logger.info(f"✨ System ready! (uptime: {time.time() - app.state.start_time:.2f}s)")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Deepfake Detection System...")
    # Cleanup code here


# ============================================
# FASTAPI APP INITIALIZATION
# ============================================

app = FastAPI(
    title="Deepfake Detection System API",
    description="AI-Based Deepfake Detection and Digital Authenticity Verification System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ========== CORS CONFIGURATION FOR REACT ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # FastAPI server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "https://yourdomain.com",  # Production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted hosts middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"],
)


# ============================================
# API ROUTES
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Deepfake Detection System API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/api/docs",
        "redoc": "/api/redoc"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        "models_loaded": app.state.models_loaded,
        "version": "1.0.0"
    }


@app.get("/api/status")
async def system_status():
    """Get system status with model information"""
    if not app.state.models_loaded:
        return {
            "status": "initializing",
            "models_loaded": False,
            "message": "Models are still loading"
        }
    
    # Get model info
    models_info = {}
    for name, model in app.state.models.items():
        if hasattr(model, 'count_parameters'):
            models_info[name] = {
                "loaded": True,
                "parameters": model.count_parameters()
            }
        else:
            models_info[name] = {"loaded": True}
    
    return {
        "status": "ready",
        "models_loaded": True,
        "models": models_info,
        "uptime": time.time() - app.state.start_time,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# ============================================
# IMAGE DETECTION ENDPOINTS
# ============================================

@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    generate_heatmap: bool = Form(False),
    store_on_blockchain: bool = Form(False)
):
    """
    Detect deepfake in uploaded image
    """
    start_time = time.time()
    
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = PROJECT_ROOT / 'uploads' / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load image
        image = cv2.imread(str(file_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_result = app.state.face_detector.detect(image_rgb)
        
        # Detect deepfake
        if face_result.num_faces > 0:
            # Use largest face
            largest_face = max(face_result.faces, key=lambda f: f.box.area)
            
            # Preprocess face
            face_preprocessor = FacePreprocessor()
            face_img = face_preprocessor.preprocess_for_model(
                largest_face, image_rgb, align=True, normalize=True
            )
            
            # Get prediction
            result = app.state.models['ensemble'].predict_single(face_img)
        else:
            # No face detected - analyze whole image
            result = app.state.models['ensemble'].predict_single(image_rgb)
        
        # Prepare response
        response = {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "result": result.to_dict() if hasattr(result, 'to_dict') else result,
            "faces_detected": face_result.num_faces,
            "face_details": [f.to_dict() for f in face_result.faces] if face_result.num_faces > 0 else [],
            "processing_time": time.time() - start_time
        }
        
        # Generate heatmap if requested
        if generate_heatmap and app.state.heatmap_generator:
            try:
                if face_result.num_faces > 0:
                    face_img_tensor = torch.from_numpy(face_img).float().permute(2, 0, 1).unsqueeze(0)
                    heatmap_result = app.state.heatmap_generator.generate(
                        face_img_tensor, 
                        face_img,
                        target_class=1 if result.prediction == 'FAKE' else 0
                    )
                else:
                    img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    heatmap_result = app.state.heatmap_generator.generate(
                        img_tensor, 
                        image_rgb
                    )
                
                response['heatmap'] = {
                    'overlay': heatmap_result.get_overlay_base64(),
                    'heatmap': heatmap_result.get_heatmap_base64(),
                    'manipulated_regions': heatmap_result.manipulated_regions
                }
            except Exception as e:
                logger.error(f"Heatmap generation error: {str(e)}")
        
        # Store on blockchain if requested
        if store_on_blockchain:
            try:
                # Hash image
                hashes = app.state.hasher.hash_image(str(file_path))
                
                # Store verification
                verification = app.state.blockchain.store_verification(
                    media_path=str(file_path),
                    media_type='image',
                    is_authentic=result.prediction == 'REAL',
                    confidence=result.confidence,
                    metadata=hashes
                )
                
                if verification:
                    response['blockchain'] = verification.to_dict()
            except Exception as e:
                logger.error(f"Blockchain error: {str(e)}")
        
        # Clean up
        os.unlink(file_path)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Image detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/image/batch")
async def detect_images_batch(files: List[UploadFile] = File(...)):
    """
    Detect deepfake in multiple images
    """
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    results = []
    
    for file in files:
        try:
            # Process each image
            result = await detect_image(file)
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {
        "success": True,
        "total": len(results),
        "results": results
    }


# ============================================
# VIDEO DETECTION ENDPOINTS
# ============================================

@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    sample_rate: float = Form(1.0),
    max_frames: int = Form(300),
    analyze_audio: bool = Form(True)
):
    """
    Detect deepfake in uploaded video
    """
    start_time = time.time()
    
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = PROJECT_ROOT / 'uploads' / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze video
        result = app.state.video_processor.analyze_video(
            video_path=str(file_path),
            sample_rate=sample_rate,
            max_frames=max_frames,
            analyze_audio=analyze_audio
        )
        
        # Generate report URL
        report_filename = f"{file_id}_report.html"
        report_path = PROJECT_ROOT / 'static' / 'reports' / report_filename
        report_path.parent.mkdir(exist_ok=True)
        
        app.state.video_processor.generate_video_report(result, str(report_path))
        
        # Clean up video file
        os.unlink(file_path)
        
        response = {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "result": result.to_dict(),
            "report_url": f"/static/reports/{report_filename}",
            "summary": result.summary(),
            "processing_time": time.time() - start_time
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Video detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# AUDIO DETECTION ENDPOINTS
# ============================================

@app.post("/api/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    """
    Detect deepfake in uploaded audio file
    """
    start_time = time.time()
    
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = PROJECT_ROOT / 'uploads' / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze audio
        result = app.state.audio_detector.detect_from_file(str(file_path))
        
        # Clean up
        os.unlink(file_path)
        
        response = {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "result": result.to_dict() if hasattr(result, 'to_dict') else result,
            "processing_time": time.time() - start_time
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Audio detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# TEXT DETECTION ENDPOINTS
# ============================================

@app.post("/api/detect/text")
async def detect_text(
    text: str = Form(...), 
    return_details: bool = Form(True)
):
    """
    Detect if text is AI-generated
    """
    start_time = time.time()
    
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        # Analyze text
        result = app.state.models['text_detector'].detect(text, return_details=return_details)
        
        response = {
            "success": True,
            "text_length": len(text),
            "word_count": len(text.split()),
            "result": result.to_dict() if hasattr(result, 'to_dict') else result,
            "processing_time": time.time() - start_time
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Text detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/text/batch")
async def detect_text_batch(texts: List[str] = Form(...)):
    """
    Detect AI-generated text in multiple texts
    """
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    results = []
    
    for text in texts:
        try:
            result = app.state.models['text_detector'].detect(text)
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "result": result.to_dict() if hasattr(result, 'to_dict') else result
            })
        except Exception as e:
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total": len(results),
        "results": results
    }


# ============================================
# URL ANALYSIS ENDPOINTS
# ============================================

@app.post("/api/analyze/url")
async def analyze_url(
    url: str = Form(...),
    extract_content: bool = Form(True),
    check_ssl: bool = Form(True),
    whois_lookup: bool = Form(True),
    analyze_images: bool = Form(False)
):
    """
    Analyze URL for threats and deepfake content
    """
    start_time = time.time()
    
    try:
        # Analyze URL
        result = app.state.link_analyzer.analyze(
            url=url,
            analyze_content=extract_content,
            check_ssl=check_ssl,
            check_whois=whois_lookup,
            analyze_images=analyze_images
        )
        
        response = {
            "success": True,
            "url": url,
            "result": result.to_dict(),
            "processing_time": time.time() - start_time
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"URL analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# BLOCKCHAIN ENDPOINTS
# ============================================

@app.post("/api/blockchain/verify")
async def blockchain_verify(file: UploadFile = File(...)):
    """
    Verify media against blockchain records
    """
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = PROJECT_ROOT / 'uploads' / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Determine media type
        if file.content_type.startswith('image/'):
            media_type = 'image'
        elif file.content_type.startswith('video/'):
            media_type = 'video'
        elif file.content_type.startswith('audio/'):
            media_type = 'audio'
        else:
            media_type = 'file'
        
        # Verify
        result = app.state.blockchain.verify_media(str(file_path), media_type)
        
        # Clean up
        os.unlink(file_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "result": result.to_dict() if hasattr(result, 'to_dict') else result
        }
        
    except Exception as e:
        logger.error(f"Blockchain verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/blockchain/store")
async def blockchain_store(
    file: UploadFile = File(...),
    is_authentic: bool = Form(True),
    confidence: float = Form(1.0)
):
    """
    Store media verification on blockchain
    """
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = PROJECT_ROOT / 'uploads' / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Determine media type
        if file.content_type.startswith('image/'):
            media_type = 'image'
        elif file.content_type.startswith('video/'):
            media_type = 'video'
        elif file.content_type.startswith('audio/'):
            media_type = 'audio'
        else:
            media_type = 'file'
        
        # Store verification
        verification = app.state.blockchain.store_verification(
            media_path=str(file_path),
            media_type=media_type,
            is_authentic=is_authentic,
            confidence=confidence,
            store_on_ipfs=True
        )
        
        # Generate certificate
        if verification:
            certificate = VerificationCertificate.generate(verification)
            cert_path = PROJECT_ROOT / 'static' / 'certificates' / f"{file_id}_certificate.html"
            cert_path.parent.mkdir(exist_ok=True)
            VerificationCertificate.save_html(certificate, str(cert_path))
        
        # Clean up
        os.unlink(file_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "verification": verification.to_dict() if verification else None,
            "certificate_url": f"/static/certificates/{file_id}_certificate.html" if verification else None
        }
        
    except Exception as e:
        logger.error(f"Blockchain storage error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# WEBSOCKET FOR REAL-TIME DETECTION
# ============================================

@app.websocket("/ws/detect/live")
async def websocket_live_detection(websocket: WebSocket):
    """
    WebSocket endpoint for real-time deepfake detection
    Used by React frontend for live camera/video call detection
    """
    await websocket.accept()
    logger.info("🔌 WebSocket connection accepted")
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Live detection ready",
            "timestamp": time.time()
        })
        
        # Initialize camera
        from app.camera import CameraManager, RealTimeDeepfakeDetector
        
        camera = CameraManager()
        if not camera.open():
            await websocket.send_json({
                "type": "error",
                "message": "Could not open camera"
            })
            await websocket.close()
            return
        
        # Start detection
        detector = RealTimeDeepfakeDetector(
            face_detector=app.state.face_detector,
            deepfake_model=app.state.models.get('ensemble')
        )
        detector.start(camera)
        
        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "message": "Camera initialized, starting detection",
            "timestamp": time.time()
        })
        
        # Continuous detection loop
        frame_count = 0
        while True:
            # Get frame
            frame_data = camera.get_frame(timeout=0.1)
            if frame_data is None:
                continue
            
            frame, timestamp, frame_num = frame_data
            frame_count += 1
            
            # Process every 5th frame
            if frame_count % 5 == 0:
                # Detect faces
                face_result = app.state.face_detector.detect(frame)
                
                if face_result.num_faces > 0:
                    # Use largest face
                    largest_face = max(face_result.faces, key=lambda f: f.box.area)
                    
                    # Preprocess face
                    face_preprocessor = FacePreprocessor()
                    face_img = face_preprocessor.preprocess_for_model(
                        largest_face, frame, align=True, normalize=True
                    )
                    
                    # Detect deepfake
                    result = app.state.models['ensemble'].predict_single(face_img)
                    
                    # Send result
                    await websocket.send_json({
                        "type": "detection",
                        "timestamp": timestamp,
                        "frame": frame_num,
                        "faces_detected": face_result.num_faces,
                        "result": result.to_dict() if hasattr(result, 'to_dict') else result,
                        "face_box": largest_face.box.to_dict()
                    })
            
            # Check for client messages
            try:
                message = await websocket.receive_text()
                if message == "stop":
                    logger.info("Stop signal received")
                    break
                elif message == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except:
                pass
            
            # Small delay
            await asyncio.sleep(0.01)
        
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # Cleanup
        if 'detector' in locals():
            detector.stop()
        if 'camera' in locals():
            camera.release()


# ============================================
# STATIC FILES AND REPORTS
# ============================================

# Mount static files directory
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")


@app.get("/api/reports/{file_id}")
async def get_report(file_id: str):
    """
    Get generated report file
    """
    report_path = PROJECT_ROOT / 'static' / 'reports' / f"{file_id}_report.html"
    if report_path.exists():
        return FileResponse(report_path)
    raise HTTPException(status_code=404, detail="Report not found")


@app.get("/api/certificates/{file_id}")
async def get_certificate(file_id: str):
    """
    Get blockchain certificate
    """
    cert_path = PROJECT_ROOT / 'static' / 'certificates' / f"{file_id}_certificate.html"
    if cert_path.exists():
        return FileResponse(cert_path)
    raise HTTPException(status_code=404, detail="Certificate not found")


# ============================================
# DETECTION HISTORY ENDPOINTS
# ============================================

@app.get("/api/detections/recent")
async def get_recent_detections(limit: int = 10):
    """
    Get recent detections (mock data for frontend)
    """
    # Mock data - replace with database in production
    mock_detections = [
        {
            "id": "DET-001",
            "type": "Image",
            "filename": "profile_photo.jpg",
            "result": "Real",
            "confidence": 98.5,
            "date": "2024-01-15"
        },
        {
            "id": "DET-002",
            "type": "Video",
            "filename": "interview.mp4",
            "result": "Fake",
            "confidence": 94.2,
            "date": "2024-01-14"
        },
        {
            "id": "DET-003",
            "type": "Audio",
            "filename": "voice_message.wav",
            "result": "Fake",
            "confidence": 87.3,
            "date": "2024-01-14"
        },
        {
            "id": "DET-004",
            "type": "Text",
            "filename": "article.txt",
            "result": "Real",
            "confidence": 96.8,
            "date": "2024-01-13"
        },
        {
            "id": "DET-005",
            "type": "URL",
            "filename": "example.com",
            "result": "Suspicious",
            "confidence": 76.5,
            "date": "2024-01-13"
        }
    ]
    
    return mock_detections[:limit]


@app.get("/api/stats")
async def get_statistics():
    """
    Get system statistics (mock data for frontend)
    """
    return {
        "totalScans": 1247,
        "deepfakes": 324,
        "authentic": 923,
        "avgResponse": 0.8,
        "dailyStats": {
            "mon": 45,
            "tue": 52,
            "wed": 48,
            "thu": 51,
            "fri": 49,
            "sat": 53,
            "sun": 47
        },
        "typeDistribution": {
            "images": 45,
            "videos": 25,
            "audio": 15,
            "text": 10,
            "urls": 5
        }
    }


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    """
    Run the FastAPI application
    """
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting server on http://{host}:{port}")
    logger.info(f"📚 API docs available at http://{host}:{port}/api/docs")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
        workers=1
    )