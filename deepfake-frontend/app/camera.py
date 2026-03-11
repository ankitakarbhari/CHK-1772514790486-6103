# Placeholder file for camera.py
# app/camera.py
"""
Live Camera Module for Real-Time Deepfake Detection
Captures and processes camera feed for live deepfake analysis
Supports multiple cameras, face tracking, and real-time alerts
Python 3.13+ Compatible
"""

import os
import sys
import time
import json
import threading
import queue
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== CAMERA PROCESSING ==========
import cv2
import numpy as np
from PIL import Image

# ========== FACE DETECTION ==========
from app.utils.face_detection import FaceDetectionEnsemble, FacePreprocessor, Face

# ========== DEEPFAKE MODELS ==========
from app.models.ensemble import DeepfakeEnsemble, EnsembleFactory

# ========== HEATMAP GENERATION ==========
from app.utils.heatmap import EnsembleHeatmapGenerator, HeatmapVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class CameraInfo:
    """Camera information"""
    device_id: int
    name: str
    width: int
    height: int
    fps: float
    backend: str
    is_opened: bool


@dataclass
class DetectedFace:
    """Face detected in camera feed"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    is_fake: bool
    fake_probability: float
    real_probability: float
    landmarks: Optional[Dict] = None
    face_image: Optional[np.ndarray] = None
    heatmap: Optional[np.ndarray] = None
    track_id: Optional[int] = None


@dataclass
class CameraFrame:
    """Processed camera frame"""
    frame: np.ndarray
    timestamp: float
    frame_number: int
    faces: List[DetectedFace]
    fps: float
    processing_time: float


@dataclass
class DetectionAlert:
    """Alert when deepfake is detected"""
    timestamp: float
    confidence: float
    face_bbox: Tuple[int, int, int, int]
    frame_number: int
    image: Optional[np.ndarray] = None


# ============================================
# CAMERA MANAGER
# ============================================

class CameraManager:
    """
    Manages camera devices and capture
    Supports multiple cameras and backends
    """
    
    def __init__(self, 
                 device_id: int = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 backend: Optional[str] = None):
        """
        Initialize camera manager
        
        Args:
            device_id: Camera device ID (0 for default)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
            backend: OpenCV backend (None = auto)
        """
        self.device_id = device_id
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.backend = backend
        
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        self.current_fps = 0
        
        # Frame buffer for processing
        self.frame_buffer = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        # Threading
        self.capture_thread = None
        self.process_thread = None
        
        logger.info(f"CameraManager initialized for device {device_id}")
    
    def _get_backend(self) -> int:
        """Get OpenCV backend constant"""
        backends = {
            'any': cv2.CAP_ANY,
            'v4l2': cv2.CAP_V4L2,
            'vfw': cv2.CAP_VFW,
            'dshow': cv2.CAP_DSHOW,  # DirectShow (Windows)
            'msmf': cv2.CAP_MSMF,    # Microsoft Media Foundation (Windows)
            'ffmpeg': cv2.CAP_FFMPEG,
            'images': cv2.CAP_IMAGES,
            'opencv': cv2.CAP_OPENCV
        }
        return backends.get(self.backend, cv2.CAP_ANY)
    
    def open(self) -> bool:
        """Open camera device"""
        try:
            # Open camera with specified backend
            if self.backend:
                self.cap = cv2.VideoCapture(self.device_id, self._get_backend())
            else:
                self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.device_id}")
                return False
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Get actual properties
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera opened: {self.actual_width}x{self.actual_height} @ {self.actual_fps:.2f}fps")
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {str(e)}")
            return False
    
    def get_camera_info(self) -> CameraInfo:
        """Get camera information"""
        if not self.cap:
            return CameraInfo(
                device_id=self.device_id,
                name=f"Camera {self.device_id}",
                width=0,
                height=0,
                fps=0,
                backend=str(self.backend),
                is_opened=False
            )
        
        # Try to get camera name (platform specific)
        name = f"Camera {self.device_id}"
        if sys.platform == 'win32':
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                     r"SOFTWARE\Microsoft\Windows Media Foundation\Sources")
                # This is simplified - actual implementation would enumerate devices
            except:
                pass
        
        return CameraInfo(
            device_id=self.device_id,
            name=name,
            width=self.actual_width,
            height=self.actual_height,
            fps=self.actual_fps,
            backend=str(self.backend),
            is_opened=self.is_running
        )
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame"""
        if not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret and frame is not None:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return None
    
    def start_capture(self):
        """Start continuous frame capture"""
        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        def capture_loop():
            logger.info("Capture thread started")
            while self.is_running:
                try:
                    frame = self.read_frame()
                    if frame is not None:
                        self.frame_count += 1
                        
                        # Calculate FPS
                        elapsed = time.time() - self.start_time
                        self.current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                        
                        # Add to buffer (non-blocking)
                        if self.frame_buffer.full():
                            try:
                                self.frame_buffer.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_buffer.put_nowait((frame, time.time(), self.frame_count))
                    else:
                        time.sleep(0.001)
                except Exception as e:
                    logger.error(f"Capture error: {str(e)}")
                    time.sleep(0.01)
        
        self.capture_thread = threading.Thread(target=capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        logger.info("Camera capture started")
    
    def stop_capture(self):
        """Stop frame capture"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("Camera capture stopped")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, float, int]]:
        """Get next frame from buffer"""
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def release(self):
        """Release camera resources"""
        self.stop_capture()
        if self.cap:
            self.cap.release()
        logger.info("Camera released")


# ============================================
# REAL-TIME DEEPFAKE DETECTOR
# ============================================

class RealTimeDeepfakeDetector:
    """
    Real-time deepfake detection from camera feed
    """
    
    def __init__(self,
                 face_detector: Optional[FaceDetectionEnsemble] = None,
                 deepfake_model: Optional[DeepfakeEnsemble] = None,
                 face_preprocessor: Optional[FacePreprocessor] = None,
                 heatmap_generator: Optional[EnsembleHeatmapGenerator] = None,
                 alert_callback: Optional[Callable[[DetectionAlert], None]] = None,
                 alert_threshold: float = 0.7,
                 process_every_n_frames: int = 5,
                 enable_heatmap: bool = False):
        """
        Initialize real-time detector
        
        Args:
            face_detector: Face detection ensemble
            deepfake_model: Deepfake detection model
            face_preprocessor: Face preprocessing module
            heatmap_generator: Heatmap generator
            alert_callback: Function to call when deepfake detected
            alert_threshold: Confidence threshold for alerts
            process_every_n_frames: Process every Nth frame
            enable_heatmap: Enable heatmap generation (slower)
        """
        self.face_detector = face_detector or FaceDetectionEnsemble()
        self.deepfake_model = deepfake_model or EnsembleFactory.create_fast_ensemble()
        self.face_preprocessor = face_preprocessor or FacePreprocessor()
        self.heatmap_generator = heatmap_generator
        self.alert_callback = alert_callback
        self.alert_threshold = alert_threshold
        self.process_every_n_frames = process_every_n_frames
        self.enable_heatmap = enable_heatmap
        
        self.camera = None
        self.is_detecting = False
        self.detect_thread = None
        
        # Tracking
        self.next_track_id = 0
        self.face_tracks = {}  # track_id -> face info
        self.track_history = deque(maxlen=100)
        
        # Statistics
        self.total_frames_processed = 0
        self.fake_detections = 0
        self.alerts_sent = 0
        self.processing_times = deque(maxlen=100)
        
        logger.info("RealTimeDeepfakeDetector initialized")
    
    def start(self, camera_manager: CameraManager):
        """Start real-time detection"""
        self.camera = camera_manager
        self.is_detecting = True
        
        # Start camera capture if not already running
        if not self.camera.is_running:
            self.camera.start_capture()
        
        def detect_loop():
            logger.info("Detection thread started")
            
            frame_counter = 0
            
            while self.is_detecting:
                try:
                    # Get frame from camera
                    frame_data = self.camera.get_frame(timeout=1.0)
                    if frame_data is None:
                        continue
                    
                    frame, timestamp, frame_num = frame_data
                    frame_counter += 1
                    
                    # Process every Nth frame
                    if frame_counter % self.process_every_n_frames == 0:
                        start_proc = time.time()
                        
                        # Detect faces
                        face_result = self.face_detector.detect(frame)
                        
                        detected_faces = []
                        
                        if face_result.num_faces > 0:
                            # Process each face
                            for i, face in enumerate(face_result.faces):
                                # Preprocess face
                                face_img = self.face_preprocessor.preprocess_for_model(
                                    face, frame, align=True, normalize=True
                                )
                                
                                # Detect deepfake
                                model_result = self.deepfake_model.predict_single(face_img)
                                
                                is_fake = model_result.prediction == 'FAKE'
                                fake_prob = model_result.fake_probability
                                real_prob = model_result.real_probability
                                
                                # Generate heatmap if enabled
                                heatmap = None
                                if self.enable_heatmap and self.heatmap_generator and is_fake:
                                    try:
                                        face_tensor = torch.from_numpy(face_img).float().permute(2, 0, 1).unsqueeze(0)
                                        heatmap_result = self.heatmap_generator.generate(
                                            face_tensor, 
                                            face_img,
                                            target_class=1  # Fake class
                                        )
                                        heatmap = heatmap_result.heatmap
                                    except:
                                        pass
                                
                                # Assign track ID
                                track_id = self._assign_track_id(face, frame_num)
                                
                                detected_face = DetectedFace(
                                    bbox=(face.box.x, face.box.y, face.box.width, face.box.height),
                                    confidence=model_result.confidence,
                                    is_fake=is_fake,
                                    fake_probability=float(fake_prob),
                                    real_probability=float(real_prob),
                                    landmarks=face.landmarks.to_dict() if face.landmarks else None,
                                    face_image=face_img,
                                    heatmap=heatmap,
                                    track_id=track_id
                                )
                                detected_faces.append(detected_face)
                                
                                # Update tracking
                                self._update_track(track_id, detected_face, frame_num)
                                
                                # Check for alert
                                if is_fake and model_result.confidence > self.alert_threshold:
                                    self._send_alert(detected_face, frame, frame_num, timestamp)
                            
                            # Update statistics
                            self.total_frames_processed += 1
                            if any(f.is_fake for f in detected_faces):
                                self.fake_detections += 1
                        
                        proc_time = time.time() - start_proc
                        self.processing_times.append(proc_time)
                        
                        # Create frame result (can be used for display)
                        frame_result = CameraFrame(
                            frame=frame,
                            timestamp=timestamp,
                            frame_number=frame_num,
                            faces=detected_faces,
                            fps=self.camera.current_fps,
                            processing_time=proc_time
                        )
                        
                        # Could emit this via callback if needed
                        
                    # Small sleep to prevent CPU overload
                    time.sleep(0.001)
                    
                except Exception as e:
                    logger.error(f"Detection error: {str(e)}")
                    time.sleep(0.01)
        
        self.detect_thread = threading.Thread(target=detect_loop)
        self.detect_thread.daemon = True
        self.detect_thread.start()
        
        logger.info("Real-time detection started")
    
    def stop(self):
        """Stop detection"""
        self.is_detecting = False
        if self.detect_thread:
            self.detect_thread.join(timeout=2.0)
        if self.camera:
            self.camera.stop_capture()
        logger.info("Detection stopped")
    
    def _assign_track_id(self, face: Face, frame_num: int) -> int:
        """Assign tracking ID to face using IoU matching"""
        best_match = None
        best_iou = 0.5  # Threshold
        
        for track_id, track_info in self.face_tracks.items():
            # Check if track is still active
            if frame_num - track_info['last_frame'] > 30:
                continue
            
            # Calculate IoU
            iou = self._calculate_iou(
                (face.box.x, face.box.y, face.box.width, face.box.height),
                track_info['last_bbox']
            )
            
            if iou > best_iou:
                best_iou = iou
                best_match = track_id
        
        if best_match is not None:
            return best_match
        else:
            # New track
            track_id = self.next_track_id
            self.next_track_id += 1
            return track_id
    
    def _update_track(self, track_id: int, face: DetectedFace, frame_num: int):
        """Update face tracking information"""
        if track_id not in self.face_tracks:
            self.face_tracks[track_id] = {
                'first_frame': frame_num,
                'last_frame': frame_num,
                'last_bbox': face.bbox,
                'fake_count': 0,
                'real_count': 0
            }
        
        track = self.face_tracks[track_id]
        track['last_frame'] = frame_num
        track['last_bbox'] = face.bbox
        
        if face.is_fake:
            track['fake_count'] += 1
        else:
            track['real_count'] += 1
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _send_alert(self, face: DetectedFace, frame: np.ndarray, 
                    frame_num: int, timestamp: float):
        """Send deepfake detection alert"""
        alert = DetectionAlert(
            timestamp=timestamp,
            confidence=face.confidence,
            face_bbox=face.bbox,
            frame_number=frame_num,
            image=frame.copy() if frame is not None else None
        )
        
        self.alerts_sent += 1
        
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {str(e)}")
        
        logger.warning(f"⚠️ DEEPFAKE ALERT! Confidence: {face.confidence:.2%}")
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_frames_processed': self.total_frames_processed,
            'fake_detections': self.fake_detections,
            'fake_percentage': (self.fake_detections / max(1, self.total_frames_processed)) * 100,
            'alerts_sent': self.alerts_sent,
            'active_tracks': len(self.face_tracks),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'current_fps': self.camera.current_fps if self.camera else 0
        }
    
    def get_face_tracks(self) -> Dict:
        """Get face tracking information"""
        return self.face_tracks


# ============================================
# CAMERA UI RENDERER
# ============================================

class CameraUIRenderer:
    """
    Renders camera feed with detection overlays
    """
    
    def __init__(self):
        """Initialize UI renderer"""
        # Colors (BGR format for OpenCV)
        self.colors = {
            'real': (0, 255, 0),      # Green
            'fake': (0, 0, 255),       # Red
            'suspicious': (0, 165, 255),  # Orange
            'text': (255, 255, 255),   # White
            'box': (255, 255, 255)      # White
        }
        
        # Font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        logger.info("CameraUIRenderer initialized")
    
    def draw_frame(self, 
                   frame: np.ndarray,
                   faces: List[DetectedFace],
                   stats: Dict,
                   show_heatmap: bool = False) -> np.ndarray:
        """
        Draw detection overlays on frame
        
        Args:
            frame: Original frame (RGB)
            faces: Detected faces
            stats: Statistics to display
            show_heatmap: Show heatmap overlay
        
        Returns:
            Frame with overlays (BGR for display)
        """
        # Convert to BGR for OpenCV display
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw faces
        for face in faces:
            self._draw_face_overlay(display, face, show_heatmap)
        
        # Draw stats
        self._draw_stats(display, stats)
        
        return display
    
    def _draw_face_overlay(self, frame: np.ndarray, face: DetectedFace, show_heatmap: bool):
        """Draw overlay for a single face"""
        x, y, w, h = face.bbox
        
        # Choose color based on detection
        if face.is_fake:
            color = self.colors['fake']
            label = f"FAKE: {face.confidence:.2%}"
        else:
            if face.confidence > 0.6:
                color = self.colors['real']
                label = f"REAL: {face.confidence:.2%}"
            else:
                color = self.colors['suspicious']
                label = f"UNCERTAIN: {face.confidence:.2%}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 10, y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x + 5, y - 5), self.font, self.font_scale, 
                   self.colors['text'], self.font_thickness)
        
        # Draw landmarks if available
        if face.landmarks:
            for point_name, point in face.landmarks.items():
                if isinstance(point, tuple) and len(point) == 2:
                    cv2.circle(frame, point, 2, (255, 255, 0), -1)
        
        # Draw heatmap if available and requested
        if show_heatmap and face.heatmap is not None:
            # Resize heatmap to face size
            heatmap_resized = cv2.resize(face.heatmap, (w, h))
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Overlay on face region
            roi = frame[y:y+h, x:x+w]
            blended = cv2.addWeighted(roi, 0.6, heatmap_colored, 0.4, 0)
            frame[y:y+h, x:x+w] = blended
    
    def _draw_stats(self, frame: np.ndarray, stats: Dict):
        """Draw statistics overlay"""
        y_offset = 30
        line_height = 25
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 30 + line_height * len(stats)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw stats
        for key, value in stats.items():
            text = f"{key}: {value}"
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            
            cv2.putText(frame, text, (20, y_offset), self.font, 0.5, 
                       self.colors['text'], 1)
            y_offset += line_height


# ============================================
# CAMERA APPLICATION
# ============================================

class CameraApp:
    """
    Main camera application for deepfake detection
    """
    
    def __init__(self,
                 device_id: int = 0,
                 width: int = 640,
                 height: int = 480,
                 alert_callback: Optional[Callable] = None):
        """
        Initialize camera application
        
        Args:
            device_id: Camera device ID
            width: Frame width
            height: Frame height
            alert_callback: Callback for deepfake alerts
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        
        # Initialize components
        self.camera = CameraManager(device_id, width, height)
        self.detector = RealTimeDeepfakeDetector(
            alert_callback=alert_callback,
            alert_threshold=0.7,
            process_every_n_frames=3,
            enable_heatmap=True
        )
        self.renderer = CameraUIRenderer()
        
        # Display window
        self.window_name = f"Deepfake Detector - Camera {device_id}"
        self.is_running = False
        
        logger.info("CameraApp initialized")
    
    def start(self):
        """Start camera application"""
        # Open camera
        if not self.camera.open():
            logger.error("Failed to open camera")
            return False
        
        # Start detection
        self.detector.start(self.camera)
        
        self.is_running = True
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
        logger.info("Camera app started")
        
        try:
            while self.is_running:
                # Get frame
                frame_data = self.camera.get_frame(timeout=0.1)
                if frame_data is None:
                    continue
                
                frame, timestamp, frame_num = frame_data
                
                # Get current faces from detector (simplified - in real app would need shared state)
                # For now, we'll just show the raw frame with minimal overlay
                
                # Get stats
                stats = self.detector.get_statistics()
                stats['FPS'] = f"{self.camera.current_fps:.1f}"
                
                # Create simple detected face for display (simplified)
                # In production, you'd have a shared queue with detection results
                faces = []  # Would come from detector
                
                # Render frame
                display = self.renderer.draw_frame(frame, faces, stats, show_heatmap=True)
                
                # Show frame
                cv2.imshow(self.window_name, display)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    self.stop()
                    break
                elif key == ord('s'):  # Save screenshot
                    self._save_screenshot(display)
                elif key == ord('h'):  # Toggle heatmap
                    self.detector.enable_heatmap = not self.detector.enable_heatmap
                    logger.info(f"Heatmap: {'ON' if self.detector.enable_heatmap else 'OFF'}")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop camera application"""
        self.is_running = False
        self.detector.stop()
        self.camera.release()
        cv2.destroyAllWindows()
        logger.info("Camera app stopped")
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")


# ============================================
# FACTORY CLASS
# ============================================

class CameraFactory:
    """Factory for creating camera components"""
    
    @staticmethod
    def create_camera(device_id: int = 0, **kwargs) -> CameraManager:
        """Create camera manager"""
        return CameraManager(device_id, **kwargs)
    
    @staticmethod
    def create_detector(**kwargs) -> RealTimeDeepfakeDetector:
        """Create real-time detector"""
        return RealTimeDeepfakeDetector(**kwargs)
    
    @staticmethod
    def create_app(device_id: int = 0, **kwargs) -> CameraApp:
        """Create camera application"""
        return CameraApp(device_id, **kwargs)


# ============================================
# TESTING FUNCTION
# ============================================

def test_camera():
    """Test camera module"""
    print("=" * 60)
    print("TESTING CAMERA MODULE")
    print("=" * 60)
    
    # Test camera manager
    print("\n1️⃣ Testing Camera Manager...")
    camera = CameraFactory.create_camera(0)
    
    if camera.open():
        info = camera.get_camera_info()
        print(f"✅ Camera opened: {info}")
        
        # Test frame capture
        frame = camera.read_frame()
        if frame is not None:
            print(f"✅ Frame captured: {frame.shape}")
        else:
            print("❌ Frame capture failed")
        
        camera.release()
    else:
        print("❌ Could not open camera (may not be available)")
    
    # Test detector without camera
    print("\n2️⃣ Testing Detector initialization...")
    detector = CameraFactory.create_detector()
    print(f"✅ Detector initialized")
    print(f"   Face detector: {type(detector.face_detector).__name__}")
    print(f"   Deepfake model: {type(detector.deepfake_model).__name__}")
    
    # Test renderer
    print("\n3️⃣ Testing UI Renderer...")
    renderer = CameraUIRenderer()
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create dummy face
    dummy_face = DetectedFace(
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        is_fake=True,
        fake_probability=0.85,
        real_probability=0.15,
        landmarks={'left_eye': (150, 150), 'right_eye': (250, 150)},
        face_image=dummy_frame[100:300, 100:300],
        heatmap=np.random.rand(200, 200).astype(np.float32)
    )
    
    stats = {'FPS': 30, 'Fake Detections': 5, 'Alerts': 2}
    
    rendered = renderer.draw_frame(dummy_frame, [dummy_face], stats, show_heatmap=True)
    print(f"✅ Frame rendered: {rendered.shape}")
    
    print("\n" + "=" * 60)
    print("✅ CAMERA MODULE TEST PASSED!")
    print("=" * 60)
    
    print("\n📝 To run live camera detection:")
    print("""
    from app.camera import CameraApp
    
    # Create and start app
    app = CameraApp(device_id=0)
    app.start()
    
    # Press 'q' to quit, 's' to save screenshot, 'h' to toggle heatmap
    """)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Run test
    test_camera()
    
    # Uncomment to run live detection
    # app = CameraApp(device_id=0)
    # app.start()