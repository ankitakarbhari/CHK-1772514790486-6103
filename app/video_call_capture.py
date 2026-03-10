# app/video_call_capture.py
"""
Video Call Capture Module for Live Deepfake Detection
Captures and analyzes video from Zoom, Google Meet, Microsoft Teams, and other platforms
Supports real-time deepfake detection during live video calls
Python 3.13+ Compatible
"""

import os
import sys
import time
import json
import base64
import logging
import threading
import queue
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== COMPUTER VISION ==========
import cv2
import numpy as np
from PIL import Image
import mss
import pyautogui
import pygetwindow as gw

# ========== FACE DETECTION ==========
from app.utils.face_detection import FaceDetectionEnsemble, FacePreprocessor, Face, FaceDetectionResult

# ========== DEEPFAKE MODELS ==========
from app.models.ensemble import DeepfakeEnsemble, EnsembleFactory, EnsembleResult
from app.utils.heatmap import EnsembleHeatmapGenerator, HeatmapVisualizer, HeatmapResult

# ========== AUDIO PROCESSING ==========
from app.utils.audio_processor import AudioCapture, AudioDeepfakeDetector, AudioProcessorFactory, VoiceActivityDetector

# ========== UI AUTOMATION ==========
try:
    import uiautomation as auto  # Windows only
    UIAUTOMATION_AVAILABLE = True
except ImportError:
    UIAUTOMATION_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class CallPlatform:
    """Video call platform information"""
    name: str  # 'zoom', 'meet', 'teams', 'webex', 'skype', 'slack', 'discord', 'whatsapp', 'facetime'
    display_name: str
    window_title_patterns: List[str]
    process_names: List[str]
    url_patterns: List[str]
    participant_selector: Optional[str] = None
    video_element_selector: Optional[str] = None
    recording_indicator: Optional[str] = None
    mute_indicator: Optional[str] = None
    icon_path: Optional[str] = None
    
    def matches_window(self, window_title: str) -> bool:
        """Check if window title matches platform"""
        window_title_lower = window_title.lower()
        for pattern in self.window_title_patterns:
            if pattern.lower() in window_title_lower:
                return True
        return False
    
    def matches_process(self, process_name: str) -> bool:
        """Check if process name matches platform"""
        process_name_lower = process_name.lower()
        for pattern in self.process_names:
            if pattern.lower() in process_name_lower:
                return True
        return False
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'display_name': self.display_name
        }


@dataclass
class CallWindow:
    """Detected call window information"""
    platform: str
    platform_display: str
    window_title: str
    window_handle: int
    window_rect: Tuple[int, int, int, int]  # x, y, width, height
    is_active: bool
    is_minimized: bool
    process_name: str
    process_id: int
    detected_at: float
    
    def to_dict(self) -> Dict:
        return {
            'platform': self.platform,
            'platform_display': self.platform_display,
            'title': self.window_title,
            'rect': self.window_rect,
            'is_active': self.is_active,
            'process_name': self.process_name
        }


@dataclass
class CallParticipant:
    """Participant in a video call"""
    participant_id: str
    name: Optional[str]
    face_box: Optional[Tuple[int, int, int, int]]
    face_confidence: float
    is_speaking: bool
    is_muted: bool
    is_video_on: bool
    deepfake_probability: float
    is_deepfake: bool
    confidence: float
    track_id: int
    last_seen: float
    screen_share: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'participant_id': self.participant_id,
            'name': self.name,
            'face_box': self.face_box,
            'is_speaking': self.is_speaking,
            'is_muted': self.is_muted,
            'is_video_on': self.is_video_on,
            'is_deepfake': self.is_deepfake,
            'confidence': self.confidence,
            'track_id': self.track_id,
            'screen_share': self.screen_share
        }


@dataclass
class CallFrame:
    """Captured frame from video call"""
    timestamp: float
    frame_number: int
    frame: np.ndarray
    frame_small: Optional[np.ndarray]  # Resized version for storage/display
    window_rect: Tuple[int, int, int, int]
    participants: List[CallParticipant]
    platform: str
    platform_display: str
    processing_time: float
    has_screen_share: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'frame_number': self.frame_number,
            'window_rect': self.window_rect,
            'participants': [p.to_dict() for p in self.participants],
            'platform': self.platform,
            'platform_display': self.platform_display,
            'processing_time': self.processing_time,
            'has_screen_share': self.has_screen_share
        }


@dataclass
class CallAlert:
    """Alert when deepfake detected in call"""
    timestamp: float
    platform: str
    platform_display: str
    participant_name: Optional[str]
    confidence: float
    frame: Optional[np.ndarray]
    face_bbox: Optional[Tuple[int, int, int, int]]
    alert_id: str
    processed: bool = False
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'platform': self.platform,
            'platform_display': self.platform_display,
            'participant_name': self.participant_name,
            'confidence': self.confidence,
            'alert_id': self.alert_id,
            'processed': self.processed,
            'acknowledged': self.acknowledged
        }


@dataclass
class CallStatistics:
    """Statistics for a video call session"""
    session_id: str
    platform: str
    start_time: float
    end_time: Optional[float]
    duration: float
    frames_captured: int
    frames_processed: int
    participants_detected: int
    deepfake_detections: int
    alerts_generated: int
    avg_processing_time: float
    peak_participants: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================
# PLATFORM DETECTOR
# ============================================

class PlatformDetector:
    """
    Detect which video call platform is being used
    """
    
    def __init__(self):
        """Initialize platform detector"""
        self.platforms = self._define_platforms()
        self.active_calls = {}
        logger.info(f"PlatformDetector initialized with {len(self.platforms)} platforms")
    
    def _define_platforms(self) -> List[CallPlatform]:
        """Define supported platforms"""
        return [
            CallPlatform(
                name="zoom",
                display_name="Zoom",
                window_title_patterns=["zoom", "zoom meeting", "zoom cloud meetings", "zoom conference"],
                process_names=["zoom.exe", "Zoom", "zoom.us"],
                url_patterns=["zoom.us", "zoom.com"],
                participant_selector=".participant-item",
                video_element_selector=".video-container",
                recording_indicator=".recording-indicator"
            ),
            CallPlatform(
                name="google_meet",
                display_name="Google Meet",
                window_title_patterns=["google meet", "meet.google.com", "meet -"],
                process_names=["chrome.exe", "msedge.exe", "brave.exe", "firefox.exe"],
                url_patterns=["meet.google.com"],
                participant_selector="[role='button'][aria-label*='participant']",
                video_element_selector="[jsname='jQJj8e']",
                recording_indicator="[aria-label*='recording']"
            ),
            CallPlatform(
                name="microsoft_teams",
                display_name="Microsoft Teams",
                window_title_patterns=["microsoft teams", "teams", "teams.ms", "microsoft teams meeting"],
                process_names=["Teams.exe", "ms-teams.exe"],
                url_patterns=["teams.microsoft.com", "teams.live.com"],
                participant_selector=".participant-name",
                video_element_selector=".video-main",
                recording_indicator="[class*='recording']"
            ),
            CallPlatform(
                name="webex",
                display_name="Webex",
                window_title_patterns=["webex", "webex meeting", "cisco webex"],
                process_names=["Webex.exe", "CiscoWebex.exe"],
                url_patterns=["webex.com"],
                participant_selector=".participant-name",
                video_element_selector=".video-container",
                recording_indicator=".recording-badge"
            ),
            CallPlatform(
                name="skype",
                display_name="Skype",
                window_title_patterns=["skype", "skype call", "skype meeting"],
                process_names=["Skype.exe", "SkypeApp.exe"],
                url_patterns=["skype.com"],
                participant_selector=".participant-name",
                video_element_selector=".video-stream",
                recording_indicator=".recording-indicator"
            ),
            CallPlatform(
                name="slack",
                display_name="Slack",
                window_title_patterns=["slack", "slack call", "slack huddle"],
                process_names=["Slack.exe"],
                url_patterns=["slack.com"],
                participant_selector=".c-call-participant",
                video_element_selector=".c-video-stream",
                recording_indicator=None
            ),
            CallPlatform(
                name="discord",
                display_name="Discord",
                window_title_patterns=["discord", "discord call", "discord voice", "discord - "],
                process_names=["Discord.exe", "DiscordPTB.exe"],
                url_patterns=["discord.com"],
                participant_selector=".voice-state",
                video_element_selector=".video-wrapper",
                recording_indicator=".recording-indicator"
            ),
            CallPlatform(
                name="whatsapp",
                display_name="WhatsApp",
                window_title_patterns=["whatsapp", "whatsapp call"],
                process_names=["WhatsApp.exe"],
                url_patterns=["web.whatsapp.com"],
                participant_selector=".participant",
                video_element_selector=".video-container",
                recording_indicator=None
            ),
            CallPlatform(
                name="facetime",
                display_name="FaceTime",
                window_title_patterns=["facetime", "face time"],
                process_names=["FaceTime.exe", "FaceTime"],
                url_patterns=[],
                participant_selector=None,
                video_element_selector=None,
                recording_indicator=None
            ),
            CallPlatform(
                name="telegram",
                display_name="Telegram",
                window_title_patterns=["telegram", "telegram call"],
                process_names=["Telegram.exe"],
                url_patterns=["telegram.org"],
                participant_selector=None,
                video_element_selector=None,
                recording_indicator=None
            ),
            CallPlatform(
                name="unknown",
                display_name="Unknown",
                window_title_patterns=[],
                process_names=[],
                url_patterns=[]
            )
        ]
    
    def detect_from_window_title(self, window_title: str) -> Optional[CallPlatform]:
        """Detect platform from window title"""
        window_title_lower = window_title.lower()
        for platform in self.platforms:
            for pattern in platform.window_title_patterns:
                if pattern.lower() in window_title_lower:
                    return platform
        return None
    
    def detect_from_process(self, process_name: str) -> Optional[CallPlatform]:
        """Detect platform from process name"""
        process_name_lower = process_name.lower()
        for platform in self.platforms:
            for pattern in platform.process_names:
                if pattern.lower() in process_name_lower:
                    return platform
        return None
    
    def detect_from_url(self, url: str) -> Optional[CallPlatform]:
        """Detect platform from URL"""
        url_lower = url.lower()
        for platform in self.platforms:
            for pattern in platform.url_patterns:
                if pattern in url_lower:
                    return platform
        return None
    
    def get_platform_by_name(self, name: str) -> Optional[CallPlatform]:
        """Get platform by name"""
        for platform in self.platforms:
            if platform.name == name:
                return platform
        return None
    
    def get_all_platforms(self) -> List[Dict]:
        """Get list of all platforms"""
        return [p.to_dict() for p in self.platforms if p.name != "unknown"]


# ============================================
# WINDOW CAPTURE
# ============================================

class WindowCapture:
    """
    Capture specific windows for video call analysis
    """
    
    def __init__(self, 
                 capture_rate: int = 5,  # frames per second
                 target_size: Tuple[int, int] = (640, 480),
                 buffer_size: int = 30):
        """
        Initialize window capture
        
        Args:
            capture_rate: Frames per second to capture
            target_size: Target size for captured frames
            buffer_size: Frame buffer size
        """
        self.capture_rate = capture_rate
        self.target_size = target_size
        self.buffer_size = buffer_size
        self.capture_interval = 1.0 / capture_rate
        
        self.is_capturing = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.window_info = None
        self.frame_count = 0
        self.dropped_frames = 0
        
        # Initialize screen capture
        self.sct = mss.mss()
        
        logger.info(f"WindowCapture initialized at {capture_rate} FPS")
    
    def find_call_windows(self) -> List[CallWindow]:
        """
        Find all potential video call windows
        
        Returns:
            List of CallWindow objects
        """
        call_windows = []
        platform_detector = PlatformDetector()
        
        # Get all windows
        windows = gw.getAllWindows()
        
        for window in windows:
            if not window.title or window.title.strip() == "":
                continue
            
            # Skip small windows (likely not call windows)
            if window.width < 300 or window.height < 200:
                continue
            
            # Try to detect platform
            platform = platform_detector.detect_from_window_title(window.title)
            
            if platform and platform.name != "unknown":
                # Get window rectangle
                rect = (window.left, window.top, window.width, window.height)
                
                # Get process info
                process_name = "unknown"
                process_id = 0
                
                if PSUTIL_AVAILABLE:
                    try:
                        # On Windows, get process ID from window handle
                        if hasattr(window, '_hWnd'):
                            import ctypes
                            user32 = ctypes.windll.user32
                            process_id = ctypes.c_ulong()
                            user32.GetWindowThreadProcessId(window._hWnd, ctypes.byref(process_id))
                            process_id = process_id.value
                            
                            if process_id:
                                process = psutil.Process(process_id)
                                process_name = process.name()
                    except:
                        pass
                
                call_window = CallWindow(
                    platform=platform.name,
                    platform_display=platform.display_name,
                    window_title=window.title,
                    window_handle=window._hWnd if hasattr(window, '_hWnd') else 0,
                    window_rect=rect,
                    is_active=window.isActive,
                    is_minimized=window.isMinimized,
                    process_name=process_name,
                    process_id=process_id,
                    detected_at=time.time()
                )
                
                call_windows.append(call_window)
        
        # Sort by active first, then by size
        call_windows.sort(key=lambda w: (not w.is_active, -w.window_rect[2] * w.window_rect[3]))
        
        logger.info(f"Found {len(call_windows)} call windows")
        return call_windows
    
    def capture_window(self, window: CallWindow) -> Optional[np.ndarray]:
        """
        Capture a specific window
        
        Args:
            window: CallWindow object
        
        Returns:
            Captured frame as numpy array (RGB)
        """
        try:
            x, y, w, h = window.window_rect
            
            # Ensure positive dimensions
            if w <= 0 or h <= 0:
                return None
            
            # Capture screen region
            monitor = {"top": y, "left": x, "width": w, "height": h}
            screenshot = self.sct.grab(monitor)
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert BGRA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            # Resize if needed
            if self.target_size:
                frame = cv2.resize(frame, self.target_size)
            
            return frame
            
        except Exception as e:
            logger.error(f"Window capture error: {str(e)}")
            return None
    
    def start_capturing(self, window: CallWindow):
        """
        Start continuous window capture
        
        Args:
            window: CallWindow to capture
        """
        if self.is_capturing:
            logger.warning("Already capturing")
            return
        
        self.window_info = window
        self.is_capturing = True
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = time.time()
        
        def capture_loop():
            logger.info(f"Started capturing window: {window.window_title}")
            
            while self.is_capturing:
                loop_start = time.time()
                
                try:
                    # Capture frame
                    frame = self.capture_window(window)
                    
                    if frame is not None:
                        self.frame_count += 1
                        
                        # Create small version for storage
                        frame_small = cv2.resize(frame, (320, 240))
                        
                        # Create frame object
                        call_frame = CallFrame(
                            timestamp=time.time(),
                            frame_number=self.frame_count,
                            frame=frame,
                            frame_small=frame_small,
                            window_rect=window.window_rect,
                            participants=[],
                            platform=window.platform,
                            platform_display=window.platform_display,
                            processing_time=0.0
                        )
                        
                        # Add to queue (non-blocking)
                        try:
                            self.frame_queue.put_nowait(call_frame)
                        except queue.Full:
                            self.dropped_frames += 1
                            # Remove oldest frame
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(call_frame)
                            except:
                                pass
                    
                    # Maintain capture rate
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.capture_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
                except Exception as e:
                    logger.error(f"Capture loop error: {str(e)}")
                    time.sleep(0.1)
            
            logger.info(f"Window capture stopped. Frames: {self.frame_count}, Dropped: {self.dropped_frames}")
        
        self.capture_thread = threading.Thread(target=capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def stop_capturing(self):
        """Stop window capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        
        logger.info("Window capture stopped")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[CallFrame]:
        """Get next captured frame"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict:
        """Get capture statistics"""
        return {
            'capture_rate': self.capture_rate,
            'frames_captured': self.frame_count,
            'frames_dropped': self.dropped_frames,
            'queue_size': self.frame_queue.qsize(),
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }


# ============================================
# VIDEO CALL ANALYZER
# ============================================

class VideoCallAnalyzer:
    """
    Analyze video calls for deepfake detection
    """
    
    def __init__(self,
                 face_detector: Optional[FaceDetectionEnsemble] = None,
                 deepfake_model: Optional[DeepfakeEnsemble] = None,
                 face_preprocessor: Optional[FacePreprocessor] = None,
                 audio_detector: Optional[AudioDeepfakeDetector] = None,
                 heatmap_generator: Optional[EnsembleHeatmapGenerator] = None,
                 alert_callback: Optional[Callable[[CallAlert], None]] = None,
                 alert_threshold: float = 0.7,
                 process_every_n_frames: int = 5,
                 enable_audio: bool = True,
                 enable_heatmap: bool = False,
                 min_face_size: int = 50):
        """
        Initialize video call analyzer
        
        Args:
            face_detector: Face detection ensemble
            deepfake_model: Deepfake detection model
            face_preprocessor: Face preprocessing module
            audio_detector: Audio deepfake detector
            heatmap_generator: Heatmap generator
            alert_callback: Callback for deepfake alerts
            alert_threshold: Confidence threshold for alerts
            process_every_n_frames: Process every Nth frame
            enable_audio: Enable audio analysis
            enable_heatmap: Enable heatmap generation
            min_face_size: Minimum face size to consider
        """
        self.face_detector = face_detector or FaceDetectionEnsemble()
        self.deepfake_model = deepfake_model or EnsembleFactory.create_fast_ensemble()
        self.face_preprocessor = face_preprocessor or FacePreprocessor()
        self.audio_detector = audio_detector
        self.heatmap_generator = heatmap_generator
        self.alert_callback = alert_callback
        self.alert_threshold = alert_threshold
        self.process_every_n_frames = process_every_n_frames
        self.enable_audio = enable_audio
        self.enable_heatmap = enable_heatmap
        self.min_face_size = min_face_size
        
        # Tracking
        self.next_track_id = 0
        self.participant_tracks = {}
        self.frame_count = 0
        self.processed_count = 0
        self.alert_count = 0
        
        # Statistics
        self.total_frames_processed = 0
        self.deepfake_detections = 0
        self.alerts_sent = 0
        self.processing_times = deque(maxlen=100)
        
        # Audio capture (if enabled)
        self.audio_capture = None
        self.audio_queue = queue.Queue()
        self.vad = VoiceActivityDetector()
        
        if enable_audio:
            try:
                self.audio_capture = AudioCapture()
                logger.info("Audio capture initialized")
            except Exception as e:
                logger.error(f"Failed to initialize audio capture: {str(e)}")
        
        logger.info("VideoCallAnalyzer initialized")
    
    def analyze_frame(self, call_frame: CallFrame) -> CallFrame:
        """
        Analyze a single frame from video call
        
        Args:
            call_frame: Frame to analyze
        
        Returns:
            Updated CallFrame with analysis results
        """
        process_start = time.time()
        self.frame_count += 1
        
        # Process every Nth frame
        if self.frame_count % self.process_every_n_frames != 0:
            return call_frame
        
        self.processed_count += 1
        
        frame = call_frame.frame
        
        # Detect faces
        face_result = self.face_detector.detect(frame)
        
        participants = []
        screen_share_detected = False
        
        for face in face_result.faces:
            # Skip small faces
            if face.box.area < self.min_face_size * self.min_face_size:
                continue
            
            # Preprocess face
            face_img = self.face_preprocessor.preprocess_for_model(
                face, frame, align=True, normalize=True
            )
            
            # Detect deepfake
            model_result = self.deepfake_model.predict_single(face_img)
            
            is_fake = model_result.prediction == 'FAKE'
            confidence = model_result.confidence
            fake_prob = model_result.fake_probability
            
            # Assign track ID
            track_id = self._assign_track_id(face, call_frame.frame_number)
            
            # Check if this might be a screen share (large face area)
            is_screen_share = face.box.area > frame.shape[0] * frame.shape[1] * 0.3
            
            # Create participant
            participant = CallParticipant(
                participant_id=f"participant_{track_id}",
                name=None,  # Would need OCR or metadata
                face_box=(face.box.x, face.box.y, face.box.width, face.box.height),
                face_confidence=face.confidence,
                is_speaking=False,  # Would need audio sync
                is_muted=False,
                is_video_on=True,
                deepfake_probability=float(fake_prob),
                is_deepfake=is_fake,
                confidence=float(confidence),
                track_id=track_id,
                last_seen=call_frame.timestamp,
                screen_share=is_screen_share
            )
            
            participants.append(participant)
            screen_share_detected = screen_share_detected or is_screen_share
            
            # Update tracking
            self._update_track(track_id, participant, call_frame.frame_number)
            
            # Check for alert
            if is_fake and confidence > self.alert_threshold:
                self._send_alert(participant, call_frame)
                self.deepfake_detections += 1
        
        # Generate heatmap for the first fake face if enabled
        if self.enable_heatmap and participants and any(p.is_deepfake for p in participants):
            fake_participant = next(p for p in participants if p.is_deepfake)
            if fake_participant and self.heatmap_generator:
                try:
                    # Extract face region
                    x, y, w, h = fake_participant.face_box
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Generate heatmap
                    heatmap_result = self.heatmap_generator.generate(face_region)
                    # Store heatmap reference (would need to add to participant)
                except:
                    pass
        
        call_frame.participants = participants
        call_frame.processing_time = time.time() - process_start
        call_frame.has_screen_share = screen_share_detected
        
        self.processing_times.append(call_frame.processing_time)
        self.total_frames_processed += 1
        
        return call_frame
    
    def _assign_track_id(self, face: Face, frame_num: int) -> int:
        """Assign tracking ID to face"""
        best_match = None
        best_iou = 0.5  # Threshold
        
        for track_id, track_info in self.participant_tracks.items():
            # Check if track is still active
            if frame_num - track_info['last_frame'] > 100:
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
            track_id = self.next_track_id
            self.next_track_id += 1
            return track_id
    
    def _update_track(self, track_id: int, participant: CallParticipant, frame_num: int):
        """Update participant tracking"""
        if track_id not in self.participant_tracks:
            self.participant_tracks[track_id] = {
                'first_frame': frame_num,
                'last_frame': frame_num,
                'last_bbox': participant.face_box,
                'deepfake_count': 0,
                'real_count': 0,
                'appearances': 1,
                'first_seen': participant.last_seen,
                'last_seen': participant.last_seen
            }
        
        track = self.participant_tracks[track_id]
        track['last_frame'] = frame_num
        track['last_bbox'] = participant.face_box
        track['appearances'] += 1
        track['last_seen'] = participant.last_seen
        
        if participant.is_deepfake:
            track['deepfake_count'] += 1
        else:
            track['real_count'] += 1
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
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
    
    def _send_alert(self, participant: CallParticipant, call_frame: CallFrame):
        """Send deepfake detection alert"""
        alert_id = f"alert_{int(time.time())}_{participant.track_id}_{self.alert_count}"
        self.alert_count += 1
        
        # Extract face region for alert
        face_img = None
        if participant.face_box:
            x, y, w, h = participant.face_box
            face_img = call_frame.frame[y:y+h, x:x+w].copy()
        
        alert = CallAlert(
            timestamp=time.time(),
            platform=call_frame.platform,
            platform_display=call_frame.platform_display,
            participant_name=participant.name,
            confidence=participant.confidence,
            frame=face_img,
            face_bbox=participant.face_box,
            alert_id=alert_id
        )
        
        self.alerts_sent += 1
        
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {str(e)}")
        
        logger.warning(f"⚠️ DEEPFAKE ALERT in {call_frame.platform_display}! "
                      f"Confidence: {participant.confidence:.2%}")
    
    def process_audio(self, audio_chunk: np.ndarray, timestamp: float):
        """Process audio chunk (to be called from audio thread)"""
        if not self.enable_audio or not self.audio_detector:
            return
        
        # Detect speech
        is_speech = self.vad.is_speech(audio_chunk)
        
        if is_speech:
            # Add to audio queue for processing
            self.audio_queue.put((audio_chunk, timestamp))
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics"""
        return {
            'total_frames': self.frame_count,
            'frames_processed': self.processed_count,
            'deepfake_detections': self.deepfake_detections,
            'alerts_sent': self.alerts_sent,
            'active_tracks': len(self.participant_tracks),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'processing_fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0
        }
    
    def get_participant_summary(self) -> List[Dict]:
        """Get summary of all tracked participants"""
        summary = []
        for track_id, track in self.participant_tracks.items():
            total = track['deepfake_count'] + track['real_count']
            fake_ratio = track['deepfake_count'] / total if total > 0 else 0
            
            summary.append({
                'track_id': track_id,
                'first_seen': track['first_seen'],
                'last_seen': track['last_seen'],
                'appearances': track['appearances'],
                'deepfake_count': track['deepfake_count'],
                'real_count': track['real_count'],
                'fake_ratio': fake_ratio,
                'is_suspicious': fake_ratio > 0.3
            })
        
        return summary
    
    def reset(self):
        """Reset analyzer state"""
        self.next_track_id = 0
        self.participant_tracks = {}
        self.frame_count = 0
        self.processed_count = 0
        self.alert_count = 0
        self.total_frames_processed = 0
        self.deepfake_detections = 0
        self.alerts_sent = 0
        self.processing_times.clear()


# ============================================
# VIDEO CALL MONITOR
# ============================================

class VideoCallMonitor:
    """
    Main video call monitoring system
    """
    
    def __init__(self,
                 analyzer: Optional[VideoCallAnalyzer] = None,
                 platform_detector: Optional[PlatformDetector] = None,
                 alert_callback: Optional[Callable[[CallAlert], None]] = None,
                 auto_select: bool = True,
                 save_snapshots: bool = False,
                 snapshot_dir: str = "call_snapshots"):
        """
        Initialize video call monitor
        
        Args:
            analyzer: VideoCallAnalyzer instance
            platform_detector: PlatformDetector instance
            alert_callback: Callback for alerts
            auto_select: Automatically select first call window
            save_snapshots: Save snapshots of detected deepfakes
            snapshot_dir: Directory to save snapshots
        """
        self.analyzer = analyzer or VideoCallAnalyzer(alert_callback=alert_callback)
        self.platform_detector = platform_detector or PlatformDetector()
        self.window_capture = WindowCapture()
        self.alert_callback = alert_callback
        self.auto_select = auto_select
        self.save_snapshots = save_snapshots
        self.snapshot_dir = Path(snapshot_dir)
        
        if save_snapshots:
            self.snapshot_dir.mkdir(exist_ok=True, parents=True)
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.selected_window = None
        self.callbacks = []
        self.session_id = None
        self.session_start = None
        self.stats = None
        
        logger.info("VideoCallMonitor initialized")
    
    def find_calls(self) -> List[Dict]:
        """
        Find active video calls
        
        Returns:
            List of detected calls
        """
        windows = self.window_capture.find_call_windows()
        
        calls = []
        for window in windows:
            calls.append({
                'platform': window.platform,
                'platform_display': window.platform_display,
                'window_title': window.window_title,
                'window_rect': window.window_rect,
                'is_active': window.is_active,
                'is_minimized': window.is_minimized,
                'process_name': window.process_name,
                'index': len(calls)
            })
        
        return calls
    
    def select_call(self, window_index: int = 0) -> bool:
        """
        Select a call to monitor
        
        Args:
            window_index: Index of window to monitor
        
        Returns:
            True if successful
        """
        windows = self.window_capture.find_call_windows()
        
        if not windows:
            logger.warning("No call windows found")
            return False
        
        if window_index >= len(windows):
            logger.error(f"Window index {window_index} out of range (max: {len(windows)-1})")
            return False
        
        self.selected_window = windows[window_index]
        logger.info(f"Selected call: {self.selected_window.platform_display} - {self.selected_window.window_title}")
        
        return True
    
    def start_monitoring(self, window_index: int = 0):
        """
        Start monitoring a video call
        
        Args:
            window_index: Index of window to monitor
        """
        if not self.select_call(window_index):
            if self.auto_select:
                windows = self.window_capture.find_call_windows()
                if windows:
                    self.selected_window = windows[0]
                    logger.info(f"Auto-selected: {self.selected_window.platform_display}")
                else:
                    logger.error("No call windows found")
                    return
            else:
                return
        
        # Reset analyzer
        self.analyzer.reset()
        
        # Start window capture
        self.window_capture.start_capturing(self.selected_window)
        
        # Initialize session
        self.session_id = f"call_{int(time.time())}"
        self.session_start = time.time()
        self.stats = CallStatistics(
            session_id=self.session_id,
            platform=self.selected_window.platform,
            start_time=self.session_start,
            end_time=None,
            duration=0,
            frames_captured=0,
            frames_processed=0,
            participants_detected=0,
            deepfake_detections=0,
            alerts_generated=0,
            avg_processing_time=0,
            peak_participants=0
        )
        
        self.is_monitoring = True
        
        def monitor_loop():
            logger.info(f"Started monitoring {self.selected_window.platform_display} call")
            
            frame_count = 0
            last_stats_time = time.time()
            
            while self.is_monitoring:
                try:
                    # Get frame
                    call_frame = self.window_capture.get_frame(timeout=1.0)
                    
                    if call_frame:
                        frame_count += 1
                        
                        # Analyze frame
                        analyzed_frame = self.analyzer.analyze_frame(call_frame)
                        
                        # Update stats
                        self.stats.frames_captured = self.window_capture.frame_count
                        self.stats.frames_processed = self.analyzer.processed_count
                        self.stats.deepfake_detections = self.analyzer.deepfake_detections
                        self.stats.alerts_generated = self.analyzer.alerts_sent
                        self.stats.participants_detected = len(self.analyzer.participant_tracks)
                        self.stats.peak_participants = max(
                            self.stats.peak_participants,
                            len(analyzed_frame.participants)
                        )
                        self.stats.avg_processing_time = np.mean(self.analyzer.processing_times) if self.analyzer.processing_times else 0
                        
                        # Save snapshot if deepfake detected and enabled
                        if self.save_snapshots and analyzed_frame.participants:
                            fake_participants = [p for p in analyzed_frame.participants if p.is_deepfake]
                            for p in fake_participants:
                                self._save_snapshot(analyzed_frame, p)
                        
                        # Call callbacks
                        for callback in self.callbacks:
                            try:
                                callback(analyzed_frame)
                            except Exception as e:
                                logger.error(f"Callback error: {str(e)}")
                        
                        # Log stats every 10 seconds
                        if time.time() - last_stats_time > 10:
                            logger.info(f"Call stats - Frames: {self.stats.frames_processed}, "
                                       f"Participants: {len(analyzed_frame.participants)}, "
                                       f"Deepfakes: {self.stats.deepfake_detections}")
                            last_stats_time = time.time()
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Monitor error: {str(e)}")
                    time.sleep(0.1)
            
            # Update end stats
            self.stats.end_time = time.time()
            self.stats.duration = self.stats.end_time - self.stats.start_time
            
            logger.info(f"Monitoring stopped. Session duration: {self.stats.duration:.1f}s")
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        self.window_capture.stop_capturing()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Monitoring stopped")
    
    def register_callback(self, callback: Callable[[CallFrame], None]):
        """Register callback for processed frames"""
        self.callbacks.append(callback)
    
    def _save_snapshot(self, frame: CallFrame, participant: CallParticipant):
        """Save snapshot of deepfake detection"""
        if not self.save_snapshots:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.snapshot_dir / f"deepfake_{timestamp}_track{participant.track_id}.jpg"
        
        # Extract face region
        if participant.face_box:
            x, y, w, h = participant.face_box
            face_img = frame.frame[y:y+h, x:x+w]
            
            # Add annotation
            annotated = frame.frame.copy()
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, f"DEEPFAKE {participant.confidence:.2%}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Save
            cv2.imwrite(str(filename), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            logger.info(f"Snapshot saved: {filename}")
    
    def get_status(self) -> Dict:
        """Get monitor status"""
        return {
            'is_monitoring': self.is_monitoring,
            'session_id': self.session_id,
            'selected_window': {
                'platform': self.selected_window.platform,
                'platform_display': self.selected_window.platform_display,
                'title': self.selected_window.window_title,
                'rect': self.selected_window.window_rect
            } if self.selected_window else None,
            'analyzer_stats': self.analyzer.get_statistics(),
            'capture_stats': self.window_capture.get_stats(),
            'session_stats': asdict(self.stats) if self.stats else None,
            'timestamp': time.time()
        }
    
    def get_participants(self) -> List[Dict]:
        """Get current participants"""
        return self.analyzer.get_participant_summary()


# ============================================
# UI OVERLAY
# ============================================

class CallOverlay:
    """
    Overlay on video call window showing detection results
    """
    
    def __init__(self):
        """Initialize call overlay"""
        self.is_showing = False
        self.overlay_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Colors (BGR)
        self.colors = {
            'real': (0, 255, 0),      # Green
            'fake': (0, 0, 255),       # Red
            'suspicious': (0, 165, 255),  # Orange
            'text': (255, 255, 255),   # White
            'box': (255, 255, 255),     # White
            'background': (0, 0, 0)      # Black
        }
        
        # Font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        
        logger.info("CallOverlay initialized")
    
    def start_overlay(self, window_name: str = "Deepfake Detection Overlay"):
        """Start overlay window"""
        self.is_showing = True
        self.window_name = window_name
        
        def overlay_loop():
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)
            
            while self.is_showing:
                with self.frame_lock:
                    if self.current_frame is not None:
                        # Create overlay frame
                        overlay = self._create_overlay(self.current_frame)
                        cv2.imshow(self.window_name, overlay)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    self.is_showing = False
                    break
                elif key == ord('s'):
                    self._save_screenshot()
            
            cv2.destroyAllWindows()
        
        self.overlay_thread = threading.Thread(target=overlay_loop)
        self.overlay_thread.daemon = True
        self.overlay_thread.start()
    
    def update_frame(self, call_frame: CallFrame):
        """Update current frame"""
        with self.frame_lock:
            self.current_frame = call_frame
    
    def _create_overlay(self, call_frame: CallFrame) -> np.ndarray:
        """Create overlay image"""
        # Create blank overlay
        height, width = 600, 800
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(overlay, f"📹 Video Call Monitor - {call_frame.platform_display}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Session info
        cv2.putText(overlay, f"Frame: {call_frame.frame_number} | Time: {call_frame.timestamp:.1f}s", 
                   (20, 60), self.font, 0.6, (200, 200, 200), 1)
        
        # Participants section
        y_offset = 100
        cv2.putText(overlay, f"👥 Participants ({len(call_frame.participants)})", 
                   (20, y_offset), self.font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        for i, participant in enumerate(call_frame.participants):
            color = self.colors['fake'] if participant.is_deepfake else self.colors['real']
            
            status = "🔴 DEEPFAKE" if participant.is_deepfake else "🟢 REAL"
            if participant.screen_share:
                status += " (Screen Share)"
            
            text = f"  {i+1}. {status} - Conf: {participant.confidence:.2%}"
            cv2.putText(overlay, text, (30, y_offset), self.font, 0.6, color, 1)
            
            y_offset += 25
        
        # Statistics section
        y_offset += 20
        cv2.putText(overlay, "📊 Statistics", (20, y_offset), self.font, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        stats = [
            f"Frames Processed: {call_frame.frame_number}",
            f"Processing Time: {call_frame.processing_time*1000:.1f}ms",
            f"Screen Share: {'Yes' if call_frame.has_screen_share else 'No'}"
        ]
        
        for stat in stats:
            cv2.putText(overlay, f"  {stat}", (30, y_offset), self.font, 0.5, (200, 200, 200), 1)
            y_offset += 20
        
        # Instructions
        cv2.putText(overlay, "Press 'q' to quit | 's' to save screenshot", 
                   (20, height - 30), self.font, 0.5, (100, 100, 100), 1)
        
        return overlay
    
    def _save_screenshot(self):
        """Save current overlay screenshot"""
        if self.current_frame is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"overlay_{timestamp}.png"
        
        with self.frame_lock:
            overlay = self._create_overlay(self.current_frame)
            cv2.imwrite(filename, overlay)
        
        logger.info(f"Screenshot saved: {filename}")
    
    def stop_overlay(self):
        """Stop overlay"""
        self.is_showing = False
        if self.overlay_thread:
            self.overlay_thread.join(timeout=2.0)


# ============================================
# BROWSER CALL DETECTOR
# ============================================

class BrowserCallDetector:
    """
    Detect video calls running in browsers
    Uses browser automation to identify call tabs
    """
    
    def __init__(self):
        """Initialize browser call detector"""
        self.browsers = ['chrome', 'firefox', 'edge', 'brave', 'opera', 'safari']
        self.platform_detector = PlatformDetector()
        logger.info("BrowserCallDetector initialized")
    
    def detect_browser_calls(self) -> List[Dict]:
        """
        Detect video calls in browser tabs
        
        Returns:
            List of detected call information
        """
        calls = []
        
        # Try to connect to browsers via debugging protocol
        # This is a placeholder - actual implementation would use
        # Chrome DevTools Protocol or similar
        
        # For now, rely on window detection for browser windows
        windows = gw.getAllWindows()
        
        for window in windows:
            title = window.title.lower()
            
            # Check if it's a browser window with call-related title
            if any(b in title for b in self.browsers):
                # Look for call platforms in title
                for platform in self.platform_detector.platforms:
                    for pattern in platform.window_title_patterns:
                        if pattern.lower() in title:
                            calls.append({
                                'platform': platform.name,
                                'platform_display': platform.display_name,
                                'window_title': window.title,
                                'browser': next((b for b in self.browsers if b in title), 'unknown'),
                                'window_rect': (window.left, window.top, window.width, window.height),
                                'is_active': window.isActive
                            })
                            break
        
        return calls


# ============================================
# FACTORY CLASS
# ============================================

class VideoCallFactory:
    """Factory for creating video call components"""
    
    @staticmethod
    def create_monitor(alert_callback: Optional[Callable] = None,
                      save_snapshots: bool = False) -> VideoCallMonitor:
        """Create video call monitor"""
        analyzer = VideoCallAnalyzer(alert_callback=alert_callback)
        return VideoCallMonitor(
            analyzer=analyzer,
            alert_callback=alert_callback,
            save_snapshots=save_snapshots
        )
    
    @staticmethod
    def create_analyzer(**kwargs) -> VideoCallAnalyzer:
        """Create video call analyzer"""
        return VideoCallAnalyzer(**kwargs)
    
    @staticmethod
    def create_platform_detector() -> PlatformDetector:
        """Create platform detector"""
        return PlatformDetector()
    
    @staticmethod
    def create_window_capture(capture_rate: int = 5) -> WindowCapture:
        """Create window capture"""
        return WindowCapture(capture_rate=capture_rate)
    
    @staticmethod
    def create_overlay() -> CallOverlay:
        """Create call overlay"""
        return CallOverlay()


# ============================================
# TESTING FUNCTION
# ============================================

def test_video_call_capture():
    """Test video call capture module"""
    print("=" * 60)
    print("TESTING VIDEO CALL CAPTURE MODULE")
    print("=" * 60)
    
    # Test platform detector
    print("\n1️⃣ Testing Platform Detector...")
    detector = PlatformDetector()
    platforms = detector.get_all_platforms()
    print(f"✅ Supported platforms: {', '.join([p['display_name'] for p in platforms])}")
    
    # Test window capture
    print("\n2️⃣ Testing Window Capture...")
    capture = WindowCapture(capture_rate=5)
    
    # Find call windows
    windows = capture.find_call_windows()
    print(f"✅ Found {len(windows)} potential call windows")
    
    for i, window in enumerate(windows[:3]):  # Show first 3
        print(f"   Window {i+1}: {window.platform_display} - {window.window_title[:50]}")
    
    # Test analyzer (without actual capture)
    print("\n3️⃣ Testing Video Call Analyzer...")
    analyzer = VideoCallAnalyzer()
    print(f"✅ Analyzer initialized")
    print(f"   Face detector: {type(analyzer.face_detector).__name__}")
    print(f"   Deepfake model: {type(analyzer.deepfake_model).__name__}")
    
    # Test with dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    call_frame = CallFrame(
        timestamp=time.time(),
        frame_number=1,
        frame=dummy_frame,
        frame_small=cv2.resize(dummy_frame, (320, 240)),
        window_rect=(0, 0, 640, 480),
        participants=[],
        platform="zoom",
        platform_display="Zoom",
        processing_time=0.0
    )
    
    analyzed = analyzer.analyze_frame(call_frame)
    print(f"✅ Frame analyzed")
    print(f"   Participants detected: {len(analyzed.participants)}")
    
    # Test monitor
    print("\n4️⃣ Testing Video Call Monitor...")
    monitor = VideoCallMonitor(analyzer=analyzer, auto_select=False)
    print(f"✅ Monitor initialized")
    
    # Test overlay
    print("\n5️⃣ Testing Call Overlay...")
    overlay = CallOverlay()
    overlay.update_frame(analyzed)
    print(f"✅ Overlay created")
    
    print("\n" + "=" * 60)
    print("✅ VIDEO CALL CAPTURE TEST PASSED!")
    print("=" * 60)
    
    print("\n📝 To run live monitoring:")
    print("""
    from app.video_call_capture import VideoCallFactory, CallOverlay
    
    # Create monitor with alert callback
    def on_alert(alert):
        print(f"⚠️ Deepfake detected: {alert}")
    
    monitor = VideoCallFactory.create_monitor(alert_callback=on_alert, save_snapshots=True)
    
    # Find available calls
    calls = monitor.find_calls()
    for i, call in enumerate(calls):
        print(f"{i}: {call['platform_display']} - {call['window_title']}")
    
    # Start monitoring (select first call)
    monitor.start_monitoring(0)
    
    # Optional: show overlay
    overlay = CallOverlay()
    overlay.start_overlay()
    
    # Register callback to update overlay
    monitor.register_callback(overlay.update_frame)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            status = monitor.get_status()
            print(f"Status: {status['session_stats']['frames_processed']} frames processed")
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        overlay.stop_overlay()
    """)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Run test
    test_video_call_capture()