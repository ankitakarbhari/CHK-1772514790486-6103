# app/utils/video_processor.py
"""
Video Processing Module for Deepfake Detection
Handles video capture, frame extraction, analysis, and real-time processing
Supports multiple video formats, live streams, and batch processing
Python 3.13+ Compatible
"""

import os
import sys
import time
import json
import math
import logging
import tempfile
import threading
import queue
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Generator, Callable
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== VIDEO PROCESSING ==========
import cv2
import numpy as np
from PIL import Image
import av  # PyAV
import imageio
import skvideo.io
from moviepy.editor import VideoFileClip

# ========== FFMPEG ==========
import ffmpeg
from ffmpeg import Error as FFmpegError

# ========== FACE DETECTION ==========
from app.utils.face_detection import FaceDetectionEnsemble, FacePreprocessor, Face, FaceDetectionResult

# ========== DEEPFAKE MODELS ==========
from app.models.ensemble import DeepfakeEnsemble, EnsembleFactory, EnsembleResult
from app.utils.heatmap import EnsembleHeatmapGenerator, HeatmapVisualizer, HeatmapResult

# ========== AUDIO PROCESSING ==========
from app.utils.audio_processor import AudioDeepfakeDetector, AudioProcessorFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class VideoInfo:
    """Video file information"""
    path: str
    filename: str
    duration: float  # seconds
    fps: float
    width: int
    height: int
    total_frames: int
    codec: str
    bitrate: int
    has_audio: bool
    audio_codec: Optional[str] = None
    file_size: int  # bytes
    creation_time: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        return result


@dataclass
class FrameResult:
    """Analysis result for a single frame"""
    frame_index: int
    timestamp: float
    face_detected: bool
    num_faces: int
    faces: List[Dict]
    is_fake: bool
    fake_probability: float
    real_probability: float
    confidence: float
    processing_time: float
    heatmap: Optional[np.ndarray] = None
    manipulated_regions: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        if self.heatmap is not None:
            result['heatmap'] = f"<heatmap shape={self.heatmap.shape}>"
        return result


@dataclass
class VideoSegment:
    """Video segment for batch processing"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    frames: List[np.ndarray]
    timestamps: List[float]


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result"""
    video_info: VideoInfo
    frames_analyzed: int
    fake_frames: int
    real_frames: int
    fake_percentage: float
    confidence_avg: float
    confidence_std: float
    verdict: str  # 'REAL', 'FAKE', 'SUSPICIOUS'
    frame_results: List[FrameResult]
    timeline: List[Dict]
    processing_time: float
    warnings: List[str]
    face_tracks: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['frame_results'] = [fr.to_dict() for fr in self.frame_results]
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def summary(self) -> str:
        """Get text summary"""
        return (f"Video Analysis: {self.video_info.filename}\n"
                f"Duration: {self.video_info.duration:.2f}s, FPS: {self.video_info.fps:.2f}\n"
                f"Frames analyzed: {self.frames_analyzed}\n"
                f"Fake frames: {self.fake_frames} ({self.fake_percentage:.2f}%)\n"
                f"Real frames: {self.real_frames}\n"
                f"Confidence: {self.confidence_avg:.3f} ± {self.confidence_std:.3f}\n"
                f"Verdict: {self.verdict}\n"
                f"Processing time: {self.processing_time:.2f}s")


@dataclass
class LiveStreamInfo:
    """Live stream information"""
    source: str  # URL, camera ID, etc.
    width: int
    height: int
    fps: float
    buffer_size: int
    is_active: bool
    frames_received: int
    frames_dropped: int
    start_time: float


# ============================================
# VIDEO CAPTURE
# ============================================

class VideoCapture:
    """
    Advanced video capture with support for multiple sources
    Handles video files, cameras, and network streams
    """
    
    def __init__(self, 
                 source: Union[str, int] = 0,
                 target_fps: Optional[float] = None,
                 target_size: Optional[Tuple[int, int]] = None,
                 buffer_size: int = 30,
                 timeout: int = 30):
        """
        Initialize video capture
        
        Args:
            source: Video source - file path, camera ID, or URL
            target_fps: Target FPS for processing (None = original)
            target_size: Target frame size (width, height)
            buffer_size: Frame buffer size
            timeout: Connection timeout for streams
        """
        self.source = source
        self.target_fps = target_fps
        self.target_size = target_size
        self.buffer_size = buffer_size
        self.timeout = timeout
        
        self.cap = None
        self.info = None
        self.is_opened = False
        self.frame_count = 0
        self.start_time = None
        self.frame_buffer = deque(maxlen=buffer_size)
        self.is_live = False
        
        logger.info(f"VideoCapture initialized with source: {source}")
    
    def open(self) -> bool:
        """Open video source"""
        try:
            # Check source type
            if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # Camera device
                device_id = int(self.source) if isinstance(self.source, str) else self.source
                self.cap = cv2.VideoCapture(device_id)
                self.is_live = True
                self._get_camera_info(device_id)
                
            elif isinstance(self.source, str) and self.source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
                # Network stream
                self.cap = cv2.VideoCapture(self.source)
                self.is_live = True
                self._get_stream_info()
                
            elif isinstance(self.source, str) and os.path.exists(self.source):
                # Video file
                self.cap = cv2.VideoCapture(self.source)
                self.is_live = False
                self._get_file_info()
                
            else:
                logger.error(f"Unsupported source: {self.source}")
                return False
            
            self.is_opened = self.cap.isOpened()
            if self.is_opened:
                logger.info(f"Video source opened successfully")
            else:
                logger.error(f"Failed to open video source: {self.source}")
            
            return self.is_opened
            
        except Exception as e:
            logger.error(f"Error opening video source: {str(e)}")
            return False
    
    def _get_file_info(self):
        """Get information from video file"""
        try:
            # Get basic properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Get file info
            file_size = os.path.getsize(self.source)
            
            # Try to get more info using ffprobe
            codec = "unknown"
            bitrate = 0
            has_audio = False
            audio_codec = None
            creation_time = None
            
            try:
                probe = ffmpeg.probe(self.source)
                video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                if video_stream:
                    codec = video_stream.get('codec_name', 'unknown')
                    bitrate = int(video_stream.get('bit_rate', 0))
                
                audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
                has_audio = audio_stream is not None
                if audio_stream:
                    audio_codec = audio_stream.get('codec_name', 'unknown')
                
                # Creation time
                if 'format' in probe and 'tags' in probe['format']:
                    creation_time = probe['format']['tags'].get('creation_time')
                    
            except Exception as e:
                logger.debug(f"FFprobe error: {str(e)}")
            
            self.info = VideoInfo(
                path=self.source,
                filename=os.path.basename(self.source),
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                codec=codec,
                bitrate=bitrate,
                has_audio=has_audio,
                audio_codec=audio_codec,
                file_size=file_size,
                creation_time=creation_time
            )
            
            logger.info(f"Video file info: {duration:.2f}s, {fps:.2f}fps, {width}x{height}")
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
    
    def _get_camera_info(self, device_id: int):
        """Get camera information"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default
        
        self.info = VideoInfo(
            path=f"camera:{device_id}",
            filename=f"Camera {device_id}",
            duration=0,  # Live, unknown
            fps=fps,
            width=width,
            height=height,
            total_frames=0,
            codec="raw",
            bitrate=0,
            has_audio=False,
            file_size=0
        )
        
        logger.info(f"Camera info: {fps:.2f}fps, {width}x{height}")
    
    def _get_stream_info(self):
        """Get stream information"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default
        
        self.info = VideoInfo(
            path=self.source,
            filename=os.path.basename(self.source),
            duration=0,  # Live stream
            fps=fps,
            width=width,
            height=height,
            total_frames=0,
            codec="stream",
            bitrate=0,
            has_audio=False,
            file_size=0
        )
        
        logger.info(f"Stream info: {fps:.2f}fps, {width}x{height}")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Read next frame from video source
        
        Returns:
            Tuple of (success, frame, timestamp)
        """
        if not self.is_opened:
            return False, None, 0
        
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            self.frame_count += 1
            timestamp = self.frame_count / self.info.fps if self.info.fps > 0 else time.time()
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if target size specified
            if self.target_size:
                frame = cv2.resize(frame, self.target_size)
            
            # Add to buffer
            self.frame_buffer.append((frame, timestamp, self.frame_count))
            
            return True, frame, timestamp
        else:
            return False, None, 0
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """Get frame at specific timestamp"""
        if not self.is_opened or self.info is None or self.info.total_frames == 0:
            return None
        
        # Calculate frame index
        frame_idx = int(timestamp * self.info.fps)
        return self.get_frame_at_index(frame_idx)
    
    def get_frame_at_index(self, index: int) -> Optional[np.ndarray]:
        """Get frame at specific index"""
        if not self.is_opened:
            return None
        
        # Set position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        
        # Read frame
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            if self.target_size:
                frame = cv2.resize(frame, self.target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        
        return None
    
    def extract_frames(self, 
                      max_frames: int = 100,
                      step: int = 1,
                      start_time: float = 0,
                      end_time: Optional[float] = None,
                      strategy: str = 'uniform') -> Generator[Tuple[np.ndarray, float, int], None, None]:
        """
        Extract frames from video
        
        Args:
            max_frames: Maximum number of frames to extract
            step: Extract every 'step' frames
            start_time: Starting timestamp
            end_time: Ending timestamp
            strategy: 'uniform', 'keyframes', 'scene_change'
        
        Yields:
            Tuple of (frame, timestamp, frame_index)
        """
        if not self.is_opened:
            return
        
        total_frames = self.info.total_frames
        
        if strategy == 'uniform':
            # Uniform sampling
            if end_time:
                end_frame = int(end_time * self.info.fps)
            else:
                end_frame = total_frames - 1
            
            start_frame = int(start_time * self.info.fps)
            frame_indices = np.linspace(start_frame, end_frame, max_frames, dtype=int)
            
            for idx in frame_indices:
                frame = self.get_frame_at_index(idx)
                if frame is not None:
                    timestamp = idx / self.info.fps
                    yield frame, timestamp, idx
        
        elif strategy == 'keyframes':
            # Extract keyframes using ffmpeg
            try:
                out, _ = (
                    ffmpeg
                    .input(self.source)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=max_frames)
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                frames = np.frombuffer(out, np.uint8).reshape(-1, self.info.height, self.info.width, 3)
                for i, frame in enumerate(frames):
                    timestamp = i / self.info.fps
                    yield frame, timestamp, i
                    
            except Exception as e:
                logger.error(f"Keyframe extraction error: {str(e)}")
                # Fallback to uniform
                yield from self.extract_frames(max_frames, step, start_time, end_time, 'uniform')
        
        elif strategy == 'scene_change':
            # Detect scene changes
            prev_frame = None
            scene_frames = []
            frame_count = 0
            
            while len(scene_frames) < max_frames:
                ret, frame, timestamp = self.read_frame()
                if not ret:
                    break
                
                if prev_frame is not None:
                    # Calculate histogram difference
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                    
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
                    
                    diff = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CHISQR)
                    
                    if diff > 0.5:  # Scene change threshold
                        scene_frames.append((frame, timestamp, frame_count))
                        yield frame, timestamp, frame_count
                
                prev_frame = frame
                frame_count += 1
            
            # Reset position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def get_live_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, float, int]]:
        """Get next frame from live stream (non-blocking)"""
        if not self.is_live or not self.is_opened:
            return None
        
        ret, frame, timestamp = self.read_frame()
        if ret:
            return frame, timestamp, self.frame_count
        return None
    
    def get_buffered_frame(self) -> Optional[Tuple[np.ndarray, float, int]]:
        """Get next frame from buffer"""
        if self.frame_buffer:
            return self.frame_buffer.popleft()
        return None
    
    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
            self.is_opened = False
            logger.info("Video capture released")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# ============================================
# VIDEO WRITER
# ============================================

class VideoWriter:
    """Write processed video with overlays"""
    
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int], codec: str = 'mp4v'):
        """
        Initialize video writer
        
        Args:
            output_path: Output file path
            fps: Frames per second
            frame_size: Frame size (width, height)
            codec: FourCC codec code
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        # Define codec
        if codec == 'mp4v':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif codec == 'avc1':
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        elif codec == 'XVID':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*codec)
        
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.frame_count = 0
        
        logger.info(f"VideoWriter initialized: {output_path}")
    
    def write_frame(self, frame: np.ndarray, overlay: Optional[np.ndarray] = None):
        """
        Write frame to video
        
        Args:
            frame: Frame to write (RGB)
            overlay: Optional overlay frame
        """
        if overlay is not None:
            # Blend frame and overlay
            output = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        else:
            output = frame
        
        # Convert RGB to BGR for OpenCV
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        if output_bgr.shape[1] != self.frame_size[0] or output_bgr.shape[0] != self.frame_size[1]:
            output_bgr = cv2.resize(output_bgr, self.frame_size)
        
        self.writer.write(output_bgr)
        self.frame_count += 1
    
    def write_annotated_frame(self, frame: np.ndarray, faces: List[Dict], stats: Dict):
        """Write frame with annotations"""
        annotated = frame.copy()
        
        # Draw face boxes
        for face in faces:
            x, y, w, h = face['bbox']
            color = (0, 0, 255) if face.get('is_fake', False) else (0, 255, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            label = f"Fake: {face['confidence']:.2f}" if face.get('is_fake') else f"Real: {face['confidence']:.2f}"
            cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw stats
        y_offset = 30
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(annotated, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        self.write_frame(annotated)
    
    def release(self):
        """Release video writer"""
        if self.writer:
            self.writer.release()
            logger.info(f"VideoWriter released: {self.frame_count} frames written")


# ============================================
# VIDEO PROCESSOR
# ============================================

class VideoProcessor:
    """
    Main video processor for deepfake detection
    Handles frame extraction, face detection, and deepfake analysis
    """
    
    def __init__(self,
                 face_detector: Optional[FaceDetectionEnsemble] = None,
                 deepfake_detector: Optional[DeepfakeEnsemble] = None,
                 audio_detector: Optional[AudioDeepfakeDetector] = None,
                 heatmap_generator: Optional[EnsembleHeatmapGenerator] = None,
                 face_preprocessor: Optional[FacePreprocessor] = None,
                 max_workers: int = 4,
                 batch_size: int = 32,
                 enable_audio: bool = True):
        """
        Initialize video processor
        
        Args:
            face_detector: Face detection ensemble
            deepfake_detector: Deepfake detection model
            audio_detector: Audio deepfake detector
            heatmap_generator: Heatmap generator
            face_preprocessor: Face preprocessing module
            max_workers: Maximum threads for parallel processing
            batch_size: Batch size for model inference
            enable_audio: Enable audio analysis
        """
        self.face_detector = face_detector or FaceDetectionEnsemble()
        self.deepfake_detector = deepfake_detector or EnsembleFactory.create_accurate_ensemble()
        self.audio_detector = audio_detector or AudioProcessorFactory.create_detector()
        self.heatmap_generator = heatmap_generator
        self.face_preprocessor = face_preprocessor or FacePreprocessor()
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_audio = enable_audio
        
        # Thread pools
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Tracking
        self.face_tracks = {}
        self.next_track_id = 0
        
        logger.info(f"VideoProcessor initialized with {max_workers} workers")
    
    def analyze_video(self,
                     video_path: str,
                     sample_rate: float = 1.0,  # frames per second
                     max_frames: int = 300,
                     return_heatmaps: bool = False,
                     analyze_audio: bool = True,
                     progress_callback: Optional[Callable] = None) -> VideoAnalysisResult:
        """
        Analyze video for deepfake detection
        
        Args:
            video_path: Path to video file
            sample_rate: Frames to analyze per second
            max_frames: Maximum frames to analyze
            return_heatmaps: Include heatmaps in results
            analyze_audio: Analyze audio track
            progress_callback: Callback for progress updates
        
        Returns:
            VideoAnalysisResult
        """
        start_time = time.time()
        warnings = []
        
        # Open video
        with VideoCapture(video_path) as cap:
            if not cap.is_opened:
                raise ValueError(f"Could not open video: {video_path}")
            
            video_info = cap.info
            
            # Calculate frame sampling
            if video_info.total_frames > 0:
                total_frames = min(video_info.total_frames, int(sample_rate * video_info.duration))
                if max_frames:
                    total_frames = min(total_frames, max_frames)
                
                frame_step = max(1, int(video_info.fps / sample_rate))
                logger.info(f"Analyzing up to {total_frames} frames from {video_path}")
            else:
                # Live or unknown length
                total_frames = max_frames
                frame_step = 1
                logger.info(f"Analyzing up to {total_frames} frames from live source")
            
            # Extract frames
            frames = []
            timestamps = []
            frame_indices = []
            
            for i, (frame, timestamp, idx) in enumerate(cap.extract_frames(
                max_frames=total_frames, 
                step=frame_step
            )):
                frames.append(frame)
                timestamps.append(timestamp)
                frame_indices.append(idx)
                
                if progress_callback:
                    progress_callback(i + 1, total_frames, "Extracting frames")
            
            # Process frames in batches
            frame_results = []
            total_batches = (len(frames) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(0, len(frames), self.batch_size):
                batch_end = min(batch_idx + self.batch_size, len(frames))
                batch_frames = frames[batch_idx:batch_end]
                batch_timestamps = timestamps[batch_idx:batch_end]
                batch_indices = frame_indices[batch_idx:batch_end]
                
                batch_results = self._process_frame_batch(
                    batch_frames,
                    batch_timestamps,
                    batch_indices,
                    return_heatmaps
                )
                
                frame_results.extend(batch_results)
                
                if progress_callback:
                    progress_callback(batch_end, len(frames), "Analyzing frames")
        
        # Analyze audio if enabled
        audio_result = None
        if analyze_audio and video_info.has_audio and self.enable_audio:
            try:
                audio_result = self.audio_detector.detect_from_file(video_path)
                if audio_result.is_fake:
                    warnings.append(f"Audio deepfake detected (confidence: {audio_result.confidence:.2f})")
            except Exception as e:
                logger.error(f"Audio analysis error: {str(e)}")
                warnings.append(f"Audio analysis failed: {str(e)}")
        
        # Compile results
        result = self._compile_results(
            video_info=video_info,
            frame_results=frame_results,
            audio_result=audio_result,
            processing_time=time.time() - start_time,
            warnings=warnings
        )
        
        logger.info(f"Video analysis complete: {result.verdict} ({result.fake_percentage:.1f}% fake)")
        
        return result
    
    def _process_frame_batch(self,
                            frames: List[np.ndarray],
                            timestamps: List[float],
                            frame_indices: List[int],
                            return_heatmaps: bool) -> List[FrameResult]:
        """Process a batch of frames"""
        results = []
        
        for frame, timestamp, idx in zip(frames, timestamps, frame_indices):
            try:
                # Detect faces
                face_result = self.face_detector.detect(frame)
                
                if face_result.num_faces > 0:
                    # Use largest face for main analysis
                    largest_face = max(face_result.faces, key=lambda f: f.box.area)
                    
                    # Preprocess face
                    face_img = self.face_preprocessor.preprocess_for_model(
                        largest_face, frame, align=True, normalize=True
                    )
                    
                    # Detect deepfake
                    model_result = self.deepfake_detector.predict_single(face_img)
                    
                    is_fake = model_result.prediction == 'FAKE'
                    fake_prob = model_result.fake_probability
                    real_prob = model_result.real_probability
                    confidence = model_result.confidence
                    
                    # Generate heatmap if requested
                    heatmap = None
                    manipulated_regions = None
                    
                    if return_heatmaps and self.heatmap_generator:
                        try:
                            heatmap_result = self.heatmap_generator.generate(face_img)
                            heatmap = heatmap_result.heatmap
                            manipulated_regions = heatmap_result.manipulated_regions
                        except Exception as e:
                            logger.error(f"Heatmap generation error: {str(e)}")
                    
                    # Prepare face data
                    faces_data = []
                    for face in face_result.faces:
                        face_dict = {
                            'bbox': (face.box.x, face.box.y, face.box.width, face.box.height),
                            'confidence': face.confidence,
                            'landmarks': face.landmarks.to_dict() if face.landmarks else None,
                            'track_id': self._assign_track_id(face, idx)
                        }
                        
                        # Update tracking
                        self._update_track(face_dict['track_id'], face_dict, idx, is_fake)
                        
                        faces_data.append(face_dict)
                    
                else:
                    # No face detected
                    is_fake = False
                    fake_prob = 0.0
                    real_prob = 0.0
                    confidence = 0.0
                    faces_data = []
                    heatmap = None
                    manipulated_regions = None
                
                # Calculate processing time
                proc_time = 0.0  # Would need to track per frame
                
                results.append(FrameResult(
                    frame_index=idx,
                    timestamp=timestamp,
                    face_detected=face_result.num_faces > 0,
                    num_faces=face_result.num_faces,
                    faces=faces_data,
                    is_fake=is_fake,
                    fake_probability=float(fake_prob),
                    real_probability=float(real_prob),
                    confidence=float(confidence),
                    processing_time=proc_time,
                    heatmap=heatmap,
                    manipulated_regions=manipulated_regions
                ))
                
            except Exception as e:
                logger.error(f"Error processing frame {idx}: {str(e)}")
                results.append(FrameResult(
                    frame_index=idx,
                    timestamp=timestamp,
                    face_detected=False,
                    num_faces=0,
                    faces=[],
                    is_fake=False,
                    fake_probability=0.0,
                    real_probability=0.0,
                    confidence=0.0,
                    processing_time=0.0
                ))
        
        return results
    
    def _assign_track_id(self, face: Face, frame_num: int) -> int:
        """Assign tracking ID to face"""
        best_match = None
        best_iou = 0.5  # Threshold
        
        for track_id, track_info in self.face_tracks.items():
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
    
    def _update_track(self, track_id: int, face_dict: Dict, frame_num: int, is_fake: bool):
        """Update face tracking"""
        if track_id not in self.face_tracks:
            self.face_tracks[track_id] = {
                'first_frame': frame_num,
                'last_frame': frame_num,
                'last_bbox': face_dict['bbox'],
                'fake_count': 0,
                'real_count': 0,
                'appearances': 1
            }
        
        track = self.face_tracks[track_id]
        track['last_frame'] = frame_num
        track['last_bbox'] = face_dict['bbox']
        track['appearances'] += 1
        
        if is_fake:
            track['fake_count'] += 1
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
    
    def _compile_results(self,
                         video_info: VideoInfo,
                         frame_results: List[FrameResult],
                         audio_result: Optional[Any],
                         processing_time: float,
                         warnings: List[str]) -> VideoAnalysisResult:
        """Compile frame results into final analysis"""
        
        if not frame_results:
            return VideoAnalysisResult(
                video_info=video_info,
                frames_analyzed=0,
                fake_frames=0,
                real_frames=0,
                fake_percentage=0.0,
                confidence_avg=0.0,
                confidence_std=0.0,
                verdict="UNKNOWN",
                frame_results=[],
                timeline=[],
                processing_time=processing_time,
                warnings=["No frames analyzed"]
            )
        
        # Count fake frames
        fake_frames = sum(1 for r in frame_results if r.is_fake)
        real_frames = len(frame_results) - fake_frames
        fake_percentage = (fake_frames / len(frame_results)) * 100
        
        # Calculate confidence statistics
        confidences = [r.confidence for r in frame_results if r.face_detected]
        confidence_avg = float(np.mean(confidences)) if confidences else 0.0
        confidence_std = float(np.std(confidences)) if confidences else 0.0
        
        # Determine verdict
        if fake_percentage > 70:
            verdict = "FAKE"
        elif fake_percentage > 30:
            verdict = "SUSPICIOUS"
        else:
            verdict = "REAL"
        
        # Create timeline
        timeline = []
        for r in frame_results:
            timeline.append({
                'timestamp': r.timestamp,
                'is_fake': r.is_fake,
                'confidence': r.confidence,
                'num_faces': r.num_faces
            })
        
        # Add audio warning if applicable
        if audio_result and audio_result.is_fake:
            warnings.append(f"Audio deepfake detected with confidence {audio_result.confidence:.2f}")
        
        return VideoAnalysisResult(
            video_info=video_info,
            frames_analyzed=len(frame_results),
            fake_frames=fake_frames,
            real_frames=real_frames,
            fake_percentage=float(fake_percentage),
            confidence_avg=confidence_avg,
            confidence_std=confidence_std,
            verdict=verdict,
            frame_results=frame_results,
            timeline=timeline,
            processing_time=processing_time,
            warnings=warnings,
            face_tracks=self.face_tracks
        )
    
    def process_live_stream(self,
                           source: Union[str, int],
                           callback: Optional[Callable[[FrameResult], None]] = None,
                           frame_interval: int = 5,
                           max_frames: Optional[int] = None) -> Generator[FrameResult, None, None]:
        """
        Process live video stream in real-time
        
        Args:
            source: Video source (camera ID, URL)
            callback: Callback for each frame result
            frame_interval: Process every Nth frame
            max_frames: Maximum frames to process
        
        Yields:
            FrameResult for each processed frame
        """
        with VideoCapture(source, target_fps=30, target_size=(640, 480)) as cap:
            if not cap.open():
                raise ValueError(f"Could not open video source: {source}")
            
            logger.info(f"Starting live processing on {source}")
            
            frame_count = 0
            processed_count = 0
            start_time = time.time()
            
            while True:
                # Check max frames
                if max_frames and processed_count >= max_frames:
                    break
                
                # Get frame
                result = cap.get_live_frame(timeout=0.1)
                if result is None:
                    continue
                
                frame, timestamp, idx = result
                frame_count += 1
                
                # Process every Nth frame
                if frame_count % frame_interval == 0:
                    # Detect faces
                    face_result = self.face_detector.detect(frame)
                    
                    if face_result.num_faces > 0:
                        # Use largest face
                        largest_face = max(face_result.faces, key=lambda f: f.box.area)
                        
                        # Preprocess face
                        face_img = self.face_preprocessor.preprocess_for_model(
                            largest_face, frame, align=True, normalize=True
                        )
                        
                        # Detect deepfake
                        model_result = self.deepfake_detector.predict_single(face_img)
                        
                        is_fake = model_result.prediction == 'FAKE'
                        confidence = model_result.confidence
                        fake_prob = model_result.fake_probability
                        real_prob = model_result.real_probability
                        
                        # Prepare face data
                        faces_data = []
                        for face in face_result.faces:
                            faces_data.append({
                                'bbox': (face.box.x, face.box.y, face.box.width, face.box.height),
                                'confidence': face.confidence,
                                'landmarks': face.landmarks.to_dict() if face.landmarks else None
                            })
                        
                        frame_result = FrameResult(
                            frame_index=processed_count,
                            timestamp=timestamp,
                            face_detected=True,
                            num_faces=face_result.num_faces,
                            faces=faces_data,
                            is_fake=is_fake,
                            fake_probability=float(fake_prob),
                            real_probability=float(real_prob),
                            confidence=float(confidence),
                            processing_time=0.0
                        )
                    else:
                        frame_result = FrameResult(
                            frame_index=processed_count,
                            timestamp=timestamp,
                            face_detected=False,
                            num_faces=0,
                            faces=[],
                            is_fake=False,
                            fake_probability=0.0,
                            real_probability=0.0,
                            confidence=0.0,
                            processing_time=0.0
                        )
                    
                    processed_count += 1
                    
                    if callback:
                        callback(frame_result)
                    
                    yield frame_result
                
                # Small delay
                time.sleep(0.001)
    
    def extract_key_frames(self,
                          video_path: str,
                          method: str = 'scene_change',
                          max_frames: int = 50) -> List[Tuple[np.ndarray, float]]:
        """
        Extract key frames from video
        
        Args:
            video_path: Path to video file
            method: 'uniform', 'scene_change', 'face_detection'
            max_frames: Maximum number of frames
        
        Returns:
            List of (frame, timestamp) tuples
        """
        key_frames = []
        
        with VideoCapture(video_path) as cap:
            if not cap.open():
                return []
            
            if method == 'uniform':
                # Uniform sampling
                total_frames = cap.info.total_frames
                step = max(1, total_frames // max_frames)
                
                for frame, timestamp, idx in cap.extract_frames(max_frames=max_frames, step=step):
                    key_frames.append((frame, timestamp))
            
            elif method == 'scene_change':
                # Detect scene changes
                prev_frame = None
                prev_hist = None
                
                for frame, timestamp, idx in cap.extract_frames(max_frames=cap.info.total_frames, step=1):
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    
                    # Calculate histogram
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist = hist / hist.sum()
                    
                    if prev_hist is not None:
                        # Calculate difference
                        diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                        
                        if diff > 0.5:  # Scene change threshold
                            key_frames.append((frame, timestamp))
                            
                            if len(key_frames) >= max_frames:
                                break
                    
                    prev_hist = hist
                    prev_frame = frame
            
            elif method == 'face_detection':
                # Extract frames with faces
                for frame, timestamp, idx in cap.extract_frames(max_frames=cap.info.total_frames, step=10):
                    face_result = self.face_detector.detect(frame)
                    
                    if face_result.num_faces > 0:
                        key_frames.append((frame, timestamp))
                        
                        if len(key_frames) >= max_frames:
                            break
        
        logger.info(f"Extracted {len(key_frames)} key frames using {method} method")
        return key_frames
    
    def generate_video_report(self, result: VideoAnalysisResult, output_path: str):
        """
        Generate HTML report for video analysis
        
        Args:
            result: Video analysis result
            output_path: Path to save HTML report
        """
        # Create HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Deepfake Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .header h1 {{ color: #333; }}
                .verdict {{ font-size: 24px; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; }}
                .REAL {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
                .FAKE {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
                .SUSPICIOUS {{ background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
                .stat-value {{ font-size: 32px; font-weight: bold; color: #007bff; }}
                .stat-label {{ font-size: 14px; color: #6c757d; margin-top: 5px; }}
                .timeline {{ margin: 40px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                .frame-dot {{ display: inline-block; width: 12px; height: 12px; margin: 2px; border-radius: 50%; }}
                .fake-dot {{ background-color: #dc3545; }}
                .real-dot {{ background-color: #28a745; }}
                .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 4px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .progress-bar {{ width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden; margin: 10px 0; }}
                .progress-fill {{ height: 100%; background-color: #dc3545; transition: width 0.3s; }}
                .footer {{ text-align: center; margin-top: 40px; color: #6c757d; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎥 Video Deepfake Analysis Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="verdict {result.verdict}">
                    <strong>Verdict: {result.verdict}</strong>
                    <p>{result.fake_percentage:.1f}% of analyzed frames show signs of manipulation</p>
                </div>
                
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {result.fake_percentage}%;"></div>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{result.frames_analyzed}</div>
                        <div class="stat-label">Frames Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{result.fake_frames}</div>
                        <div class="stat-label">Fake Frames</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{result.real_frames}</div>
                        <div class="stat-label">Real Frames</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{result.confidence_avg:.2%}</div>
                        <div class="stat-label">Avg Confidence</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{result.processing_time:.2f}s</div>
                        <div class="stat-label">Processing Time</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{result.video_info.duration:.2f}s</div>
                        <div class="stat-label">Video Duration</div>
                    </div>
                </div>
                
                <h2>📹 Video Information</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Filename</td><td>{result.video_info.filename}</td></tr>
                    <tr><td>Resolution</td><td>{result.video_info.width}x{result.video_info.height}</td></tr>
                    <tr><td>FPS</td><td>{result.video_info.fps:.2f}</td></tr>
                    <tr><td>Codec</td><td>{result.video_info.codec}</td></tr>
                    <tr><td>Has Audio</td><td>{'Yes' if result.video_info.has_audio else 'No'}</td></tr>
                    <tr><td>File Size</td><td>{result.video_info.file_size / (1024*1024):.2f} MB</td></tr>
                </table>
                
                <h2>📊 Detection Timeline</h2>
                <div class="timeline">
        """
        
        # Add timeline visualization
        for r in result.frame_results:
            dot_class = 'fake-dot' if r.is_fake else 'real-dot'
            html += f'<span class="frame-dot {dot_class}" title="t={r.timestamp:.2f}s, conf={r.confidence:.2%}"></span>'
        
        html += """
                </div>
                
                <h2>⚠️ Warnings</h2>
        """
        
        if result.warnings:
            for warning in result.warnings:
                html += f'<div class="warning">⚠️ {warning}</div>'
        else:
            html += "<p>No warnings detected.</p>"
        
        html += """
                <h2>📋 Frame Details (First 50)</h2>
                <table>
                    <tr>
                        <th>Frame</th>
                        <th>Timestamp</th>
                        <th>Faces</th>
                        <th>Detection</th>
                        <th>Confidence</th>
                    </tr>
        """
        
        for i, r in enumerate(result.frame_results[:50]):
            detection = "FAKE" if r.is_fake else "REAL"
            color = "#dc3545" if r.is_fake else "#28a745"
            html += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{r.timestamp:.2f}s</td>
                        <td>{r.num_faces}</td>
                        <td style="color: {color}; font-weight: bold;">{detection}</td>
                        <td>{r.confidence:.2%}</td>
                    </tr>
            """
        
        html += """
                </table>
                
                <div class="footer">
                    <p>Report generated by Deepfake Detection System v1.0</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Report saved to {output_path}")


# ============================================
# BATCH PROCESSOR
# ============================================

class BatchVideoProcessor:
    """Process multiple videos in batch"""
    
    def __init__(self, processor: VideoProcessor, max_workers: int = 4):
        """
        Initialize batch processor
        
        Args:
            processor: VideoProcessor instance
            max_workers: Maximum parallel workers
        """
        self.processor = processor
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.results = []
        
        logger.info(f"BatchVideoProcessor initialized with {max_workers} workers")
    
    def process_videos(self, video_paths: List[str], **kwargs) -> List[VideoAnalysisResult]:
        """
        Process multiple videos in parallel
        
        Args:
            video_paths: List of video paths
            **kwargs: Arguments for analyze_video
        
        Returns:
            List of results
        """
        from concurrent.futures import as_completed
        
        futures = []
        for path in video_paths:
            future = self.executor.submit(self.processor.analyze_video, path, **kwargs)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
        
        return results
    
    def generate_batch_report(self, output_path: str):
        """Generate HTML report for batch processing"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Video Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #333; text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .fake {{ background-color: #f8d7da; }}
                .real {{ background-color: #d4edda; }}
                .suspicious {{ background-color: #fff3cd; }}
                .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 30px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Batch Video Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Videos Processed: {len(self.results)}</p>
                
                <div class="summary">
                    <div class="stat-card">
                        <div class="stat-value">{sum(1 for r in self.results if r.verdict == 'FAKE')}</div>
                        <div>Fake Videos</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{sum(1 for r in self.results if r.verdict == 'REAL')}</div>
                        <div>Real Videos</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{sum(1 for r in self.results if r.verdict == 'SUSPICIOUS')}</div>
                        <div>Suspicious Videos</div>
                    </div>
                </div>
                
                <table>
                    <tr>
                        <th>Filename</th>
                        <th>Duration</th>
                        <th>Frames</th>
                        <th>Fake %</th>
                        <th>Verdict</th>
                        <th>Confidence</th>
                    </tr>
        """
        
        for r in self.results:
            row_class = r.verdict.lower()
            html += f"""
                    <tr class="{row_class}">
                        <td>{r.video_info.filename}</td>
                        <td>{r.video_info.duration:.2f}s</td>
                        <td>{r.frames_analyzed}</td>
                        <td>{r.fake_percentage:.1f}%</td>
                        <td><strong>{r.verdict}</strong></td>
                        <td>{r.confidence_avg:.2%}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Batch report saved to {output_path}")


# ============================================
# VIDEO EDITING UTILITIES
# ============================================

class VideoEditor:
    """Video editing utilities"""
    
    @staticmethod
    def trim_video(input_path: str, output_path: str, start_time: float, end_time: float):
        """Trim video to specified time range"""
        try:
            (
                ffmpeg
                .input(input_path, ss=start_time, to=end_time)
                .output(output_path, c='copy')
                .run(overwrite_output=True)
            )
            logger.info(f"Video trimmed: {output_path}")
        except FFmpegError as e:
            logger.error(f"Trim error: {str(e)}")
    
    @staticmethod
    def extract_audio(input_path: str, output_path: str):
        """Extract audio from video"""
        try:
            (
                ffmpeg
                .input(input_path)
                .output(output_path, acodec='mp3')
                .run(overwrite_output=True)
            )
            logger.info(f"Audio extracted: {output_path}")
        except FFmpegError as e:
            logger.error(f"Audio extraction error: {str(e)}")
    
    @staticmethod
    def compress_video(input_path: str, output_path: str, target_size_mb: float):
        """Compress video to target size"""
        try:
            # Calculate bitrate
            probe = ffmpeg.probe(input_path)
            duration = float(probe['format']['duration'])
            target_bitrate = int((target_size_mb * 8 * 1024 * 1024) / duration)
            
            (
                ffmpeg
                .input(input_path)
                .output(output_path, video_bitrate=f"{target_bitrate}k")
                .run(overwrite_output=True)
            )
            logger.info(f"Video compressed: {output_path}")
        except FFmpegError as e:
            logger.error(f"Compression error: {str(e)}")
    
    @staticmethod
    def create_thumbnail(input_path: str, output_path: str, time: float = 1.0):
        """Create thumbnail from video"""
        try:
            (
                ffmpeg
                .input(input_path, ss=time)
                .output(output_path, vframes=1)
                .run(overwrite_output=True)
            )
            logger.info(f"Thumbnail created: {output_path}")
        except FFmpegError as e:
            logger.error(f"Thumbnail error: {str(e)}")


# ============================================
# FACTORY CLASS
# ============================================

class VideoProcessorFactory:
    """Factory for creating video processors"""
    
    @staticmethod
    def create_processor(**kwargs) -> VideoProcessor:
        """Create video processor"""
        return VideoProcessor(**kwargs)
    
    @staticmethod
    def create_batch_processor(processor: Optional[VideoProcessor] = None, 
                              max_workers: int = 4) -> BatchVideoProcessor:
        """Create batch processor"""
        if processor is None:
            processor = VideoProcessor()
        return BatchVideoProcessor(processor, max_workers)
    
    @staticmethod
    def create_capture(source: Union[str, int] = 0, **kwargs) -> VideoCapture:
        """Create video capture"""
        return VideoCapture(source, **kwargs)
    
    @staticmethod
    def create_writer(output_path: str, fps: float, frame_size: Tuple[int, int]) -> VideoWriter:
        """Create video writer"""
        return VideoWriter(output_path, fps, frame_size)


# ============================================
# TESTING FUNCTION
# ============================================

def test_video_processor():
    """Test video processor module"""
    print("=" * 60)
    print("TESTING VIDEO PROCESSOR MODULE")
    print("=" * 60)
    
    # Create test video
    print("\n1️⃣ Creating test video...")
    test_frames = 30
    test_video = "test_video.mp4"
    
    out = cv2.VideoWriter(test_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
    for i in range(test_frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    print(f"✅ Test video created: {test_video}")
    
    try:
        # Test video capture
        print("\n2️⃣ Testing VideoCapture...")
        cap = VideoCapture(test_video)
        if cap.open():
            info = cap.info
            print(f"✅ Video opened: {info.duration:.2f}s, {info.fps:.2f}fps")
            
            # Read frames
            frames = []
            for i in range(5):
                ret, frame, ts = cap.read_frame()
                if ret:
                    frames.append(frame)
            print(f"✅ Read {len(frames)} frames")
            
            cap.release()
        
        # Test video processor
        print("\n3️⃣ Testing VideoProcessor...")
        processor = VideoProcessor()
        
        # Analyze video
        result = processor.analyze_video(test_video, max_frames=10)
        print(f"✅ Video analyzed")
        print(f"   Verdict: {result.verdict}")
        print(f"   Fake frames: {result.fake_frames}/{result.frames_analyzed}")
        print(f"   Confidence: {result.confidence_avg:.3f}")
        
        # Test keyframe extraction
        print("\n4️⃣ Testing keyframe extraction...")
        keyframes = processor.extract_key_frames(test_video, method='uniform', max_frames=5)
        print(f"✅ Extracted {len(keyframes)} keyframes")
        
        # Generate report
        print("\n5️⃣ Generating report...")
        processor.generate_video_report(result, "test_report.html")
        print(f"✅ Report generated: test_report.html")
        
        print("\n" + "=" * 60)
        print("✅ VIDEO PROCESSOR TEST PASSED!")
        print("=" * 60)
        
    finally:
        # Clean up
        if os.path.exists(test_video):
            os.remove(test_video)
        if os.path.exists("test_report.html"):
            os.remove("test_report.html")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    test_video_processor()
    
    print("\n📝 Example usage:")
    print("""
    from app.utils.video_processor import VideoProcessor, VideoProcessorFactory
    
    # Create processor
    processor = VideoProcessorFactory.create_processor()
    
    # Analyze video
    result = processor.analyze_video('path/to/video.mp4', sample_rate=2, max_frames=100)
    
    print(result.summary())
    
    # Generate report
    processor.generate_video_report(result, 'analysis_report.html')
    
    # Live processing
    for frame_result in processor.process_live_stream(0):  # camera 0
        if frame_result.is_fake:
            print(f"⚠️ Deepfake detected! Confidence: {frame_result.confidence:.2f}")
    """)