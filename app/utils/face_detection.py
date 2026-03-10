# Placeholder file for face_detection.py
# app/utils/face_detection.py
"""
Face Detection Module for Deepfake Analysis
Detects and extracts faces from images and video frames
Supports multiple face detection backends: MTCNN, MediaPipe, OpenCV, Dlib
Python 3.13+ Compatible
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== IMAGE PROCESSING ==========
import cv2
from PIL import Image
import numpy as np

# ========== FACE DETECTION LIBRARIES ==========
# MTCNN
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logging.warning("MTCNN not installed. Install with: pip install mtcnn")

# MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not installed. Install with: pip install mediapipe")

# Dlib
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logging.warning("Dlib not installed. Install with: pip install dlib")

# InsightFace
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not installed. Install with: pip install insightface")

# Face Recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("Face Recognition not installed. Install with: pip install face-recognition")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class FaceBox:
    """Face bounding box"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence
        }


@dataclass
class FacialLandmarks:
    """Facial landmarks"""
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    nose: Tuple[int, int]
    mouth_left: Tuple[int, int]
    mouth_right: Tuple[int, int]
    chin: Tuple[int, int]
    left_eyebrow: List[Tuple[int, int]] = None
    right_eyebrow: List[Tuple[int, int]] = None
    landmarks_68: List[Tuple[int, int]] = None  # Full 68-point landmarks
    
    def to_dict(self) -> Dict:
        return {
            'left_eye': self.left_eye,
            'right_eye': self.right_eye,
            'nose': self.nose,
            'mouth_left': self.mouth_left,
            'mouth_right': self.mouth_right,
            'chin': self.chin
        }


@dataclass
class Face:
    """Detected face with all attributes"""
    box: FaceBox
    landmarks: Optional[FacialLandmarks] = None
    confidence: float = 1.0
    face_image: Optional[np.ndarray] = None
    face_embedding: Optional[np.ndarray] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    is_real: Optional[bool] = None  # For liveness detection
    detection_method: str = "unknown"
    
    def to_dict(self) -> Dict:
        result = {
            'box': self.box.to_dict(),
            'confidence': self.confidence,
            'detection_method': self.detection_method
        }
        if self.landmarks:
            result['landmarks'] = self.landmarks.to_dict()
        if self.age:
            result['age'] = self.age
        if self.gender:
            result['gender'] = self.gender
        if self.emotion:
            result['emotion'] = self.emotion
        if self.is_real is not None:
            result['is_real'] = self.is_real
        return result


@dataclass
class FaceDetectionResult:
    """Complete face detection result"""
    faces: List[Face]
    image_shape: Tuple[int, int, int]
    num_faces: int
    detection_time: float
    method_used: str
    preprocessing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'num_faces': self.num_faces,
            'faces': [f.to_dict() for f in self.faces],
            'image_shape': self.image_shape,
            'detection_time': self.detection_time,
            'method_used': self.method_used
        }


# ============================================
# MTCNN DETECTOR
# ============================================

class MTCNNDetector:
    """
    Face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
    Good accuracy, works well for most use cases
    """
    
    def __init__(self, min_face_size: int = 20, thresholds: List[float] = None):
        """
        Initialize MTCNN detector
        
        Args:
            min_face_size: Minimum face size in pixels
            thresholds: Detection thresholds for each stage [p_threshold, r_threshold, o_threshold]
        """
        if not MTCNN_AVAILABLE:
            raise ImportError("MTCNN not installed")
        
        self.min_face_size = min_face_size
        self.thresholds = thresholds or [0.6, 0.7, 0.7]
        
        self.detector = MTCNN(
            min_face_size=min_face_size,
            thresholds=self.thresholds
        )
        
        logger.info(f"MTCNN detector initialized (min_face_size={min_face_size})")
    
    def detect(self, image: np.ndarray) -> List[Face]:
        """
        Detect faces in image
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            List of Face objects
        """
        if image is None:
            return []
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            # Assume BGR if from OpenCV
            if image[0, 0, 2] < image[0, 0, 0]:  # Rough BGR check
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces
            results = self.detector.detect_faces(image)
            
            faces = []
            for result in results:
                # Get bounding box
                x, y, w, h = result['box']
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                box = FaceBox(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=result['confidence']
                )
                
                # Get landmarks if available
                landmarks = None
                if 'keypoints' in result:
                    keypoints = result['keypoints']
                    landmarks = FacialLandmarks(
                        left_eye=tuple(keypoints.get('left_eye', (0, 0))),
                        right_eye=tuple(keypoints.get('right_eye', (0, 0))),
                        nose=tuple(keypoints.get('nose', (0, 0))),
                        mouth_left=tuple(keypoints.get('mouth_left', (0, 0))),
                        mouth_right=tuple(keypoints.get('mouth_right', (0, 0))),
                        chin=(0, 0)  # MTCNN doesn't provide chin
                    )
                
                # Extract face image
                face_img = image[y:y+h, x:x+w]
                
                face = Face(
                    box=box,
                    landmarks=landmarks,
                    confidence=result['confidence'],
                    face_image=face_img,
                    detection_method="mtcnn"
                )
                faces.append(face)
            
            return faces
            
        except Exception as e:
            logger.error(f"MTCNN detection error: {str(e)}")
            return []
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Face]]:
        """Detect faces in multiple images"""
        return [self.detect(img) for img in images]


# ============================================
# MEDIAPIPE DETECTOR
# ============================================

class MediaPipeDetector:
    """
    Face detection using MediaPipe
    Fast, good for real-time applications
    """
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize MediaPipe detector
        
        Args:
            min_detection_confidence: Minimum confidence for detection
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not installed")
        
        self.min_detection_confidence = min_detection_confidence
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range, 1 for long-range
            min_detection_confidence=min_detection_confidence
        )
        
        logger.info(f"MediaPipe detector initialized (confidence={min_detection_confidence})")
    
    def detect(self, image: np.ndarray) -> List[Face]:
        """
        Detect faces in image
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            List of Face objects
        """
        if image is None:
            return []
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        try:
            # Process image
            results = self.detector.process(image)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure within bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width <= 0 or height <= 0:
                        continue
                    
                    box = FaceBox(
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        confidence=detection.score[0]
                    )
                    
                    # Get landmarks
                    landmarks = None
                    if detection.location_data.relative_keypoints:
                        keypoints = detection.location_data.relative_keypoints
                        if len(keypoints) >= 6:
                            landmarks = FacialLandmarks(
                                left_eye=(int(keypoints[0].x * w), int(keypoints[0].y * h)),
                                right_eye=(int(keypoints[1].x * w), int(keypoints[1].y * h)),
                                nose=(int(keypoints[2].x * w), int(keypoints[2].y * h)),
                                mouth_left=(int(keypoints[3].x * w), int(keypoints[3].y * h)),
                                mouth_right=(int(keypoints[4].x * w), int(keypoints[4].y * h)),
                                chin=(int(keypoints[5].x * w), int(keypoints[5].y * h))
                            )
                    
                    # Extract face image
                    face_img = image[y:y+height, x:x+width]
                    
                    face = Face(
                        box=box,
                        landmarks=landmarks,
                        confidence=detection.score[0],
                        face_image=face_img,
                        detection_method="mediapipe"
                    )
                    faces.append(face)
            
            return faces
            
        except Exception as e:
            logger.error(f"MediaPipe detection error: {str(e)}")
            return []


# ============================================
# OPENCV DETECTOR (HAAR CASCADE)
# ============================================

class OpenCVDetector:
    """
    Face detection using OpenCV Haar Cascades
    Fast but less accurate, good for simple use cases
    """
    
    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize OpenCV detector
        
        Args:
            cascade_path: Path to Haar cascade XML file
        """
        if cascade_path is None:
            # Use default cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")
        
        logger.info(f"OpenCV detector initialized with {cascade_path}")
    
    def detect(self, image: np.ndarray, scale_factor: float = 1.1, 
               min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[Face]:
        """
        Detect faces in image
        
        Args:
            image: Image as numpy array
            scale_factor: Scale factor for detection
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size
        
        Returns:
            List of Face objects
        """
        if image is None:
            return []
        
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        try:
            # Detect faces
            boxes = self.cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
            
            faces = []
            for (x, y, w, h) in boxes:
                box = FaceBox(x=x, y=y, width=w, height=h, confidence=1.0)
                
                # Extract face image
                face_img = image[y:y+h, x:x+w]
                
                face = Face(
                    box=box,
                    confidence=1.0,
                    face_image=face_img,
                    detection_method="opencv"
                )
                faces.append(face)
            
            return faces
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {str(e)}")
            return []


# ============================================
# INSIGHTFACE DETECTOR
# ============================================

class InsightFaceDetector:
    """
    Face detection using InsightFace
    Very accurate, provides embeddings and attributes
    """
    
    def __init__(self, ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)):
        """
        Initialize InsightFace detector
        
        Args:
            ctx_id: GPU context ID (-1 for CPU)
            det_size: Detection image size
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not installed")
        
        self.ctx_id = ctx_id
        self.det_size = det_size
        
        # Initialize app
        self.app = FaceAnalysis(
            name='buffalo_l',
            root='~/.insightface',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        
        logger.info(f"InsightFace detector initialized (ctx_id={ctx_id}, det_size={det_size})")
    
    def detect(self, image: np.ndarray) -> List[Face]:
        """
        Detect faces in image
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            List of Face objects with embeddings
        """
        if image is None:
            return []
        
        try:
            # Detect faces
            faces_data = self.app.get(image)
            
            faces = []
            for face_data in faces_data:
                # Get bounding box
                bbox = face_data.bbox.astype(int)
                x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                w = x2 - x
                h = y2 - y
                
                box = FaceBox(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=float(face_data.det_score)
                )
                
                # Get landmarks
                landmarks = None
                if hasattr(face_data, 'landmark') and face_data.landmark is not None:
                    lm = face_data.landmark.astype(int)
                    if len(lm) >= 5:
                        landmarks = FacialLandmarks(
                            left_eye=tuple(lm[0]),
                            right_eye=tuple(lm[1]),
                            nose=tuple(lm[2]),
                            mouth_left=tuple(lm[3]),
                            mouth_right=tuple(lm[4]),
                            chin=(0, 0)
                        )
                
                # Extract face image
                face_img = image[y:y2, x:x2]
                
                # Get embedding
                embedding = None
                if hasattr(face_data, 'embedding'):
                    embedding = face_data.embedding
                
                # Get attributes
                age = None
                gender = None
                if hasattr(face_data, 'age'):
                    age = int(face_data.age)
                if hasattr(face_data, 'gender'):
                    gender = 'male' if face_data.gender == 1 else 'female'
                
                face = Face(
                    box=box,
                    landmarks=landmarks,
                    confidence=float(face_data.det_score),
                    face_image=face_img,
                    face_embedding=embedding,
                    age=age,
                    gender=gender,
                    detection_method="insightface"
                )
                faces.append(face)
            
            return faces
            
        except Exception as e:
            logger.error(f"InsightFace detection error: {str(e)}")
            return []


# ============================================
# DLIB DETECTOR
# ============================================

class DlibDetector:
    """
    Face detection using Dlib (HOG + SVM or CNN)
    Good accuracy, provides 68-point landmarks
    """
    
    def __init__(self, use_cnn: bool = False, model_path: Optional[str] = None):
        """
        Initialize Dlib detector
        
        Args:
            use_cnn: Use CNN detector (slower but more accurate)
            model_path: Path to CNN model (if use_cnn=True)
        """
        if not DLIB_AVAILABLE:
            raise ImportError("Dlib not installed")
        
        self.use_cnn = use_cnn
        
        if use_cnn:
            if model_path:
                self.detector = dlib.cnn_face_detection_model_v1(model_path)
            else:
                # Use default CNN model
                self.detector = dlib.get_frontal_face_detector()
        else:
            self.detector = dlib.get_frontal_face_detector()
        
        # Load landmark predictor
        try:
            landmark_path = Path(__file__).parent / "shape_predictor_68_face_landmarks.dat"
            if not landmark_path.exists():
                # Download from dlib's repository
                import urllib.request
                url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                logger.info(f"Downloading landmark predictor...")
                urllib.request.urlretrieve(url, str(landmark_path) + ".bz2")
                import bz2
                with bz2.BZ2File(str(landmark_path) + ".bz2", 'rb') as f_in:
                    with open(landmark_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                logger.info(f"Landmark predictor downloaded")
            
            self.predictor = dlib.shape_predictor(str(landmark_path))
        except Exception as e:
            logger.warning(f"Could not load landmark predictor: {str(e)}")
            self.predictor = None
        
        logger.info(f"Dlib detector initialized (use_cnn={use_cnn})")
    
    def detect(self, image: np.ndarray) -> List[Face]:
        """
        Detect faces in image
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            List of Face objects with 68-point landmarks
        """
        if image is None:
            return []
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        try:
            # Detect faces
            if self.use_cnn:
                detections = self.detector(image, 1)
                boxes = [d.rect for d in detections]
                confidences = [d.confidence for d in detections]
            else:
                boxes = self.detector(image, 1)
                confidences = [1.0] * len(boxes)
            
            faces = []
            for i, box in enumerate(boxes):
                x = max(0, box.left())
                y = max(0, box.top())
                w = min(box.right() - x, image.shape[1] - x)
                h = min(box.bottom() - y, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                face_box = FaceBox(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=confidences[i] if i < len(confidences) else 1.0
                )
                
                # Get landmarks if predictor is available
                landmarks = None
                if self.predictor:
                    shape = self.predictor(image, box)
                    if shape.num_parts >= 68:
                        # Extract key landmarks
                        points = [(shape.part(j).x, shape.part(j).y) for j in range(68)]
                        landmarks = FacialLandmarks(
                            left_eye=points[36],  # Left eye corner
                            right_eye=points[45],  # Right eye corner
                            nose=points[30],  # Nose tip
                            mouth_left=points[48],  # Mouth left corner
                            mouth_right=points[54],  # Mouth right corner
                            chin=points[8],  # Chin
                            left_eyebrow=points[17:22],
                            right_eyebrow=points[22:27],
                            landmarks_68=points
                        )
                
                # Extract face image
                face_img = image[y:y+h, x:x+w]
                
                face = Face(
                    box=face_box,
                    landmarks=landmarks,
                    confidence=face_box.confidence,
                    face_image=face_img,
                    detection_method="dlib"
                )
                faces.append(face)
            
            return faces
            
        except Exception as e:
            logger.error(f"Dlib detection error: {str(e)}")
            return []


# ============================================
# FACE RECOGNITION DETECTOR
# ============================================

class FaceRecognitionDetector:
    """
    Face detection using face_recognition library
    Simple API, good for face comparison
    """
    
    def __init__(self):
        """Initialize face_recognition detector"""
        if not FACE_RECOGNITION_AVAILABLE:
            raise ImportError("face_recognition not installed")
        
        logger.info("FaceRecognition detector initialized")
    
    def detect(self, image: np.ndarray) -> List[Face]:
        """
        Detect faces in image
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            List of Face objects
        """
        if image is None:
            return []
        
        try:
            # Detect face locations
            locations = face_recognition.face_locations(image)
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image, locations)
            
            # Get landmarks
            landmarks_list = face_recognition.face_landmarks(image, locations)
            
            faces = []
            for i, (loc, encoding) in enumerate(zip(locations, encodings)):
                top, right, bottom, left = loc
                
                x = left
                y = top
                w = right - left
                h = bottom - top
                
                box = FaceBox(x=x, y=y, width=w, height=h, confidence=1.0)
                
                # Convert landmarks
                landmarks = None
                if i < len(landmarks_list):
                    lm = landmarks_list[i]
                    landmarks = FacialLandmarks(
                        left_eye=lm.get('left_eye', [(0,0)])[0] if lm.get('left_eye') else (0,0),
                        right_eye=lm.get('right_eye', [(0,0)])[0] if lm.get('right_eye') else (0,0),
                        nose=lm.get('nose_tip', [(0,0)])[0] if lm.get('nose_tip') else (0,0),
                        mouth_left=lm.get('bottom_lip', [(0,0)])[0] if lm.get('bottom_lip') else (0,0),
                        mouth_right=lm.get('bottom_lip', [(0,0)])[-1] if lm.get('bottom_lip') else (0,0),
                        chin=(0,0)
                    )
                
                # Extract face image
                face_img = image[y:bottom, x:right]
                
                face = Face(
                    box=box,
                    landmarks=landmarks,
                    confidence=1.0,
                    face_image=face_img,
                    face_embedding=encoding,
                    detection_method="face_recognition"
                )
                faces.append(face)
            
            return faces
            
        except Exception as e:
            logger.error(f"FaceRecognition detection error: {str(e)}")
            return []
    
    def compare_faces(self, face1: Face, face2: Face, threshold: float = 0.6) -> bool:
        """Compare two faces using embeddings"""
        if face1.face_embedding is None or face2.face_embedding is None:
            return False
        
        distance = np.linalg.norm(face1.face_embedding - face2.face_embedding)
        return distance < threshold


# ============================================
# FACE DETECTION ENSEMBLE
# ============================================

class FaceDetectionEnsemble:
    """
    Ensemble of multiple face detectors for maximum accuracy
    Combines results from multiple detection methods
    """
    
    def __init__(self, detectors: List[str] = None):
        """
        Initialize face detection ensemble
        
        Args:
            detectors: List of detector names to use
                      Options: 'mtcnn', 'mediapipe', 'opencv', 'insightface', 'dlib', 'face_recognition'
        """
        if detectors is None:
            detectors = ['mtcnn', 'mediapipe', 'insightface']
        
        self.detectors = []
        self.detector_names = []
        
        for name in detectors:
            try:
                if name == 'mtcnn' and MTCNN_AVAILABLE:
                    self.detectors.append(MTCNNDetector())
                    self.detector_names.append('mtcnn')
                elif name == 'mediapipe' and MEDIAPIPE_AVAILABLE:
                    self.detectors.append(MediaPipeDetector())
                    self.detector_names.append('mediapipe')
                elif name == 'opencv':
                    self.detectors.append(OpenCVDetector())
                    self.detector_names.append('opencv')
                elif name == 'insightface' and INSIGHTFACE_AVAILABLE:
                    self.detectors.append(InsightFaceDetector())
                    self.detector_names.append('insightface')
                elif name == 'dlib' and DLIB_AVAILABLE:
                    self.detectors.append(DlibDetector())
                    self.detector_names.append('dlib')
                elif name == 'face_recognition' and FACE_RECOGNITION_AVAILABLE:
                    self.detectors.append(FaceRecognitionDetector())
                    self.detector_names.append('face_recognition')
            except Exception as e:
                logger.warning(f"Failed to load detector {name}: {str(e)}")
        
        logger.info(f"FaceDetectionEnsemble initialized with {len(self.detectors)} detectors: {self.detector_names}")
    
    def detect(self, image: np.ndarray, iou_threshold: float = 0.5) -> FaceDetectionResult:
        """
        Detect faces using ensemble of detectors
        
        Args:
            image: Input image
            iou_threshold: IoU threshold for merging detections
        
        Returns:
            FaceDetectionResult with merged detections
        """
        start_time = time.time()
        
        if image is None:
            return FaceDetectionResult(
                faces=[],
                image_shape=(0,0,0),
                num_faces=0,
                detection_time=0,
                method_used="ensemble"
            )
        
        # Store original shape
        orig_shape = image.shape
        
        # Get detections from all detectors
        all_faces = []
        detection_times = []
        
        for i, detector in enumerate(self.detectors):
            det_start = time.time()
            faces = detector.detect(image)
            det_time = time.time() - det_start
            detection_times.append(det_time)
            
            for face in faces:
                face.detection_method = self.detector_names[i]
                all_faces.append(face)
            
            logger.debug(f"{self.detector_names[i]}: {len(faces)} faces in {det_time:.3f}s")
        
        # Merge overlapping detections using NMS
        merged_faces = self._non_max_suppression(all_faces, iou_threshold)
        
        total_time = time.time() - start_time
        
        return FaceDetectionResult(
            faces=merged_faces,
            image_shape=orig_shape,
            num_faces=len(merged_faces),
            detection_time=total_time,
            method_used=f"ensemble_{'_'.join(self.detector_names)}"
        )
    
    def _non_max_suppression(self, faces: List[Face], iou_threshold: float) -> List[Face]:
        """Non-maximum suppression to merge overlapping detections"""
        if not faces:
            return []
        
        # Sort by confidence
        faces = sorted(faces, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while faces:
            # Take the face with highest confidence
            best = faces.pop(0)
            keep.append(best)
            
            # Remove faces with high IoU
            to_remove = []
            for i, face in enumerate(faces):
                iou = self._compute_iou(best.box, face.box)
                if iou > iou_threshold:
                    to_remove.append(i)
            
            # Remove in reverse order
            for i in reversed(to_remove):
                faces.pop(i)
        
        return keep
    
    def _compute_iou(self, box1: FaceBox, box2: FaceBox) -> float:
        """Compute Intersection over Union for two boxes"""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.area
        area2 = box2.area
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


# ============================================
# FACE PREPROCESSOR
# ============================================

class FacePreprocessor:
    """
    Preprocess faces for deepfake detection
    Alignment, normalization, augmentation
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize face preprocessor
        
        Args:
            target_size: Target size for face images
        """
        self.target_size = target_size
    
    def align_face(self, face: Face, image: np.ndarray) -> np.ndarray:
        """
        Align face using landmarks
        
        Args:
            face: Face object with landmarks
            image: Original image
        
        Returns:
            Aligned face image
        """
        if face.landmarks is None:
            # No landmarks, just crop and resize
            x, y, w, h = face.box.x, face.box.y, face.box.width, face.box.height
            face_img = image[y:y+h, x:x+w]
            return cv2.resize(face_img, self.target_size)
        
        # Get eye positions
        left_eye = face.landmarks.left_eye
        right_eye = face.landmarks.right_eye
        
        # Calculate angle for alignment
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get center of face
        center = face.box.center
        
        # Get rotation matrix
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        aligned = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
        
        # Crop face region
        x, y, w, h = face.box.x, face.box.y, face.box.width, face.box.height
        face_img = aligned[y:y+h, x:x+w]
        
        # Resize
        return cv2.resize(face_img, self.target_size)
    
    def normalize(self, face_img: np.ndarray) -> np.ndarray:
        """Normalize face image to [0,1] range"""
        if face_img.max() > 1.0:
            face_img = face_img.astype(np.float32) / 255.0
        return face_img
    
    def standardize(self, face_img: np.ndarray) -> np.ndarray:
        """Standardize face image (zero mean, unit variance)"""
        mean = np.mean(face_img)
        std = np.std(face_img)
        if std > 0:
            return (face_img - mean) / std
        return face_img
    
    def preprocess_for_model(self, face: Face, image: np.ndarray, 
                            align: bool = True, normalize: bool = True) -> np.ndarray:
        """
        Preprocess face for model input
        
        Returns:
            Preprocessed face image
        """
        if align:
            face_img = self.align_face(face, image)
        else:
            x, y, w, h = face.box.x, face.box.y, face.box.width, face.box.height
            face_img = image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, self.target_size)
        
        if normalize:
            face_img = self.normalize(face_img)
        
        return face_img


# ============================================
# FACT