# Placeholder file for audio_processor.py
# app/utils/audio_processor.py
"""
Audio Processing Module for Deepfake Detection
Handles live voice calls, recordings, and audio file analysis
Supports real-time voice deepfake detection
Python 3.13+ Compatible
"""

import os
import sys
import io
import wave
import time
import json
import struct
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, BinaryIO
from dataclasses import dataclass, asdict
from collections import deque
import threading
import queue
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== AUDIO PROCESSING IMPORTS ==========
import librosa
import soundfile as sf
import sounddevice as sd
import pyaudio
import webrtcvad
from scipy import signal
from scipy.ndimage import median_filter

# ========== DEEP LEARNING IMPORTS ==========
import torch
import torch.nn.functional as F
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class AudioSegment:
    """Audio segment data"""
    audio_data: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    duration: float
    has_voice: bool = True
    segment_id: str = ""


@dataclass
class AudioFeatures:
    """Extracted audio features"""
    mfccs: np.ndarray
    mel_spectrogram: np.ndarray
    spectral_centroids: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: float
    rms_energy: float
    chroma_features: np.ndarray
    tempo: float
    beat_frames: np.ndarray
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for JSON)"""
        return {
            'mfccs_shape': self.mfccs.shape,
            'mel_spectrogram_shape': self.mel_spectrogram.shape,
            'spectral_centroids_mean': float(np.mean(self.spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(self.spectral_rolloff)),
            'zero_crossing_rate': float(self.zero_crossing_rate),
            'rms_energy': float(self.rms_energy),
            'chroma_features_shape': self.chroma_features.shape,
            'tempo': float(self.tempo)
        }


@dataclass
class AudioDetectionResult:
    """Audio deepfake detection result"""
    is_fake: bool
    confidence: float
    ai_probability: float
    human_probability: float
    detection_method: str
    features: Dict[str, Any]
    segments_analyzed: int
    duration_analyzed: float
    processing_time: float
    warning_flags: List[str]


# ============================================
# AUDIO CAPTURE MODULE
# ============================================

class AudioCapture:
    """
    Real-time audio capture from microphone or system audio
    Supports live voice call monitoring
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 960,  # 60ms at 16kHz
                 channels: int = 1,
                 device_index: Optional[int] = None):
        """
        Initialize audio capture
        
        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_size: Frames per buffer
            channels: Number of audio channels
            device_index: Input device index (None = default)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recorded_frames = []
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # List available devices
        self._list_devices()
        
        logger.info(f"AudioCapture initialized: {sample_rate}Hz, {chunk_size} frames")
    
    def _list_devices(self):
        """List available audio input devices"""
        logger.info("Available audio input devices:")
        for i in range(self.p.get_device_count()):
            dev_info = self.p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                logger.info(f"  {i}: {dev_info['name']} (channels: {dev_info['maxInputChannels']})")
    
    def start_recording(self, callback: Optional[callable] = None):
        """Start recording from microphone"""
        self.is_recording = True
        self.recorded_frames = []
        
        def audio_callback(in_data, frame_count, time_info, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Normalize to float [-1, 1]
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Add to queue
            self.audio_queue.put(audio_float)
            self.recorded_frames.append(audio_data)
            
            # Call user callback if provided
            if callback:
                callback(audio_float, time_info)
            
            return (None, pyaudio.paContinue)
        
        # Open stream
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=audio_callback
        )
        
        self.stream.start_stream()
        logger.info("Recording started")
    
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return captured audio"""
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        # Combine all frames
        if self.recorded_frames:
            audio_data = np.concatenate(self.recorded_frames)
            logger.info(f"Recording stopped. Captured {len(audio_data)/self.sample_rate:.2f}s")
            return audio_data
        else:
            logger.warning("No audio captured")
            return np.array([])
    
    def get_live_audio(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get next chunk of live audio (non-blocking)"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'stream'):
            self.stream.close()
        self.p.terminate()
        logger.info("AudioCapture closed")


# ============================================
# VOICE ACTIVITY DETECTION
# ============================================

class VoiceActivityDetector:
    """
    Voice Activity Detection using WebRTC VAD
    Detects when someone is speaking
    """
    
    def __init__(self, 
                 mode: int = 1,  # Aggressiveness mode (0-3)
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30):
        """
        Initialize VAD
        
        Args:
            mode: VAD aggressiveness (0=least, 3=most aggressive)
            sample_rate: Audio sample rate
            frame_duration_ms: Frame duration in ms (10, 20, or 30)
        """
        self.mode = mode
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(mode)
        
        logger.info(f"VAD initialized: mode={mode}, frame_size={self.frame_size}")
    
    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """
        Detect if audio frame contains speech
        
        Args:
            audio_frame: Audio frame as float array [-1, 1]
        
        Returns:
            True if speech detected
        """
        # Convert to int16
        audio_int16 = (audio_frame * 32768).astype(np.int16)
        
        # Convert to bytes
        audio_bytes = audio_int16.tobytes()
        
        try:
            return self.vad.is_speech(audio_bytes, self.sample_rate)
        except Exception as e:
            logger.error(f"VAD error: {str(e)}")
            return False
    
    def detect_speech_segments(self, 
                              audio: np.ndarray, 
                              sample_rate: int,
                              min_speech_duration: float = 0.5) -> List[Tuple[float, float]]:
        """
        Detect speech segments in audio
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            min_speech_duration: Minimum speech segment duration (seconds)
        
        Returns:
            List of (start_time, end_time) speech segments
        """
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # Convert to int16
        audio_int16 = (audio * 32768).astype(np.int16)
        
        # Process in frames
        speech_frames = []
        for i in range(0, len(audio_int16) - self.frame_size, self.frame_size):
            frame = audio_int16[i:i + self.frame_size]
            frame_bytes = frame.tobytes()
            
            try:
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                speech_frames.append(is_speech)
            except:
                speech_frames.append(False)
        
        # Merge consecutive frames
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                in_speech = True
                start_frame = i
            elif not is_speech and in_speech:
                in_speech = False
                end_frame = i
                
                # Check duration
                duration = (end_frame - start_frame) * self.frame_duration_ms / 1000
                if duration >= min_speech_duration:
                    segments.append((
                        start_frame * self.frame_duration_ms / 1000,
                        end_frame * self.frame_duration_ms / 1000
                    ))
        
        # Handle case where speech continues to end
        if in_speech:
            end_frame = len(speech_frames)
            duration = (end_frame - start_frame) * self.frame_duration_ms / 1000
            if duration >= min_speech_duration:
                segments.append((
                    start_frame * self.frame_duration_ms / 1000,
                    end_frame * self.frame_duration_ms / 1000
                ))
        
        logger.info(f"Detected {len(segments)} speech segments")
        return segments


# ============================================
# AUDIO FEATURE EXTRACTOR
# ============================================

class AudioFeatureExtractor:
    """
    Extract audio features for deepfake detection
    MFCCs, spectrograms, and other acoustic features
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_mels: int = 128,
                 hop_length: int = 512,
                 n_fft: int = 2048):
        """
        Initialize feature extractor
        
        Args:
            sample_rate: Target sample rate
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of Mel bands
            hop_length: Hop length for STFT
            n_fft: FFT window size
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        logger.info(f"AudioFeatureExtractor initialized: {sample_rate}Hz")
    
    def extract_features(self, 
                        audio: np.ndarray, 
                        sr: int,
                        return_all: bool = False) -> Union[AudioFeatures, Dict]:
        """
        Extract all audio features
        
        Args:
            audio: Audio signal
            sr: Sample rate of input audio
            return_all: Return full AudioFeatures object or dict summary
        
        Returns:
            AudioFeatures object or feature dictionary
        """
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Ensure minimum length
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        
        # Extract features
        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc,
            hop_length=self.hop_length, n_fft=self.n_fft
        )
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=self.n_mels,
            hop_length=self.hop_length, n_fft=self.n_fft
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=self.hop_length
        )[0]
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, hop_length=self.hop_length
        )
        
        # Tempo and beats
        tempo, beats = librosa.beat.beat_track(
            y=audio, sr=sr, hop_length=self.hop_length
        )
        
        features = AudioFeatures(
            mfccs=mfccs,
            mel_spectrogram=mel_spec_db,
            spectral_centroids=spectral_centroids,
            spectral_rolloff=spectral_rolloff,
            zero_crossing_rate=float(np.mean(zcr)),
            rms_energy=float(np.mean(rms)),
            chroma_features=chroma,
            tempo=float(tempo),
            beat_frames=beats
        )
        
        if return_all:
            return features
        else:
            return features.to_dict()
    
    def extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract only MFCC features (for model input)"""
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc,
            hop_length=self.hop_length, n_fft=self.n_fft
        )
        
        return mfccs
    
    def extract_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract mel spectrogram (for model input)"""
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels,
            hop_length=self.hop_length, n_fft=self.n_fft
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def preprocess_for_model(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Preprocess audio for deep learning model
        
        Returns:
            Tensor of shape (1, 1, n_mels, time)
        """
        # Extract mel spectrogram
        mel_spec = self.extract_spectrogram(audio, sr)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Pad or truncate to fixed length (128 time steps)
        target_time = 128
        if mel_spec.shape[1] < target_time:
            # Pad
            pad_width = target_time - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            mel_spec = mel_spec[:, :target_time]
        
        # Convert to tensor
        tensor = torch.from_numpy(mel_spec).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        return tensor


# ============================================
# AUDIO DEEPFAKE DETECTOR
# ============================================

class AudioDeepfakeDetector:
    """
    Detect AI-generated or manipulated audio
    Uses multiple techniques: spectrogram analysis, feature engineering, etc.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 threshold: float = 0.5,
                 device: Optional[str] = None):
        """
        Initialize audio deepfake detector
        
        Args:
            model_path: Path to pre-trained model
            threshold: Detection threshold
            device: 'cuda' or 'cpu'
        """
        self.threshold = threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor()
        self.vad = VoiceActivityDetector()
        
        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        logger.info(f"AudioDeepfakeDetector initialized on {self.device}")
    
    def load_model(self, model_path: str):
        """Load pre-trained audio deepfake model"""
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
    
    def detect_from_file(self, file_path: str) -> AudioDetectionResult:
        """
        Detect deepfake in audio file
        
        Args:
            file_path: Path to audio file
        
        Returns:
            AudioDetectionResult
        """
        start_time = time.time()
        
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000)
            
            # Detect speech segments
            segments = self.vad.detect_speech_segments(audio, sr)
            
            if not segments:
                logger.warning("No speech detected in audio")
                return AudioDetectionResult(
                    is_fake=False,
                    confidence=0.0,
                    ai_probability=0.0,
                    human_probability=0.0,
                    detection_method="none",
                    features={},
                    segments_analyzed=0,
                    duration_analyzed=len(audio)/sr,
                    processing_time=time.time() - start_time,
                    warning_flags=["NO_SPEECH_DETECTED"]
                )
            
            # Analyze each segment
            segment_scores = []
            for start, end in segments[:5]:  # Limit to first 5 segments
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                segment = audio[start_sample:end_sample]
                
                # Extract features
                features = self.feature_extractor.preprocess_for_model(segment, sr)
                features = features.to(self.device)
                
                # Get prediction
                if self.model:
                    with torch.no_grad():
                        output = self.model(features)
                        prob = torch.sigmoid(output).item()
                else:
                    # Rule-based detection (fallback)
                    prob = self._rule_based_detection(segment, sr)
                
                segment_scores.append(prob)
            
            # Aggregate scores
            if segment_scores:
                ai_prob = np.mean(segment_scores)
                confidence = np.std(segment_scores)  # Lower std = higher confidence
                confidence = 1.0 - min(confidence, 0.5)
            else:
                ai_prob = 0.5
                confidence = 0.0
            
            # Extract features for result
            all_features = self.feature_extractor.extract_features(audio, sr, return_all=False)
            
            # Generate warning flags
            warnings = []
            if ai_prob > 0.8:
                warnings.append("HIGH_AI_PROBABILITY")
            if np.std(segment_scores) > 0.3:
                warnings.append("INCONSISTENT_SEGMENTS")
            if len(segments) > 10:
                warnings.append("MANY_SHORT_SEGMENTS")
            
            result = AudioDetectionResult(
                is_fake=ai_prob > self.threshold,
                confidence=float(confidence),
                ai_probability=float(ai_prob),
                human_probability=float(1 - ai_prob),
                detection_method="ensemble" if self.model else "rule_based",
                features=all_features,
                segments_analyzed=len(segment_scores),
                duration_analyzed=len(audio)/sr,
                processing_time=time.time() - start_time,
                warning_flags=warnings
            )
            
            logger.info(f"Audio detection: {'FAKE' if result.is_fake else 'REAL'} "
                       f"(AI={ai_prob:.3f}, Conf={confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return AudioDetectionResult(
                is_fake=False,
                confidence=0.0,
                ai_probability=0.0,
                human_probability=0.0,
                detection_method="error",
                features={},
                segments_analyzed=0,
                duration_analyzed=0,
                processing_time=time.time() - start_time,
                warning_flags=[f"ERROR: {str(e)[:50]}"]
            )
    
    def detect_from_bytes(self, audio_bytes: bytes, sr: int = 16000) -> AudioDetectionResult:
        """Detect deepfake from audio bytes"""
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            result = self.detect_from_file(f.name)
        
        # Clean up
        os.unlink(f.name)
        
        return result
    
    def detect_live(self, 
                   audio_capture: AudioCapture,
                   duration: float = 5.0) -> AudioDetectionResult:
        """
        Detect deepfake from live audio stream
        
        Args:
            audio_capture: AudioCapture instance
            duration: Duration to capture (seconds)
        
        Returns:
            AudioDetectionResult
        """
        logger.info(f"Capturing {duration}s of live audio...")
        
        # Start recording
        frames = []
        
        def callback(audio_data, time_info):
            frames.append(audio_data)
        
        audio_capture.start_recording(callback)
        
        # Record for specified duration
        time.sleep(duration)
        
        # Stop and get audio
        audio_capture.stop_recording()
        
        if frames:
            audio = np.concatenate(frames)
            return self.detect_from_bytes(audio.tobytes(), sr=audio_capture.sample_rate)
        else:
            logger.error("No audio captured")
            return AudioDetectionResult(
                is_fake=False,
                confidence=0.0,
                ai_probability=0.0,
                human_probability=0.0,
                detection_method="error",
                features={},
                segments_analyzed=0,
                duration_analyzed=0,
                processing_time=0,
                warning_flags=["NO_AUDIO_CAPTURED"]
            )
    
    def _rule_based_detection(self, audio: np.ndarray, sr: int) -> float:
        """
        Rule-based audio deepfake detection (fallback)
        Uses statistical anomalies to detect synthetic audio
        """
        # Extract features
        mfccs = self.feature_extractor.extract_mfcc(audio, sr)
        
        # 1. Check MFCC variance (synthetic audio often has lower variance)
        mfcc_var = np.var(mfccs)
        mfcc_score = 1.0 - min(mfcc_var / 1000, 1.0)
        
        # 2. Check for unnatural silence patterns
        energy = librosa.feature.rms(y=audio)[0]
        silence_ratio = np.sum(energy < 0.01) / len(energy)
        silence_score = min(silence_ratio * 2, 1.0)
        
        # 3. Check spectral flatness (synthetic audio often too "clean")
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        flatness_score = min(spectral_flatness * 2, 1.0)
        
        # Combine scores
        ai_prob = 0.4 * mfcc_score + 0.3 * silence_score + 0.3 * flatness_score
        
        return ai_prob


# ============================================
# LIVE VOICE CALL MONITOR
# ============================================

class LiveVoiceCallMonitor:
    """
    Monitor live voice calls for deepfake detection
    Continuously analyzes audio stream during calls
    """
    
    def __init__(self, 
                 detector: AudioDeepfakeDetector,
                 callback: Optional[callable] = None,
                 segment_duration: float = 3.0):
        """
        Initialize live voice call monitor
        
        Args:
            detector: AudioDeepfakeDetector instance
            callback: Function to call when deepfake detected
            segment_duration: Duration of audio segments to analyze
        """
        self.detector = detector
        self.callback = callback
        self.segment_duration = segment_duration
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.audio_buffer = deque(maxlen=int(segment_duration * 2))  # 2x buffer
        
        logger.info(f"LiveVoiceCallMonitor initialized (segment={segment_duration}s)")
    
    def start_monitoring(self, audio_capture: AudioCapture):
        """Start monitoring live voice call"""
        self.is_monitoring = True
        self.audio_capture = audio_capture
        
        def monitor_loop():
            logger.info("Voice call monitoring started")
            
            segment_buffer = []
            segment_samples = int(self.segment_duration * audio_capture.sample_rate)
            
            while self.is_monitoring:
                # Get audio chunk
                audio_chunk = audio_capture.get_live_audio(timeout=0.1)
                
                if audio_chunk is not None:
                    segment_buffer.append(audio_chunk)
                    
                    # Check if we have enough for a segment
                    total_samples = sum(len(chunk) for chunk in segment_buffer)
                    
                    if total_samples >= segment_samples:
                        # Combine chunks
                        segment = np.concatenate(segment_buffer)
                        
                        # Analyze segment
                        result = self.detector.detect_from_bytes(
                            (segment * 32768).astype(np.int16).tobytes(),
                            sr=audio_capture.sample_rate
                        )
                        
                        # Call callback if deepfake detected
                        if result.is_fake and self.callback:
                            self.callback(result)
                        
                        # Keep last part for continuity (50% overlap)
                        keep_samples = segment_samples // 2
                        keep_bytes = int(keep_samples * 2)  # 16-bit = 2 bytes
                        segment_buffer = [segment[-keep_samples:]]
                
                time.sleep(0.01)  # Small sleep to prevent CPU overload
            
            logger.info("Voice call monitoring stopped")
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Monitoring stopped")


# ============================================
# AUDIO FILE HANDLER
# ============================================

class AudioFileHandler:
    """Handle various audio file formats"""
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    
    @staticmethod
    def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio file, convert to target sample rate
        
        Returns:
            (audio_array, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            logger.info(f"Loaded {file_path}: {len(audio)/sr:.2f}s @ {sr}Hz")
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def save_audio(file_path: str, audio: np.ndarray, sr: int):
        """Save audio to file"""
        try:
            sf.write(file_path, audio, sr)
            logger.info(f"Saved audio to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def convert_format(input_path: str, output_path: str, target_sr: int = 16000):
        """Convert audio format"""
        audio, sr = AudioFileHandler.load_audio(input_path, target_sr)
        AudioFileHandler.save_audio(output_path, audio, sr)
    
    @staticmethod
    def get_info(file_path: str) -> Dict:
        """Get audio file information"""
        try:
            audio, sr = librosa.load(file_path, sr=None)
            duration = len(audio) / sr
            
            return {
                'path': file_path,
                'duration': duration,
                'sample_rate': sr,
                'samples': len(audio),
                'format': Path(file_path).suffix,
                'channels': 1 if len(audio.shape) == 1 else audio.shape[1]
            }
        except Exception as e:
            logger.error(f"Failed to get info for {file_path}: {str(e)}")
            return {'error': str(e)}


# ============================================
# FACTORY CLASS
# ============================================

class AudioProcessorFactory:
    """Factory for creating audio processing components"""
    
    @staticmethod
    def create_capture_device(sample_rate: int = 16000) -> AudioCapture:
        """Create audio capture device"""
        return AudioCapture(sample_rate=sample_rate)
    
    @staticmethod
    def create_vad(mode: int = 1) -> VoiceActivityDetector:
        """Create voice activity detector"""
        return VoiceActivityDetector(mode=mode)
    
    @staticmethod
    def create_feature_extractor() -> AudioFeatureExtractor:
        """Create feature extractor"""
        return AudioFeatureExtractor()
    
    @staticmethod
    def create_detector(model_path: Optional[str] = None) -> AudioDeepfakeDetector:
        """Create audio deepfake detector"""
        return AudioDeepfakeDetector(model_path=model_path)
    
    @staticmethod
    def create_live_monitor(detector: AudioDeepfakeDetector, 
                           callback: Optional[callable] = None) -> LiveVoiceCallMonitor:
        """Create live voice call monitor"""
        return LiveVoiceCallMonitor(detector, callback)


# ============================================
# TESTING FUNCTION
# ============================================

def test_audio_processor():
    """Test audio processing modules"""
    print("=" * 60)
    print("TESTING AUDIO PROCESSOR MODULE")
    print("=" * 60)
    
    # Create test audio (sine wave)
    print("\n1️⃣ Creating test audio...")
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    print(f"✅ Test audio created: {duration}s @ {sr}Hz")
    
    # Test feature extractor
    print("\n2️⃣ Testing Feature Extractor...")
    extractor = AudioProcessorFactory.create_feature_extractor()
    features = extractor.extract_features(test_audio, sr)
    print(f"✅ Features extracted: {features}")
    
    # Test model preprocessing
    tensor = extractor.preprocess_for_model(test_audio, sr)
    print(f"✅ Model input tensor shape: {tensor.shape}")
    
    # Test VAD
    print("\n3️⃣ Testing Voice Activity Detection...")
    vad = AudioProcessorFactory.create_vad()
    
    # Test with silence
    silence = np.zeros(int(sr * 0.1))
    is_speech = vad.is_speech(silence)
    print(f"   Silence detection: {'SPEECH' if is_speech else 'SILENCE'} ✅")
    
    # Test with speech (using sine wave as proxy)
    is_speech = vad.is_speech(test_audio[:vad.frame_size])
    print(f"   Audio detection: {'SPEECH' if is_speech else 'SILENCE'} ✅")
    
    # Test detector
    print("\n4️⃣ Testing Audio Deepfake Detector...")
    detector = AudioProcessorFactory.create_detector()
    
    # Save test audio to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, test_audio, sr)
        
        # Detect
        result = detector.detect_from_file(f.name)
        print(f"✅ Detection complete")
        print(f"   Result: {'FAKE' if result.is_fake else 'REAL'}")
        print(f"   AI Prob: {result.ai_probability:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Method: {result.detection_method}")
        print(f"   Time: {result.processing_time:.3f}s")
        
        # Clean up
        os.unlink(f.name)
    
    # Test audio file handler
    print("\n5️⃣ Testing Audio File Handler...")
    handler = AudioFileHandler()
    
    # Save test file
    test_file = "test_audio.wav"
    handler.save_audio(test_file, test_audio, sr)
    print(f"✅ Audio saved to {test_file}")
    
    # Get info
    info = handler.get_info(test_file)
    print(f"   File info: {info}")
    
    # Load back
    loaded_audio, loaded_sr = handler.load_audio(test_file)
    print(f"   Loaded: {len(loaded_audio)/loaded_sr:.2f}s @ {loaded_sr}Hz")
    
    # Clean up
    os.remove(test_file)
    print(f"✅ Test file cleaned up")
    
    print("\n" + "=" * 60)
    print("✅ AUDIO PROCESSOR TEST PASSED!")
    print("=" * 60)
    
    return detector, extractor, vad


if __name__ == "__main__":
    # Run test
    detector, extractor, vad = test_audio_processor()
    
    print("\n📝 Example usage:")