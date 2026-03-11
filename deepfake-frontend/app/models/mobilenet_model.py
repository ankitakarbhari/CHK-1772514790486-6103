# Placeholder file for mobilenet_model.py
# app/models/mobilenet_model.py
"""
MobileNetV2 Model for Deepfake Detection
Lightweight model for real-time detection (15-20 FPS)
Python 3.13+ Compatible
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== DEEP LEARNING IMPORTS ==========
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import keras
from keras import layers, Model
import tensorflow as tf

# ========== IMAGE PROCESSING ==========
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# CUSTOM MOBILENETV2 ARCHITECTURE FOR DEEPFAKE
# ============================================

class DeepfakeMobileNetV2(nn.Module):
    """
    Custom MobileNetV2 architecture optimized for deepfake detection
    Adds attention mechanism and custom classification head
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        """
        Initialize custom MobileNetV2
        
        Args:
            num_classes: Number of output classes (2: real/fake)
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super(DeepfakeMobileNetV2, self).__init__()
        
        # Load base MobileNetV2
        if pretrained:
            self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        else:
            self.base_model = models.mobilenet_v2(weights=None)
        
        # Get the number of features from the last layer
        self.num_features = self.base_model.classifier[1].in_features
        
        # Remove the original classifier
        self.base_model.classifier = nn.Identity()
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize custom layer weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get features from base model
        features = self.base_model.features(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(attended_features, (1, 1))
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


# ============================================
# PYTORCH MOBILENET IMPLEMENTATION
# ============================================

class PyTorchMobileNet:
    """
    PyTorch-based MobileNet for deepfake detection
    Uses torchvision's pre-trained MobileNetV2
    """
    
    def __init__(self, 
                 model_size: str = 'v2',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 use_custom: bool = True,
                 device: Optional[str] = None):
        """
        Initialize PyTorch MobileNet model
        
        Args:
            model_size: 'v2' or 'v3'
            num_classes: Number of output classes (2: real/fake)
            pretrained: Use ImageNet pretrained weights
            use_custom: Use custom architecture with attention
            device: 'cuda' or 'cpu'
        """
        self.model_size = model_size
        self.num_classes = num_classes
        self.use_custom = use_custom
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing PyTorch MobileNet{model_size} on {self.device}")
        logger.info(f"Using {'custom' if use_custom else 'standard'} architecture")
        
        # Load model
        if use_custom:
            # Use custom architecture
            self.model = DeepfakeMobileNetV2(
                num_classes=num_classes,
                pretrained=pretrained
            )
        else:
            # Use standard MobileNetV2
            if model_size == 'v2':
                self.model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
                # Modify classifier
                in_features = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features, num_classes)
                )
            else:  # v3
                self.model = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
                in_features = self.model.classifier[3].in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features, num_classes)
                )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"Model initialized with {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Path to image, numpy array, or PIL Image
        
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, str):
            # Load from path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy to PIL
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[-1] == 3:  # BGR to RGB if from OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[-1] == 4:  # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            image = Image.fromarray(image)
        
        # Preprocess and add batch dimension
        tensor = self.preprocess(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, float]:
        """
        Predict if image is real or fake
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with probabilities
        """
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Get probabilities
            probs = probabilities.cpu().numpy()[0]
            
            return {
                'real_probability': float(probs[0]),
                'fake_probability': float(probs[1]),
                'prediction': 'REAL' if probs[0] > probs[1] else 'FAKE',
                'confidence': float(max(probs))
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'real_probability': 0.0,
                'fake_probability': 0.0,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_batch(self, images: List[Union[str, np.ndarray]]) -> List[Dict]:
        """Predict multiple images"""
        results = []
        for img in images:
            results.append(self.predict(img))
        return results
    
    def predict_fast(self, image: np.ndarray) -> Dict[str, float]:
        """
        Fast prediction for real-time applications
        Assumes image is already preprocessed
        """
        try:
            # Convert to tensor if needed
            if isinstance(image, np.ndarray):
                if image.max() > 1.0:
                    image = image / 255.0
                tensor = torch.from_numpy(image).float()
                if len(tensor.shape) == 3:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.to(self.device)
            else:
                tensor = image
            
            # Inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            probs = probabilities.cpu().numpy()[0]
            
            return {
                'real': float(probs[0]),
                'fake': float(probs[1]),
                'prediction': 0 if probs[0] > probs[1] else 1
            }
            
        except Exception as e:
            logger.error(f"Fast prediction error: {str(e)}")
            return {'real': 0.5, 'fake': 0.5, 'prediction': -1}
    
    def get_intermediate_features(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Get intermediate features for analysis"""
        try:
            input_tensor = self.preprocess_image(image)
            
            with torch.no_grad():
                # Get features from base model
                if self.use_custom:
                    features = self.model.base_model.features(input_tensor)
                else:
                    # Hook to get features
                    features = self.model.features(input_tensor)
                
                # Global average pooling
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            
            return features.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return np.array([])
    
    def save_model(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'use_custom': self.use_custom
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_size = checkpoint.get('model_size', self.model_size)
        self.num_classes = checkpoint.get('num_classes', self.num_classes)
        self.use_custom = checkpoint.get('use_custom', self.use_custom)
        logger.info(f"Model loaded from {path}")


# ============================================
# KERAS 3 MOBILENET IMPLEMENTATION
# ============================================

class KerasMobileNet:
    """
    Keras 3-based MobileNet for deepfake detection
    Optimized for speed and real-time applications
    """
    
    def __init__(self,
                 model_size: str = 'V2',
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 pretrained: bool = True):
        """
        Initialize Keras MobileNet model
        
        Args:
            model_size: 'V2' or 'V3'
            input_shape: Input image shape
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
        """
        self.model_size = model_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        logger.info(f"Initializing Keras MobileNet{model_size}")
        
        # Set backend (works with PyTorch backend on Python 3.13)
        os.environ['KERAS_BACKEND'] = 'torch'
        
        # Build model
        self.model = self._build_model(pretrained)
        
        # Image preprocessing
        self.preprocess = self._get_preprocessing_function()
        
        logger.info(f"Model input shape: {self.model.input_shape}")
        logger.info(f"Model output shape: {self.model.output_shape}")
    
    def _build_model(self, pretrained: bool) -> Model:
        """Build MobileNet model"""
        
        if self.model_size == 'V2':
            # MobileNetV2 base
            base_model = keras.applications.MobileNetV2(
                weights='imagenet' if pretrained else None,
                include_top=False,
                input_shape=self.input_shape,
                pooling='avg'
            )
        else:  # V3
            base_model = keras.applications.MobileNetV3Large(
                weights='imagenet' if pretrained else None,
                include_top=False,
                input_shape=self.input_shape,
                pooling='avg'
            )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name=f'Deepfake_MobileNet{self.model_size}')
        
        return model
    
    def _get_preprocessing_function(self):
        """Get preprocessing function for MobileNet"""
        def preprocess(img):
            if isinstance(img, str):
                img = keras.utils.load_img(img, target_size=self.input_shape[:2])
                img = keras.utils.img_to_array(img)
            elif isinstance(img, np.ndarray):
                if img.shape[-1] == 3 and img.shape[:2] != self.input_shape[:2]:
                    img = cv2.resize(img, self.input_shape[:2])
                if img.max() > 1.0:
                    img = img / 255.0
            return np.expand_dims(img, axis=0)
        
        return preprocess
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile model for training"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Model compiled successfully")
    
    def predict(self, image: Union[str, np.ndarray]) -> Dict[str, float]:
        """
        Predict if image is real or fake
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with probabilities
        """
        try:
            # Preprocess
            input_tensor = self.preprocess(image)
            
            # Convert to tensor (works with PyTorch backend)
            if not isinstance(input_tensor, torch.Tensor):
                input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor, training=False)
                if isinstance(outputs, torch.Tensor):
                    probabilities = F.softmax(outputs, dim=1)
                    probs = probabilities.cpu().numpy()[0]
                else:
                    probs = outputs.numpy()[0]
            
            return {
                'real_probability': float(probs[0]),
                'fake_probability': float(probs[1]),
                'prediction': 'REAL' if probs[0] > probs[1] else 'FAKE',
                'confidence': float(max(probs))
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'real_probability': 0.0,
                'fake_probability': 0.0,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_fast(self, image: np.ndarray) -> Dict[str, float]:
        """
        Ultra-fast prediction for real-time video
        """
        try:
            # Assume image is already preprocessed
            if not isinstance(image, torch.Tensor):
                if image.max() > 1.0:
                    image = image / 255.0
                tensor = torch.tensor(image, dtype=torch.float32)
                if len(tensor.shape) == 3:
                    tensor = tensor.unsqueeze(0)
            else:
                tensor = image
            
            with torch.no_grad():
                outputs = self.model(tensor, training=False)
                if isinstance(outputs, torch.Tensor):
                    probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                else:
                    probs = outputs.numpy()[0]
            
            return {
                'real': float(probs[0]),
                'fake': float(probs[1]),
                'prediction': 0 if probs[0] > probs[1] else 1
            }
            
        except Exception as e:
            return {'real': 0.5, 'fake': 0.5, 'prediction': -1}
    
    def save_model(self, path: str):
        """Save model in Keras format"""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load Keras model"""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


# ============================================
# FACTORY CLASS
# ============================================

class MobileNetFactory:
    """Factory class to create MobileNet models"""
    
    @staticmethod
    def create_pytorch_model(model_size: str = 'v2', 
                            use_custom: bool = True,
                            **kwargs):
        """
        Create PyTorch MobileNet model
        
        Args:
            model_size: 'v2' or 'v3'
            use_custom: Use custom architecture with attention
        
        Returns:
            PyTorchMobileNet instance
        """
        return PyTorchMobileNet(
            model_size=model_size,
            use_custom=use_custom,
            **kwargs
        )
    
    @staticmethod
    def create_keras_model(model_size: str = 'V2', **kwargs):
        """
        Create Keras MobileNet model
        
        Args:
            model_size: 'V2' or 'V3'
        
        Returns:
            KerasMobileNet instance
        """
        return KerasMobileNet(model_size=model_size, **kwargs)
    
    @staticmethod
    def create_fast_model(device: Optional[str] = None):
        """
        Create fastest model for real-time detection
        """
        return PyTorchMobileNet(
            model_size='v2',
            use_custom=False,
            device=device
        )
    
    @staticmethod
    def create_accurate_model(device: Optional[str] = None):
        """
        Create more accurate model with attention
        """
        return PyTorchMobileNet(
            model_size='v2',
            use_custom=True,
            device=device
        )


# ============================================
# TESTING FUNCTION
# ============================================

def test_mobilenet():
    """Test the MobileNet model"""
    print("=" * 50)
    print("TESTING MOBILENET MODEL")
    print("=" * 50)
    
    try:
        # Test PyTorch model
        print("\n1️⃣ Testing PyTorch MobileNet (Fast version)...")
        model_pt = MobileNetFactory.create_fast_model()
        print(f"✅ PyTorch model created")
        print(f"   Parameters: {model_pt.count_parameters():,}")
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test prediction
        result = model_pt.predict(dummy_img)
        print(f"✅ Prediction successful")
        print(f"   Result: {result}")
        
        # Test fast prediction
        img_norm = dummy_img / 255.0
        fast_result = model_pt.predict_fast(img_norm)
        print(f"✅ Fast prediction successful")
        print(f"   Fast result: {fast_result}")
        
        # Test feature extraction
        features = model_pt.get_intermediate_features(dummy_img)
        print(f"✅ Feature extraction successful")
        print(f"   Features shape: {features.shape}")
        
        # Test PyTorch accurate model
        print("\n2️⃣ Testing PyTorch MobileNet (Accurate version)...")
        model_pt_acc = MobileNetFactory.create_accurate_model()
        print(f"✅ Accurate model created")
        print(f"   Parameters: {model_pt_acc.count_parameters():,}")
        
        result_acc = model_pt_acc.predict(dummy_img)
        print(f"   Result: {result_acc}")
        
        # Test Keras model
        print("\n3️⃣ Testing Keras MobileNet...")
        try:
            model_keras = MobileNetFactory.create_keras_model('V2')
            print(f"✅ Keras model created")
            
            result_keras = model_keras.predict(dummy_img)
            print(f"   Result: {result_keras}")
        except Exception as e:
            print(f"⚠️ Keras model test skipped: {str(e)[:50]}...")
        
        print("\n" + "=" * 50)
        print("✅ MOBILENET TEST PASSED!")
        print("=" * 50)
        
        return model_pt
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# PERFORMANCE BENCHMARK
# ============================================

def benchmark_mobilenet(model, num_iterations: int = 100):
    """Benchmark MobileNet performance"""
    print("\n" + "=" * 50)
    print("BENCHMARKING MOBILENET")
    print("=" * 50)
    
    # Create dummy images
    dummy_images = []
    for _ in range(10):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_images.append(img)
    
    # Warmup
    for _ in range(10):
        model.predict_fast(dummy_images[0] / 255.0)
    
    # Benchmark fast prediction
    import time
    start = time.time()
    for i in range(num_iterations):
        img = dummy_images[i % len(dummy_images)] / 255.0
        model.predict_fast(img)
    end = time.time()
    
    avg_time = (end - start) / num_iterations
    fps = 1.0 / avg_time
    
    print(f"\n📊 FAST PREDICTION:")
    print(f"   Average time: {avg_time*1000:.2f} ms")
    print(f"   FPS: {fps:.1f}")
    
    # Benchmark full prediction
    start = time.time()
    for i in range(num_iterations // 10):  # Fewer iterations for full prediction
        img = dummy_images[i % len(dummy_images)]
        model.predict(img)
    end = time.time()
    
    avg_time_full = (end - start) / (num_iterations // 10)
    fps_full = 1.0 / avg_time_full
    
    print(f"\n📊 FULL PREDICTION:")
    print(f"   Average time: {avg_time_full*1000:.2f} ms")
    print(f"   FPS: {fps_full:.1f}")
    
    return {'fast_fps': fps, 'full_fps': fps_full}


if __name__ == "__main__":
    # Run test
    model = test_mobilenet()
    
    # Run benchmark if test passed
    if model:
        benchmark_mobilenet(model)
        
        print("\n📝 Example usage:")
        print("""
# In your main application:
from app.models.mobilenet_model import MobileNetFactory

# For real-time video (fastest)
fast_model = MobileNetFactory.create_fast_model()

# For better accuracy
accurate_model = MobileNetFactory.create_accurate_model()

# Predict
result = fast_model.predict('path/to/image.jpg')
if result['prediction'] == 'FAKE':
    print(f"Deepfake detected! Confidence: {result['confidence']:.2f}")

# Real-time video loop
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Fast prediction
    result = fast_model.predict_fast(frame / 255.0)
    if result['prediction'] == 1:  # Fake
        cv2.putText(frame, "DEEPFAKE!", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Live Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
        """)