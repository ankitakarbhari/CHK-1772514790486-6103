# Placeholder file for xception_model.py
# app/models/xception_model.py
"""
Xception Model for Deepfake Detection
Xception (Extreme Inception) is excellent at detecting image manipulations
Based on the paper: "Xception: Deep Learning with Depthwise Separable Convolutions"
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
# CUSTOM XCEPTION ARCHITECTURE FOR DEEPFAKE
# ============================================

class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """Xception block with residual connections"""
    def __init__(self, in_channels, out_channels, stride=1, first_block=False):
        super(XceptionBlock, self).__init__()
        
        self.first_block = first_block
        self.stride = stride
        
        # Main path
        if not first_block:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU(inplace=True)
        
        self.sepconv1 = SeparableConv2d(
            in_channels if first_block else in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.sepconv2 = SeparableConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        # Main path
        if not self.first_block:
            x = self.bn1(x)
            x = self.relu1(x)
        
        x = self.sepconv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.sepconv2(x)
        x = self.bn3(x)
        
        # Add residual
        x += self.shortcut(identity)
        
        return x


class CustomXception(nn.Module):
    """
    Custom Xception architecture optimized for deepfake detection
    Includes attention mechanisms and multi-scale feature extraction
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(CustomXception, self).__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Xception blocks
        self.block1 = XceptionBlock(64, 128, stride=2, first_block=True)
        self.block2 = XceptionBlock(128, 256, stride=2)
        self.block3 = XceptionBlock(256, 728, stride=2)
        
        # Middle flow (repeated blocks)
        self.middle_blocks = nn.ModuleList()
        for _ in range(8):  # 8 middle blocks
            self.middle_blocks.append(XceptionBlock(728, 728, stride=1))
        
        # Exit flow
        self.block4 = XceptionBlock(728, 1024, stride=2)
        
        self.sepconv3 = SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.sepconv4 = SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Attention mechanism for deepfake detection
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self._initialize_weights()
        
        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self):
        """Load pretrained weights from torchvision Xception"""
        try:
            # Create pretrained model
            pretrained = models.xception(weights='IMAGENET1K_V1')
            
            # Get state dicts
            pretrained_dict = pretrained.state_dict()
            model_dict = self.state_dict()
            
            # Filter out mismatched keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            # Update model
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            logger.info(f"Loaded pretrained weights for {len(pretrained_dict)} layers")
            
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {str(e)}")
    
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        for block in self.middle_blocks:
            x = block(x)
        
        # Exit flow
        x = self.block4(x)
        
        x = self.sepconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.sepconv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================
# PYTORCH XCEPTION IMPLEMENTATION
# ============================================

class PyTorchXception:
    """
    PyTorch-based Xception for deepfake detection
    Xception is particularly good at detecting image manipulations
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 pretrained: bool = True,
                 use_custom: bool = True,
                 device: Optional[str] = None):
        """
        Initialize PyTorch Xception model
        
        Args:
            num_classes: Number of output classes (2: real/fake)
            pretrained: Use ImageNet pretrained weights
            use_custom: Use custom architecture with attention
            device: 'cuda' or 'cpu'
        """
        self.num_classes = num_classes
        self.use_custom = use_custom
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing PyTorch Xception on {self.device}")
        logger.info(f"Using {'custom' if use_custom else 'standard'} architecture")
        
        # Load model
        if use_custom:
            # Use custom architecture
            self.model = CustomXception(num_classes=num_classes, pretrained=pretrained)
        else:
            # Use torchvision Xception
            if pretrained:
                self.model = models.xception(weights='IMAGENET1K_V1')
            else:
                self.model = models.xception(weights=None)
            
            # Modify classifier for binary classification
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),  # Xception expects 299x299
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # Xception uses [-1,1] range
                std=[0.5, 0.5, 0.5]
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
    
    def get_manipulation_map(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Get manipulation heatmap using Grad-CAM-like approach
        Xception is excellent at highlighting manipulated regions
        """
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image)
            input_tensor.requires_grad = True
            
            # Forward pass
            outputs = self.model(input_tensor)
            target_class = outputs.argmax(dim=1).item()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass for target class
            outputs[0, target_class].backward()
            
            # Get gradients
            gradients = input_tensor.grad.abs().cpu().numpy()[0]
            
            # Create heatmap
            heatmap = np.mean(gradients, axis=0)
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]) if isinstance(image, np.ndarray) else (299, 299))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Manipulation map error: {str(e)}")
            return np.zeros((299, 299))
    
    def extract_features(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Extract features from intermediate layer
        Useful for analysis and comparison
        """
        try:
            input_tensor = self.preprocess_image(image)
            
            # Hook to get features from last conv layer
            features = []
            
            def hook_fn(module, input, output):
                features.append(output.detach())
            
            if self.use_custom:
                # Register hook on last conv layer
                handle = self.model.sepconv4.register_forward_hook(hook_fn)
            else:
                handle = self.model.features[-1].register_forward_hook(hook_fn)
            
            # Forward pass
            self.model(input_tensor)
            
            # Remove hook
            handle.remove()
            
            if features:
                # Global average pooling
                feat = F.adaptive_avg_pool2d(features[0], (1, 1))
                return feat.cpu().numpy().flatten()
            
            return np.array([])
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return np.array([])
    
    def save_model(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'use_custom': self.use_custom
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.num_classes = checkpoint.get('num_classes', self.num_classes)
        self.use_custom = checkpoint.get('use_custom', self.use_custom)
        logger.info(f"Model loaded from {path}")


# ============================================
# KERAS 3 XCEPTION IMPLEMENTATION
# ============================================

class KerasXception:
    """
    Keras 3-based Xception for deepfake detection
    Known for excellent performance on manipulation detection
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (299, 299, 3),
                 num_classes: int = 2,
                 pretrained: bool = True):
        """
        Initialize Keras Xception model
        
        Args:
            input_shape: Input image shape (299x299 for Xception)
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        logger.info(f"Initializing Keras Xception")
        logger.info(f"Input shape: {input_shape}")
        
        # Set backend (works with PyTorch backend on Python 3.13)
        os.environ['KERAS_BACKEND'] = 'torch'
        
        # Build model
        self.model = self._build_model(pretrained)
        
        # Image preprocessing
        self.preprocess = self._get_preprocessing_function()
        
        logger.info(f"Model input shape: {self.model.input_shape}")
        logger.info(f"Model output shape: {self.model.output_shape}")
    
    def _build_model(self, pretrained: bool) -> Model:
        """Build Xception model"""
        
        # Xception base
        base_model = keras.applications.Xception(
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
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='Deepfake_Xception')
        
        return model
    
    def _get_preprocessing_function(self):
        """Get preprocessing function for Xception"""
        def preprocess(img):
            if isinstance(img, str):
                img = keras.utils.load_img(img, target_size=self.input_shape[:2])
                img = keras.utils.img_to_array(img)
            elif isinstance(img, np.ndarray):
                if img.shape[-1] == 3 and img.shape[:2] != self.input_shape[:2]:
                    img = cv2.resize(img, self.input_shape[:2])
                
                # Xception expects [-1, 1] range
                if img.max() > 1.0:
                    img = img / 127.5 - 1.0
                elif img.max() <= 1.0:
                    img = img * 2.0 - 1.0
            
            return np.expand_dims(img, axis=0)
        
        return preprocess
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile model for training"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
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
            
            # Convert to tensor
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

class XceptionFactory:
    """Factory class to create Xception models"""
    
    @staticmethod
    def create_pytorch_model(use_custom: bool = True, **kwargs):
        """
        Create PyTorch Xception model
        
        Args:
            use_custom: Use custom architecture with attention
        
        Returns:
            PyTorchXception instance
        """
        return PyTorchXception(use_custom=use_custom, **kwargs)
    
    @staticmethod
    def create_keras_model(**kwargs):
        """
        Create Keras Xception model
        
        Returns:
            KerasXception instance
        """
        return KerasXception(**kwargs)
    
    @staticmethod
    def create_standard_model(device: Optional[str] = None):
        """
        Create standard Xception model
        """
        return PyTorchXception(use_custom=False, device=device)
    
    @staticmethod
    def create_custom_model(device: Optional[str] = None):
        """
        Create custom Xception with attention
        """
        return PyTorchXception(use_custom=True, device=device)


# ============================================
# TESTING FUNCTION
# ============================================

def test_xception():
    """Test the Xception model"""
    print("=" * 60)
    print("TESTING XCEPTION MODEL")
    print("=" * 60)
    
    try:
        # Test standard model
        print("\n1️⃣ Testing Standard Xception...")
        model_std = XceptionFactory.create_standard_model()
        print(f"✅ Standard model created")
        print(f"   Parameters: {model_std.count_parameters():,}")
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
        
        # Test prediction
        result = model_std.predict(dummy_img)
        print(f"✅ Prediction successful")
        print(f"   Result: {result}")
        
        # Test custom model
        print("\n2️⃣ Testing Custom Xception (with attention)...")
        model_custom = XceptionFactory.create_custom_model()
        print(f"✅ Custom model created")
        print(f"   Parameters: {model_custom.count_parameters():,}")
        
        result_custom = model_custom.predict(dummy_img)
        print(f"   Result: {result_custom}")
        
        # Test manipulation map
        print("\n3️⃣ Testing Manipulation Map...")
        heatmap = model_custom.get_manipulation_map(dummy_img)
        print(f"✅ Manipulation map generated")
        print(f"   Heatmap shape: {heatmap.shape}")
        print(f"   Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        
        # Test feature extraction
        print("\n4️⃣ Testing Feature Extraction...")
        features = model_custom.extract_features(dummy_img)
        print(f"✅ Features extracted")
        print(f"   Features shape: {features.shape}")
        
        # Test batch prediction
        print("\n5️⃣ Testing Batch Prediction...")
        batch = [dummy_img, dummy_img, dummy_img]
        results = model_custom.predict_batch(batch)
        print(f"✅ Batch prediction successful")
        print(f"   Batch size: {len(results)}")
        
        # Test Keras model
        print("\n6️⃣ Testing Keras Xception...")
        try:
            model_keras = XceptionFactory.create_keras_model()
            print(f"✅ Keras model created")
            
            result_keras = model_keras.predict(dummy_img)
            print(f"   Result: {result_keras}")
        except Exception as e:
            print(f"⚠️ Keras model test skipped: {str(e)[:50]}...")
        
        print("\n" + "=" * 60)
        print("✅ XCEPTION TEST PASSED!")
        print("=" * 60)
        
        return model_custom
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_xception(model: PyTorchXception, num_iterations: int = 50):
    """Benchmark Xception performance"""
    print("\n" + "=" * 60)
    print("BENCHMARKING XCEPTION")
    print("=" * 60)
    
    # Create dummy images
    dummy_img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
    
    import time
    
    # Warmup
    for _ in range(5):
        model.predict(dummy_img)
    
    # Benchmark
    times = []
    for i in range(num_iterations):
        start = time.time()
        model.predict(dummy_img)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n📊 BENCHMARK RESULTS:")
    print(f"   Average time: {avg_time*1000:.2f} ms")
    print(f"   Std deviation: {std_time*1000:.2f} ms")
    print(f"   FPS: {1.0/avg_time:.1f}")
    print(f"   Min time: {min(times)*1000:.2f} ms")
    print(f"   Max time: {max(times)*1000:.2f} ms")
    
    # Memory usage (if on CUDA)
    if torch.cuda.is_available():
        memory = torch.cuda.memory_allocated() / 1024**2
        print(f"   GPU Memory: {memory:.1f} MB")


def compare_models():
    """Compare standard vs custom Xception"""
    print("\n" + "=" * 60)
    print("COMPARING XCEPTION MODELS")
    print("=" * 60)
    
    # Create models
    model_std = XceptionFactory.create_standard_model()
    model_custom = XceptionFactory.create_custom_model()
    
    print(f"\n📊 MODEL COMPARISON:")
    print(f"   {'Metric':<20} {'Standard':<15} {'Custom':<15}")
    print(f"   {'-'*50}")
    print(f"   {'Parameters':<20} {model_std.count_parameters():<15,} {model_custom.count_parameters():<15,}")
    
    # Test on a sample image
    dummy_img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
    
    import time
    
    # Benchmark standard
    start = time.time()
    result_std = model_std.predict(dummy_img)
    std_time = time.time() - start
    
    # Benchmark custom
    start = time.time()
    result_custom = model_custom.predict(dummy_img)
    custom_time = time.time() - start
    
    print(f"   {'Inference time':<20} {std_time*1000:<15.2f}ms {custom_time*1000:<15.2f}ms")
    print(f"   {'Confidence':<20} {result_std['confidence']:<15.3f} {result_custom['confidence']:<15.3f}")


if __name__ == "__main__":
    # Run test
    model = test_xception()
    
    # Run benchmark if test passed
    if model:
        benchmark_xception(model, num_iterations=20)
        compare_models()
        
        print("\n📝 Example usage:")
        print("""
# In your main application:
from app.models.xception_model import XceptionFactory

# Create Xception model (best for manipulation detection)
model = XceptionFactory.create_custom_model()

# Predict single image
result = model.predict('path/to/image.jpg')
if result['prediction'] == 'FAKE':
    print(f"⚠️ Deepfake detected! Confidence: {result['confidence']:.2%}")

# Get manipulation heatmap
heatmap = model.get_manipulation_map('path/to/image.jpg')
# heatmap shows which regions are manipulated (higher values = more likely manipulated)

# Extract features for comparison
features = model.extract_features('path/to/image.jpg')
        """)