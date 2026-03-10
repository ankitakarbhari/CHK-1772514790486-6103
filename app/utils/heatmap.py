# app/utils/heatmap.py
"""
Heatmap Generation Module for Deepfake Visualization
Creates visual explanations of model decisions using Grad-CAM, Grad-CAM++, Score-CAM, and more
Helps identify manipulated regions in images with color-coded heatmaps
Supports PyTorch models and integrates with ensemble detector
Python 3.13+ Compatible
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from collections import OrderedDict
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== DEEP LEARNING IMPORTS ==========
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ========== IMAGE PROCESSING ==========
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ========== GRAD-CAM (Optional) ==========
try:
    from pytorch_grad_cam import (
        GradCAM, 
        GradCAMPlusPlus,
        XGradCAM,
        EigenCAM,
        ScoreCAM,
        AblationCAM,
        LayerCAM
    )
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    logging.warning("pytorch_grad_cam not installed. Install with: pip install grad-cam")

# ========== CAPTUM (Alternative) ==========
try:
    from captum.attr import (
        IntegratedGradients,
        Saliency,
        GuidedGradCam,
        Occlusion,
        LayerActivation,
        FeatureAblation
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logging.warning("captum not installed. Install with: pip install captum")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation"""
    method: str = 'gradcam'  # 'gradcam', 'gradcam++', 'scorecam', 'eigencam', 'layerCAM'
    target_layer: Optional[str] = None
    normalize: bool = True
    colormap: str = 'jet'
    alpha: float = 0.5
    use_cuda: bool = True
    resize_to_input: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class HeatmapResult:
    """Heatmap generation result"""
    heatmap: np.ndarray
    overlay: np.ndarray
    original_image: np.ndarray
    method: str
    confidence: float
    prediction: str
    prediction_idx: int
    processing_time: float
    manipulated_regions: List[Dict] = None
    attention_weights: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        result = {
            'heatmap_shape': self.heatmap.shape,
            'overlay_shape': self.overlay.shape,
            'method': self.method,
            'confidence': float(self.confidence),
            'prediction': self.prediction,
            'prediction_idx': int(self.prediction_idx),
            'processing_time': float(self.processing_time),
            'manipulated_regions': self.manipulated_regions or []
        }
        return result
    
    def get_heatmap_base64(self) -> str:
        """Get heatmap as base64 string"""
        import base64
        heatmap_bgr = cv2.cvtColor(self.heatmap, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', heatmap_bgr)
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_overlay_base64(self) -> str:
        """Get overlay as base64 string"""
        import base64
        overlay_bgr = cv2.cvtColor(self.overlay, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', overlay_bgr)
        return base64.b64encode(buffer).decode('utf-8')
    
    def save(self, output_dir: str, filename: str):
        """Save heatmap and overlay to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save heatmap
        heatmap_path = os.path.join(output_dir, f"{filename}_heatmap.png")
        heatmap_bgr = cv2.cvtColor(self.heatmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, heatmap_bgr)
        
        # Save overlay
        overlay_path = os.path.join(output_dir, f"{filename}_overlay.png")
        overlay_bgr = cv2.cvtColor(self.overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(overlay_path, overlay_bgr)
        
        # Save original
        original_path = os.path.join(output_dir, f"{filename}_original.png")
        original_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(original_path, original_bgr)
        
        logger.info(f"Saved heatmap to {heatmap_path}")
        logger.info(f"Saved overlay to {overlay_path}")
        logger.info(f"Saved original to {original_path}")


# ============================================
# BASE HEATMAP GENERATOR
# ============================================

class BaseHeatmapGenerator:
    """Base class for all heatmap generators"""
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        """
        Initialize base generator
        
        Args:
            model: PyTorch model
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        
        # Default image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"BaseHeatmapGenerator initialized on {self.device}")
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image
        
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
    
    def get_target_layer(self, layer_name: Optional[str] = None) -> nn.Module:
        """Get target layer for CAM methods"""
        if layer_name is None:
            # Try to find last convolutional layer
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    logger.info(f"Auto-selected target layer: {name}")
                    return module
            
            # If no conv layer found, use last layer
            last_layer = list(self.model.modules())[-1]
            logger.warning(f"No conv layer found, using last layer: {last_layer.__class__.__name__}")
            return last_layer
        else:
            # Get layer by name
            parts = layer_name.split('.')
            module = self.model
            for part in parts:
                module = getattr(module, part)
            return module
    
    def generate(self, image: Union[str, np.ndarray, Image.Image], 
                target_class: Optional[int] = None) -> HeatmapResult:
        """
        Generate heatmap (to be overridden by subclasses)
        """
        raise NotImplementedError("Subclasses must implement generate()")


# ============================================
# GRAD-CAM GENERATOR
# ============================================

class GradCAMGenerator(BaseHeatmapGenerator):
    """
    Generate heatmaps using Grad-CAM (Gradient-weighted Class Activation Mapping)
    Highlights regions that are important for the model's decision
    """
    
    def __init__(self, 
                 model: nn.Module,
                 target_layer: Optional[nn.Module] = None,
                 device: Optional[str] = None):
        """
        Initialize Grad-CAM generator
        
        Args:
            model: PyTorch model
            target_layer: Target layer for CAM
            device: 'cuda' or 'cpu'
        """
        super().__init__(model, device)
        
        if not GRAD_CAM_AVAILABLE:
            logger.warning("pytorch_grad_cam not available. Install for better heatmaps.")
            self.cam = None
        else:
            try:
                if target_layer is None:
                    target_layer = self.get_target_layer()
                
                self.cam = GradCAM(model=self.model, target_layers=[target_layer])
                logger.info(f"GradCAM initialized with target layer: {target_layer.__class__.__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize GradCAM: {str(e)}")
                self.cam = None
    
    def generate(self, 
                image: Union[str, np.ndarray, Image.Image],
                target_class: Optional[int] = None,
                normalize: bool = True) -> HeatmapResult:
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Input image
            target_class: Target class (None = use predicted class)
            normalize: Normalize heatmap to [0,1]
        
        Returns:
            HeatmapResult
        """
        start_time = time.time()
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Get prediction first
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(dim=1).item()
            confidence = float(probs[0, pred_idx].cpu().numpy())
            prediction = 'FAKE' if pred_idx == 1 else 'REAL'
        
        # Generate heatmap
        if self.cam is not None:
            try:
                # Set target
                if target_class is not None:
                    targets = [ClassifierOutputTarget(target_class)]
                else:
                    targets = [ClassifierOutputTarget(pred_idx)]
                
                # Generate CAM
                grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension
                
                if normalize:
                    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
                
            except Exception as e:
                logger.error(f"Grad-CAM generation error: {str(e)}")
                grayscale_cam = self._fallback_heatmap(input_tensor)
        else:
            grayscale_cam = self._fallback_heatmap(input_tensor)
        
        # Resize heatmap to match original image
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        elif isinstance(image, str):
            img = cv2.imread(image)
            h, w = img.shape[:2]
        else:  # PIL Image
            w, h = image.size
        
        heatmap_resized = cv2.resize(grayscale_cam, (w, h))
        
        # Create colored heatmap
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            original = image.copy()
        else:
            original = np.array(image)
        
        overlay = self._overlay_heatmap(original, heatmap_resized, alpha=0.5)
        
        # Find manipulated regions
        regions = self._find_manipulated_regions(heatmap_resized, threshold=0.7)
        
        processing_time = time.time() - start_time
        
        return HeatmapResult(
            heatmap=heatmap_color,
            overlay=overlay,
            original_image=original,
            method='gradcam',
            confidence=confidence,
            prediction=prediction,
            prediction_idx=pred_idx,
            processing_time=processing_time,
            manipulated_regions=regions
        )
    
    def _fallback_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Fallback method when Grad-CAM is not available"""
        logger.warning("Using fallback saliency map")
        
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(input_tensor)
        target = outputs[0, outputs.argmax(dim=1).item()]
        
        # Backward pass
        self.model.zero_grad()
        target.backward()
        
        # Get gradients
        gradients = input_tensor.grad.abs().cpu().numpy()[0]
        
        # Create heatmap
        heatmap = np.mean(gradients, axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def _overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay heatmap on image"""
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Apply colormap
        heatmap_colored = cm.jet(heatmap)[:, :, :3]
        
        # Overlay
        overlay = (1 - alpha) * image + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        return (overlay * 255).astype(np.uint8)
    
    def _find_manipulated_regions(self, heatmap: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """Find manipulated regions from heatmap"""
        # Create binary mask
        mask = (heatmap > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 100:  # Filter small regions
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get average heatmap value in region
            region_heatmap = heatmap[y:y+h, x:x+w]
            avg_intensity = float(np.mean(region_heatmap))
            
            regions.append({
                'id': i,
                'area': int(area),
                'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'center': {'x': int(x + w/2), 'y': int(y + h/2)},
                'confidence': float(avg_intensity)
            })
        
        # Sort by intensity
        regions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return regions


# ============================================
# GRAD-CAM++ GENERATOR
# ============================================

class GradCAMPlusPlusGenerator(BaseHeatmapGenerator):
    """
    Generate heatmaps using Grad-CAM++
    Improved version with better localization
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, device: Optional[str] = None):
        super().__init__(model, device)
        
        if not GRAD_CAM_AVAILABLE:
            self.cam = None
        else:
            try:
                if target_layer is None:
                    target_layer = self.get_target_layer()
                
                self.cam = GradCAMPlusPlus(model=self.model, target_layers=[target_layer])
                logger.info(f"Grad-CAM++ initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Grad-CAM++: {str(e)}")
                self.cam = None
    
    def generate(self, image: Union[str, np.ndarray, Image.Image], 
                target_class: Optional[int] = None) -> HeatmapResult:
        """Generate Grad-CAM++ heatmap"""
        start_time = time.time()
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(dim=1).item()
            confidence = float(probs[0, pred_idx].cpu().numpy())
            prediction = 'FAKE' if pred_idx == 1 else 'REAL'
        
        # Generate heatmap
        if self.cam is not None:
            try:
                targets = [ClassifierOutputTarget(target_class or pred_idx)]
                grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
            except Exception as e:
                logger.error(f"Grad-CAM++ error: {str(e)}")
                grayscale_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        else:
            grayscale_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Resize and create overlay
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = 224, 224
        
        heatmap_resized = cv2.resize(grayscale_cam, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            original = image.copy()
        else:
            original = np.array(image)
        
        # Simple overlay
        overlay = original.copy()
        heatmap_resized_3ch = np.stack([heatmap_resized]*3, axis=-1)
        overlay = (overlay * 0.5 + heatmap_color * 0.5).astype(np.uint8)
        
        processing_time = time.time() - start_time
        
        return HeatmapResult(
            heatmap=heatmap_color,
            overlay=overlay,
            original_image=original,
            method='gradcam++',
            confidence=confidence,
            prediction=prediction,
            prediction_idx=pred_idx,
            processing_time=processing_time
        )


# ============================================
# SCORE-CAM GENERATOR
# ============================================

class ScoreCAMGenerator(BaseHeatmapGenerator):
    """
    Generate heatmaps using Score-CAM
    Gradient-free method using score-based weighting
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, device: Optional[str] = None):
        super().__init__(model, device)
        
        if not GRAD_CAM_AVAILABLE:
            self.cam = None
        else:
            try:
                if target_layer is None:
                    target_layer = self.get_target_layer()
                
                self.cam = ScoreCAM(model=self.model, target_layers=[target_layer])
                logger.info(f"Score-CAM initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Score-CAM: {str(e)}")
                self.cam = None
    
    def generate(self, image: Union[str, np.ndarray, Image.Image], 
                target_class: Optional[int] = None) -> HeatmapResult:
        """Generate Score-CAM heatmap"""
        start_time = time.time()
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(dim=1).item()
            confidence = float(probs[0, pred_idx].cpu().numpy())
            prediction = 'FAKE' if pred_idx == 1 else 'REAL'
        
        if self.cam is not None:
            try:
                targets = [ClassifierOutputTarget(target_class or pred_idx)]
                grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
            except:
                grayscale_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        else:
            grayscale_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Resize and create output
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = 224, 224
        
        heatmap_resized = cv2.resize(grayscale_cam, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_INFERNO)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            original = image.copy()
        else:
            original = np.array(image)
        
        overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
        
        processing_time = time.time() - start_time
        
        return HeatmapResult(
            heatmap=heatmap_color,
            overlay=overlay,
            original_image=original,
            method='scorecam',
            confidence=confidence,
            prediction=prediction,
            prediction_idx=pred_idx,
            processing_time=processing_time
        )


# ============================================
# EIGEN-CAM GENERATOR
# ============================================

class EigenCAMGenerator(BaseHeatmapGenerator):
    """
    Generate heatmaps using Eigen-CAM
    Uses principal components of activations, no gradients needed
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, device: Optional[str] = None):
        super().__init__(model, device)
        
        if not GRAD_CAM_AVAILABLE:
            self.cam = None
        else:
            try:
                if target_layer is None:
                    target_layer = self.get_target_layer()
                
                self.cam = EigenCAM(model=self.model, target_layers=[target_layer])
                logger.info(f"Eigen-CAM initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Eigen-CAM: {str(e)}")
                self.cam = None
    
    def generate(self, image: Union[str, np.ndarray, Image.Image]) -> HeatmapResult:
        """Generate Eigen-CAM heatmap"""
        start_time = time.time()
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(dim=1).item()
            confidence = float(probs[0, pred_idx].cpu().numpy())
            prediction = 'FAKE' if pred_idx == 1 else 'REAL'
        
        if self.cam is not None:
            try:
                grayscale_cam = self.cam(input_tensor=input_tensor)
                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
            except:
                grayscale_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        else:
            grayscale_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Resize and create output
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = 224, 224
        
        heatmap_resized = cv2.resize(grayscale_cam, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_VIRIDIS)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            original = image.copy()
        else:
            original = np.array(image)
        
        overlay = cv2.addWeighted(original, 0.5, heatmap_color, 0.5, 0)
        
        processing_time = time.time() - start_time
        
        return HeatmapResult(
            heatmap=heatmap_color,
            overlay=overlay,
            original_image=original,
            method='eigencam',
            confidence=confidence,
            prediction=prediction,
            prediction_idx=pred_idx,
            processing_time=processing_time
        )


# ============================================
# LAYER-CAM GENERATOR
# ============================================

class LayerCAMGenerator(BaseHeatmapGenerator):
    """
    Generate heatmaps using Layer-CAM
    Provides finer-grained localization
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, device: Optional[str] = None):
        super().__init__(model, device)
        
        if not GRAD_CAM_AVAILABLE:
            self.cam = None
        else:
            try:
                if target_layer is None:
                    target_layer = self.get_target_layer()
                
                self.cam = LayerCAM(model=self.model, target_layers=[target_layer])
                logger.info(f"Layer-CAM initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Layer-CAM: {str(e)}")
                self.cam = None
    
    def generate(self, image: Union[str, np.ndarray, Image.Image], 
                target_class: Optional[int] = None) -> HeatmapResult:
        """Generate Layer-CAM heatmap"""
        start_time = time.time()
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(dim=1).item()
            confidence = float(probs[0, pred_idx].cpu().numpy())
            prediction = 'FAKE' if pred_idx == 1 else 'REAL'
        
        if self.cam is not None:
            try:
                targets = [ClassifierOutputTarget(target_class or pred_idx)]
                grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
            except:
                grayscale_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        else:
            grayscale_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Resize and create output
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = 224, 224
        
        heatmap_resized = cv2.resize(grayscale_cam, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_MAGMA)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            original = image.copy()
        else:
            original = np.array(image)
        
        overlay = cv2.addWeighted(original, 0.5, heatmap_color, 0.5, 0)
        
        processing_time = time.time() - start_time
        
        return HeatmapResult(
            heatmap=heatmap_color,
            overlay=overlay,
            original_image=original,
            method='layercam',
            confidence=confidence,
            prediction=prediction,
            prediction_idx=pred_idx,
            processing_time=processing_time
        )


# ============================================
# INTEGRATED GRADIENTS GENERATOR (Captum)
# ============================================

class IntegratedGradientsGenerator(BaseHeatmapGenerator):
    """
    Generate heatmaps using Integrated Gradients (Captum)
    Provides attribution based on path integration
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        super().__init__(model, device)
        
        if not CAPTUM_AVAILABLE:
            logger.warning("captum not available. Install with: pip install captum")
            self.ig = None
        else:
            self.ig = IntegratedGradients(self.model)
            logger.info(f"Integrated Gradients initialized")
    
    def generate(self, image: Union[str, np.ndarray, Image.Image], 
                target_class: Optional[int] = None,
                steps: int = 50) -> HeatmapResult:
        """Generate Integrated Gradients heatmap"""
        start_time = time.time()
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(dim=1).item()
            confidence = float(probs[0, pred_idx].cpu().numpy())
            prediction = 'FAKE' if pred_idx == 1 else 'REAL'
        
        if self.ig is not None:
            try:
                target = target_class or pred_idx
                attributions = self.ig.attribute(input_tensor, target=target, n_steps=steps)
                heatmap = attributions.squeeze().cpu().detach().numpy()
                heatmap = np.mean(np.abs(heatmap), axis=0)
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            except Exception as e:
                logger.error(f"Integrated Gradients error: {str(e)}")
                heatmap = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        else:
            heatmap = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Resize and create output
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = 224, 224
        
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_PLASMA)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            original = image.copy()
        else:
            original = np.array(image)
        
        overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
        
        processing_time = time.time() - start_time
        
        return HeatmapResult(
            heatmap=heatmap_color,
            overlay=overlay,
            original_image=original,
            method='integrated_gradients',
            confidence=confidence,
            prediction=prediction,
            prediction_idx=pred_idx,
            processing_time=processing_time
        )


# ============================================
# ENSEMBLE HEATMAP GENERATOR
# ============================================

class EnsembleHeatmapGenerator:
    """
    Combine multiple heatmap methods for better visualization
    Uses weighted averaging of different CAM methods
    """
    
    def __init__(self, 
                 model: nn.Module,
                 methods: List[str] = None,
                 device: Optional[str] = None):
        """
        Initialize ensemble heatmap generator
        
        Args:
            model: PyTorch model
            methods: List of methods to use
                    Options: 'gradcam', 'gradcam++', 'scorecam', 'eigencam', 'layercam', 'ig'
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        
        if methods is None:
            methods = ['gradcam', 'gradcam++', 'eigencam']
        
        self.methods = []
        self.method_names = []
        self.weights = {
            'gradcam': 0.3,
            'gradcam++': 0.3,
            'scorecam': 0.2,
            'eigencam': 0.1,
            'layercam': 0.05,
            'ig': 0.05
        }
        
        # Initialize generators
        for method in methods:
            try:
                if method == 'gradcam':
                    self.methods.append(GradCAMGenerator(model, device=self.device))
                    self.method_names.append('gradcam')
                elif method == 'gradcam++':
                    self.methods.append(GradCAMPlusPlusGenerator(model, device=self.device))
                    self.method_names.append('gradcam++')
                elif method == 'scorecam':
                    self.methods.append(ScoreCAMGenerator(model, device=self.device))
                    self.method_names.append('scorecam')
                elif method == 'eigencam':
                    self.methods.append(EigenCAMGenerator(model, device=self.device))
                    self.method_names.append('eigencam')
                elif method == 'layercam':
                    self.methods.append(LayerCAMGenerator(model, device=self.device))
                    self.method_names.append('layercam')
                elif method == 'ig':
                    self.methods.append(IntegratedGradientsGenerator(model, device=self.device))
                    self.method_names.append('ig')
            except Exception as e:
                logger.warning(f"Failed to initialize {method}: {str(e)}")
        
        logger.info(f"EnsembleHeatmapGenerator initialized with methods: {self.method_names}")
    
    def generate(self, 
                image: Union[str, np.ndarray, Image.Image],
                target_class: Optional[int] = None,
                weights: Optional[List[float]] = None) -> HeatmapResult:
        """
        Generate ensemble heatmap
        
        Args:
            image: Input image
            target_class: Target class
            weights: Custom weights for each method
        
        Returns:
            Combined HeatmapResult
        """
        start_time = time.time()
        
        if not self.methods:
            raise RuntimeError("No heatmap methods available")
        
        # Get individual heatmaps
        individual_results = []
        for method in self.methods:
            try:
                result = method.generate(image, target_class)
                individual_results.append(result)
            except Exception as e:
                logger.error(f"Error in {method.__class__.__name__}: {str(e)}")
        
        if not individual_results:
            raise RuntimeError("All heatmap methods failed")
        
        # Get prediction from first result
        first_result = individual_results[0]
        
        # Combine heatmaps (weighted average)
        h, w = first_result.heatmap.shape[:2]
        combined_heatmap = np.zeros((h, w), dtype=np.float32)
        total_weight = 0
        
        for i, result in enumerate(individual_results):
            # Convert to grayscale if needed
            if len(result.heatmap.shape) == 3:
                gray = cv2.cvtColor(result.heatmap, cv2.COLOR_RGB2GRAY)
            else:
                gray = result.heatmap
            
            gray = gray.astype(np.float32) / 255.0
            
            # Apply weight
            method_name = self.method_names[i] if i < len(self.method_names) else 'unknown'
            weight = weights[i] if weights and i < len(weights) else self.weights.get(method_name, 0.2)
            
            combined_heatmap += gray * weight
            total_weight += weight
        
        if total_weight > 0:
            combined_heatmap /= total_weight
        
        # Create combined visualization
        combined_color = cv2.applyColorMap(np.uint8(255 * combined_heatmap), cv2.COLORMAP_JET)
        combined_color = cv2.cvtColor(combined_color, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            original = image.copy()
        else:
            original = np.array(image)
        
        overlay = cv2.addWeighted(original, 0.5, combined_color, 0.5, 0)
        
        processing_time = time.time() - start_time
        
        return HeatmapResult(
            heatmap=combined_color,
            overlay=overlay,
            original_image=original,
            method=f"ensemble_{'_'.join(self.method_names)}",
            confidence=first_result.confidence,
            prediction=first_result.prediction,
            prediction_idx=first_result.prediction_idx,
            processing_time=processing_time
        )


# ============================================
# HEATMAP VISUALIZER
# ============================================

class HeatmapVisualizer:
    """
    Advanced heatmap visualization utilities
    """
    
    def __init__(self):
        """Initialize heatmap visualizer"""
        # Custom colormaps
        self.deepfake_cmap = LinearSegmentedColormap.from_list(
            'deepfake',
            ['green', 'yellow', 'red']
        )
        
        self.manipulation_cmap = LinearSegmentedColormap.from_list(
            'manipulation',
            ['blue', 'cyan', 'yellow', 'red']
        )
    
    def overlay(self, 
               image: np.ndarray,
               heatmap: np.ndarray,
               alpha: float = 0.5,
               colormap: str = 'jet') -> np.ndarray:
        """
        Overlay heatmap on image with specified colormap
        
        Args:
            image: Original image (RGB)
            heatmap: Heatmap array
            alpha: Transparency of heatmap
            colormap: Colormap name
        
        Returns:
            Overlay image
        """
        # Ensure image is in correct format
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Resize heatmap to match image
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        if colormap == 'deepfake':
            colored = self.deepfake_cmap(heatmap)[:, :, :3]
        elif colormap == 'manipulation':
            colored = self.manipulation_cmap(heatmap)[:, :, :3]
        else:
            # Use matplotlib colormap
            cmap = cm.get_cmap(colormap)
            colored = cmap(heatmap)[:, :, :3]
        
        # Overlay
        overlay = (1 - alpha) * image + alpha * colored
        overlay = np.clip(overlay, 0, 1)
        
        return (overlay * 255).astype(np.uint8)
    
    def blend(self, image: np.ndarray, heatmap: np.ndarray, mode: str = 'screen') -> np.ndarray:
        """
        Blend image and heatmap using different blending modes
        
        Args:
            image: Original image
            heatmap: Heatmap array
            mode: 'screen', 'multiply', 'overlay', 'soft_light'
        
        Returns:
            Blended image
        """
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        if heatmap.max() > 1.0:
            heatmap = heatmap / 255.0
        
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Create 3-channel heatmap
        heatmap_3ch = np.stack([heatmap] * 3, axis=-1)
        
        if mode == 'screen':
            result = 1 - (1 - image) * (1 - heatmap_3ch)
        elif mode == 'multiply':
            result = image * heatmap_3ch
        elif mode == 'overlay':
            mask = image < 0.5
            result = np.zeros_like(image)
            result[mask] = 2 * image[mask] * heatmap_3ch[mask]
            result[~mask] = 1 - 2 * (1 - image[~mask]) * (1 - heatmap_3ch[~mask])
        elif mode == 'soft_light':
            result = (1 - 2 * heatmap_3ch) * image ** 2 + 2 * heatmap_3ch * image
        else:
            result = (image + heatmap_3ch) / 2
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    def create_grid(self, 
                   original: np.ndarray,
                   heatmap: np.ndarray,
                   overlay: np.ndarray,
                   contours: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create grid visualization with original, heatmap, overlay, and contours
        
        Returns:
            Grid image
        """
        h, w = original.shape[:2]
        
        # Convert heatmap to RGB if grayscale
        if len(heatmap.shape) == 2:
            heatmap_rgb = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
        else:
            heatmap_rgb = heatmap
        
        # Create grid
        grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Place images
        grid[:h, :w] = original
        grid[:h, w:w*2] = heatmap_rgb
        grid[h:h*2, :w] = overlay
        
        if contours is not None:
            grid[h:h*2, w:w*2] = contours
        else:
            # Create blended version
            blended = cv2.addWeighted(original, 0.5, heatmap_rgb, 0.5, 0)
            grid[h:h*2, w:w*2] = blended
        
        return grid
    
    def add_contours(self,
                    image: np.ndarray,
                    heatmap: np.ndarray,
                    threshold: float = 0.6,
                    color: Tuple[int, int, int] = (255, 0, 0),
                    thickness: int = 2) -> np.ndarray:
        """
        Add contour lines around high-activation regions
        
        Args:
            image: Original image
            heatmap: Heatmap array
            threshold: Threshold for contour
            color: Contour color (BGR)
            thickness: Line thickness
        
        Returns:
            Image with contours
        """
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Create binary mask
        mask = (heatmap > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        result = image.copy()
        cv2.drawContours(result, contours, -1, color, thickness)
        
        return result
    
    def highlight_regions(self,
                         image: np.ndarray,
                         heatmap: np.ndarray,
                         threshold: float = 0.7,
                         color: Tuple[int, int, int] = (255, 0, 0),
                         alpha: float = 0.3) -> np.ndarray:
        """
        Highlight high-activation regions with colored overlay
        
        Args:
            image: Original image
            heatmap: Heatmap array
            threshold: Threshold for highlighting
            color: Highlight color (BGR)
            alpha: Transparency of highlight
        
        Returns:
            Image with highlighted regions
        """
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Create mask
        mask = (heatmap > threshold).astype(np.float32)
        
        # Create colored overlay
        overlay = np.zeros_like(image, dtype=np.float32)
        overlay[:, :] = color
        
        # Blend
        result = image.astype(np.float32)
        for c in range(3):
            result[:, :, c] = (1 - mask * alpha) * result[:, :, c] + mask * alpha * overlay[:, :, c]
        
        return result.astype(np.uint8)
    
    def find_regions(self,
                    heatmap: np.ndarray,
                    threshold: float = 0.7,
                    min_area: int = 100) -> List[Dict]:
        """
        Find regions with high heatmap values
        
        Args:
            heatmap: Heatmap array
            threshold: Threshold for region detection
            min_area: Minimum area for region
        
        Returns:
            List of region dictionaries
        """
        # Create binary mask
        mask = (heatmap > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get heatmap stats in region
            region_heatmap = heatmap[y:y+h, x:x+w]
            avg_value = float(np.mean(region_heatmap))
            max_value = float(np.max(region_heatmap))
            
            # Get contour points (simplified)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2).tolist()
            
            regions.append({
                'id': i,
                'area': int(area),
                'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'center': {'x': int(x + w/2), 'y': int(y + h/2)},
                'avg_intensity': avg_value,
                'max_intensity': max_value,
                'contour_points': points
            })
        
        # Sort by intensity
        regions.sort(key=lambda x: x['max_intensity'], reverse=True)
        
        return regions


# ============================================
# FACTORY CLASS
# ============================================

class HeatmapFactory:
    """Factory for creating heatmap generators"""
    
    @staticmethod
    def create_generator(model: nn.Module, 
                        method: str = 'gradcam',
                        **kwargs) -> BaseHeatmapGenerator:
        """
        Create heatmap generator
        
        Args:
            model: PyTorch model
            method: 'gradcam', 'gradcam++', 'scorecam', 'eigencam', 'layercam', 'ig', 'ensemble'
            **kwargs: Additional arguments
        
        Returns:
            Heatmap generator
        """
        if method == 'gradcam':
            return GradCAMGenerator(model, **kwargs)
        elif method == 'gradcam++':
            return GradCAMPlusPlusGenerator(model, **kwargs)
        elif method == 'scorecam':
            return ScoreCAMGenerator(model, **kwargs)
        elif method == 'eigencam':
            return EigenCAMGenerator(model, **kwargs)
        elif method == 'layercam':
            return LayerCAMGenerator(model, **kwargs)
        elif method == 'ig':
            return IntegratedGradientsGenerator(model, **kwargs)
        elif method == 'ensemble':
            return EnsembleHeatmapGenerator(model, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def create_visualizer() -> HeatmapVisualizer:
        """Create heatmap visualizer"""
        return HeatmapVisualizer()


# ============================================
# TESTING FUNCTION
# ============================================

def test_heatmap():
    """Test heatmap generation"""
    print("=" * 60)
    print("TESTING HEATMAP GENERATION")
    print("=" * 60)
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 2, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(2, 2)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = TestModel()
    model.eval()
    
    # Create test image
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_img_path = "test_heatmap.jpg"
    cv2.imwrite(test_img_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    
    try:
        # Test Grad-CAM
        print("\n1️⃣ Testing Grad-CAM...")
        generator = HeatmapFactory.create_generator(model, method='gradcam')
        result = generator.generate(test_img_path)
        print(f"✅ Grad-CAM generated")
        print(f"   Heatmap shape: {result.heatmap.shape}")
        print(f"   Prediction: {result.prediction} ({result.confidence:.4f})")
        print(f"   Time: {result.processing_time:.3f}s")
        
        # Test visualizer
        print("\n2️⃣ Testing Visualizer...")
        visualizer = HeatmapVisualizer()
        overlay = visualizer.overlay(test_img, result.heatmap, alpha=0.6)
        print(f"✅ Overlay created: {overlay.shape}")
        
        # Test region detection
        regions = visualizer.find_regions(result.heatmap, threshold=0.7)
        print(f"\n3️⃣ Found {len(regions)} manipulated regions")
        for i, region in enumerate(regions[:3]):
            print(f"   Region {i+1}: area={region['area']}, confidence={region['avg_intensity']:.3f}")
        
        # Test ensemble
        print("\n4️⃣ Testing Ensemble...")
        ensemble = EnsembleHeatmapGenerator(model, methods=['gradcam', 'eigencam'])
        ensemble_result = ensemble.generate(test_img_path)
        print(f"✅ Ensemble generated")
        print(f"   Heatmap shape: {ensemble_result.heatmap.shape}")
        
        # Clean up
        os.remove(test_img_path)
        
        print("\n" + "=" * 60)
        print("✅ HEATMAP TEST PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up
        if os.path.exists(test_img_path):
            os.remove(test_img_path)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    test_heatmap()
    
    print("\n📝 Example usage:")
    print("""
    from app.utils.heatmap import HeatmapFactory, HeatmapVisualizer
    from app.models.ensemble import EnsembleFactory
    
    # Load model
    model = EnsembleFactory.create_accurate_ensemble().model
    
    # Create heatmap generator
    generator = HeatmapFactory.create_generator(model, method='ensemble')
    
    # Generate heatmap
    result = generator.generate('path/to/image.jpg')
    
    # Visualize
    visualizer = HeatmapVisualizer()
    overlay = visualizer.overlay(result.original_image, result.heatmap)
    
    # Find manipulated regions
    regions = visualizer.find_regions(result.heatmap, threshold=0.7)
    for region in regions:
        print(f"Manipulated region at {region['bounding_box']} with confidence {region['avg_intensity']:.3f}")
    """)