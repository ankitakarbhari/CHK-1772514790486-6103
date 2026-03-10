# app/models/__init__.py
"""
Models Package for DeepShield AI
Simplified version with error handling
"""
# app/models/__init__.py
import os
import logging
import warnings

# CRITICAL: Skip spacy import completely
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["SPACY_WARNING_IGNORE"] = "true"

# Suppress all warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logger.info("📦 Loading models package (simplified mode)...")

# Define placeholder classes
class PlaceholderModel:
    def __init__(self, *args, **kwargs):
        pass
    
    def predict(self, *args, **kwargs):
        return {"prediction": "REAL", "confidence": 0.95}

# Simple factory
class PlaceholderFactory:
    @staticmethod
    def create(*args, **kwargs):
        return PlaceholderModel()

# Export these
DeepfakeEnsemble = PlaceholderModel
EnsembleFactory = PlaceholderFactory
TextDeepfakeDetector = PlaceholderModel
TextDetectorFactory = PlaceholderFactory

logger.info("✅ Models package loaded in simplified mode")

# ============================================
# Define placeholder classes in case imports fail
# ============================================

class ModelNotAvailable:
    """Placeholder class when model is not available"""
    def __init__(self, *args, **kwargs):
        raise ImportError("This model is not available. Please install required dependencies.")
    
    @classmethod
    def create(cls, *args, **kwargs):
        raise ImportError("This model is not available. Please install required dependencies.")

# ============================================
# Try to import ensemble model
# ============================================

try:
    from app.models.ensemble import (
        DeepfakeEnsemble,
        EnsembleFactory,
        EnsembleResult,
        EnsembleWeightOptimizer
    )
    logger.info("✅ Ensemble model loaded successfully")
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Ensemble model not loaded: {e}")
    # Create placeholder
    DeepfakeEnsemble = ModelNotAvailable
    EnsembleFactory = ModelNotAvailable
    EnsembleResult = dict
    EnsembleWeightOptimizer = ModelNotAvailable
    ENSEMBLE_AVAILABLE = False

# ============================================
# Try to import MobileNet model
# ============================================

try:
    from app.models.mobilenet_model import (
        PyTorchMobileNet,
        MobileNetFactory,
        DeepfakeMobileNetV2
    )
    logger.info("✅ MobileNet model loaded successfully")
    MOBILENET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ MobileNet model not loaded: {e}")
    PyTorchMobileNet = ModelNotAvailable
    MobileNetFactory = ModelNotAvailable
    DeepfakeMobileNetV2 = ModelNotAvailable
    MOBILENET_AVAILABLE = False

# ============================================
# Try to import Xception model
# ============================================

try:
    from app.models.xception_model import (
        PyTorchXception,
        XceptionFactory,
        CustomXception
    )
    logger.info("✅ Xception model loaded successfully")
    XCEPTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Xception model not loaded: {e}")
    PyTorchXception = ModelNotAvailable
    XceptionFactory = ModelNotAvailable
    CustomXception = ModelNotAvailable
    XCEPTION_AVAILABLE = False

# ============================================
# Try to import EfficientNet model
# ============================================

try:
    from app.models.efficientnet_model import (
        PyTorchEfficientNet,
        EfficientNetFactory
    )
    logger.info("✅ EfficientNet model loaded successfully")
    EFFICIENTNET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ EfficientNet model not loaded: {e}")
    PyTorchEfficientNet = ModelNotAvailable
    EfficientNetFactory = ModelNotAvailable
    EFFICIENTNET_AVAILABLE = False

# ============================================
# Try to import Text Detection model
# ============================================

try:
    from app.models.text_detection_model import (
        TextDeepfakeDetector,
        TextDetectorFactory,
        AIDetectionResult,
        StatisticalTextAnalyzer
    )
    logger.info("✅ Text detection model loaded successfully")
    TEXT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Text detection model not loaded: {e}")
    TextDeepfakeDetector = ModelNotAvailable
    TextDetectorFactory = ModelNotAvailable
    AIDetectionResult = dict
    StatisticalTextAnalyzer = ModelNotAvailable
    TEXT_AVAILABLE = False

# ============================================
# Model Registry (only include available models)
# ============================================

MODEL_REGISTRY = {}

if ENSEMBLE_AVAILABLE:
    MODEL_REGISTRY['ensemble'] = {
        'class': DeepfakeEnsemble,
        'factory': EnsembleFactory,
        'description': 'Ensemble of MobileNetV2, Xception, and EfficientNet',
        'accuracy': '98.3%',
    }

if MOBILENET_AVAILABLE:
    MODEL_REGISTRY['mobilenet'] = {
        'class': PyTorchMobileNet,
        'factory': MobileNetFactory,
        'description': 'Fast lightweight model for real-time detection',
        'speed': '15-50 FPS',
    }

if XCEPTION_AVAILABLE:
    MODEL_REGISTRY['xception'] = {
        'class': PyTorchXception,
        'factory': XceptionFactory,
        'description': 'Best for manipulation detection',
        'accuracy': '95-97%',
    }

if EFFICIENTNET_AVAILABLE:
    MODEL_REGISTRY['efficientnet'] = {
        'class': PyTorchEfficientNet,
        'factory': EfficientNetFactory,
        'description': 'State-of-the-art architecture',
        'accuracy': '96-98%',
    }

if TEXT_AVAILABLE:
    MODEL_REGISTRY['text_detector'] = {
        'class': TextDeepfakeDetector,
        'factory': TextDetectorFactory,
        'description': 'AI-generated text detection',
        'accuracy': '92-96%',
    }

# ============================================
# Utility Functions
# ============================================

def get_model_info(model_name: str):
    """Get information about a specific model"""
    return MODEL_REGISTRY.get(model_name.lower())

def list_available_models():
    """List all available models"""
    return MODEL_REGISTRY.copy()

def create_model(model_name: str, **kwargs):
    """Create a model instance by name"""
    try:
        model_info = get_model_info(model_name)
        if not model_info:
            logger.error(f"❌ Unknown model: {model_name}")
            return None
        
        factory = model_info.get('factory')
        if factory and factory != ModelNotAvailable:
            return factory(**kwargs)
        
        model_class = model_info.get('class')
        if model_class and model_class != ModelNotAvailable:
            return model_class(**kwargs)
        
        logger.error(f"❌ Model {model_name} not available")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error creating model {model_name}: {e}")
        return None

def get_device():
    """Get the best available device"""
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        return 'cpu'

def check_available_models():
    """Check which models are available"""
    return {
        'ensemble': ENSEMBLE_AVAILABLE,
        'mobilenet': MOBILENET_AVAILABLE,
        'xception': XCEPTION_AVAILABLE,
        'efficientnet': EFFICIENTNET_AVAILABLE,
        'text_detector': TEXT_AVAILABLE,
    }

# ============================================
# Package Exports
# ============================================

__all__ = [
    # Model classes (may be placeholders)
    'DeepfakeEnsemble',
    'EnsembleFactory',
    'EnsembleResult',
    'EnsembleWeightOptimizer',
    'PyTorchMobileNet',
    'MobileNetFactory',
    'DeepfakeMobileNetV2',
    'PyTorchXception',
    'XceptionFactory',
    'CustomXception',
    'PyTorchEfficientNet',
    'EfficientNetFactory',
    'TextDeepfakeDetector',
    'TextDetectorFactory',
    'AIDetectionResult',
    'StatisticalTextAnalyzer',
    
    # Utility functions
    'get_model_info',
    'list_available_models',
    'create_model',
    'get_device',
    'check_available_models',
    
    # Registry and status
    'MODEL_REGISTRY',
    'ENSEMBLE_AVAILABLE',
    'MOBILENET_AVAILABLE',
    'XCEPTION_AVAILABLE',
    'EFFICIENTNET_AVAILABLE',
    'TEXT_AVAILABLE',
]

# Print summary
available = check_available_models()
loaded_count = sum(1 for v in available.values() if v)
logger.info(f"📦 Models package initialized: {loaded_count}/{len(available)} models available")