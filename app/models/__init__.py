# app/models/__init__.py
"""
Models Package for Deepfake Detection System
Exports all model classes and factory functions for easy importing
Supports image, video, audio, and text deepfake detection models
Version: 1.0.0
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================
# VERSION INFORMATION
# ============================================

__version__ = "1.0.0"
__author__ = "Deepfake Detection System"
__description__ = "AI-Based Deepfake Detection Models for Images, Video, Audio, and Text"


# ============================================
# IMPORT ALL MODELS
# ============================================

# Ensemble Model
from app.models.ensemble import (
    DeepfakeEnsemble,
    EnsembleFactory,
    EnsembleResult,
    ModelPrediction,
    EnsembleWeightOptimizer
)

# MobileNet Model
from app.models.mobilenet_model import (
    PyTorchMobileNet,
    KerasMobileNet,
    MobileNetFactory,
    DeepfakeMobileNetV2
)

# Xception Model
from app.models.xception_model import (
    PyTorchXception,
    KerasXception,
    XceptionFactory,
    CustomXception,
    SeparableConv2d,
    XceptionBlock
)

# EfficientNet Model
from app.models.efficientnet_model import (
    PyTorchEfficientNet,
    KerasEfficientNet,
    EfficientNetFactory
)

# Text Detection Model
from app.models.text_detection_model import (
    TextDeepfakeDetector,
    TextDetectorFactory,
    AIDetectionResult,
    TextMetrics,
    StatisticalTextAnalyzer,
    PerplexityDetector,
    BERTAIDetector,
    GPTZeroStyleDetector,
    PlagiarismDetector
)


# ============================================
# MODEL REGISTRY
# ============================================

MODEL_REGISTRY = {
    # Image/Video Models
    'ensemble': {
        'class': DeepfakeEnsemble,
        'factory': EnsembleFactory,
        'description': 'Ensemble of MobileNetV2, Xception, and EfficientNet',
        'accuracy': '98%+',
        'speed': 'Medium',
        'type': 'image',
        'input_size': '224x224 (299x299 for Xception)',
        'parameters': '~45M'
    },
    'mobilenet': {
        'class': PyTorchMobileNet,
        'factory': MobileNetFactory,
        'description': 'MobileNetV2 - Fast, lightweight model for real-time detection',
        'accuracy': '92-95%',
        'speed': 'Fast (15-50 FPS)',
        'type': 'image',
        'input_size': '224x224',
        'parameters': '~3.5M'
    },
    'xception': {
        'class': PyTorchXception,
        'factory': XceptionFactory,
        'description': 'Xception - Best for manipulation detection and face swaps',
        'accuracy': '95-97%',
        'speed': 'Medium',
        'type': 'image',
        'input_size': '299x299',
        'parameters': '~22M'
    },
    'efficientnet': {
        'class': PyTorchEfficientNet,
        'factory': EfficientNetFactory,
        'description': 'EfficientNet - State-of-the-art architecture',
        'accuracy': '96-98%',
        'speed': 'Slow',
        'type': 'image',
        'input_size': '224x224',
        'parameters': '~12M (B3)'
    },
    
    # Text Models
    'text_detector': {
        'class': TextDeepfakeDetector,
        'factory': TextDetectorFactory,
        'description': 'AI-generated text detection (GPT, Claude, LLaMA)',
        'accuracy': '92-96%',
        'speed': 'Fast',
        'type': 'text',
        'input_size': 'Up to 512 tokens',
        'parameters': '~125M (with BERT)'
    },
    'bert_detector': {
        'class': BERTAIDetector,
        'description': 'BERT/RoBERTa-based text classifier',
        'accuracy': '94-97%',
        'speed': 'Medium',
        'type': 'text',
        'input_size': 'Up to 512 tokens',
        'parameters': '~125M'
    },
    'perplexity': {
        'class': PerplexityDetector,
        'description': 'GPT-2 perplexity based detector',
        'accuracy': '85-90%',
        'speed': 'Medium',
        'type': 'text',
        'input_size': 'Up to 1024 tokens',
        'parameters': '~124M'
    }
}


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_model_info(model_name: str) -> Optional[Dict]:
    """
    Get information about a specific model
    
    Args:
        model_name: Name of the model ('ensemble', 'mobilenet', 'xception', etc.)
    
    Returns:
        Dictionary with model information or None if not found
    
    Example:
        >>> info = get_model_info('mobilenet')
        >>> print(info['description'])
        MobileNetV2 - Fast, lightweight model for real-time detection
    """
    return MODEL_REGISTRY.get(model_name.lower())


def list_available_models(model_type: Optional[str] = None) -> Dict[str, Dict]:
    """
    List all available models with their information
    
    Args:
        model_type: Filter by type ('image', 'text', or None for all)
    
    Returns:
        Dictionary of model names and their info
    
    Example:
        >>> image_models = list_available_models('image')
        >>> for name, info in image_models.items():
        ...     print(f"{name}: {info['description']}")
    """
    if model_type:
        return {name: info for name, info in MODEL_REGISTRY.items() 
                if info.get('type') == model_type}
    return MODEL_REGISTRY.copy()


def create_model(model_name: str, **kwargs) -> Optional[Any]:
    """
    Create a model instance by name with automatic device selection
    
    Args:
        model_name: Name of the model to create
        **kwargs: Additional arguments for the model constructor
    
    Returns:
        Model instance or None if creation fails
    
    Example:
        >>> # Create ensemble model on GPU
        >>> model = create_model('ensemble', device='cuda')
        >>> 
        >>> # Create fast MobileNet for real-time
        >>> model = create_model('mobilenet', model_size='v2', use_custom=False)
        >>> 
        >>> # Create text detector
        >>> detector = create_model('text_detector')
    """
    try:
        model_info = get_model_info(model_name)
        if not model_info:
            logger.error(f"Unknown model: {model_name}")
            print(f"❌ Unknown model: {model_name}")
            print(f"   Available models: {', '.join(MODEL_REGISTRY.keys())}")
            return None
        
        # Use factory if available (preferred method)
        factory = model_info.get('factory')
        if factory:
            if model_name == 'ensemble':
                # For ensemble, create accurate version by default
                return factory.create_accurate_ensemble(**kwargs)
            elif model_name == 'mobilenet':
                # For MobileNet, create fast version by default
                return factory.create_fast_model(**kwargs)
            elif model_name == 'xception':
                # For Xception, create custom version by default
                return factory.create_custom_model(**kwargs)
            elif model_name == 'efficientnet':
                # For EfficientNet, create PyTorch version
                return factory.create_pytorch_model(**kwargs)
            elif model_name == 'text_detector':
                # For text detector, create full version
                return factory.create_full_detector(**kwargs)
        
        # Direct instantiation if no factory
        model_class = model_info.get('class')
        if model_class:
            return model_class(**kwargs)
        
        logger.warning(f"No factory or class found for {model_name}")
        return None
        
    except Exception as e:
        logger.error(f"Error creating model {model_name}: {str(e)}")
        return None


def load_model_from_path(model_name: str, model_path: Union[str, Path], **kwargs) -> Optional[Any]:
    """
    Load a model from a saved file
    
    Args:
        model_name: Name of the model
        model_path: Path to the saved model file
        **kwargs: Additional arguments for model creation
    
    Returns:
        Loaded model instance or None if loading fails
    
    Example:
        >>> model = load_model_from_path('mobilenet', 'models/mobilenetv2_deepfake.pt')
    """
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    model = create_model(model_name, **kwargs)
    if model and hasattr(model, 'load_model'):
        try:
            model.load_model(str(model_path))
            logger.info(f"✅ Loaded {model_name} from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    else:
        logger.error(f"Model {model_name} does not have load_model method")
        return None


def get_model_versions() -> Dict[str, str]:
    """
    Get versions of all underlying libraries
    
    Returns:
        Dictionary of library versions
    
    Example:
        >>> versions = get_model_versions()
        >>> print(versions['torch'])
        2.7.0+cu118
    """
    versions = {
        'models_package': __version__,
    }
    
    # PyTorch
    try:
        import torch
        versions['torch'] = torch.__version__
        versions['cuda_available'] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions['cuda_version'] = torch.version.cuda
    except:
        versions['torch'] = 'not installed'
    
    # TensorFlow / Keras
    try:
        import tensorflow as tf
        versions['tensorflow'] = tf.__version__
    except:
        versions['tensorflow'] = 'not installed'
    
    try:
        import keras
        versions['keras'] = keras.__version__
    except:
        versions['keras'] = 'not installed'
    
    # Transformers
    try:
        import transformers
        versions['transformers'] = transformers.__version__
    except:
        versions['transformers'] = 'not installed'
    
    # NLP libraries
    try:
        import nltk
        versions['nltk'] = nltk.__version__
    except:
        versions['nltk'] = 'not installed'
    
    try:
        import spacy
        versions['spacy'] = spacy.__version__
    except:
        versions['spacy'] = 'not installed'
    
    return versions


def get_device(device: Optional[str] = None) -> str:
    """
    Get the best available device
    
    Args:
        device: Preferred device ('cuda', 'cpu', or None for auto)
    
    Returns:
        Device string ('cuda:0' or 'cpu')
    
    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
    """
    if device is not None:
        return device
    
    import torch
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def print_model_summary(models: Optional[Dict] = None):
    """
    Print a formatted summary of models
    
    Args:
        models: Dictionary of model instances (optional)
    
    Example:
        >>> print_model_summary()
        >>> # or
        >>> models = initialize_models()
        >>> print_model_summary(models)
    """
    print("\n" + "=" * 80)
    print("📊 DEEPFAKE DETECTION MODELS SUMMARY")
    print("=" * 80)
    
    if models:
        # Print loaded instances
        print("\n📦 Loaded Models:")
        for name, model in models.items():
            if hasattr(model, 'count_parameters'):
                params = model.count_parameters()
                print(f"  • {name}: {params:,} parameters")
            else:
                print(f"  • {name}: {type(model).__name__}")
    else:
        # Print registry
        for model_type in ['image', 'text']:
            print(f"\n🖼️  {model_type.upper()} MODELS:")
            models_of_type = list_available_models(model_type)
            for name, info in models_of_type.items():
                print(f"  • {name}:")
                print(f"    Description: {info['description']}")
                print(f"    Accuracy: {info['accuracy']}, Speed: {info['speed']}")
                print(f"    Input: {info['input_size']}, Params: {info['parameters']}")
    
    print("\n" + "=" * 80)


# ============================================
# MODEL INITIALIZATION
# ============================================

def initialize_models(device: Optional[str] = None, 
                     verbose: bool = True,
                     model_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Initialize all models (lazy loading)
    
    Args:
        device: Device to load models on ('cuda' or 'cpu')
        verbose: Whether to print initialization info
        model_types: List of model types to initialize ('image', 'text')
    
    Returns:
        Dictionary of initialized models
    
    Example:
        >>> # Initialize all models
        >>> models = initialize_models()
        >>> 
        >>> # Initialize only image models on GPU
        >>> models = initialize_models(device='cuda', model_types=['image'])
    """
    if device is None:
        device = get_device()
    
    if verbose:
        print("\n" + "=" * 60)
        print("🚀 INITIALIZING DEEPFAKE DETECTION MODELS")
        print("=" * 60)
        print(f"📦 Using device: {device}")
    
    models = {}
    
    if model_types is None or 'image' in model_types:
        if verbose:
            print("\n🖼️  Image/Video Models:")
        
        # Initialize ensemble model
        try:
            models['ensemble'] = EnsembleFactory.create_accurate_ensemble(device=device)
            if verbose:
                params = models['ensemble'].model.count_parameters() if hasattr(models['ensemble'], 'model') else '?'
                print(f"  ✅ Ensemble Model (MobileNet+Xception+EfficientNet) - {params:,} params")
        except Exception as e:
            if verbose:
                print(f"  ❌ Ensemble Model: {str(e)[:100]}")
        
        # Initialize MobileNet
        try:
            models['mobilenet'] = MobileNetFactory.create_fast_model(device=device)
            if verbose:
                params = models['mobilenet'].count_parameters()
                print(f"  ✅ MobileNetV2 (Fast, 15-50 FPS) - {params:,} params")
        except Exception as e:
            if verbose:
                print(f"  ❌ MobileNetV2: {str(e)[:100]}")
        
        # Initialize Xception
        try:
            models['xception'] = XceptionFactory.create_custom_model(device=device)
            if verbose:
                params = models['xception'].count_parameters()
                print(f"  ✅ Xception (Manipulation Detection) - {params:,} params")
        except Exception as e:
            if verbose:
                print(f"  ❌ Xception: {str(e)[:100]}")
        
        # Initialize EfficientNet
        try:
            models['efficientnet'] = EfficientNetFactory.create_pytorch_model(device=device)
            if verbose:
                params = models['efficientnet'].count_parameters()
                print(f"  ✅ EfficientNet (State-of-the-art) - {params:,} params")
        except Exception as e:
            if verbose:
                print(f"  ❌ EfficientNet: {str(e)[:100]}")
    
    if model_types is None or 'text' in model_types:
        if verbose:
            print("\n📝 Text Models:")
        
        # Initialize text detector
        try:
            models['text_detector'] = TextDetectorFactory.create_full_detector()
            if verbose:
                print(f"  ✅ Text Detector (AI-generated text detection)")
        except Exception as e:
            if verbose:
                print(f"  ❌ Text Detector: {str(e)[:100]}")
    
    if verbose:
        print(f"\n✅ Initialized {len(models)} models successfully")
        print("=" * 60)
    
    return models


def get_model_by_type(model_type: str, device: Optional[str] = None) -> Optional[Any]:
    """
    Get the best model for a specific type
    
    Args:
        model_type: Type of model ('image', 'text')
        device: Device to load on
    
    Returns:
        Best model instance for the type
    
    Example:
        >>> image_model = get_model_by_type('image')
        >>> text_model = get_model_by_type('text')
    """
    if model_type == 'image':
        try:
            return EnsembleFactory.create_accurate_ensemble(device=device)
        except:
            try:
                return XceptionFactory.create_custom_model(device=device)
            except:
                return MobileNetFactory.create_fast_model(device=device)
    elif model_type == 'text':
        try:
            return TextDetectorFactory.create_full_detector()
        except:
            return TextDetectorFactory.create_lightweight_detector()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================
# CLEANUP FUNCTIONS
# ============================================

def cleanup_models(models: Dict):
    """
    Clean up models and free GPU memory
    
    Args:
        models: Dictionary of models to clean up
    
    Example:
        >>> models = initialize_models()
        >>> # ... use models ...
        >>> cleanup_models(models)
    """
    for name, model in models.items():
        try:
            # Move to CPU
            if hasattr(model, 'to') and callable(getattr(model, 'to')):
                model.to('cpu')
            if hasattr(model, 'model') and hasattr(model.model, 'to'):
                model.model.to('cpu')
            
            # Delete model
            del model
            
        except Exception as e:
            logger.warning(f"Error cleaning up {name}: {str(e)}")
    
    # Clear GPU cache
    import torch
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    
    logger.info("🧹 Models cleaned up, GPU memory freed")


def clear_cache():
    """Clear model cache and free memory"""
    import torch
    import gc
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    
    logger.info("🧹 Cache cleared")


# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    # Ensemble exports
    'DeepfakeEnsemble',
    'EnsembleFactory',
    'EnsembleResult',
    'ModelPrediction',
    'EnsembleWeightOptimizer',
    
    # MobileNet exports
    'PyTorchMobileNet',
    'KerasMobileNet',
    'MobileNetFactory',
    'DeepfakeMobileNetV2',
    
    # Xception exports
    'PyTorchXception',
    'KerasXception',
    'XceptionFactory',
    'CustomXception',
    'SeparableConv2d',
    'XceptionBlock',
    
    # EfficientNet exports
    'PyTorchEfficientNet',
    'KerasEfficientNet',
    'EfficientNetFactory',
    
    # Text Detection exports
    'TextDeepfakeDetector',
    'TextDetectorFactory',
    'AIDetectionResult',
    'TextMetrics',
    'StatisticalTextAnalyzer',
    'PerplexityDetector',
    'BERTAIDetector',
    'GPTZeroStyleDetector',
    'PlagiarismDetector',
    
    # Utility functions
    'get_model_info',
    'list_available_models',
    'create_model',
    'load_model_from_path',
    'get_model_versions',
    'get_device',
    'initialize_models',
    'get_model_by_type',
    'print_model_summary',
    'cleanup_models',
    'clear_cache',
    
    # Registry and version info
    'MODEL_REGISTRY',
    '__version__',
    '__author__',
    '__description__'
]


# ============================================
# PACKAGE INITIALIZATION
# ============================================

# Log package initialization
logger.debug(f"Models package v{__version__} loaded")
logger.debug(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")