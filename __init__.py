# echogrid/__init__.py
"""
EchoGrid: Community-powered text-to-speech library.

EchoGrid provides access to thousands of community-created voice models
for text-to-speech generation, with support for both local inference
and hosted API endpoints.

This library makes it easy to:

- Browse and discover community voice models
- Generate high-quality speech from text
- Clone voices from audio samples
- Fine-tune existing models
- Train new models from scratch
- Process and enhance audio files

Examples:
    Basic text-to-speech generation::

        import echogrid as eg
        
        model = eg.load("alice/uk-teen-female")
        audio = model.tts("Hello, world!")
        audio.save("output.wav")

    Voice cloning::

        cloned = eg.clone_voice(
            reference_audio="my_voice.wav",
            name="my-personal-voice"
        )
        audio = cloned.tts("This is my cloned voice!")

    Browse community models::

        models = eg.browse(language="en", category="narrator")
        for model in models:
            print(f"{model.name}: {model.rating}/5")

Note:
    This library requires Python 3.8 or later. Some features require
    additional dependencies which can be installed using extras:
    
    - ``pip install echogrid[local]`` - Local inference with PyTorch
    - ``pip install echogrid[training]`` - Model training capabilities  
    - ``pip install echogrid[audio]`` - Advanced audio processing
    - ``pip install echogrid[full]`` - All optional dependencies
"""

import sys
import warnings
from typing import Optional, List

# Version information
from .version import __version__, __version_info__, get_version_info

# Configuration management
from .config import configure, get_config, configure_session

# Core exceptions
from .exceptions import (
    EchoGridError,
    ModelNotFoundError,
    InsufficientMemoryError,
    APIRateLimitError,
    TrainingError,
    AudioProcessingError,
    ConfigurationError,
    NetworkError,
    ValidationError,
    AuthenticationError,
    PermissionError,
    ModelArchitectureError,
)

# Core functionality - wrapped in try/except for graceful degradation
try:
    from .core.discovery import browse, search, info
    from .core.models import load, list_models
    from .core.cache import cache
    _core_available = True
except ImportError as e:
    warnings.warn(f"Core functionality not available: {e}")
    _core_available = False

# API client - optional dependency
try:
    from .api.client import set_api_key, configure_api, api
    _api_available = True
except ImportError:
    _api_available = False
    
    def set_api_key(*args, **kwargs):
        raise ImportError("API client not available. Install with: pip install echogrid[api]")
    
    def configure_api(*args, **kwargs):
        raise ImportError("API client not available. Install with: pip install echogrid[api]")
    
    api = None

# Audio processing - optional dependency
try:
    from .audio.file import AudioFile
    from .audio.processing import process_audio_batch
    from .audio.conversion import convert_audio, convert_batch
    _audio_available = True
except ImportError:
    _audio_available = False
    
    class AudioFile:
        def __init__(self, *args, **kwargs):
            raise ImportError("AudioFile not available. Install with: pip install echogrid[audio]")
    
    def process_audio_batch(*args, **kwargs):
        raise ImportError("Audio processing not available. Install with: pip install echogrid[audio]")
    
    def convert_audio(*args, **kwargs):
        raise ImportError("Audio conversion not available. Install with: pip install echogrid[audio]")
    
    def convert_batch(*args, **kwargs):
        raise ImportError("Audio conversion not available. Install with: pip install echogrid[audio]")

# Training functionality - optional dependency
try:
    from .training.clone import clone_voice
    from .training.fine_tune import fine_tune
    from .training.train import train_model
    from .training.datasets import AudioDataset
    _training_available = True
except ImportError:
    _training_available = False
    
    def clone_voice(*args, **kwargs):
        raise ImportError("Voice cloning not available. Install with: pip install echogrid[training]")
    
    def fine_tune(*args, **kwargs):
        raise ImportError("Model fine-tuning not available. Install with: pip install echogrid[training]")
    
    def train_model(*args, **kwargs):
        raise ImportError("Model training not available. Install with: pip install echogrid[training]")
    
    class AudioDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("AudioDataset not available. Install with: pip install echogrid[training]")

# Hub integration - optional dependency
try:
    from .hub.upload import upload_model
    from .hub.download import download
    _hub_available = True
except ImportError:
    _hub_available = False
    
    def upload_model(*args, **kwargs):
        raise ImportError("Model upload not available. Install with: pip install echogrid[hub]")
    
    def download(*args, **kwargs):
        raise ImportError("Model download not available. Install with: pip install echogrid[hub]")

# Utilities - always available
from .utils.logging import set_log_level, get_logger
from .utils.device_utils import get_available_devices, get_device_info

# CLI functionality - optional
try:
    from .cli.main import main as cli_main
    _cli_available = True
except ImportError:
    _cli_available = False
    cli_main = None

# Web integrations - optional
try:
    from .integrations.gradio import create_gradio_interface
except ImportError:
    create_gradio_interface = None

try:
    from .integrations.streamlit import create_streamlit_app
except ImportError:
    create_streamlit_app = None

try:
    from .integrations.fastapi import create_fastapi_app
except ImportError:
    create_fastapi_app = None

# Plugin system - optional
try:
    from .plugins.manager import PluginManager
    plugins = PluginManager()
except ImportError:
    plugins = None

# Version compatibility check
if sys.version_info < (3, 8):
    raise RuntimeError(
        f"EchoGrid requires Python 3.8 or later. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}"
    )

# Initialize default configuration on import
try:
    configure()
except Exception as e:
    warnings.warn(f"Failed to initialize default configuration: {e}")

# Module-level convenience functions
def get_version() -> str:
    """Get EchoGrid version string.
    
    Returns:
        str: Version string (e.g., "0.1.0")
    """
    return __version__

def get_build_info() -> dict:
    """Get detailed build information.
    
    Returns:
        dict: Build information including version, commit, date, etc.
    """
    return get_version_info()

def check_installation() -> dict:
    """Check which optional features are available.
    
    Returns:
        dict: Dictionary showing which features are installed
        
    Example:
        >>> import echogrid as eg
        >>> status = eg.check_installation()
        >>> print(status)
        {
            'core': True,
            'api': True,
            'audio': False,
            'training': True,
            'hub': True,
            'cli': False,
            'gradio': None,
            'streamlit': None
        }
    """
    return {
        'core': _core_available,
        'api': _api_available,
        'audio': _audio_available,
        'training': _training_available,
        'hub': _hub_available,
        'cli': _cli_available,
        'gradio': create_gradio_interface is not None,
        'streamlit': create_streamlit_app is not None,
        'fastapi': create_fastapi_app is not None,
        'plugins': plugins is not None,
    }

def install_guide() -> str:
    """Get installation guide for missing features.
    
    Returns:
        str: Installation instructions for missing features
    """
    status = check_installation()
    missing = [name for name, available in status.items() if not available]
    
    if not missing:
        return "All features are installed!"
    
    guide = "Missing features and installation commands:\n\n"
    
    install_map = {
        'api': 'pip install echogrid[api]',
        'audio': 'pip install echogrid[audio]',
        'training': 'pip install echogrid[training]',
        'hub': 'pip install echogrid[hub]',
        'cli': 'pip install echogrid[cli]',
        'gradio': 'pip install echogrid[web]',
        'streamlit': 'pip install echogrid[web]',
        'fastapi': 'pip install echogrid[web]',
        'plugins': 'pip install echogrid[plugins]',
    }
    
    for feature in missing:
        if feature in install_map:
            guide += f"- {feature}: {install_map[feature]}\n"
    
    guide += "\nOr install everything: pip install echogrid[full]"
    
    return guide

# Export list for explicit imports
__all__ = [
    # Version information
    "__version__",
    "__version_info__",
    "get_version",
    "get_version_info", 
    "get_build_info",
    
    # Configuration
    "configure",
    "get_config",
    "configure_session",
    
    # Exceptions
    "EchoGridError",
    "ModelNotFoundError",
    "InsufficientMemoryError", 
    "APIRateLimitError",
    "TrainingError",
    "AudioProcessingError",
    "ConfigurationError",
    "NetworkError",
    "ValidationError",
    "AuthenticationError",
    "PermissionError",
    "ModelArchitectureError",
    
    # Core functions
    "browse",
    "search",
    "info", 
    "load",
    "list_models",
    "cache",
    
    # API
    "set_api_key",
    "configure_api",
    "api",
    
    # Audio
    "AudioFile",
    "process_audio_batch",
    "convert_audio",
    "convert_batch",
    
    # Training
    "clone_voice",
    "fine_tune",
    "train_model", 
    "AudioDataset",
    
    # Hub
    "upload_model",
    "download",
    
    # Utilities
    "set_log_level",
    "get_logger",
    "get_available_devices",
    "get_device_info",
    
    # Integrations (if available)
    "create_gradio_interface",
    "create_streamlit_app",
    "create_fastapi_app",
    
    # Plugin system
    "plugins",
    
    # CLI
    "cli_main",
    
    # Installation helpers
    "check_installation",
    "install_guide",
]

# Package metadata
__author__ = "EchoGrid Team"
__email__ = "team@echogrid.dev"
__license__ = "Apache-2.0"
__url__ = "https://echogrid.dev"
__description__ = "Community-powered text-to-speech with thousands of free voice models"