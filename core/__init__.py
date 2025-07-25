# echogrid/core/__init__.py
"""
Core functionality for EchoGrid.

This module contains the core functionality for model discovery, loading,
caching, and backend management. It provides the foundation for all
text-to-speech operations in EchoGrid.

The core module includes:

- **Model Discovery**: Browse and search for community voice models
- **Model Loading**: Load models for local or remote inference
- **Caching**: Manage downloaded models and cache storage
- **Backend Management**: Handle different inference backends (local/API)
- **Session Management**: Manage model instances and memory usage

Submodules:
    discovery: Model browsing, searching, and information retrieval
    models: Model loading and management functionality
    cache: Local model caching and storage management
    backends: Inference backend implementations and management
    session: Session state and model lifecycle management

Examples:
    Basic model discovery and loading::

        from echogrid.core import browse, load
        
        # Browse available models
        models = browse(language="en", category="narrator")
        
        # Load a specific model
        model = load("alice/uk-teen-female")
        audio = model.tts("Hello world")

    Cache management::

        from echogrid.core import cache
        
        # List cached models
        cached = cache.list()
        
        # Clear old models
        cache.clear(older_than=30)

    Backend configuration::

        from echogrid.core import get_backend, BackendType
        
        # Get local backend
        backend = get_backend(BackendType.LOCAL)
        capabilities = backend.get_capabilities()
"""

from typing import List, Dict, Any, Optional, Union

# Model discovery and information
from .discovery import (
    browse,
    search, 
    info,
    get_model_info,
    ModelInfo
)

# Model loading and management
from .models import (
    load,
    aload,
    list_models,
    get_model_info_cached,
    unload_model,
    reload_model,
    get_backend_info
)

# Cache management
from .cache import (
    cache,
    CacheManager,
    CachedModel,
    CacheStats
)

# Backend management
from .backends import (
    BaseBackend,
    LocalBackend,
    APIBackend,
    BackendType,
    get_backend,
    list_backends,
    get_optimal_backend,
    register_backend,
    clear_backend_cache
)

# Session management
from .session import (
    SessionManager,
    ModelSession,
    get_session_manager,
    get_session_model,
    clear_session,
    list_session_models,
    get_session_stats,
    optimize_session_memory,
    cleanup_unused_models,
    SessionContext
)

# Utility functions for core operations
def discover_models(
    query: Optional[str] = None,
    **filters
) -> List[ModelInfo]:
    """Discover models using search or browse.
    
    This is a convenience function that automatically chooses between
    search and browse based on whether a query is provided.
    
    Args:
        query: Optional text query for searching
        **filters: Filter parameters for browsing/searching
        
    Returns:
        List[ModelInfo]: List of discovered models
        
    Examples:
        Search for models::
        
            models = discover_models("british narrator")
            
        Browse with filters::
        
            models = discover_models(language="en", category="narrator")
    """
    if query:
        return search(query, **filters)
    else:
        return browse(**filters)

def get_model_suggestions(
    preferences: Dict[str, Any]
) -> List[ModelInfo]:
    """Get model suggestions based on user preferences.
    
    Args:
        preferences: Dictionary of user preferences including:
            - language: Preferred language(s)
            - use_case: Intended use case
            - quality: Minimum quality level
            - voice_characteristics: Desired voice traits
            
    Returns:
        List[ModelInfo]: Recommended models sorted by relevance
        
    Example:
        Get suggestions for podcast narration::
        
            suggestions = get_model_suggestions({
                'language': 'en',
                'use_case': 'podcast',
                'quality': 'high',
                'voice_characteristics': ['professional', 'clear']
            })
    """
    filters = {}
    
    # Map preferences to browse filters
    if 'language' in preferences:
        filters['language'] = preferences['language']
    
    if 'use_case' in preferences:
        use_case_mapping = {
            'podcast': 'narrator',
            'audiobook': 'narrator',
            'assistant': 'conversational',
            'character': 'character',
            'gaming': 'character'
        }
        filters['category'] = use_case_mapping.get(preferences['use_case'])
    
    if 'quality' in preferences:
        quality_mapping = {
            'high': 4.0,
            'medium': 3.0,
            'low': 2.0
        }
        filters['min_rating'] = quality_mapping.get(preferences['quality'], 3.0)
    
    # Get models and sort by rating
    models = browse(**filters)
    return sorted(models, key=lambda m: m.rating, reverse=True)

def quick_start() -> Dict[str, Any]:
    """Get quick start information for new users.
    
    Returns:
        Dict[str, Any]: Quick start guide with popular models and examples
        
    Example:
        Get popular models to try::
        
            guide = quick_start()
            popular_models = guide['popular_models']
            example_code = guide['example_code']
    """
    # Get popular models across different categories
    popular = {
        'conversational': browse(category='conversational', sort_by='popularity', limit=3),
        'narrator': browse(category='narrator', sort_by='popularity', limit=3),
        'character': browse(category='character', sort_by='popularity', limit=3)
    }
    
    example_code = '''
import echogrid as eg

# Load a popular model
model = eg.load("alice/uk-teen-female")

# Generate speech
audio = model.tts("Hello! Welcome to EchoGrid.")

# Save the audio
audio.save("welcome.wav")

# Try different emotions
excited = model.tts("This is amazing!", emotion="excited")
calm = model.tts("Take a deep breath.", emotion="calm")
'''
    
    return {
        'popular_models': popular,
        'example_code': example_code,
        'installation_status': check_core_features(),
        'next_steps': [
            "Try cloning your own voice with eg.clone_voice()",
            "Explore audio processing with eg.AudioFile()",
            "Browse models by language with eg.browse(language='es')",
            "Check out training tutorials for custom models"
        ]
    }

def check_core_features() -> Dict[str, bool]:
    """Check availability of core features.
    
    Returns:
        Dict[str, bool]: Status of core feature availability
    """
    features = {}
    
    # Check model loading
    try:
        from .models import load
        features['model_loading'] = True
    except ImportError:
        features['model_loading'] = False
    
    # Check local inference
    try:
        from ..inference.local import LocalInferenceEngine
        features['local_inference'] = True
    except ImportError:
        features['local_inference'] = False
    
    # Check API inference  
    try:
        from ..inference.remote import APIInferenceEngine
        features['api_inference'] = True
    except ImportError:
        features['api_inference'] = False
    
    # Check cache functionality
    try:
        from .cache import cache
        features['caching'] = True
    except ImportError:
        features['caching'] = False
    
    return features

def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status.
    
    Returns:
        Dict[str, Any]: System status including backends, cache, memory
    """
    from ..utils.device_utils import get_available_devices, get_memory_info
    
    status = {
        'version': get_version_info(),
        'features': check_core_features(),
        'backends': get_backend_info(),
        'devices': get_available_devices(),
        'memory': get_memory_info(),
        'cache': cache.stats() if 'caching' in check_core_features() else None,
        'session': get_session_stats() if 'model_loading' in check_core_features() else None
    }
    
    return status

# Import version info for status functions
try:
    from ..version import get_version_info
except ImportError:
    def get_version_info():
        return {'version': 'unknown'}

__all__ = [
    # Discovery functions
    'browse',
    'search',
    'info', 
    'get_model_info',
    'ModelInfo',
    
    # Model management
    'load',
    'aload',
    'list_models',
    'get_model_info_cached',
    'unload_model',
    'reload_model',
    'get_backend_info',
    
    # Cache management
    'cache',
    'CacheManager', 
    'CachedModel',
    'CacheStats',
    
    # Backend management
    'BaseBackend',
    'LocalBackend',
    'APIBackend', 
    'BackendType',
    'get_backend',
    'list_backends',
    'get_optimal_backend',
    'register_backend',
    'clear_backend_cache',
    
    # Session management
    'SessionManager',
    'ModelSession',
    'get_session_manager',
    'get_session_model',
    'clear_session',
    'list_session_models',
    'get_session_stats',
    'optimize_session_memory',
    'cleanup_unused_models',
    'SessionContext',
    
    # Utility functions
    'discover_models',
    'get_model_suggestions',
    'quick_start',
    'check_core_features',
    'get_system_status',
]