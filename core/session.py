# echogrid/core/session.py
"""
Session management for EchoGrid.

This module handles session state, model management, and resource cleanup
for efficient model usage across multiple operations.
"""

import logging
import threading
import weakref
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..config import get_config
from ..models.voice_model import VoiceModel
from ..exceptions import ConfigurationError, InsufficientMemoryError
from ..utils.device_utils import get_memory_info

logger = logging.getLogger(__name__)


@dataclass
class ModelSession:
    """
    Information about a model in the current session.
    
    Attributes:
        model: Reference to the loaded model
        load_time: When the model was loaded
        last_used: When the model was last used
        use_count: Number of times model has been used
        memory_usage: Estimated memory usage in MB
        device: Device the model is running on
        backend: Backend type used for the model
    """
    model: VoiceModel
    load_time: datetime
    last_used: datetime
    use_count: int = 0
    memory_usage: Optional[float] = None
    device: Optional[str] = None
    backend: Optional[str] = None
    
    def update_usage(self):
        """Update usage statistics."""
        self.last_used = datetime.now()
        self.use_count += 1


class SessionManager:
    """
    Manages model sessions and resource allocation.
    
    This class handles loading, caching, and cleanup of models within
    a session to optimize memory usage and performance.
    """
    
    def __init__(self):
        """Initialize session manager."""
        self.config = get_config()
        self._models: Dict[str, ModelSession] = {}
        self._lock = threading.RLock()
        self._max_models = self.config.model_cache_size
        self._cleanup_threshold = 0.9  # Cleanup when 90% of limit reached
        
        # Weak references to track model lifecycle
        self._model_refs: Set[weakref.ref] = set()
        
        logger.debug("Initialized session manager")
    
    def get_model(
        self,
        model_name: str,
        version: str = "latest",
        **load_kwargs
    ) -> VoiceModel:
        """
        Get a model from the session cache or load it.
        
        Args:
            model_name: Model identifier
            version: Model version
            **load_kwargs: Arguments passed to model loading
        
        Returns:
            VoiceModel: Loaded voice model
        """
        with self._lock:
            session_key = f"{model_name}:{version}"
            
            # Check if model is already loaded
            if session_key in self._models:
                session = self._models[session_key]
                
                # Verify model is still valid
                if self._is_model_valid(session.model):
                    session.update_usage()
                    logger.debug(f"Using cached model: {model_name}")
                    return session.model
                else:
                    # Remove invalid model
                    self._remove_model(session_key)
            
            # Load new model
            return self._load_and_cache_model(model_name, version, **load_kwargs)
    
    def preload_models(
        self,
        model_names: List[str],
        **load_kwargs
    ) -> Dict[str, VoiceModel]:
        """
        Preload multiple models into the session cache.
        
        Args:
            model_names: List of model identifiers to preload
            **load_kwargs: Arguments passed to model loading
        
        Returns:
            Dict[str, VoiceModel]: Dictionary of loaded models
        
        Example:
            >>> session = SessionManager()
            >>> models = session.preload_models([
            ...     "alice/uk-teen-female",
            ...     "bob/us-male-narrator"
            ... ])
        """
        results = {}
        
        for model_name in model_names:
            try:
                model = self.get_model(model_name, **load_kwargs)
                results[model_name] = model
                logger.info(f"Preloaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """
        List all models currently loaded in the session.
        
        Returns:
            List[Dict[str, Any]]: List of loaded model information
        """
        with self._lock:
            models = []
            
            for session_key, session in self._models.items():
                model_info = {
                    "name": session.model.name,
                    "session_key": session_key,
                    "load_time": session.load_time,
                    "last_used": session.last_used,
                    "use_count": session.use_count,
                    "memory_usage_mb": session.memory_usage,
                    "device": session.device,
                    "backend": session.backend,
                    "age_minutes": (datetime.now() - session.load_time).total_seconds() / 60
                }
                models.append(model_info)
            
            # Sort by last used (most recent first)
            models.sort(key=lambda x: x["last_used"], reverse=True)
            return models
    
    def unload_model(self, model_name: str, version: str = "latest") -> bool:
        """
        Unload a specific model from the session.
        
        Args:
            model_name: Model identifier
            version: Model version
        
        Returns:
            bool: True if model was unloaded, False if not found
        """
        with self._lock:
            session_key = f"{model_name}:{version}"
            return self._remove_model(session_key)
    
    def clear_session(self):
        """
        Clear all loaded models from the session.
        
        This frees up all memory used by cached models.
        """
        with self._lock:
            model_keys = list(self._models.keys())
            for session_key in model_keys:
                self._remove_model(session_key)
            
            logger.info("Cleared all models from session")
    
    def cleanup_unused(self, max_age_minutes: int = 30):
        """
        Clean up models that haven't been used recently.
        
        Args:
            max_age_minutes: Maximum age for unused models in minutes
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
            to_remove = []
            
            for session_key, session in self._models.items():
                if session.last_used < cutoff_time:
                    to_remove.append(session_key)
            
            for session_key in to_remove:
                self._remove_model(session_key)
                logger.debug(f"Cleaned up unused model: {session_key}")
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} unused models")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for the session.
        
        Returns:
            Dict[str, Any]: Memory usage information
        """
        with self._lock:
            total_memory = 0
            model_count = len(self._models)
            
            memory_by_model = {}
            for session_key, session in self._models.items():
                if session.memory_usage:
                    memory_by_model[session_key] = session.memory_usage
                    total_memory += session.memory_usage
            
            system_memory = get_memory_info()
            
            return {
                "total_session_memory_mb": total_memory,
                "total_session_memory_gb": total_memory / 1024,
                "model_count": model_count,
                "memory_by_model": memory_by_model,
                "system_memory": system_memory,
                "cache_utilization": model_count / self._max_models if self._max_models > 0 else 0
            }
    
    def optimize_memory(self, target_memory_gb: Optional[float] = None):
        """
        Optimize memory usage by unloading least recently used models.
        
        Args:
            target_memory_gb: Target memory usage in GB
        """
        with self._lock:
            if not target_memory_gb:
                # Default to 80% of available memory
                system_memory = get_memory_info()
                target_memory_gb = system_memory.get("available_gb", 8) * 0.8
            
            current_usage = self.get_memory_usage()
            current_memory_gb = current_usage["total_session_memory_gb"]
            
            if current_memory_gb <= target_memory_gb:
                return  # Already within target
            
            # Sort models by last used (oldest first)
            sessions = list(self._models.items())
            sessions.sort(key=lambda x: x[1].last_used)
            
            # Remove models until under target
            for session_key, session in sessions:
                if current_memory_gb <= target_memory_gb:
                    break
                
                if session.memory_usage:
                    current_memory_gb -= session.memory_usage / 1024
                
                self._remove_model(session_key)
                logger.info(f"Unloaded model for memory optimization: {session_key}")
    
    def _load_and_cache_model(
        self,
        model_name: str,
        version: str,
        **load_kwargs
    ) -> VoiceModel:
        """Load a model and add it to the session cache."""
        from .models import load
        
        # Check if we need to make room
        self._maybe_cleanup()
        
        # Load the model
        model = load(model_name, version=version, **load_kwargs)
        
        # Create session entry
        session = ModelSession(
            model=model,
            load_time=datetime.now(),
            last_used=datetime.now(),
            use_count=1,
            memory_usage=self._estimate_model_memory(model),
            device=getattr(model, 'device', None),
            backend=getattr(model, 'backend', None)
        )
        
        session_key = f"{model_name}:{version}"
        self._models[session_key] = session
        
        # Add weak reference for lifecycle tracking
        model_ref = weakref.ref(model, lambda ref: self._cleanup_dead_reference(ref))
        self._model_refs.add(model_ref)
        
        logger.info(f"Loaded and cached model: {model_name}")
        return model
    
    def _remove_model(self, session_key: str) -> bool:
        """Remove a model from the session cache."""
        if session_key not in self._models:
            return False
        
        session = self._models[session_key]
        
        # Attempt to unload the model
        try:
            if hasattr(session.model, 'unload'):
                session.model.unload()
        except Exception as e:
            logger.warning(f"Error unloading model: {e}")
        
        # Remove from cache
        del self._models[session_key]
        logger.debug(f"Removed model from session: {session_key}")
        return True
    
    def _maybe_cleanup(self):
        """Clean up models if cache is getting full."""
        if len(self._models) >= self._max_models * self._cleanup_threshold:
            # Remove least recently used model
            if self._models:
                oldest_key = min(
                    self._models.keys(),
                    key=lambda k: self._models[k].last_used
                )
                self._remove_model(oldest_key)
                logger.debug(f"Auto-removed oldest model: {oldest_key}")
    
    def _is_model_valid(self, model: VoiceModel) -> bool:
        """Check if a model is still valid and usable."""
        try:
            # Basic checks
            if not hasattr(model, 'name'):
                return False
            
            # Check if model is still loaded
            if hasattr(model, '_backend') and hasattr(model._backend, 'is_loaded'):
                return model._backend.is_loaded()
            
            return True
        except Exception:
            return False
    
    def _estimate_model_memory(self, model: VoiceModel) -> Optional[float]:
        """Estimate memory usage of a model in MB."""
        try:
            if hasattr(model, 'get_memory_usage'):
                return model.get_memory_usage()
            
            # Fallback estimation based on model info
            if hasattr(model, 'model_info') and model.model_info:
                if hasattr(model.model_info, 'memory_usage'):
                    return model.model_info.memory_usage
                
                if hasattr(model.model_info, 'model_size'):
                    # Rough estimate: model size * 1.5 for runtime overhead
                    return (model.model_info.model_size / (1024 * 1024)) * 1.5
            
            # Default estimate
            return 1024.0  # 1GB default
        except Exception:
            return None
    
    def _cleanup_dead_reference(self, ref: weakref.ref):
        """Clean up dead model references."""
        if ref in self._model_refs:
            self._model_refs.remove(ref)


# Global session manager instance
_session_manager = None
_session_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """
    Get the global session manager instance.
    
    Returns:
        SessionManager: Global session manager
    """
    global _session_manager
    
    if _session_manager is None:
        with _session_lock:
            if _session_manager is None:
                _session_manager = SessionManager()
    
    return _session_manager


def get_session_model(model_name: str, **kwargs) -> VoiceModel:
    """
    Get a model from the global session cache.
    
    Args:
        model_name: Model identifier
        **kwargs: Additional arguments passed to model loading
    
    Returns:
        VoiceModel: Loaded voice model
    
    Example:
        >>> import echogrid as eg
        >>> model = eg.get_session_model("alice/uk-teen-female")
        >>> # Model is cached for subsequent calls
        >>> same_model = eg.get_session_model("alice/uk-teen-female")
    """
    session_manager = get_session_manager()
    return session_manager.get_model(model_name, **kwargs)


def clear_session():
    """
    Clear the global model session.
    
    This unloads all cached models and frees memory.
    
    Example:
        >>> import echogrid as eg
        >>> eg.clear_session()  # Free all cached models
    """
    session_manager = get_session_manager()
    session_manager.clear_session()


def list_session_models() -> List[Dict[str, Any]]:
    """
    List all models in the current session.
    
    Returns:
        List[Dict[str, Any]]: List of session model information
    
    Example:
        >>> import echogrid as eg
        >>> models = eg.list_session_models()
        >>> for model_info in models:
        ...     print(f"{model_info['name']}: {model_info['use_count']} uses")
    """
    session_manager = get_session_manager()
    return session_manager.list_loaded_models()


def get_session_stats() -> Dict[str, Any]:
    """
    Get session memory and usage statistics.
    
    Returns:
        Dict[str, Any]: Session statistics
    
    Example:
        >>> import echogrid as eg
        >>> stats = eg.get_session_stats()
        >>> print(f"Session memory: {stats['total_session_memory_gb']:.1f}GB")
        >>> print(f"Models loaded: {stats['model_count']}")
    """
    session_manager = get_session_manager()
    return session_manager.get_memory_usage()


def optimize_session_memory(target_memory_gb: Optional[float] = None):
    """
    Optimize session memory usage.
    
    Args:
        target_memory_gb: Target memory usage in GB
    
    Example:
        >>> import echogrid as eg
        >>> eg.optimize_session_memory(target_memory_gb=4.0)  # Limit to 4GB
    """
    session_manager = get_session_manager()
    session_manager.optimize_memory(target_memory_gb)


def cleanup_unused_models(max_age_minutes: int = 30):
    """
    Clean up models that haven't been used recently.
    
    Args:
        max_age_minutes: Maximum age for unused models in minutes
    
    Example:
        >>> import echogrid as eg
        >>> eg.cleanup_unused_models(max_age_minutes=15)  # Remove unused models older than 15 minutes
    """
    session_manager = get_session_manager()
    session_manager.cleanup_unused(max_age_minutes)


class SessionContext:
    """
    Context manager for temporary sessions.
    
    This allows you to create isolated model sessions that are
    automatically cleaned up when the context exits.
    
    Example:
        >>> import echogrid as eg
        >>> 
        >>> with eg.SessionContext() as session:
        ...     model = session.get_model("alice/uk-teen-female")
        ...     audio = model.tts("Hello world")
        ... # Model is automatically unloaded here
    """
    
    def __init__(self, max_models: int = 3):
        """
        Initialize session context.
        
        Args:
            max_models: Maximum number of models to keep in this session
        """
        self.max_models = max_models
        self.session_manager = None
    
    def __enter__(self) -> SessionManager:
        """Enter the session context."""
        self.session_manager = SessionManager()
        self.session_manager._max_models = self.max_models
        return self.session_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the session context and clean up."""
        if self.session_manager:
            self.session_manager.clear_session()


# Auto-cleanup thread for the global session
import atexit

def _cleanup_global_session():
    """Clean up the global session on exit."""
    global _session_manager
    if _session_manager:
        _session_manager.clear_session()

# Register cleanup function
atexit.register(_cleanup_global_session)