# echogrid/core/models.py
"""
Model loading and management for EchoGrid.

This module handles loading voice models, managing backends, and providing
the main interface for model operations.
"""

import logging
import asyncio
from typing import Optional, Union, Dict, Any, List
from pathlib import Path

from ..config import get_config
from ..constants import SUPPORTED_ARCHITECTURES
from ..exceptions import (
    ModelNotFoundError, InsufficientMemoryError, ValidationError,
    NetworkError, ConfigurationError
)
from .discovery import info as get_model_info
from .cache import cache
from .backends import get_backend, BackendType
from ..models.voice_model import VoiceModel
from ..utils.device_utils import get_available_devices, get_memory_info
from ..utils.validation import validate_model_name

logger = logging.getLogger(__name__)


def load(
    model_name: str,
    backend: str = "auto",
    device: str = "auto",
    compute_precision: str = "auto",
    max_memory: Optional[str] = None,
    cache_dir: Optional[str] = None,
    version: str = "latest",
    timeout: int = 30,
    low_memory: bool = False,
    compile_model: Optional[bool] = None,
    trust_remote_code: bool = False,
    use_auth_token: Optional[str] = None,
    local_files_only: bool = False,
    force_download: bool = False,
    progress_callback: Optional[callable] = None,
    warm_up: bool = True,
    **model_kwargs
) -> VoiceModel:
    """
    Load a voice model for inference.
    
    This is the main function for loading voice models in EchoGrid. It handles
    automatic backend selection, model downloading, caching, and initialization.
    
    Args:
        model_name: Model identifier in "username/model-name" format
        backend: Inference backend ("auto", "local", "api")
        device: Computing device ("auto", "cpu", "cuda", "cuda:0", "mps")
        compute_precision: Computation precision ("auto", "fp32", "fp16", "bf16", "int8")
        max_memory: Maximum memory limit (e.g., "4GB", "8GB")
        cache_dir: Custom cache directory
        version: Model version to load
        timeout: Loading timeout in seconds
        low_memory: Use memory-efficient loading strategies
        compile_model: Compile model for faster inference
        trust_remote_code: Allow loading custom model code
        use_auth_token: Authentication token for private models
        local_files_only: Only use locally cached files
        force_download: Re-download even if cached
        progress_callback: Function to call with loading progress
        warm_up: Pre-compile model with test input
        **model_kwargs: Additional model-specific parameters
    
    Returns:
        VoiceModel: Loaded voice model ready for inference
    
    Raises:
        ModelNotFoundError: If the model doesn't exist
        InsufficientMemoryError: If insufficient memory for model
        ValidationError: If invalid parameters provided
        NetworkError: If download fails
        ConfigurationError: If backend configuration is invalid
    
    Example:
        >>> import echogrid as eg
        >>> 
        >>> # Simple loading with auto-detection
        >>> model = eg.load("alice/uk-teen-female")
        >>> 
        >>> # Force local inference with specific device
        >>> model = eg.load(
        ...     "alice/uk-teen-female",
        ...     backend="local",
        ...     device="cuda:0",
        ...     compute_precision="fp16"
        ... )
        >>> 
        >>> # Memory-constrained loading
        >>> model = eg.load(
        ...     "large-model/studio-voice",
        ...     max_memory="4GB",
        ...     low_memory=True,
        ...     compile_model=False
        ... )
    """
    config = get_config()
    
    # Validate model name
    if not validate_model_name(model_name):
        raise ValidationError(
            f"Invalid model name format: {model_name}",
            field="model_name",
            value=model_name,
            constraint="Must be in 'username/model-name' format"
        )
    
    # Override config with function parameters
    if cache_dir:
        config.cache_dir = cache_dir
    
    # Progress callback wrapper
    def progress_update(stage: str, current: int, total: int, **kwargs):
        if progress_callback:
            progress_callback(stage, current, total, **kwargs)
        logger.debug(f"Loading {model_name}: {stage} ({current}/{total})")
    
    try:
        progress_update("initializing", 0, 5)
        
        # Get model information
        try:
            model_info = get_model_info(model_name, include_technical=True)
        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.warning(f"Could not fetch model info: {e}")
            model_info = None
        
        progress_update("checking_cache", 1, 5)
        
        # Check if model is cached
        model_path = None
        if not force_download:
            model_path = cache.get_path(model_name, version)
        
        progress_update("determining_backend", 2, 5)
        
        # Determine backend
        backend_type = _determine_backend(
            backend, model_info, model_path, local_files_only
        )
        
        progress_update("loading_model", 3, 5)
        
        # Get backend instance
        model_backend = get_backend(backend_type)
        
        # Load model using selected backend
        voice_model = model_backend.load_model(
            model_name=model_name,
            model_info=model_info,
            model_path=model_path,
            device=device,
            compute_precision=compute_precision,
            max_memory=max_memory,
            version=version,
            timeout=timeout,
            low_memory=low_memory,
            compile_model=compile_model,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            progress_callback=progress_update,
            **model_kwargs
        )
        
        progress_update("warming_up", 4, 5)
        
        # Warm up model if requested
        if warm_up and backend_type == BackendType.LOCAL:
            try:
                _warm_up_model(voice_model)
            except Exception as e:
                logger.warning(f"Model warm-up failed: {e}")
        
        progress_update("complete", 5, 5)
        
        logger.info(f"Successfully loaded model: {model_name} ({backend_type.value} backend)")
        return voice_model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


async def aload(
    model_name: str,
    **kwargs
) -> VoiceModel:
    """
    Asynchronous version of load().
    
    Args:
        model_name: Model identifier
        **kwargs: Same arguments as load()
    
    Returns:
        VoiceModel: Loaded voice model
    
    Example:
        >>> import asyncio
        >>> import echogrid as eg
        >>> 
        >>> async def main():
        ...     model = await eg.aload("alice/uk-teen-female")
        ...     audio = await model.atts("Hello world")
        ...     return audio
        >>> 
        >>> audio = asyncio.run(main())
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: load(model_name, **kwargs))


def _determine_backend(
    backend_preference: str,
    model_info: Optional[Any],
    cached_path: Optional[str],
    local_files_only: bool
) -> BackendType:
    """
    Determine which backend to use for model loading.
    
    Args:
        backend_preference: User's backend preference
        model_info: Model metadata
        cached_path: Path to cached model (if any)
        local_files_only: Whether to restrict to local files
    
    Returns:
        BackendType: Selected backend type
    """
    config = get_config()
    
    # Handle explicit backend selection
    if backend_preference == "local":
        return BackendType.LOCAL
    elif backend_preference == "api":
        if local_files_only:
            raise ConfigurationError(
                "Cannot use API backend with local_files_only=True"
            )
        return BackendType.API
    
    # Auto-selection logic
    elif backend_preference == "auto":
        # Prefer local if available and user preference
        if config.prefer_local_if_available:
            # Check if we have local capabilities
            devices = get_available_devices()
            has_gpu = any("cuda" in device or "mps" in device for device in devices)
            
            # Check memory requirements
            if model_info:
                required_memory = _estimate_memory_requirements(model_info)
                available_memory = get_memory_info().get("available_gb", 0)
                
                if required_memory and available_memory < required_memory:
                    logger.info(
                        f"Insufficient memory for local inference: "
                        f"{required_memory:.1f}GB required, {available_memory:.1f}GB available"
                    )
                    if not config.api_fallback or local_files_only:
                        raise InsufficientMemoryError(required_memory, available_memory)
                    return BackendType.API
            
            # Check if model is cached locally
            if cached_path and Path(cached_path).exists():
                return BackendType.LOCAL
            
            # If we have GPU and model is not too large, prefer local
            if has_gpu and not local_files_only:
                return BackendType.LOCAL
        
        # Fall back to API if local not preferred or not available
        if not local_files_only and config.api_fallback:
            return BackendType.API
        
        # Default to local
        return BackendType.LOCAL
    
    else:
        raise ValidationError(
            f"Invalid backend: {backend_preference}",
            field="backend",
            value=backend_preference,
            constraint="Must be 'auto', 'local', or 'api'"
        )


def _estimate_memory_requirements(model_info: Any) -> Optional[float]:
    """
    Estimate memory requirements for a model.
    
    Args:
        model_info: Model metadata
    
    Returns:
        float: Estimated memory requirement in GB
    """
    if not model_info:
        return None
    
    # Check if model_info has memory_usage
    if hasattr(model_info, 'memory_usage') and model_info.memory_usage:
        return model_info.memory_usage / 1024  # Convert MB to GB
    
    # Check model size
    if hasattr(model_info, 'model_size') and model_info.model_size:
        # Rough estimate: model size * 2-3 for inference
        model_size_gb = model_info.model_size / (1024 ** 3)
        return model_size_gb * 2.5
    
    # Architecture-based estimates
    if hasattr(model_info, 'architecture'):
        architecture_memory = {
            "xtts-v2": 4.0,
            "styletts2": 2.0,
            "bark": 6.0,
            "openvoice": 3.0,
            "f5-tts": 2.5,
        }
        return architecture_memory.get(model_info.architecture, 4.0)
    
    # Default estimate
    return 4.0


def _warm_up_model(model: VoiceModel):
    """
    Warm up model by running a test inference.
    
    Args:
        model: Voice model to warm up
    """
    try:
        # Run a short test generation to compile model
        test_text = "Hello"
        model.tts(
            test_text,
            temperature=0.5,
            speed=1.0,
            # Minimal generation for warm-up
        )
        logger.debug("Model warm-up completed successfully")
    except Exception as e:
        logger.warning(f"Model warm-up failed: {e}")


def list_models(
    backend: Optional[str] = None,
    device: Optional[str] = None,
    **filter_kwargs
) -> List[Dict[str, Any]]:
    """
    List available models with compatibility information.
    
    Args:
        backend: Filter by backend compatibility
        device: Filter by device compatibility
        **filter_kwargs: Additional filters passed to browse()
    
    Returns:
        List[Dict[str, Any]]: List of models with compatibility info
    
    Example:
        >>> import echogrid as eg
        >>> 
        >>> # List all models compatible with current system
        >>> models = eg.list_models()
        >>> 
        >>> # List models compatible with CUDA
        >>> cuda_models = eg.list_models(device="cuda")
        >>> 
        >>> # List cached models
        >>> cached_models = eg.list_models(backend="local")
    """
    from .discovery import browse
    
    # Get base model list
    models = browse(**filter_kwargs)
    
    # Add compatibility information
    result = []
    for model_info in models:
        model_dict = {
            "name": model_info.name,
            "description": model_info.description,
            "creator": model_info.creator,
            "language": model_info.language,
            "category": model_info.category,
            "rating": model_info.rating,
            "downloads": model_info.downloads,
            "architecture": model_info.architecture,
            "cached": cache.is_cached(model_info.name),
            "compatible": True,  # Default to compatible
            "memory_estimate": _estimate_memory_requirements(model_info)
        }
        
        # Check backend compatibility
        if backend:
            if backend == "local":
                # Check if model can run locally
                memory_req = model_dict["memory_estimate"]
                available_memory = get_memory_info().get("available_gb", 0)
                if memory_req and available_memory < memory_req:
                    model_dict["compatible"] = False
                    model_dict["incompatible_reason"] = "Insufficient memory"
            elif backend == "api":
                # All models should be API compatible
                pass
        
        # Check device compatibility
        if device:
            if device.startswith("cuda") and "cuda" not in get_available_devices():
                model_dict["compatible"] = False
                model_dict["incompatible_reason"] = "CUDA not available"
            elif device == "mps" and "mps" not in get_available_devices():
                model_dict["compatible"] = False
                model_dict["incompatible_reason"] = "MPS not available"
        
        result.append(model_dict)
    
    return result


def get_model_info_cached(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get cached model information without network requests.
    
    Args:
        model_name: Model identifier
    
    Returns:
        Dict[str, Any]: Cached model information or None
    
    Example:
        >>> import echogrid as eg
        >>> info = eg.get_model_info_cached("alice/uk-teen-female")
        >>> if info:
        ...     print(f"Model is cached at: {info['path']}")
    """
    cached_model = cache.get(model_name)
    if cached_model:
        return {
            "name": cached_model.name,
            "version": cached_model.version,
            "path": cached_model.path,
            "size_mb": cached_model.size_mb,
            "download_date": cached_model.download_date,
            "last_used": cached_model.last_used,
            "format": cached_model.format,
            "architecture": cached_model.architecture,
            "metadata": cached_model.metadata
        }
    return None


def unload_model(model: VoiceModel):
    """
    Explicitly unload a model to free memory.
    
    Args:
        model: Voice model to unload
    
    Example:
        >>> import echogrid as eg
        >>> model = eg.load("alice/uk-teen-female")
        >>> # Use model...
        >>> eg.unload_model(model)  # Free memory
    """
    try:
        if hasattr(model, '_backend') and hasattr(model._backend, 'unload'):
            model._backend.unload()
        
        # Clear references
        if hasattr(model, '_model'):
            del model._model
        if hasattr(model, '_processor'):
            del model._processor
            
        logger.info(f"Unloaded model: {model.name}")
    except Exception as e:
        logger.warning(f"Error unloading model: {e}")


def reload_model(model: VoiceModel, **kwargs) -> VoiceModel:
    """
    Reload a model with new parameters.
    
    Args:
        model: Voice model to reload
        **kwargs: New loading parameters
    
    Returns:
        VoiceModel: Reloaded model
    
    Example:
        >>> import echogrid as eg
        >>> model = eg.load("alice/uk-teen-female", device="cpu")
        >>> # Reload on GPU
        >>> model = eg.reload_model(model, device="cuda:0")
    """
    model_name = model.name
    unload_model(model)
    return load(model_name, **kwargs)


def get_backend_info() -> Dict[str, Any]:
    """
    Get information about available backends.
    
    Returns:
        Dict[str, Any]: Backend availability and capabilities
    
    Example:
        >>> import echogrid as eg
        >>> backends = eg.get_backend_info()
        >>> print(f"Local backend available: {backends['local']['available']}")
        >>> print(f"API backend available: {backends['api']['available']}")
    """
    config = get_config()
    devices = get_available_devices()
    memory_info = get_memory_info()
    
    return {
        "local": {
            "available": True,  # Local backend always available
            "devices": devices,
            "memory": memory_info,
            "compiled_models": config.compile_models,
            "cache_dir": config.cache_dir
        },
        "api": {
            "available": bool(config.api_key or config.api_endpoint),
            "endpoint": config.api_endpoint,
            "authenticated": bool(config.api_key),
            "timeout": config.api_timeout,
            "retries": config.api_retries
        },
        "default": config.default_backend,
        "prefer_local": config.prefer_local_if_available,
        "api_fallback": config.api_fallback
    }