# echogrid/core/backends.py
"""
Backend management for EchoGrid inference.

This module handles different inference backends (local, API) and provides
a unified interface for model loading and inference.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any, Dict, Callable

from ..config import get_config
from ..exceptions import ConfigurationError, NetworkError
from ..models.voice_model import VoiceModel

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available backend types."""
    LOCAL = "local"
    API = "api"


class BaseBackend(ABC):
    """
    Abstract base class for inference backends.
    
    All backends must implement this interface to provide model loading
    and inference capabilities.
    """
    
    def __init__(self):
        """Initialize backend."""
        self.config = get_config()
    
    @abstractmethod
    def load_model(
        self,
        model_name: str,
        model_info: Optional[Any] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        compute_precision: str = "auto",
        max_memory: Optional[str] = None,
        version: str = "latest",
        timeout: int = 30,
        low_memory: bool = False,
        compile_model: Optional[bool] = None,
        trust_remote_code: bool = False,
        use_auth_token: Optional[str] = None,
        local_files_only: bool = False,
        force_download: bool = False,
        progress_callback: Optional[Callable] = None,
        **model_kwargs
    ) -> VoiceModel:
        """
        Load a voice model.
        
        Args:
            model_name: Model identifier
            model_info: Model metadata
            model_path: Local path to model (if cached)
            device: Computing device
            compute_precision: Computation precision
            max_memory: Memory limit
            version: Model version
            timeout: Loading timeout
            low_memory: Use memory-efficient loading
            compile_model: Compile model for speed
            trust_remote_code: Allow custom code
            use_auth_token: Authentication token
            local_files_only: Only use local files
            force_download: Force re-download
            progress_callback: Progress callback
            **model_kwargs: Additional model parameters
        
        Returns:
            VoiceModel: Loaded voice model
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if backend is available.
        
        Returns:
            bool: True if backend can be used
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get backend capabilities.
        
        Returns:
            Dict[str, Any]: Backend capabilities and limitations
        """
        pass


class LocalBackend(BaseBackend):
    """
    Local inference backend.
    
    This backend runs models locally using PyTorch, ONNX, or other
    inference engines.
    """
    
    def __init__(self):
        """Initialize local backend."""
        super().__init__()
        self._check_dependencies()
    
    def load_model(
        self,
        model_name: str,
        model_info: Optional[Any] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        compute_precision: str = "auto",
        max_memory: Optional[str] = None,
        version: str = "latest",
        timeout: int = 30,
        low_memory: bool = False,
        compile_model: Optional[bool] = None,
        trust_remote_code: bool = False,
        use_auth_token: Optional[str] = None,
        local_files_only: bool = False,
        force_download: bool = False,
        progress_callback: Optional[Callable] = None,
        **model_kwargs
    ) -> VoiceModel:
        """Load model for local inference."""
        from ..inference.local import LocalInferenceEngine
        from ..hub.download import download_model
        
        # Download model if not cached
        if not model_path or force_download:
            if progress_callback:
                progress_callback("downloading", 0, 100)
            
            model_path = download_model(
                model_name=model_name,
                version=version,
                format="auto",
                force=force_download,
                progress_callback=lambda current, total: progress_callback(
                    "downloading", int(current/total*100), 100
                ) if progress_callback else None
            )
        
        if progress_callback:
            progress_callback("loading", 0, 100)
        
        # Create inference engine
        inference_engine = LocalInferenceEngine(
            device=device,
            compute_precision=compute_precision,
            max_memory=max_memory,
            low_memory=low_memory,
            compile_model=compile_model if compile_model is not None else self.config.compile_models,
            trust_remote_code=trust_remote_code
        )
        
        # Load model
        model = inference_engine.load_model(
            model_path=model_path,
            model_info=model_info,
            progress_callback=lambda current, total: progress_callback(
                "loading", int(current/total*100), 100
            ) if progress_callback else None,
            **model_kwargs
        )
        
        # Create VoiceModel wrapper
        voice_model = VoiceModel(
            name=model_name,
            backend=BackendType.LOCAL,
            inference_engine=inference_engine,
            model_info=model_info,
            **model_kwargs
        )
        
        logger.info(f"Loaded model locally: {model_name}")
        return voice_model
    
    def is_available(self) -> bool:
        """Check if local backend is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get local backend capabilities."""
        from ..utils.device_utils import get_available_devices, get_memory_info
        
        capabilities = {
            "type": "local",
            "available": self.is_available(),
            "devices": get_available_devices(),
            "memory": get_memory_info(),
            "supports_compilation": True,
            "supports_quantization": True,
            "offline_capable": True
        }
        
        # Check for specific libraries
        try:
            import torch
            capabilities["pytorch_version"] = torch.__version__
            capabilities["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                capabilities["cuda_version"] = torch.version.cuda
                capabilities["gpu_count"] = torch.cuda.device_count()
        except ImportError:
            pass
        
        try:
            import onnxruntime
            capabilities["onnx_available"] = True
            capabilities["onnx_version"] = onnxruntime.__version__
        except ImportError:
            capabilities["onnx_available"] = False
        
        return capabilities
    
    def _check_dependencies(self):
        """Check required dependencies for local backend."""
        try:
            import torch
            import torchaudio
        except ImportError as e:
            raise ConfigurationError(
                f"Local backend dependencies not available: {e}. "
                f"Install with: pip install echogrid[local]"
            )


class APIBackend(BaseBackend):
    """
    API inference backend.
    
    This backend uses the EchoGrid hosted API for model inference.
    """
    
    def __init__(self):
        """Initialize API backend."""
        super().__init__()
        self._check_configuration()
    
    def load_model(
        self,
        model_name: str,
        model_info: Optional[Any] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        compute_precision: str = "auto",
        max_memory: Optional[str] = None,
        version: str = "latest",
        timeout: int = 30,
        low_memory: bool = False,
        compile_model: Optional[bool] = None,
        trust_remote_code: bool = False,
        use_auth_token: Optional[str] = None,
        local_files_only: bool = False,
        force_download: bool = False,
        progress_callback: Optional[Callable] = None,
        **model_kwargs
    ) -> VoiceModel:
        """Load model for API inference."""
        from ..inference.remote import APIInferenceEngine
        
        if local_files_only:
            raise ConfigurationError(
                "Cannot use API backend with local_files_only=True"
            )
        
        if progress_callback:
            progress_callback("connecting", 50, 100)
        
        # Create API inference engine
        inference_engine = APIInferenceEngine(
            api_endpoint=self.config.api_endpoint,
            api_key=use_auth_token or self.config.api_key,
            timeout=timeout,
            region=getattr(self.config, 'api_region', 'auto')
        )
        
        # Validate model availability
        if not inference_engine.is_model_available(model_name, version):
            raise ModelNotFoundError(
                f"Model {model_name}:{version} not available via API"
            )
        
        if progress_callback:
            progress_callback("ready", 100, 100)
        
        # Create VoiceModel wrapper
        voice_model = VoiceModel(
            name=model_name,
            backend=BackendType.API,
            inference_engine=inference_engine,
            model_info=model_info,
            **model_kwargs
        )
        
        logger.info(f"Connected to API model: {model_name}")
        return voice_model
    
    def is_available(self) -> bool:
        """Check if API backend is available."""
        try:
            import requests
            
            # Test API connection
            response = requests.get(
                f"{self.config.api_endpoint}/health",
                timeout=5,
                headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get API backend capabilities."""
        capabilities = {
            "type": "api",
            "available": self.is_available(),
            "endpoint": self.config.api_endpoint,
            "authenticated": bool(self.config.api_key),
            "supports_compilation": False,
            "supports_quantization": False,
            "offline_capable": False,
            "supports_streaming": True,
            "supports_batch": True
        }
        
        # Get API info if available
        if self.is_available():
            try:
                import requests
                response = requests.get(
                    f"{self.config.api_endpoint}/info",
                    timeout=5,
                    headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
                )
                if response.status_code == 200:
                    api_info = response.json()
                    capabilities.update({
                        "api_version": api_info.get("version"),
                        "supported_models": api_info.get("supported_models", []),
                        "rate_limits": api_info.get("rate_limits", {}),
                        "regions": api_info.get("regions", [])
                    })
            except Exception as e:
                logger.warning(f"Could not fetch API info: {e}")
        
        return capabilities
    
    def _check_configuration(self):
        """Check API backend configuration."""
        if not self.config.api_endpoint:
            raise ConfigurationError(
                "API endpoint not configured. Set ECHOGRID_API_ENDPOINT or use eg.configure(api_endpoint=...)"
            )


# Backend registry
_backends = {
    BackendType.LOCAL: LocalBackend,
    BackendType.API: APIBackend
}

# Backend instances cache
_backend_instances = {}


def get_backend(backend_type: BackendType) -> BaseBackend:
    """
    Get backend instance.
    
    Args:
        backend_type: Type of backend to get
    
    Returns:
        BaseBackend: Backend instance
    
    Raises:
        ConfigurationError: If backend type is not supported
    """
    if backend_type not in _backends:
        raise ConfigurationError(f"Unsupported backend type: {backend_type}")
    
    # Return cached instance if available
    if backend_type in _backend_instances:
        return _backend_instances[backend_type]
    
    # Create new instance
    backend_class = _backends[backend_type]
    backend_instance = backend_class()
    
    # Cache instance
    _backend_instances[backend_type] = backend_instance
    
    return backend_instance


def list_backends() -> Dict[str, Dict[str, Any]]:
    """
    List all available backends with their capabilities.
    
    Returns:
        Dict[str, Dict[str, Any]]: Backend information
    
    Example:
        >>> import echogrid as eg
        >>> backends = eg.list_backends()
        >>> for name, info in backends.items():
        ...     print(f"{name}: available={info['available']}")
    """
    result = {}
    
    for backend_type in BackendType:
        try:
            backend = get_backend(backend_type)
            result[backend_type.value] = backend.get_capabilities()
        except Exception as e:
            result[backend_type.value] = {
                "type": backend_type.value,
                "available": False,
                "error": str(e)
            }
    
    return result


def get_optimal_backend(
    model_info: Optional[Any] = None,
    requirements: Optional[Dict[str, Any]] = None
) -> BackendType:
    """
    Get optimal backend for given requirements.
    
    Args:
        model_info: Model metadata for optimization
        requirements: Specific requirements (e.g., {"offline": True, "speed": "high"})
    
    Returns:
        BackendType: Recommended backend type
    
    Example:
        >>> import echogrid as eg
        >>> backend = eg.get_optimal_backend(requirements={"offline": True})
        >>> print(f"Recommended backend: {backend.value}")
    """
    config = get_config()
    requirements = requirements or {}
    
    # Check if offline is required
    if requirements.get("offline", False):
        return BackendType.LOCAL
    
    # Check local backend availability and capabilities
    try:
        local_backend = get_backend(BackendType.LOCAL)
        local_caps = local_backend.get_capabilities()
        
        if not local_caps["available"]:
            return BackendType.API
        
        # Check memory requirements
        if model_info and hasattr(model_info, 'memory_usage'):
            required_memory = model_info.memory_usage / 1024  # MB to GB
            available_memory = local_caps.get("memory", {}).get("available_gb", 0)
            
            if required_memory > available_memory:
                logger.info(
                    f"Insufficient local memory ({available_memory:.1f}GB available, "
                    f"{required_memory:.1f}GB required), using API backend"
                )
                return BackendType.API
        
        # Prefer local if user configured it
        if config.prefer_local_if_available:
            return BackendType.LOCAL
    
    except Exception as e:
        logger.warning(f"Local backend not available: {e}")
    
    # Check API backend
    try:
        api_backend = get_backend(BackendType.API)
        if api_backend.is_available():
            return BackendType.API
    except Exception as e:
        logger.warning(f"API backend not available: {e}")
    
    # Default to local if nothing else works
    return BackendType.LOCAL


def register_backend(backend_type: BackendType, backend_class: type):
    """
    Register a custom backend.
    
    Args:
        backend_type: Backend type identifier
        backend_class: Backend class implementing BaseBackend
    
    Example:
        >>> import echogrid as eg
        >>> 
        >>> class CustomBackend(eg.BaseBackend):
        ...     def load_model(self, **kwargs):
        ...         # Custom implementation
        ...         pass
        ...     
        ...     def is_available(self):
        ...         return True
        ...     
        ...     def get_capabilities(self):
        ...         return {"type": "custom"}
        >>> 
        >>> eg.register_backend(BackendType.CUSTOM, CustomBackend)
    """
    if not issubclass(backend_class, BaseBackend):
        raise ValueError("Backend class must inherit from BaseBackend")
    
    _backends[backend_type] = backend_class
    
    # Clear cached instance if it exists
    if backend_type in _backend_instances:
        del _backend_instances[backend_type]
    
    logger.info(f"Registered custom backend: {backend_type.value}")


def clear_backend_cache():
    """
    Clear cached backend instances.
    
    This forces backends to be re-initialized on next use,
    which can be useful after configuration changes.
    """
    global _backend_instances
    _backend_instances.clear()