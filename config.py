# echogrid/config.py
"""
Configuration management for EchoGrid.

This module handles global configuration for the EchoGrid library, including
settings for backends, caching, API access, and default parameters. Configuration
can be set through code, environment variables, or configuration files.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

from .constants import (
    DEFAULT_API_ENDPOINT, DEFAULT_CACHE_DIR, DEFAULT_SAMPLE_RATE,
    ENV_VARS, SUPPORTED_ARCHITECTURES, DEFAULT_MAX_MEMORY_GB
)
from .exceptions import ConfigurationError


@dataclass
class EchoGridConfig:
    """
    Global configuration for EchoGrid.
    
    This class contains all configuration options for the EchoGrid library.
    Settings can be modified at runtime and are automatically validated.
    
    Attributes:
        default_backend: Default inference backend ("auto", "local", "api")
        prefer_local_if_available: Prefer local inference when available
        api_fallback: Fall back to API if local inference fails
        local_device: Default device for local inference ("auto", "cpu", "cuda", "mps")
        local_precision: Default precision for local inference ("auto", "fp32", "fp16", "int8")
        max_local_memory: Maximum memory limit for local models (e.g., "8GB")
        compile_models: Enable model compilation for faster inference
        cache_models: Keep models in memory after loading
        model_cache_size: Maximum number of models to keep in memory
        api_endpoint: API endpoint URL
        api_key: API authentication key
        api_timeout: API request timeout in seconds
        api_retries: Number of API retry attempts
        api_region: Preferred API region ("auto", "us-east-1", "eu-west-1")
        cache_dir: Directory for model cache and temporary files
        temp_dir: Directory for temporary files (None = use cache_dir/temp)
        default_sample_rate: Default audio sample rate
        default_format: Default audio format for output
        default_quality: Default quality setting for audio generation
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_to_file: Enable logging to file
        log_file: Log file path (None = auto-generate)
        max_workers: Maximum number of worker threads (None = auto)
        enable_profiling: Enable performance profiling
        allow_custom_models: Allow loading custom/untrusted models
        verify_downloads: Verify integrity of downloaded files
        anonymous_usage_stats: Allow anonymous usage statistics collection
    """
    
    # Backend preferences
    default_backend: str = "auto"
    prefer_local_if_available: bool = True
    api_fallback: bool = True
    
    # Local inference settings
    local_device: str = "auto"
    local_precision: str = "auto"
    max_local_memory: Optional[str] = None
    compile_models: bool = True
    cache_models: bool = True
    model_cache_size: int = 3
    
    # API settings
    api_endpoint: str = DEFAULT_API_ENDPOINT
    api_key: Optional[str] = None
    api_timeout: int = 30
    api_retries: int = 3
    api_region: str = "auto"
    
    # Cache and storage
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    temp_dir: Optional[str] = None
    
    # Audio defaults
    default_sample_rate: int = DEFAULT_SAMPLE_RATE
    default_format: str = "wav"
    default_quality: str = "high"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: Optional[str] = None
    
    # Performance
    max_workers: Optional[int] = None
    enable_profiling: bool = False
    
    # Security and validation
    allow_custom_models: bool = True
    verify_downloads: bool = True
    anonymous_usage_stats: bool = True


# Global configuration instance
_config: Optional[EchoGridConfig] = None


def configure(**kwargs) -> EchoGridConfig:
    """
    Configure EchoGrid with custom settings.
    
    This function initializes or updates the global EchoGrid configuration.
    Settings are loaded from (in order of precedence):
    1. Keyword arguments passed to this function
    2. Environment variables
    3. Configuration files
    4. Default values
    
    Args:
        **kwargs: Configuration options to set. Can be any attribute of EchoGridConfig.
    
    Returns:
        EchoGridConfig: The updated configuration object
    
    Raises:
        ConfigurationError: If invalid configuration values are provided
    
    Example:
        >>> import echogrid as eg
        >>> config = eg.configure(
        ...     default_backend="local",
        ...     local_device="cuda:0",
        ...     cache_dir="./my_models"
        ... )
        >>> print(config.default_backend)
        local
    """
    global _config
    
    if _config is None:
        _config = EchoGridConfig()
    
    # Update from environment variables
    _update_from_env(_config)
    
    # Update from config file
    _update_from_config_file(_config)
    
    # Update from kwargs
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
        else:
            raise ConfigurationError(
                f"Unknown configuration option: {key}",
                config_key=key,
                invalid_value=str(value)
            )
    
    # Validate configuration
    _validate_config(_config)
    
    # Setup logging
    _setup_logging(_config)
    
    return _config


def get_config() -> EchoGridConfig:
    """
    Get the current global configuration.
    
    If no configuration has been set, this will initialize it with default values.
    
    Returns:
        EchoGridConfig: The current configuration object
    
    Example:
        >>> import echogrid as eg
        >>> config = eg.get_config()
        >>> print(config.cache_dir)
        /home/user/.echogrid
    """
    if _config is None:
        return configure()
    return _config


def reset_config():
    """
    Reset configuration to default values.
    
    This function resets the global configuration to its default state,
    clearing any custom settings that have been applied.
    
    Example:
        >>> import echogrid as eg
        >>> eg.configure(cache_dir="/custom/path")
        >>> eg.reset_config()
        >>> config = eg.get_config()
        >>> print(config.cache_dir)  # Back to default
        /home/user/.echogrid
    """
    global _config
    _config = None
    configure()


def save_config(path: Optional[Union[str, Path]] = None):
    """
    Save current configuration to a file.
    
    Args:
        path: Path to save the configuration file. If None, saves to the default
              location (~/.echogrid/config.yaml)
    
    Raises:
        ConfigurationError: If the configuration cannot be saved
    
    Example:
        >>> import echogrid as eg
        >>> eg.configure(default_backend="local")
        >>> eg.save_config("my_config.yaml")
    """
    config = get_config()
    
    if path is None:
        path = Path(config.cache_dir) / "config.yaml"
    else:
        path = Path(path)
    
    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dict, excluding None values
    config_dict = {}
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        if value is not None:
            config_dict[field_name] = value
    
    try:
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration: {e}")


def load_config(path: Union[str, Path]):
    """
    Load configuration from a file.
    
    Args:
        path: Path to the configuration file
    
    Raises:
        ConfigurationError: If the configuration file cannot be loaded or contains invalid values
    
    Example:
        >>> import echogrid as eg
        >>> eg.load_config("my_config.yaml")
        >>> config = eg.get_config()
    """
    path = Path(path)
    
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if config_data:
            configure(**config_data)
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration from {path}: {e}")


def _update_from_env(config: EchoGridConfig):
    """Update configuration from environment variables."""
    env_mapping = {
        ENV_VARS["API_KEY"]: ("api_key", str),
        ENV_VARS["API_ENDPOINT"]: ("api_endpoint", str),
        ENV_VARS["CACHE_DIR"]: ("cache_dir", str),
        ENV_VARS["DEVICE"]: ("local_device", str),
        ENV_VARS["DEFAULT_BACKEND"]: ("default_backend", str),
        ENV_VARS["LOG_LEVEL"]: ("log_level", str),
        "ECHOGRID_API_TIMEOUT": ("api_timeout", int),
        "ECHOGRID_API_RETRIES": ("api_retries", int),
        "ECHOGRID_MODEL_CACHE_SIZE": ("model_cache_size", int),
        "ECHOGRID_DEFAULT_SAMPLE_RATE": ("default_sample_rate", int),
        "ECHOGRID_PREFER_LOCAL": ("prefer_local_if_available", bool),
        "ECHOGRID_API_FALLBACK": ("api_fallback", bool),
        "ECHOGRID_COMPILE_MODELS": ("compile_models", bool),
        "ECHOGRID_CACHE_MODELS": ("cache_models", bool),
        "ECHOGRID_VERIFY_DOWNLOADS": ("verify_downloads", bool),
    }
    
    for env_var, (config_attr, value_type) in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            try:
                if value_type == bool:
                    value = value.lower() in ("true", "1", "yes", "on")
                elif value_type == int:
                    value = int(value)
                # str values need no conversion
                
                setattr(config, config_attr, value)
            except ValueError:
                logging.warning(f"Invalid value for {env_var}: {value}")


def _update_from_config_file(config: EchoGridConfig):
    """Update configuration from YAML config files."""
    config_paths = [
        Path.home() / ".echogrid" / "config.yaml",
        Path.cwd() / "echogrid.yaml",
        Path(__file__).parent / "data" / "configs" / "default.yaml",
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                
                if file_config:
                    for key, value in file_config.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                break
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {e}")
                continue


def _validate_config(config: EchoGridConfig):
    """Validate configuration values."""
    # Expand user paths
    config.cache_dir = str(Path(config.cache_dir).expanduser())
    if config.temp_dir:
        config.temp_dir = str(Path(config.temp_dir).expanduser())
    
    # Create directories
    cache_path = Path(config.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    if config.temp_dir:
        temp_path = Path(config.temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
    
    # Validate backend
    if config.default_backend not in ["auto", "local", "api"]:
        raise ConfigurationError(
            f"Invalid default_backend: {config.default_backend}",
            config_key="default_backend",
            invalid_value=config.default_backend,
            valid_values=["auto", "local", "api"]
        )
    
    # Validate device
    valid_devices = ["auto", "cpu", "cuda", "mps"]
    if (config.local_device not in valid_devices and 
        not config.local_device.startswith("cuda:")):
        raise ConfigurationError(
            f"Invalid local_device: {config.local_device}",
            config_key="local_device",
            invalid_value=config.local_device,
            valid_values=valid_devices + ["cuda:N"]
        )
    
    # Validate precision
    valid_precisions = ["auto", "fp32", "fp16", "bf16", "int8", "int4"]
    if config.local_precision not in valid_precisions:
        raise ConfigurationError(
            f"Invalid local_precision: {config.local_precision}",
            config_key="local_precision",
            invalid_value=config.local_precision,
            valid_values=valid_precisions
        )
    
    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.log_level not in valid_log_levels:
        raise ConfigurationError(
            f"Invalid log_level: {config.log_level}",
            config_key="log_level",
            invalid_value=config.log_level,
            valid_values=valid_log_levels
        )
    
    # Validate numeric values
    if config.api_timeout <= 0:
        raise ConfigurationError("api_timeout must be positive")
    
    if config.api_retries < 0:
        raise ConfigurationError("api_retries must be non-negative")
    
    if config.model_cache_size < 0:
        raise ConfigurationError("model_cache_size must be non-negative")
    
    if config.default_sample_rate <= 0:
        raise ConfigurationError("default_sample_rate must be positive")


def _setup_logging(config: EchoGridConfig):
    """Setup logging based on configuration."""
    # Get numeric log level
    numeric_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if config.log_to_file:
        log_file = config.log_file
        if log_file is None:
            log_file = str(Path(config.cache_dir) / "echogrid.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


class configure_session:
    """
    Context manager for temporary configuration changes.
    
    This context manager allows you to temporarily modify the EchoGrid configuration
    within a specific scope, automatically restoring the original values when exiting.
    
    Args:
        **kwargs: Configuration options to temporarily set
    
    Example:
        >>> import echogrid as eg
        >>> with eg.configure_session(default_backend="api", api_timeout=60):
        ...     # Use API backend with 60s timeout
        ...     model = eg.load("alice/voice")
        ...     audio = model.tts("Hello")
        >>> # Configuration automatically restored here
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.original_values = {}
    
    def __enter__(self) -> EchoGridConfig:
        """Enter the context and apply temporary configuration."""
        config = get_config()
        
        # Store original values
        for key in self.kwargs:
            if hasattr(config, key):
                self.original_values[key] = getattr(config, key)
                setattr(config, key, self.kwargs[key])
            else:
                raise ConfigurationError(f"Unknown configuration option: {key}")
        
        # Validate the temporary configuration
        _validate_config(config)
        
        return config
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original configuration."""
        config = get_config()
        
        # Restore original values
        for key, value in self.original_values.items():
            setattr(config, key, value)
        
        # Re-validate configuration
        _validate_config(config)





