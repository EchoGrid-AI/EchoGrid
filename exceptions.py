# echogrid/exceptions.py
"""
Exception classes for EchoGrid.

This module defines custom exception classes used throughout the EchoGrid library
to provide clear error handling and user feedback.
"""

from typing import List, Optional


class EchoGridError(Exception):
    """
    Base exception for all EchoGrid errors.
    
    This is the parent class for all custom exceptions in the EchoGrid library.
    It provides common functionality for error handling and logging.
    
    Args:
        message: Error message describing what went wrong
        code: Optional error code for programmatic handling
        details: Optional additional details about the error
    """
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)
    
    def __str__(self):
        return self.message
    
    def to_dict(self):
        """
        Convert exception to dictionary representation.
        
        Returns:
            dict: Dictionary containing error information
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details
        }


class ModelNotFoundError(EchoGridError):
    """
    Raised when a requested model cannot be found.
    
    This exception is raised when attempting to load, download, or access
    a model that doesn't exist in the local cache or remote repository.
    
    Args:
        model_name: Name of the model that wasn't found
        suggestions: List of similar model names that might be intended
        available_models: List of all available models
    
    Example:
        >>> try:
        ...     model = eg.load("nonexistent/model")
        ... except ModelNotFoundError as e:
        ...     print(f"Model not found: {e.model_name}")
        ...     print(f"Suggestions: {e.suggestions}")
    """
    
    def __init__(self, model_name: str, suggestions: Optional[List[str]] = None, 
                 available_models: Optional[List[str]] = None):
        self.model_name = model_name
        self.suggestions = suggestions or []
        self.available_models = available_models or []
        
        message = f"Model '{model_name}' not found"
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions[:3])}?"
        
        super().__init__(message, code="MODEL_NOT_FOUND", details={
            "model_name": model_name,
            "suggestions": suggestions,
            "available_models": available_models
        })


class InsufficientMemoryError(EchoGridError):
    """
    Raised when there's insufficient memory for an operation.
    
    This exception is raised when attempting to load a model or perform
    an operation that requires more memory than is available.
    
    Args:
        required_gb: Amount of memory required in GB
        available_gb: Amount of memory available in GB
        operation: Description of the operation that failed
    
    Example:
        >>> try:
        ...     model = eg.load("large-model", device="cuda")
        ... except InsufficientMemoryError as e:
        ...     print(f"Need {e.required_gb}GB, have {e.available_gb}GB")
    """
    
    def __init__(self, required_gb: float, available_gb: float, operation: str = "operation"):
        self.required_gb = required_gb
        self.available_gb = available_gb
        self.operation = operation
        
        message = (f"Insufficient memory for {operation}: {required_gb:.1f}GB required, "
                  f"{available_gb:.1f}GB available")
        
        super().__init__(message, code="INSUFFICIENT_MEMORY", details={
            "required_gb": required_gb,
            "available_gb": available_gb,
            "operation": operation
        })


class APIRateLimitError(EchoGridError):
    """
    Raised when API rate limit is exceeded.
    
    This exception is raised when the API client has exceeded the allowed
    number of requests per time period.
    
    Args:
        retry_after: Number of seconds to wait before retrying
        limit: The rate limit that was exceeded
        current_usage: Current usage count
    
    Example:
        >>> try:
        ...     audio = model.tts("text", backend="api")
        ... except APIRateLimitError as e:
        ...     time.sleep(e.retry_after)
        ...     # Retry the request
    """
    
    def __init__(self, retry_after: Optional[int] = None, limit: Optional[int] = None, 
                 current_usage: Optional[int] = None):
        self.retry_after = retry_after
        self.limit = limit
        self.current_usage = current_usage
        
        message = "API rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        if limit and current_usage:
            message += f" ({current_usage}/{limit} requests)"
        
        super().__init__(message, code="RATE_LIMIT_EXCEEDED", details={
            "retry_after": retry_after,
            "limit": limit,
            "current_usage": current_usage
        })


class TrainingError(EchoGridError):
    """
    Raised when model training fails.
    
    This exception is raised when training, fine-tuning, or voice cloning
    operations encounter errors.
    
    Args:
        stage: Training stage where error occurred (e.g., "preprocessing", "training", "validation")
        epoch: Training epoch when error occurred (if applicable)
        original_error: The original exception that caused the training to fail
    
    Example:
        >>> try:
        ...     model = eg.train_model(dataset, "my-model")
        ... except TrainingError as e:
        ...     print(f"Training failed at {e.stage}: {e.message}")
    """
    
    def __init__(self, message: str, stage: Optional[str] = None, 
                 epoch: Optional[int] = None, original_error: Optional[Exception] = None):
        self.stage = stage
        self.epoch = epoch
        self.original_error = original_error
        
        full_message = f"Training failed: {message}"
        if stage:
            full_message = f"Training failed at {stage}: {message}"
        if epoch is not None:
            full_message += f" (epoch {epoch})"
        
        super().__init__(full_message, code="TRAINING_FAILED", details={
            "stage": stage,
            "epoch": epoch,
            "original_error": str(original_error) if original_error else None
        })


class AudioProcessingError(EchoGridError):
    """
    Raised when audio processing operations fail.
    
    This exception is raised when audio loading, conversion, enhancement,
    or other processing operations encounter errors.
    
    Args:
        operation: The audio operation that failed
        file_path: Path to the audio file being processed (if applicable)
        original_error: The original exception that caused the failure
    
    Example:
        >>> try:
        ...     audio = eg.AudioFile("corrupted.wav")
        ... except AudioProcessingError as e:
        ...     print(f"Failed to load audio: {e.message}")
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 file_path: Optional[str] = None, original_error: Optional[Exception] = None):
        self.operation = operation
        self.file_path = file_path
        self.original_error = original_error
        
        full_message = message
        if operation:
            full_message = f"Audio {operation} failed: {message}"
        
        super().__init__(full_message, code="AUDIO_PROCESSING_FAILED", details={
            "operation": operation,
            "file_path": file_path,
            "original_error": str(original_error) if original_error else None
        })


class ConfigurationError(EchoGridError):
    """
    Raised for configuration-related errors.
    
    This exception is raised when there are issues with configuration
    files, environment variables, or invalid configuration values.
    
    Args:
        config_key: The configuration key that has an invalid value
        invalid_value: The invalid value that was provided
        valid_values: List of valid values for this configuration key
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 invalid_value: Optional[str] = None, valid_values: Optional[List[str]] = None):
        self.config_key = config_key
        self.invalid_value = invalid_value
        self.valid_values = valid_values
        
        super().__init__(message, code="CONFIGURATION_ERROR", details={
            "config_key": config_key,
            "invalid_value": invalid_value,
            "valid_values": valid_values
        })


class NetworkError(EchoGridError):
    """
    Raised for network-related errors.
    
    This exception is raised when network operations like downloading
    models, API requests, or hub communication fail.
    
    Args:
        operation: The network operation that failed
        status_code: HTTP status code (if applicable)
        url: The URL that was being accessed
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 status_code: Optional[int] = None, url: Optional[str] = None):
        self.operation = operation
        self.status_code = status_code
        self.url = url
        
        super().__init__(message, code="NETWORK_ERROR", details={
            "operation": operation,
            "status_code": status_code,
            "url": url
        })


class ValidationError(EchoGridError):
    """
    Raised for validation errors.
    
    This exception is raised when input validation fails, such as
    invalid parameter values, malformed data, or constraint violations.
    
    Args:
        field: The field or parameter that failed validation
        value: The invalid value
        constraint: Description of the validation constraint that was violated
    """
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[str] = None, constraint: Optional[str] = None):
        self.field = field
        self.value = value
        self.constraint = constraint
        
        super().__init__(message, code="VALIDATION_ERROR", details={
            "field": field,
            "value": value,
            "constraint": constraint
        })


class AuthenticationError(EchoGridError):
    """
    Raised for authentication-related errors.
    
    This exception is raised when API authentication fails due to
    invalid credentials, expired tokens, or missing API keys.
    """
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, code="AUTHENTICATION_ERROR")


class PermissionError(EchoGridError):
    """
    Raised when user lacks permission for an operation.
    
    This exception is raised when attempting to perform operations
    that require elevated permissions or access to restricted resources.
    
    Args:
        resource: The resource that access was denied to
        required_permission: The permission level required
    """
    
    def __init__(self, message: str, resource: Optional[str] = None, 
                 required_permission: Optional[str] = None):
        self.resource = resource
        self.required_permission = required_permission
        
        super().__init__(message, code="PERMISSION_DENIED", details={
            "resource": resource,
            "required_permission": required_permission
        })


class ModelArchitectureError(EchoGridError):
    """
    Raised when there are issues with model architecture or compatibility.
    
    This exception is raised when attempting to load models with
    unsupported architectures or version mismatches.
    """
    
    def __init__(self, message: str, architecture: Optional[str] = None, 
                 supported_architectures: Optional[List[str]] = None):
        self.architecture = architecture
        self.supported_architectures = supported_architectures
        
        super().__init__(message, code="MODEL_ARCHITECTURE_ERROR", details={
            "architecture": architecture,
            "supported_architectures": supported_architectures
        })