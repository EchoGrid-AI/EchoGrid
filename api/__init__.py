# echogrid/api/__init__.py
"""
API client for EchoGrid hosted inference.

This module provides a complete API client for accessing EchoGrid's hosted
text-to-speech services, including authentication, rate limiting, and streaming support.

The API client supports:
- Voice model discovery and information retrieval
- Text-to-speech generation with various options
- Asynchronous and streaming generation
- Model uploading and management
- Usage tracking and account management
- Comprehensive error handling and retry logic

Example:
    Basic usage with API key::

        import echogrid
        
        # Configure API key
        echogrid.set_api_key("your-api-key")
        
        # List available models
        models = echogrid.api.list_models()
        
        # Generate speech
        audio = echogrid.api.generate_speech(
            text="Hello world",
            model="alice/voice",
            language="en"
        )

    Advanced client configuration::

        from echogrid.api import APIClient
        
        client = APIClient(
            api_key="your-key",
            endpoint="https://api.echogrid.dev/v1",
            timeout=60,
            max_retries=5
        )
        
        # Use client for all operations
        result = client.generate_speech("Hello world", "alice/voice")

.. note::
   All API operations require a valid API key. You can obtain one
   by registering at https://echogrid.dev/signup

.. warning::
   API keys should be kept secure and never committed to version control.
   Consider using environment variables or configuration files.
"""

from .client import APIClient, get_api_client, set_api_key, configure_api
from .auth import APIAuth
from .endpoints import APIEndpoints
from .retry import RetryHandler
from .rate_limiting import RateLimiter
from .streaming import StreamingClient, StreamSession, StreamBuffer

# Create global API instance
api = get_api_client()

__all__ = [
    "APIClient",
    "APIAuth", 
    "APIEndpoints",
    "RetryHandler",
    "RateLimiter",
    "StreamingClient",
    "StreamSession",
    "StreamBuffer",
    "get_api_client",
    "set_api_key", 
    "configure_api",
    "api"
]