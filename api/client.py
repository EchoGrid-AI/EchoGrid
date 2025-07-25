# echogrid/api/client.py
"""
Main API client for EchoGrid hosted services.

This module provides the primary interface for communicating with EchoGrid's
hosted API endpoints, handling authentication, request management, and response parsing.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union, Iterator
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import get_config
from ..exceptions import (
    NetworkError, APIRateLimitError, AuthenticationError, 
    ValidationError, ModelNotFoundError
)
from .auth import APIAuth
from .endpoints import APIEndpoints
from .retry import RetryHandler
from .rate_limiting import RateLimiter
from .streaming import StreamingClient

logger = logging.getLogger(__name__)


class APIClient:
    """
    Main API client for EchoGrid hosted services.
    
    This class provides a high-level interface for all API operations including
    model discovery, text-to-speech generation, and account management.
    
    Args:
        api_key: API authentication key
        endpoint: API base URL
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        user_agent: Custom user agent string
        
    Example:
        >>> client = APIClient(api_key="your-key")
        >>> models = client.list_models()
        >>> audio = client.generate_speech("Hello world", "alice/voice")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        verify_ssl: bool = True,
        proxy: Optional[str] = None
    ):
        """Initialize API client."""
        self.config = get_config()
        
        # Configuration
        self.api_key = api_key or self.config.api_key
        self.endpoint = endpoint or self.config.api_endpoint
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        if not self.api_key:
            logger.warning("No API key provided. Some features may be unavailable.")
        
        # Initialize components
        self.auth = APIAuth(self.api_key)
        self.endpoints = APIEndpoints(self.endpoint)
        self.retry_handler = RetryHandler(max_retries=max_retries)
        self.rate_limiter = RateLimiter()
        self.streaming = StreamingClient(self)
        
        # Setup HTTP session
        self.session = requests.Session()
        
        # Configure session headers
        headers = {
            "User-Agent": user_agent or f"echogrid-python/{self._get_version()}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.session.headers.update(headers)
        
        # Configure retries
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            respect_retry_after_header=True
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Configure proxy
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
        
        # Performance tracking
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0
        
        logger.debug(f"Initialized API client for {self.endpoint}")
    
    def list_models(
        self,
        language: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        List available voice models.
        
        Args:
            language: Filter by language code
            category: Filter by model category
            limit: Maximum results to return
            offset: Pagination offset
            **filters: Additional filter parameters
            
        Returns:
            List of model information dictionaries
            
        Raises:
            NetworkError: If the API request fails
            ValidationError: If invalid parameters provided
        """
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if language:
            params["language"] = language
        if category:
            params["category"] = category
        
        # Add additional filters
        params.update(filters)
        
        try:
            response = self._make_request(
                "GET",
                self.endpoints.models_list,
                params=params
            )
            
            data = response.json()
            return data.get("models", [])
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise NetworkError(f"Failed to list models: {e}") from e
    
    def search_models(
        self,
        query: str,
        language: Optional[str] = None,
        limit: int = 20,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for voice models.
        
        Args:
            query: Search query string
            language: Filter by language
            limit: Maximum results
            **kwargs: Additional search parameters
            
        Returns:
            List of matching model information
        """
        params = {
            "q": query,
            "limit": limit
        }
        
        if language:
            params["language"] = language
        
        params.update(kwargs)
        
        try:
            response = self._make_request(
                "GET", 
                self.endpoints.models_search,
                params=params
            )
            
            data = response.json()
            return data.get("results", [])
            
        except Exception as e:
            logger.error(f"Model search failed: {e}")
            raise NetworkError(f"Model search failed: {e}") from e
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Model identifier (username/model-name)
            
        Returns:
            Model information dictionary
            
        Raises:
            ModelNotFoundError: If model doesn't exist
            NetworkError: If API request fails
        """
        try:
            response = self._make_request(
                "GET",
                self.endpoints.model_info(model_name)
            )
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(model_name) from e
            raise NetworkError(f"Failed to get model info: {e}") from e
    
    def generate_speech(
        self,
        text: str,
        model_name: str,
        **generation_params
    ) -> Dict[str, Any]:
        """
        Generate speech from text using specified model.
        
        Args:
            text: Input text to synthesize
            model_name: Voice model to use
            **generation_params: Additional generation parameters
            
        Returns:
            Generation result with audio data and metadata
            
        Raises:
            ValidationError: If invalid parameters provided
            NetworkError: If generation fails
            APIRateLimitError: If rate limit exceeded
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty")
        
        if len(text) > 10000:  # API limit
            raise ValidationError("Text too long (max 10,000 characters)")
        
        # Prepare request payload
        payload = {
            "text": text,
            "model": model_name,
            "parameters": generation_params
        }
        
        try:
            # Check rate limits
            self.rate_limiter.check_limit()
            
            response = self._make_request(
                "POST",
                self.endpoints.generate,
                json=payload,
                timeout=generation_params.get("timeout", self.timeout)
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                self.rate_limiter.handle_rate_limit(retry_after)
                raise APIRateLimitError(retry_after=retry_after)
            
            return response.json()
            
        except APIRateLimitError:
            raise
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise NetworkError(f"Speech generation failed: {e}") from e
    
    def generate_speech_async(
        self,
        text: str,
        model_name: str,
        callback_url: str,
        **generation_params
    ) -> Dict[str, Any]:
        """
        Start asynchronous speech generation.
        
        Args:
            text: Input text to synthesize
            model_name: Voice model to use
            callback_url: Webhook URL for completion notification
            **generation_params: Additional generation parameters
            
        Returns:
            Job information with ID and status
        """
        payload = {
            "text": text,
            "model": model_name,
            "callback_url": callback_url,
            "parameters": generation_params
        }
        
        try:
            response = self._make_request(
                "POST",
                self.endpoints.generate_async,
                json=payload
            )
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            raise NetworkError(f"Async generation failed: {e}") from e
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of asynchronous job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        try:
            response = self._make_request(
                "GET",
                self.endpoints.job_status(job_id)
            )
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise NetworkError(f"Failed to get job status: {e}") from e
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel asynchronous job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancellation was successful
        """
        try:
            response = self._make_request(
                "DELETE",
                self.endpoints.job_cancel(job_id)
            )
            
            return response.status_code == 200
            
        except Exception:
            return False
    
    def stream_speech(
        self,
        text: str,
        model_name: str,
        **generation_params
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream speech generation for real-time playback.
        
        Args:
            text: Input text to synthesize
            model_name: Voice model to use
            **generation_params: Additional generation parameters
            
        Yields:
            Audio chunks as they're generated
        """
        return self.streaming.stream_speech(text, model_name, **generation_params)
    
    def upload_model(
        self,
        model_data: Dict[str, Any],
        files: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload a voice model to the community hub.
        
        Args:
            model_data: Model metadata and configuration
            files: Model files to upload
            
        Returns:
            Upload result with model URL and status
        """
        try:
            # Prepare multipart upload
            if files:
                response = self._make_request(
                    "POST",
                    self.endpoints.models_upload,
                    data=model_data,
                    files=files
                )
            else:
                response = self._make_request(
                    "POST",
                    self.endpoints.models_upload,
                    json=model_data
                )
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Model upload failed: {e}")
            raise NetworkError(f"Model upload failed: {e}") from e
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics for current account.
        
        Returns:
            Usage statistics including requests, characters, and limits
        """
        try:
            response = self._make_request("GET", self.endpoints.usage_stats)
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            raise NetworkError(f"Failed to get usage stats: {e}") from e
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection and authentication.
        
        Returns:
            Connection test results including latency and status
        """
        start_time = time.time()
        
        try:
            response = self._make_request("GET", self.endpoints.health)
            latency = time.time() - start_time
            
            result = response.json()
            result["latency"] = latency
            result["authenticated"] = bool(self.api_key)
            
            return result
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "latency": time.time() - start_time,
                "authenticated": bool(self.api_key)
            }
    
    def _make_request(
        self,
        method: str,
        url: str,
        timeout: Optional[int] = None,
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with error handling and retries.
        
        Args:
            method: HTTP method
            url: Request URL
            timeout: Request timeout
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response object
            
        Raises:
            NetworkError: If request fails after retries
            AuthenticationError: If authentication fails
        """
        timeout = timeout or self.timeout
        
        try:
            # Update request count
            self._request_count += 1
            
            # Make request with retry logic
            start_time = time.time()
            
            response = self.session.request(
                method=method,
                url=url,
                timeout=timeout,
                verify=self.verify_ssl,
                **kwargs
            )
            
            # Track latency
            self._total_latency += time.time() - start_time
            
            # Handle authentication errors
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key or unauthorized")
            
            # Handle other HTTP errors
            response.raise_for_status()
            
            return response
            
        except requests.exceptions.RequestException as e:
            self._error_count += 1
            logger.error(f"API request failed: {e}")
            raise NetworkError(f"API request failed: {e}") from e
    
    def get_client_stats(self) -> Dict[str, Any]:
        """
        Get client-side statistics.
        
        Returns:
            Client performance and usage statistics
        """
        avg_latency = (
            self._total_latency / self._request_count 
            if self._request_count > 0 else 0
        )
        
        success_rate = (
            (self._request_count - self._error_count) / self._request_count
            if self._request_count > 0 else 0
        )
        
        return {
            "requests_made": self._request_count,
            "errors": self._error_count,
            "success_rate": success_rate,
            "average_latency": avg_latency
        }
    
    def _get_version(self) -> str:
        """Get library version."""
        try:
            from ..version import __version__
            return __version__
        except ImportError:
            return "unknown"
    
    def close(self):
        """Close HTTP session and cleanup resources."""
        if self.session:
            self.session.close()
        
        logger.debug("API client session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global API client instance
_api_client: Optional[APIClient] = None


def get_api_client(**kwargs) -> APIClient:
    """
    Get or create global API client instance.
    
    Args:
        **kwargs: API client configuration options
        
    Returns:
        Global API client instance
    """
    global _api_client
    
    if _api_client is None:
        _api_client = APIClient(**kwargs)
    
    return _api_client


def set_api_key(api_key: str, persist: bool = True):
    """
    Set API key for global client.
    
    Args:
        api_key: EchoGrid API key
        persist: Save to configuration file
    """
    global _api_client
    
    # Update global config
    config = get_config()
    config.api_key = api_key
    
    # Reset client to use new key
    if _api_client:
        _api_client.close()
        _api_client = None
    
    # Persist to config file if requested
    if persist:
        try:
            from ..config import save_config
            save_config()
        except Exception as e:
            logger.warning(f"Failed to persist API key: {e}")
    
    logger.info("API key updated")


def configure_api(**kwargs):
    """
    Configure global API client settings.
    
    Args:
        **kwargs: API configuration options
    """
    global _api_client
    
    # Update global config
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, f"api_{key}"):
            setattr(config, f"api_{key}", value)
        elif key in ["endpoint", "timeout", "retries"]:
            setattr(config, f"api_{key}", value)
    
    # Reset client to use new configuration
    if _api_client:
        _api_client.close()
        _api_client = None
    
    logger.debug("API configuration updated")