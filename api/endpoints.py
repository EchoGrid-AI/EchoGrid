# echogrid/api/endpoints.py
"""
API endpoint definitions and URL construction for EchoGrid services.

This module provides a centralized way to manage all API endpoints used by the
EchoGrid service, handling URL construction, parameter encoding, and versioning.
"""

from typing import Dict, Any, Optional, List
from urllib.parse import urljoin, quote, urlencode


class APIEndpoints:
    """
    API endpoint definitions for EchoGrid services.
    
    This class provides methods to construct API URLs for different operations,
    handling URL encoding, parameter substitution, and API versioning. It serves
    as the single source of truth for all EchoGrid API endpoints.
    
    Args:
        base_url (str): Base API URL including version (e.g., "https://api.echogrid.dev/v1")
        api_version (str, optional): API version override. Defaults to version in base_url.
        
    Attributes:
        base_url (str): The base API URL
        api_version (str): Current API version
        
    Example:
        Initialize endpoints and construct URLs::

            endpoints = APIEndpoints("https://api.echogrid.dev/v1")
            
            # Get model info URL
            model_url = endpoints.model_info("alice/voice")
            print(model_url)  # https://api.echogrid.dev/v1/models/alice%2Fvoice
            
            # Get search URL with parameters
            search_url = endpoints.models_search()
            params = {"q": "female english", "language": "en"}
            full_url = f"{search_url}?{urlencode(params)}"

    Note:
        All URLs are properly encoded to handle special characters in model names,
        usernames, and other parameters. The class follows REST conventions for
        resource naming and HTTP methods.
    """
    
    def __init__(self, base_url: str, api_version: Optional[str] = None):
        """Initialize endpoints with base URL and optional version override."""
        self.base_url = base_url.rstrip('/')
        
        # Extract version from URL if not explicitly provided
        if api_version:
            self.api_version = api_version
        elif '/v' in self.base_url:
            # Extract version from URL like https://api.echogrid.dev/v1
            self.api_version = self.base_url.split('/v')[-1]
        else:
            self.api_version = "1"
        
        # Core system endpoints
        self.health = self._url("health")
        self.info = self._url("info")
        self.status = self._url("status")
        self.version = self._url("version")
        
        # Authentication endpoints
        self.auth_validate = self._url("auth/validate")
        self.auth_token = self._url("auth/token")
        self.auth_refresh = self._url("auth/refresh")
        self.auth_authorize = self._url("auth/authorize")
        self.auth_revoke = self._url("auth/revoke")
        
        # Model discovery and management endpoints
        self.models_list = self._url("models")
        self.models_search = self._url("models/search")
        self.models_categories = self._url("models/categories")
        self.models_languages = self._url("models/languages")
        self.models_upload = self._url("models/upload")
        self.models_featured = self._url("models/featured")
        self.models_trending = self._url("models/trending")
        
        # Text-to-speech generation endpoints
        self.generate = self._url("generate")
        self.generate_async = self._url("generate/async")
        self.generate_stream = self._url("generate/stream")
        self.generate_batch = self._url("generate/batch")
        
        # Job management endpoints
        self.jobs = self._url("jobs")
        self.jobs_queue = self._url("jobs/queue")
        
        # Account and usage endpoints
        self.account_info = self._url("account")
        self.account_settings = self._url("account/settings")
        self.usage_stats = self._url("usage")
        self.usage_history = self._url("usage/history")
        self.billing_info = self._url("billing")
        self.subscription = self._url("subscription")
        
        # User and community endpoints
        self.users = self._url("users")
        self.community_stats = self._url("community/stats")
        self.leaderboard = self._url("community/leaderboard")
        
        # Analytics and monitoring
        self.analytics = self._url("analytics")
        self.metrics = self._url("metrics")
        
        # Webhook and notification endpoints
        self.webhooks = self._url("webhooks")
        self.notifications = self._url("notifications")
    
    def model_info(self, model_name: str) -> str:
        """
        Get URL for model information endpoint.
        
        Args:
            model_name (str): Model identifier in format "username/model-name"
            
        Returns:
            str: Model information endpoint URL
            
        Example:
            >>> endpoints = APIEndpoints("https://api.echogrid.dev/v1")
            >>> url = endpoints.model_info("alice/newsreader")
            >>> print(url)
            https://api.echogrid.dev/v1/models/alice%2Fnewsreader
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}")
    
    def model_download(self, model_name: str, version: str = "latest", format: str = "safetensors") -> str:
        """
        Get URL for model download endpoint.
        
        Args:
            model_name (str): Model identifier
            version (str): Model version. Defaults to "latest"
            format (str): Download format ("safetensors", "pytorch", "onnx")
            
        Returns:
            str: Model download endpoint URL
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}/download/{version}?format={format}")
    
    def model_versions(self, model_name: str) -> str:
        """
        Get URL for model versions endpoint.
        
        Args:
            model_name (str): Model identifier
            
        Returns:
            str: Model versions endpoint URL
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}/versions")
    
    def model_samples(self, model_name: str, sample_type: str = "preview") -> str:
        """
        Get URL for model audio samples endpoint.
        
        Args:
            model_name (str): Model identifier
            sample_type (str): Type of samples ("preview", "full", "comparison")
            
        Returns:
            str: Model samples endpoint URL
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}/samples?type={sample_type}")
    
    def model_stats(self, model_name: str) -> str:
        """
        Get URL for model statistics endpoint.
        
        Args:
            model_name (str): Model identifier
            
        Returns:
            str: Model statistics endpoint URL
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}/stats")
    
    def model_reviews(self, model_name: str) -> str:
        """
        Get URL for model reviews endpoint.
        
        Args:
            model_name (str): Model identifier
            
        Returns:
            str: Model reviews endpoint URL
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}/reviews")
    
    def model_fork(self, model_name: str) -> str:
        """
        Get URL for model forking endpoint.
        
        Args:
            model_name (str): Model identifier to fork
            
        Returns:
            str: Model fork endpoint URL
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}/fork")
    
    def job_status(self, job_id: str) -> str:
        """
        Get URL for job status endpoint.
        
        Args:
            job_id (str): Job identifier (UUID)
            
        Returns:
            str: Job status endpoint URL
        """
        return self._url(f"jobs/{job_id}")
    
    def job_cancel(self, job_id: str) -> str:
        """
        Get URL for job cancellation endpoint.
        
        Args:
            job_id (str): Job identifier
            
        Returns:
            str: Job cancellation endpoint URL
        """
        return self._url(f"jobs/{job_id}/cancel")
    
    def job_result(self, job_id: str) -> str:
        """
        Get URL for job result endpoint.
        
        Args:
            job_id (str): Job identifier
            
        Returns:
            str: Job result endpoint URL
        """
        return self._url(f"jobs/{job_id}/result")
    
    def job_logs(self, job_id: str) -> str:
        """
        Get URL for job logs endpoint.
        
        Args:
            job_id (str): Job identifier
            
        Returns:
            str: Job logs endpoint URL
        """
        return self._url(f"jobs/{job_id}/logs")
    
    def user_profile(self, username: str) -> str:
        """
        Get URL for user profile endpoint.
        
        Args:
            username (str): Username
            
        Returns:
            str: User profile endpoint URL
        """
        encoded_username = quote(username, safe='')
        return self._url(f"users/{encoded_username}")
    
    def user_models(self, username: str) -> str:
        """
        Get URL for user's models endpoint.
        
        Args:
            username (str): Username
            
        Returns:
            str: User models endpoint URL
        """
        encoded_username = quote(username, safe='')
        return self._url(f"users/{encoded_username}/models")
    
    def user_stats(self, username: str) -> str:
        """
        Get URL for user statistics endpoint.
        
        Args:
            username (str): Username
            
        Returns:
            str: User statistics endpoint URL
        """
        encoded_username = quote(username, safe='')
        return self._url(f"users/{encoded_username}/stats")
    
    def user_following(self, username: str) -> str:
        """
        Get URL for user's following list endpoint.
        
        Args:
            username (str): Username
            
        Returns:
            str: User following endpoint URL
        """
        encoded_username = quote(username, safe='')
        return self._url(f"users/{encoded_username}/following")
    
    def user_followers(self, username: str) -> str:
        """
        Get URL for user's followers list endpoint.
        
        Args:
            username (str): Username
            
        Returns:
            str: User followers endpoint URL
        """
        encoded_username = quote(username, safe='')
        return self._url(f"users/{encoded_username}/followers")
    
    def webhook_endpoint(self, webhook_id: str) -> str:
        """
        Get URL for specific webhook endpoint.
        
        Args:
            webhook_id (str): Webhook identifier
            
        Returns:
            str: Webhook endpoint URL
        """
        return self._url(f"webhooks/{webhook_id}")
    
    def webhook_test(self, webhook_id: str) -> str:
        """
        Get URL for webhook testing endpoint.
        
        Args:
            webhook_id (str): Webhook identifier
            
        Returns:
            str: Webhook test endpoint URL
        """
        return self._url(f"webhooks/{webhook_id}/test")
    
    def analytics_models(self, timeframe: str = "30d") -> str:
        """
        Get URL for model analytics endpoint.
        
        Args:
            timeframe (str): Analytics timeframe ("1d", "7d", "30d", "90d", "1y")
            
        Returns:
            str: Model analytics endpoint URL
        """
        return self._url(f"analytics/models?timeframe={timeframe}")
    
    def analytics_usage(self, timeframe: str = "30d") -> str:
        """
        Get URL for usage analytics endpoint.
        
        Args:
            timeframe (str): Analytics timeframe
            
        Returns:
            str: Usage analytics endpoint URL
        """
        return self._url(f"analytics/usage?timeframe={timeframe}")
    
    def search_with_params(self, **params) -> str:
        """
        Get search URL with query parameters.
        
        Args:
            **params: Search parameters (q, language, category, etc.)
            
        Returns:
            str: Search URL with encoded parameters
            
        Example:
            >>> endpoints = APIEndpoints("https://api.echogrid.dev/v1")
            >>> url = endpoints.search_with_params(q="female voice", language="en", limit=10)
            >>> print(url)
            https://api.echogrid.dev/v1/models/search?q=female+voice&language=en&limit=10
        """
        if params:
            return f"{self.models_search}?{urlencode(params)}"
        return self.models_search
    
    def generate_with_callback(self, callback_url: str) -> str:
        """
        Get generation URL with callback parameter.
        
        Args:
            callback_url (str): Webhook callback URL
            
        Returns:
            str: Generation endpoint URL with callback
        """
        return f"{self.generate_async}?callback={quote(callback_url)}"
    
    def paginated_url(self, base_endpoint: str, page: int = 1, per_page: int = 20) -> str:
        """
        Add pagination parameters to any endpoint.
        
        Args:
            base_endpoint (str): Base endpoint URL
            page (int): Page number (1-based)
            per_page (int): Items per page
            
        Returns:
            str: Paginated endpoint URL
        """
        params = {"page": page, "per_page": per_page}
        separator = "&" if "?" in base_endpoint else "?"
        return f"{base_endpoint}{separator}{urlencode(params)}"
    
    def _url(self, path: str) -> str:
        """
        Construct full URL from path.
        
        Args:
            path (str): API endpoint path
            
        Returns:
            str: Complete API URL
        """
        return urljoin(self.base_url + '/', path)
    
    def get_endpoint_info(self) -> Dict[str, Any]:
        """
        Get information about configured endpoints.
        
        Returns:
            Dict[str, Any]: Endpoint configuration information
        """
        return {
            "base_url": self.base_url,
            "api_version": self.api_version,
            "total_endpoints": len([attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]),
            "auth_endpoints": [
                self.auth_validate,
                self.auth_token,
                self.auth_refresh,
                self.auth_authorize,
                self.auth_revoke
            ],
            "model_endpoints": [
                self.models_list,
                self.models_search,
                self.models_upload,
                self.models_featured
            ],
            "generation_endpoints": [
                self.generate,
                self.generate_async,
                self.generate_stream,
                self.generate_batch
            ]
        }
    
    def validate_model_name(self, model_name: str) -> bool:
        """
        Validate model name format.
        
        Args:
            model_name (str): Model name to validate
            
        Returns:
            bool: True if model name is valid
        """
        if not model_name or not isinstance(model_name, str):
            return False
        
        # Must contain exactly one slash
        parts = model_name.split('/')
        if len(parts) != 2:
            return False
        
        username, modelname = parts
        
        # Validate username and model name
        if not username or not modelname:
            return False
        
        # Check for invalid characters (basic validation)
        invalid_chars = set('<>:"|?*\\')
        if any(char in invalid_chars for char in model_name):
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation of endpoints configuration."""
        return f"APIEndpoints(base_url='{self.base_url}', version='{self.api_version}')"