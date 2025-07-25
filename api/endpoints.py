# echogrid/api/endpoints.py
"""
API endpoint definitions and URL construction.

This module defines all API endpoints used by the EchoGrid service,
providing a centralized way to manage API URLs and versioning.
"""

from typing import Dict, Any
from urllib.parse import urljoin, quote


class APIEndpoints:
    """
    API endpoint definitions for EchoGrid services.
    
    This class provides methods to construct API URLs for different
    operations, handling URL encoding and parameter substitution.
    
    Args:
        base_url: Base API URL (e.g., "https://api.echogrid.dev/v1")
        
    Example:
        >>> endpoints = APIEndpoints("https://api.echogrid.dev/v1")
        >>> url = endpoints.model_info("alice/voice")
        >>> print(url)
        https://api.echogrid.dev/v1/models/alice%2Fvoice
    """
    
    def __init__(self, base_url: str):
        """Initialize endpoints with base URL."""
        self.base_url = base_url.rstrip('/')
        
        # Core endpoints
        self.health = self._url("health")
        self.info = self._url("info")
        
        # Model endpoints
        self.models_list = self._url("models")
        self.models_search = self._url("models/search")
        self.models_upload = self._url("models/upload")
        
        # Generation endpoints
        self.generate = self._url("generate")
        self.generate_async = self._url("generate/async")
        self.generate_stream = self._url("generate/stream")
        
        # Job management
        self.jobs = self._url("jobs")
        
        # Account and usage
        self.usage_stats = self._url("usage")
        self.account_info = self._url("account")
        
        # Authentication
        self.auth_token = self._url("auth/token")
        self.auth_refresh = self._url("auth/refresh")
    
    def model_info(self, model_name: str) -> str:
        """
        Get URL for model information endpoint.
        
        Args:
            model_name: Model identifier (username/model-name)
            
        Returns:
            Model info endpoint URL
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}")
    
    def model_download(self, model_name: str, version: str = "latest") -> str:
        """
        Get URL for model download endpoint.
        
        Args:
            model_name: Model identifier
            version: Model version
            
        Returns:
            Model download endpoint URL
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}/download/{version}")
    
    def model_versions(self, model_name: str) -> str:
        """
        Get URL for model versions endpoint.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Model versions endpoint URL
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}/versions")
    
    def model_samples(self, model_name: str) -> str:
        """
        Get URL for model audio samples endpoint.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Model samples endpoint URL
        """
        encoded_name = quote(model_name, safe='')
        return self._url(f"models/{encoded_name}/samples")
    
    def job_status(self, job_id: str) -> str:
        """
        Get URL for job status endpoint.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status endpoint URL
        """
        return self._url(f"jobs/{job_id}")
    
    def job_cancel(self, job_id: str) -> str:
        """
        Get URL for job cancellation endpoint.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job cancellation endpoint URL
        """
        return self._url(f"jobs/{job_id}/cancel")
    
    def job_result(self, job_id: str) -> str:
        """
        Get URL for job result endpoint.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job result endpoint URL
        """
        return self._url(f"jobs/{job_id}/result")
    
    def user_models(self, username: str) -> str:
        """
        Get URL for user's models endpoint.
        
        Args:
            username: Username
            
        Returns:
            User models endpoint URL
        """
        return self._url(f"users/{username}/models")
    
    def user_profile(self, username: str) -> str:
        """
        Get URL for user profile endpoint.
        
        Args:
            username: Username
            
        Returns:
            User profile endpoint URL
        """
        return self._url(f"users/{username}")
    
    def _url(self, path: str) -> str:
        """
        Construct full URL from path.
        
        Args:
            path: API endpoint path
            
        Returns:
            Complete API URL
        """
        return urljoin(self.base_url + '/', path)