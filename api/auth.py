# echogrid/api/auth.py
"""
Authentication handling for EchoGrid API.

This module provides authentication mechanisms including API key management,
token refresh, and OAuth integration for EchoGrid services.
"""

import logging
import time
from typing import Dict, Any, Optional
import requests

from ..exceptions import AuthenticationError, NetworkError

logger = logging.getLogger(__name__)


class APIAuth:
    """
    API authentication handler.
    
    This class manages authentication for EchoGrid API requests,
    including API key validation, token refresh, and OAuth flows.
    
    Args:
        api_key: API authentication key
        auth_method: Authentication method ("bearer", "api_key", "oauth")
        
    Example:
        >>> auth = APIAuth("your-api-key")
        >>> headers = auth.get_auth_headers()
        >>> # Use headers in API requests
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        auth_method: str = "bearer",
        oauth_client_id: Optional[str] = None,
        oauth_client_secret: Optional[str] = None
    ):
        """Initialize authentication handler."""
        self.api_key = api_key
        self.auth_method = auth_method
        self.oauth_client_id = oauth_client_id
        self.oauth_client_secret = oauth_client_secret
        
        # Token management
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None
        
        # Authentication state
        self._is_authenticated = bool(api_key)
        self._last_auth_check = 0.0
        
        logger.debug(f"Initialized API auth with method: {auth_method}")
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Returns:
            Dictionary of authentication headers
            
        Raises:
            AuthenticationError: If authentication fails
        """
        if not self.is_authenticated():
            return {}
        
        headers = {}
        
        if self.auth_method == "bearer":
            if self._access_token:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        
        elif self.auth_method == "api_key":
            if self.api_key:
                headers["X-API-Key"] = self.api_key
        
        elif self.auth_method == "oauth":
            if self._access_token:
                headers["Authorization"] = f"Bearer {self._access_token}"
            else:
                raise AuthenticationError("OAuth token not available")
        
        return headers
    
    def is_authenticated(self) -> bool:
        """
        Check if client is authenticated.
        
        Returns:
            True if authenticated and tokens are valid
        """
        if not self._is_authenticated:
            return False
        
        # Check token expiration for OAuth
        if self.auth_method == "oauth" and self._token_expires_at:
            if time.time() >= self._token_expires_at - 60:  # 1 minute buffer
                return self._refresh_oauth_token()
        
        return True
    
    def validate_api_key(self, endpoint: str) -> bool:
        """
        Validate API key with the server.
        
        Args:
            endpoint: API endpoint URL for validation
            
        Returns:
            True if API key is valid
        """
        if not self.api_key:
            return False
        
        try:
            headers = self.get_auth_headers()
            response = requests.get(
                f"{endpoint}/auth/validate",
                headers=headers,
                timeout=10
            )
            
            self._is_authenticated = response.status_code == 200
            self._last_auth_check = time.time()
            
            return self._is_authenticated
            
        except Exception as e:
            logger.warning(f"API key validation failed: {e}")
            return False
    
    def oauth_login(self, endpoint: str, username: str, password: str) -> bool:
        """
        Perform OAuth login flow.
        
        Args:
            endpoint: OAuth endpoint URL
            username: User username or email
            password: User password
            
        Returns:
            True if login successful
            
        Raises:
            AuthenticationError: If OAuth login fails
        """
        if not self.oauth_client_id or not self.oauth_client_secret:
            raise AuthenticationError("OAuth credentials not configured")
        
        try:
            # OAuth token request
            response = requests.post(
                f"{endpoint}/auth/token",
                data={
                    "grant_type": "password",
                    "username": username,
                    "password": password,
                    "client_id": self.oauth_client_id,
                    "client_secret": self.oauth_client_secret
                },
                timeout=30
            )
            
            if response.status_code != 200:
                raise AuthenticationError("OAuth login failed")
            
            token_data = response.json()
            
            self._access_token = token_data.get("access_token")
            self._refresh_token = token_data.get("refresh_token")
            
            # Calculate token expiration
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in
            
            self._is_authenticated = True
            
            logger.info("OAuth login successful")
            return True
            
        except Exception as e:
            logger.error(f"OAuth login failed: {e}")
            raise AuthenticationError(f"OAuth login failed: {e}") from e
    
    def _refresh_oauth_token(self) -> bool:
        """
        Refresh OAuth access token.
        
        Returns:
            True if token refresh successful
        """
        if not self._refresh_token:
            return False
        
        try:
            # Token refresh request
            response = requests.post(
                f"{self._oauth_endpoint}/auth/refresh",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id": self.oauth_client_id,
                    "client_secret": self.oauth_client_secret
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.warning("OAuth token refresh failed")
                self._is_authenticated = False
                return False
            
            token_data = response.json()
            
            self._access_token = token_data.get("access_token")
            
            # Update refresh token if provided
            if "refresh_token" in token_data:
                self._refresh_token = token_data["refresh_token"]
            
            # Update expiration
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in
            
            logger.debug("OAuth token refreshed")
            return True
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            self._is_authenticated = False
            return False
    
    def logout(self):
        """Clear authentication tokens and logout."""
        self._access_token = None
        self._refresh_token = None
        self._token_expires