# echogrid/api/auth.py
"""
Authentication handling for EchoGrid API.

This module provides comprehensive authentication mechanisms for the EchoGrid API,
including API key management, OAuth 2.0 flows, token refresh, and session management.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, Tuple
import requests
import jwt
from urllib.parse import urlencode, parse_qs, urlparse

from ..exceptions import AuthenticationError, NetworkError, ValidationError

logger = logging.getLogger(__name__)


class APIAuth:
    """
    API authentication handler for EchoGrid services.
    
    This class manages all aspects of authentication including API key validation,
    OAuth 2.0 flows, token refresh, and secure credential storage. It supports
    multiple authentication methods and automatically handles token expiration.
    
    Args:
        api_key (str, optional): API authentication key. If not provided,
            will attempt to load from environment variable ECHOGRID_API_KEY.
        auth_method (str): Authentication method. Options:
            - "bearer": Bearer token authentication (default)
            - "api_key": API key in custom header
            - "oauth": OAuth 2.0 flow
        oauth_client_id (str, optional): OAuth client ID for OAuth flows
        oauth_client_secret (str, optional): OAuth client secret for OAuth flows
            
    Attributes:
        api_key (str): Current API key
        auth_method (str): Active authentication method
        is_authenticated (bool): Current authentication status
        
    Example:
        Basic API key authentication::

            auth = APIAuth("your-api-key")
            headers = auth.get_auth_headers()
            
            # Use headers in API requests
            response = requests.get(url, headers=headers)

        OAuth 2.0 authentication::

            auth = APIAuth(
                auth_method="oauth",
                oauth_client_id="your-client-id",
                oauth_client_secret="your-client-secret"
            )
            
            # Perform OAuth login
            success = auth.oauth_login("https://api.echogrid.dev", "username", "password")
            
            if success:
                headers = auth.get_auth_headers()

    Note:
        For production use, store credentials securely and never hardcode
        them in source code. Use environment variables or secure configuration files.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        auth_method: str = "bearer",
        oauth_client_id: Optional[str] = None,
        oauth_client_secret: Optional[str] = None,
        token_storage_path: Optional[str] = None
    ):
        """Initialize authentication handler with configuration."""
        # Load API key from environment if not provided
        self.api_key = api_key or os.getenv("ECHOGRID_API_KEY")
        self.auth_method = auth_method
        self.oauth_client_id = oauth_client_id or os.getenv("ECHOGRID_CLIENT_ID")
        self.oauth_client_secret = oauth_client_secret or os.getenv("ECHOGRID_CLIENT_SECRET")
        
        # Token storage configuration
        self.token_storage_path = token_storage_path or os.path.expanduser("~/.echogrid/tokens")
        
        # OAuth token management
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None
        self._token_type: str = "Bearer"
        self._scope: Optional[str] = None
        
        # Authentication state tracking
        self._is_authenticated = bool(self.api_key)
        self._last_auth_check = 0.0
        self._auth_check_interval = 300  # 5 minutes
        self._oauth_endpoint: Optional[str] = None
        
        # Load stored tokens if available
        if self.auth_method == "oauth":
            self._load_stored_tokens()
        
        logger.debug(f"Initialized API auth with method: {auth_method}")
    
    @property
    def is_authenticated(self) -> bool:
        """
        Check if client is currently authenticated.
        
        This property checks authentication status and automatically
        refreshes tokens if needed for OAuth flows.
        
        Returns:
            bool: True if authenticated and tokens are valid
        """
        if not self._is_authenticated:
            return False
        
        # For API key auth, check periodically
        if self.auth_method in ["bearer", "api_key"]:
            current_time = time.time()
            if current_time - self._last_auth_check > self._auth_check_interval:
                # Re-validate if it's been a while
                self._last_auth_check = current_time
                return self.api_key is not None
        
        # For OAuth, check token expiration
        elif self.auth_method == "oauth":
            if self._token_expires_at:
                # Refresh if token expires within 5 minutes
                if time.time() >= self._token_expires_at - 300:
                    return self._refresh_oauth_token()
            return self._access_token is not None
        
        return self._is_authenticated
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Returns the appropriate authentication headers based on the
        configured authentication method. Automatically handles token
        refresh for OAuth flows.
        
        Returns:
            Dict[str, str]: Dictionary of HTTP headers for authentication.
                Common headers include:
                - Authorization: Bearer token or API key
                - X-API-Key: API key for custom header auth
                - X-Client-ID: Client identification
                
        Raises:
            AuthenticationError: If authentication is not configured or tokens are invalid
            
        Example:
            Get headers and make authenticated request::

                auth = APIAuth("your-api-key")
                headers = auth.get_auth_headers()
                
                response = requests.get(
                    "https://api.echogrid.dev/v1/models",
                    headers=headers
                )
        """
        if not self.is_authenticated:
            logger.warning("Authentication not configured or invalid")
            return {}
        
        headers = {}
        
        if self.auth_method == "bearer":
            # Bearer token authentication
            if self._access_token:
                headers["Authorization"] = f"{self._token_type} {self._access_token}"
            elif self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        
        elif self.auth_method == "api_key":
            # Custom API key header
            if self.api_key:
                headers["X-API-Key"] = self.api_key
                headers["X-API-Version"] = "v1"
        
        elif self.auth_method == "oauth":
            # OAuth Bearer token
            if self._access_token:
                headers["Authorization"] = f"{self._token_type} {self._access_token}"
                if self._scope:
                    headers["X-OAuth-Scope"] = self._scope
            else:
                raise AuthenticationError(
                    "OAuth token not available. Please authenticate first."
                )
        
        # Add client identification
        if self.oauth_client_id:
            headers["X-Client-ID"] = self.oauth_client_id
        
        return headers
    
    def validate_api_key(self, endpoint: str, timeout: int = 10) -> bool:
        """
        Validate API key with the authentication server.
        
        Performs a validation request to verify that the API key
        is valid and has the necessary permissions.
        
        Args:
            endpoint (str): API base endpoint URL
            timeout (int): Request timeout in seconds
            
        Returns:
            bool: True if API key is valid and authorized
            
        Raises:
            NetworkError: If validation request fails
            
        Example:
            Validate API key before making requests::

                auth = APIAuth("your-api-key")
                
                if auth.validate_api_key("https://api.echogrid.dev/v1"):
                    print("API key is valid")
                    headers = auth.get_auth_headers()
                else:
                    print("Invalid API key")
        """
        if not self.api_key:
            logger.warning("No API key provided for validation")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{endpoint.rstrip('/')}/auth/validate",
                headers=headers,
                timeout=timeout
            )
            
            is_valid = response.status_code == 200
            self._is_authenticated = is_valid
            self._last_auth_check = time.time()
            
            if is_valid:
                # Extract additional info from response
                try:
                    data = response.json()
                    self._scope = data.get("scope")
                    logger.info(f"API key validated successfully. Scope: {self._scope}")
                except (ValueError, KeyError):
                    logger.info("API key validated successfully")
            else:
                logger.warning(f"API key validation failed: {response.status_code}")
            
            return is_valid
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API key validation request failed: {e}")
            raise NetworkError(f"Failed to validate API key: {e}") from e
    
    def oauth_login(
        self, 
        endpoint: str, 
        username: str, 
        password: str,
        scope: Optional[str] = None
    ) -> bool:
        """
        Perform OAuth 2.0 Resource Owner Password Credentials flow.
        
        Authenticates using username/password and obtains access and refresh tokens.
        This method should only be used for trusted applications.
        
        Args:
            endpoint (str): OAuth authorization server endpoint
            username (str): User username or email address
            password (str): User password
            scope (str, optional): Requested OAuth scope
            
        Returns:
            bool: True if authentication successful
            
        Raises:
            AuthenticationError: If OAuth credentials are not configured or login fails
            NetworkError: If the authentication request fails
            ValidationError: If required parameters are missing
            
        Example:
            Authenticate with OAuth::

                auth = APIAuth(
                    auth_method="oauth",
                    oauth_client_id="your-client-id",
                    oauth_client_secret="your-client-secret"
                )
                
                success = auth.oauth_login(
                    "https://api.echogrid.dev/v1",
                    "your-username",
                    "your-password",
                    scope="models:read models:write"
                )
                
                if success:
                    print("Authentication successful")
                    headers = auth.get_auth_headers()

        Warning:
            The Resource Owner Password Credentials flow should only be used
            when other OAuth flows are not feasible. It requires transmitting
            user credentials to the client application.
        """
        if not self.oauth_client_id or not self.oauth_client_secret:
            raise AuthenticationError(
                "OAuth client credentials not configured. "
                "Please provide oauth_client_id and oauth_client_secret."
            )
        
        if not username or not password:
            raise ValidationError("Username and password are required for OAuth login")
        
        self._oauth_endpoint = endpoint.rstrip('/')
        
        # Prepare OAuth token request
        token_data = {
            "grant_type": "password",
            "username": username,
            "password": password,
            "client_id": self.oauth_client_id,
            "client_secret": self.oauth_client_secret
        }
        
        if scope:
            token_data["scope"] = scope
        
        try:
            response = requests.post(
                f"{self._oauth_endpoint}/auth/token",
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            
            if response.status_code != 200:
                error_data = {}
                try:
                    error_data = response.json()
                except ValueError:
                    pass
                
                error_msg = error_data.get("error_description", "OAuth login failed")
                raise AuthenticationError(f"OAuth login failed: {error_msg}")
            
            token_data = response.json()
            
            # Store tokens
            self._access_token = token_data.get("access_token")
            self._refresh_token = token_data.get("refresh_token")
            self._token_type = token_data.get("token_type", "Bearer")
            self._scope = token_data.get("scope")
            
            # Calculate token expiration
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in
            
            self._is_authenticated = True
            
            # Save tokens to disk
            self._save_tokens()
            
            logger.info(f"OAuth login successful. Scope: {self._scope}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OAuth login request failed: {e}")
            raise NetworkError(f"OAuth login failed: {e}") from e
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during OAuth login: {e}")
            raise AuthenticationError(f"OAuth login failed: {e}") from e
    
    def oauth_authorization_url(
        self, 
        endpoint: str, 
        redirect_uri: str, 
        state: Optional[str] = None,
        scope: Optional[str] = None
    ) -> str:
        """
        Generate OAuth 2.0 authorization URL for authorization code flow.
        
        Creates the authorization URL that users should visit to grant
        permission to the application.
        
        Args:
            endpoint (str): OAuth authorization server endpoint
            redirect_uri (str): Callback URL for authorization response
            state (str, optional): CSRF protection state parameter
            scope (str, optional): Requested OAuth scope
            
        Returns:
            str: Authorization URL for user redirection
            
        Raises:
            AuthenticationError: If OAuth client ID is not configured
            ValidationError: If required parameters are missing
            
        Example:
            Generate authorization URL::

                auth = APIAuth(
                    auth_method="oauth",
                    oauth_client_id="your-client-id"
                )
                
                auth_url = auth.oauth_authorization_url(
                    "https://api.echogrid.dev/v1",
                    "https://yourapp.com/callback",
                    state="random-csrf-token",
                    scope="models:read models:write"
                )
                
                print(f"Visit: {auth_url}")
        """
        if not self.oauth_client_id:
            raise AuthenticationError("OAuth client ID not configured")
        
        if not redirect_uri:
            raise ValidationError("Redirect URI is required")
        
        params = {
            "response_type": "code",
            "client_id": self.oauth_client_id,
            "redirect_uri": redirect_uri
        }
        
        if state:
            params["state"] = state
        if scope:
            params["scope"] = scope
        
        return f"{endpoint.rstrip('/')}/auth/authorize?{urlencode(params)}"
    
    def oauth_token_exchange(
        self, 
        endpoint: str, 
        authorization_code: str, 
        redirect_uri: str
    ) -> bool:
        """
        Exchange authorization code for access token.
        
        Completes the OAuth 2.0 authorization code flow by exchanging
        the authorization code for access and refresh tokens.
        
        Args:
            endpoint (str): OAuth token endpoint
            authorization_code (str): Authorization code from callback
            redirect_uri (str): Original redirect URI used in authorization
            
        Returns:
            bool: True if token exchange successful
            
        Raises:
            AuthenticationError: If token exchange fails
            NetworkError: If the token request fails
        """
        if not self.oauth_client_id or not self.oauth_client_secret:
            raise AuthenticationError("OAuth client credentials not configured")
        
        token_data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": redirect_uri,
            "client_id": self.oauth_client_id,
            "client_secret": self.oauth_client_secret
        }
        
        try:
            response = requests.post(
                f"{endpoint.rstrip('/')}/auth/token",
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            
            if response.status_code != 200:
                raise AuthenticationError("Token exchange failed")
            
            token_response = response.json()
            
            # Store tokens
            self._access_token = token_response.get("access_token")
            self._refresh_token = token_response.get("refresh_token")
            self._token_type = token_response.get("token_type", "Bearer")
            self._scope = token_response.get("scope")
            
            # Calculate expiration
            expires_in = token_response.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in
            
            self._is_authenticated = True
            self._oauth_endpoint = endpoint.rstrip('/')
            
            # Save tokens
            self._save_tokens()
            
            logger.info("OAuth token exchange successful")
            return True
            
        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            raise AuthenticationError(f"Token exchange failed: {e}") from e
    
    def _refresh_oauth_token(self) -> bool:
        """
        Refresh OAuth access token using refresh token.
        
        Returns:
            bool: True if token refresh successful
        """
        if not self._refresh_token or not self._oauth_endpoint:
            logger.warning("Cannot refresh token: missing refresh token or endpoint")
            return False
        
        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self.oauth_client_id,
            "client_secret": self.oauth_client_secret
        }
        
        try:
            response = requests.post(
                f"{self._oauth_endpoint}/auth/token",
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.warning("OAuth token refresh failed")
                self._is_authenticated = False
                return False
            
            token_response = response.json()
            
            # Update tokens
            self._access_token = token_response.get("access_token")
            
            # Update refresh token if provided
            if "refresh_token" in token_response:
                self._refresh_token = token_response["refresh_token"]
            
            # Update expiration
            expires_in = token_response.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in
            
            # Save updated tokens
            self._save_tokens()
            
            logger.debug("OAuth token refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            self._is_authenticated = False
            return False
    
    def _save_tokens(self):
        """Save OAuth tokens to secure storage."""
        if not self._access_token:
            return
        
        try:
            os.makedirs(os.path.dirname(self.token_storage_path), exist_ok=True)
            
            token_data = {
                "access_token": self._access_token,
                "refresh_token": self._refresh_token,
                "token_type": self._token_type,
                "expires_at": self._token_expires_at,
                "scope": self._scope,
                "endpoint": self._oauth_endpoint
            }
            
            import json
            with open(self.token_storage_path, 'w') as f:
                json.dump(token_data, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(self.token_storage_path, 0o600)
            
        except Exception as e:
            logger.warning(f"Failed to save tokens: {e}")
    
    def _load_stored_tokens(self):
        """Load OAuth tokens from secure storage."""
        try:
            if not os.path.exists(self.token_storage_path):
                return
            
            import json
            with open(self.token_storage_path, 'r') as f:
                token_data = json.load(f)
            
            self._access_token = token_data.get("access_token")
            self._refresh_token = token_data.get("refresh_token")
            self._token_type = token_data.get("token_type", "Bearer")
            self._token_expires_at = token_data.get("expires_at")
            self._scope = token_data.get("scope")
            self._oauth_endpoint = token_data.get("endpoint")
            
            # Check if tokens are still valid
            if self._token_expires_at and time.time() < self._token_expires_at:
                self._is_authenticated = True
                logger.debug("Loaded valid tokens from storage")
            elif self._refresh_token:
                # Try to refresh
                self._is_authenticated = self._refresh_oauth_token()
            
        except Exception as e:
            logger.warning(f"Failed to load stored tokens: {e}")
    
    def get_token_info(self) -> Dict[str, Any]:
        """
        Get information about current authentication tokens.
        
        Returns:
            Dict[str, Any]: Token information including expiration and scope
        """
        if not self.is_authenticated:
            return {"authenticated": False}
        
        info = {
            "authenticated": True,
            "auth_method": self.auth_method,
            "has_api_key": bool(self.api_key),
            "has_access_token": bool(self._access_token),
            "has_refresh_token": bool(self._refresh_token)
        }
        
        if self._token_expires_at:
            info["expires_at"] = self._token_expires_at
            info["expires_in"] = max(0, self._token_expires_at - time.time())
        
        if self._scope:
            info["scope"] = self._scope
        
        return info
    
    def logout(self):
        """
        Clear authentication tokens and logout.
        
        Clears all stored authentication data including tokens,
        API keys, and cached authentication state.
        """
        self._access_token = None
        self._refresh_token = None
        self._token_expires_at = None
        self._token_type = "Bearer"
        self._scope = None
        self._is_authenticated = False
        self._last_auth_check = 0.0
        
        # Remove stored tokens
        try:
            if os.path.exists(self.token_storage_path):
                os.remove(self.token_storage_path)
        except Exception as e:
            logger.warning(f"Failed to remove stored tokens: {e}")
        
        logger.info("Logged out successfully")
    
    def __repr__(self) -> str:
        """String representation of auth handler."""
        return (
            f"APIAuth(method={self.auth_method}, "
            f"authenticated={self.is_authenticated}, "
            f"has_api_key={bool(self.api_key)})"
        )