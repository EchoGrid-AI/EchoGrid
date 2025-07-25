# echogrid/api/rate_limiting.py
"""
Rate limiting and quota management for EchoGrid API requests.

This module provides client-side rate limiting to respect API quotas and
prevent hitting server-side rate limits. It implements various rate limiting
algorithms including token bucket and sliding window.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from collections import deque
from threading import Lock
import math

from ..exceptions import APIRateLimitError

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Client-side rate limiter for API requests.
    
    This class implements multiple rate limiting algorithms to ensure
    API requests stay within configured limits and handle server-side
    rate limiting gracefully.
    
    Args:
        requests_per_second (float): Maximum requests per second. Defaults to 10.0.
        requests_per_minute (int): Maximum requests per minute. Defaults to 600.
        requests_per_hour (int): Maximum requests per hour. Defaults to 10000.
        burst_size (int): Maximum burst size for token bucket. Defaults to 20.
        algorithm (str): Rate limiting algorithm ("token_bucket", "sliding_window").
            Defaults to "token_bucket".
        
    Attributes:
        requests_per_second (float): Current RPS limit
        total_requests (int): Total requests made
        rejected_requests (int): Requests rejected due to rate limiting
        
    Example:
        Basic rate limiting::

            rate_limiter = RateLimiter(
                requests_per_second=5.0,
                requests_per_minute=300,
                burst_size=10
            )
            
            # Check if request is allowed
            if rate_limiter.is_allowed():
                # Make API request
                response = make_api_call()
                rate_limiter.record_request()
            else:
                # Wait or handle rate limit
                wait_time = rate_limiter.time_until_allowed()
                time.sleep(wait_time)

        Automatic rate limiting::

            @rate_limiter.limit_requests
            def api_function():
                return requests.get("https://api.echogrid.dev/v1/models")

    Note:
        The rate limiter uses thread-safe implementations and can be
        shared across multiple threads or async tasks.
    """
    
    def __init__(
        self,
        requests_per_second: float = 10.0,
        requests_per_minute: int = 600,
        requests_per_hour: int = 10000,
        burst_size: int = 20,
        algorithm: str = "token_bucket"
    ):
        """Initialize rate limiter with configuration."""
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.algorithm = algorithm
        
        # Thread safety
        self._lock = Lock()
        
        # Token bucket algorithm state
        self._tokens = float(burst_size)
        self._last_refill = time.time()
        
        # Sliding window algorithm state
        self._request_times: deque = deque()
        self._minute_requests: deque = deque()
        self._hour_requests: deque = deque()
        
        # Statistics
        self.total_requests = 0
        self.rejected_requests = 0
        self.last_request_time = 0.0
        
        # Server rate limit tracking
        self._server_rate_limit_reset = 0.0
        self._server_remaining_requests = None
        self._server_rate_limit_window = 0.0
        
        logger.debug(f"Initialized rate limiter: {requests_per_second} RPS, algorithm={algorithm}")
    
    def is_allowed(self, cost: int = 1) -> bool:
        """
        Check if a request is allowed under current rate limits.
        
        Args:
            cost (int): Request cost in tokens/credits. Defaults to 1.
            
        Returns:
            bool: True if request is allowed
            
        Example:
            Check before making expensive request::

                # Heavy request costs more tokens
                if rate_limiter.is_allowed(cost=5):
                    result = expensive_api_call()
                else:
                    print("Rate limited, waiting...")
                    time.sleep(rate_limiter.time_until_allowed())
        """
        with self._lock:
            current_time = time.time()
            
            # Check server-side rate limit
            if self._is_server_rate_limited(current_time):
                return False
            
            if self.algorithm == "token_bucket":
                return self._token_bucket_allowed(current_time, cost)
            elif self.algorithm == "sliding_window":
                return self._sliding_window_allowed(current_time, cost)
            else:
                logger.warning(f"Unknown algorithm {self.algorithm}, defaulting to token bucket")
                return self._token_bucket_allowed(current_time, cost)
    
    def record_request(self, cost: int = 1, server_headers: Optional[Dict[str, str]] = None):
        """
        Record a completed request and update rate limiting state.
        
        Args:
            cost (int): Request cost in tokens. Defaults to 1.
            server_headers (Dict[str, str], optional): HTTP response headers
                containing server rate limit information.
                
        Example:
            Record request with server headers::

                response = requests.get(url)
                rate_limiter.record_request(
                    cost=1,
                    server_headers=dict(response.headers)
                )
        """
        with self._lock:
            current_time = time.time()
            
            # Update statistics
            self.total_requests += 1
            self.last_request_time = current_time
            
            # Update algorithm state
            if self.algorithm == "token_bucket":
                self._tokens = max(0, self._tokens - cost)
            elif self.algorithm == "sliding_window":
                self._request_times.append(current_time)
                self._minute_requests.append(current_time)
                self._hour_requests.append(current_time)
                self._cleanup_old_requests(current_time)
            
            # Update server rate limit info
            if server_headers:
                self._update_server_rate_limit_info(server_headers)
    
    def time_until_allowed(self, cost: int = 1) -> float:
        """
        Calculate time until next request is allowed.
        
        Args:
            cost (int): Request cost in tokens
            
        Returns:
            float: Time in seconds until request is allowed
            
        Example:
            Wait for rate limit to reset::

                wait_time = rate_limiter.time_until_allowed()
                if wait_time > 0:
                    print(f"Waiting {wait_time:.2f}s for rate limit reset")
                    time.sleep(wait_time)
        """
        with self._lock:
            current_time = time.time()
            
            # Check server rate limit first
            server_wait = self._server_rate_limit_wait_time(current_time)
            if server_wait > 0:
                return server_wait
            
            if self.algorithm == "token_bucket":
                return self._token_bucket_wait_time(current_time, cost)
            elif self.algorithm == "sliding_window":
                return self._sliding_window_wait_time(current_time, cost)
            else:
                return 0.0
    
    def check_limit(self, cost: int = 1) -> bool:
        """
        Check rate limit and raise exception if exceeded.
        
        Args:
            cost (int): Request cost
            
        Returns:
            bool: True if allowed
            
        Raises:
            APIRateLimitError: If rate limit is exceeded
            
        Example:
            Use in API client::

                try:
                    rate_limiter.check_limit()
                    response = make_request()
                except APIRateLimitError as e:
                    print(f"Rate limited, retry after {e.retry_after}s")
        """
        if not self.is_allowed(cost):
            wait_time = self.time_until_allowed(cost)
            self.rejected_requests += 1
            
            raise APIRateLimitError(
                f"Rate limit exceeded. Retry after {wait_time:.2f} seconds.",
                retry_after=wait_time
            )
        
        return True
    
    def handle_rate_limit(self, retry_after: Optional[float] = None):
        """
        Handle server-side rate limit response.
        
        Args:
            retry_after (float, optional): Server-specified retry delay
        """
        with self._lock:
            current_time = time.time()
            
            if retry_after:
                self._server_rate_limit_reset = current_time + retry_after
                logger.info(f"Server rate limit hit, reset in {retry_after}s")
            else:
                # Default backoff
                self._server_rate_limit_reset = current_time + 60
                logger.info("Server rate limit hit, backing off for 60s")
    
    def limit_requests(self, func):
        """
        Decorator to automatically apply rate limiting to a function.
        
        Args:
            func: Function to wrap with rate limiting
            
        Returns:
            Wrapped function that respects rate limits
            
        Example:
            Apply rate limiting decorator::

                @rate_limiter.limit_requests
                def get_models():
                    return requests.get("https://api.echogrid.dev/v1/models")
                
                # Function will automatically wait for rate limits
                models = get_models()
        """
        def wrapper(*args, **kwargs):
            # Wait if necessary
            wait_time = self.time_until_allowed()
            if wait_time > 0:
                logger.debug(f"Rate limit wait: {wait_time:.2f}s")
                time.sleep(wait_time)
            
            # Check limit before proceeding
            self.check_limit()
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful request
                headers = {}
                if hasattr(result, 'headers'):
                    headers = dict(result.headers)
                
                self.record_request(server_headers=headers)
                return result
                
            except Exception as e:
                # Handle rate limit responses
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code == 429:
                        retry_after = e.response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                self.handle_rate_limit(float(retry_after))
                            except ValueError:
                                self.handle_rate_limit()
                
                raise
        
        return wrapper
    
    def _token_bucket_allowed(self, current_time: float, cost: int) -> bool:
        """Check if request is allowed using token bucket algorithm."""
        # Refill tokens
        time_passed = current_time - self._last_refill
        tokens_to_add = time_passed * self.requests_per_second
        self._tokens = min(self.burst_size, self._tokens + tokens_to_add)
        self._last_refill = current_time
        
        # Check if we have enough tokens
        return self._tokens >= cost
    
    def _token_bucket_wait_time(self, current_time: float, cost: int) -> float:
        """Calculate wait time for token bucket algorithm."""
        # Update tokens first
        self._token_bucket_allowed(current_time, 0)
        
        if self._tokens >= cost:
            return 0.0
        
        tokens_needed = cost - self._tokens
        return tokens_needed / self.requests_per_second
    
    def _sliding_window_allowed(self, current_time: float, cost: int) -> bool:
        """Check if request is allowed using sliding window algorithm."""
        self._cleanup_old_requests(current_time)
        
        # Check per-second limit
        recent_requests = sum(1 for t in self._request_times if current_time - t <= 1.0)
        if recent_requests >= self.requests_per_second:
            return False
        
        # Check per-minute limit
        if len(self._minute_requests) >= self.requests_per_minute:
            return False
        
        # Check per-hour limit
        if len(self._hour_requests) >= self.requests_per_hour:
            return False
        
        return True
    
    def _sliding_window_wait_time(self, current_time: float, cost: int) -> float:
        """Calculate wait time for sliding window algorithm."""
        self._cleanup_old_requests(current_time)
        
        wait_times = []
        
        # Check per-second limit
        recent_requests = [t for t in self._request_times if current_time - t <= 1.0]
        if len(recent_requests) >= self.requests_per_second:
            oldest_in_window = min(recent_requests)
            wait_times.append(1.0 - (current_time - oldest_in_window))
        
        # Check per-minute limit
        if len(self._minute_requests) >= self.requests_per_minute:
            oldest_in_minute = self._minute_requests[0]
            wait_times.append(60.0 - (current_time - oldest_in_minute))
        
        # Check per-hour limit
        if len(self._hour_requests) >= self.requests_per_hour:
            oldest_in_hour = self._hour_requests[0]
            wait_times.append(3600.0 - (current_time - oldest_in_hour))
        
        return max(wait_times) if wait_times else 0.0
    
    def _cleanup_old_requests(self, current_time: float):
        """Remove old requests from sliding windows."""
        # Clean per-second window
        while self._request_times and current_time - self._request_times[0] > 1.0:
            self._request_times.popleft()
        
        # Clean per-minute window
        while self._minute_requests and current_time - self._minute_requests[0] > 60.0:
            self._minute_requests.popleft()
        
        # Clean per-hour window
        while self._hour_requests and current_time - self._hour_requests[0] > 3600.0:
            self._hour_requests.popleft()
    
    def _is_server_rate_limited(self, current_time: float) -> bool:
        """Check if we're currently server rate limited."""
        return current_time < self._server_rate_limit_reset
    
    def _server_rate_limit_wait_time(self, current_time: float) -> float:
        """Get wait time for server rate limit."""
        if current_time < self._server_rate_limit_reset:
            return self._server_rate_limit_reset - current_time
        return 0.0
    
    def _update_server_rate_limit_info(self, headers: Dict[str, str]):
        """Update server rate limit info from response headers."""
        # Common rate limit headers
        remaining = headers.get('X-RateLimit-Remaining') or headers.get('X-Rate-Limit-Remaining')
        reset = headers.get('X-RateLimit-Reset') or headers.get('X-Rate-Limit-Reset')
        window = headers.get('X-RateLimit-Window') or headers.get('X-Rate-Limit-Window')
        
        try:
            if remaining is not None:
                self._server_remaining_requests = int(remaining)
            
            if reset is not None:
                # Reset can be timestamp or seconds
                reset_value = int(reset)
                if reset_value > 1000000000:  # Looks like timestamp
                    self._server_rate_limit_reset = float(reset_value)
                else:  # Seconds from now
                    self._server_rate_limit_reset = time.time() + reset_value
            
            if window is not None:
                self._server_rate_limit_window = float(window)
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse rate limit headers: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Returns:
            Dict[str, Any]: Statistics including request counts and limits
        """
        with self._lock:
            current_time = time.time()
            
            stats = {
                "total_requests": self.total_requests,
                "rejected_requests": self.rejected_requests,
                "requests_per_second": self.requests_per_second,
                "requests_per_minute": self.requests_per_minute,
                "requests_per_hour": self.requests_per_hour,
                "burst_size": self.burst_size,
                "algorithm": self.algorithm,
                "current_tokens": self._tokens if self.algorithm == "token_bucket" else None,
                "server_rate_limited": self._is_server_rate_limited(current_time),
                "server_remaining_requests": self._server_remaining_requests,
                "time_until_server_reset": max(0, self._server_rate_limit_reset - current_time),
            }
            
            if self.algorithm == "sliding_window":
                self._cleanup_old_requests(current_time)
                stats.update({
                    "requests_last_second": len([t for t in self._request_times if current_time - t <= 1.0]),
                    "requests_last_minute": len(self._minute_requests),
                    "requests_last_hour": len(self._hour_requests),
                })
            
            return stats
    
    def reset_statistics(self):
        """Reset rate limiter statistics."""
        with self._lock:
            self.total_requests = 0
            self.rejected_requests = 0
            self.last_request_time = 0.0
            
            if self.algorithm == "sliding_window":
                self._request_times.clear()
                self._minute_requests.clear()
                self._hour_requests.clear()
    
    def update_limits(
        self,
        requests_per_second: Optional[float] = None,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
        burst_size: Optional[int] = None
    ):
        """
        Update rate limiting configuration.
        
        Args:
            requests_per_second (float, optional): New RPS limit
            requests_per_minute (int, optional): New per-minute limit
            requests_per_hour (int, optional): New per-hour limit
            burst_size (int, optional): New burst size
        """
        with self._lock:
            if requests_per_second is not None:
                self.requests_per_second = requests_per_second
            if requests_per_minute is not None:
                self.requests_per_minute = requests_per_minute
            if requests_per_hour is not None:
                self.requests_per_hour = requests_per_hour
            if burst_size is not None:
                self.burst_size = burst_size
                # Adjust current tokens if needed
                if self.algorithm == "token_bucket":
                    self._tokens = min(self._tokens, burst_size)
        
        logger.debug("Rate limiter configuration updated")
    
    def __repr__(self) -> str:
        """String representation of rate limiter."""
        return (
            f"RateLimiter(rps={self.requests_per_second}, "
            f"rpm={self.requests_per_minute}, "
            f"rph={self.requests_per_hour}, "
            f"algorithm={self.algorithm})"
        )