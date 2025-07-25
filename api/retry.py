# echogrid/api/retry.py
"""
Retry logic and error handling for EchoGrid API requests.

This module provides sophisticated retry mechanisms with exponential backoff,
jitter, and intelligent error handling for robust API communication.
"""

import logging
import random
import time
from typing import Callable, Any, Optional, List, Set, Type
from functools import wraps
import requests

from ..exceptions import (
    NetworkError, APIRateLimitError, AuthenticationError,
    ValidationError, EchoGridError
)

logger = logging.getLogger(__name__)


class RetryHandler:
    """
    Advanced retry handler for API requests with exponential backoff.
    
    This class implements intelligent retry logic that handles different types
    of failures appropriately, with configurable backoff strategies and
    failure conditions.
    
    Args:
        max_retries (int): Maximum number of retry attempts. Defaults to 3.
        base_delay (float): Base delay between retries in seconds. Defaults to 1.0.
        max_delay (float): Maximum delay between retries. Defaults to 60.0.
        backoff_multiplier (float): Exponential backoff multiplier. Defaults to 2.0.
        jitter (bool): Whether to add random jitter to delays. Defaults to True.
        retryable_status_codes (Set[int]): HTTP status codes that should trigger retries.
        retryable_exceptions (Set[Type[Exception]]): Exception types that should trigger retries.
        
    Attributes:
        max_retries (int): Maximum retry attempts
        total_retries (int): Total retries attempted across all requests
        successful_retries (int): Number of successful retries
        
    Example:
        Basic retry configuration::

            retry_handler = RetryHandler(
                max_retries=5,
                base_delay=2.0,
                max_delay=120.0
            )
            
            @retry_handler.retry_on_failure
            def make_api_request():
                response = requests.get("https://api.echogrid.dev/v1/models")
                response.raise_for_status()
                return response.json()

        Custom retry conditions::

            # Only retry on specific errors
            retry_handler = RetryHandler(
                retryable_status_codes={500, 502, 503, 504, 429},
                retryable_exceptions={requests.ConnectionError, requests.Timeout}
            )

    Note:
        The retry handler uses exponential backoff with jitter by default
        to prevent thundering herd problems when multiple clients retry
        simultaneously.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
        retryable_status_codes: Optional[Set[int]] = None,
        retryable_exceptions: Optional[Set[Type[Exception]]] = None
    ):
        """Initialize retry handler with configuration."""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        
        # Default retryable conditions
        self.retryable_status_codes = retryable_status_codes or {
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
            507,  # Insufficient Storage
            508,  # Loop Detected
            511,  # Network Authentication Required
        }
        
        self.retryable_exceptions = retryable_exceptions or {
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
            NetworkError,
        }
        
        # Statistics tracking
        self.total_retries = 0
        self.successful_retries = 0
        self.failed_retries = 0
        self._request_count = 0
        
        logger.debug(f"Initialized retry handler: max_retries={max_retries}, base_delay={base_delay}s")
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if a request should be retried based on the exception.
        
        Args:
            exception (Exception): The exception that occurred
            attempt (int): Current attempt number (1-based)
            
        Returns:
            bool: True if the request should be retried
            
        Example:
            Check if an exception should trigger a retry::

                try:
                    response = requests.get(url)
                    response.raise_for_status()
                except Exception as e:
                    if retry_handler.should_retry(e, attempt_num):
                        # Retry the request
                        continue
                    else:
                        # Give up and re-raise
                        raise
        """
        if attempt > self.max_retries:
            return False
        
        # Check for non-retryable exceptions first
        if isinstance(exception, (AuthenticationError, ValidationError)):
            logger.debug(f"Not retrying {type(exception).__name__}: not retryable")
            return False
        
        # Check for rate limiting with retry-after header
        if isinstance(exception, APIRateLimitError):
            retry_after = getattr(exception, 'retry_after', None)
            if retry_after and retry_after > self.max_delay:
                logger.debug(f"Rate limit retry-after ({retry_after}s) exceeds max delay")
                return False
            return True
        
        # Check HTTP status codes
        if hasattr(exception, 'response') and exception.response is not None:
            status_code = exception.response.status_code
            should_retry = status_code in self.retryable_status_codes
            logger.debug(f"HTTP {status_code}: {'retrying' if should_retry else 'not retrying'}")
            return should_retry
        
        # Check exception types
        exception_type = type(exception)
        should_retry = any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)
        
        logger.debug(f"{exception_type.__name__}: {'retrying' if should_retry else 'not retrying'}")
        return should_retry
    
    def calculate_delay(self, attempt: int, exception: Optional[Exception] = None) -> float:
        """
        Calculate delay before next retry attempt.
        
        Uses exponential backoff with optional jitter and respects
        rate limit retry-after headers when available.
        
        Args:
            attempt (int): Current attempt number (1-based)
            exception (Exception, optional): The exception that triggered the retry
            
        Returns:
            float: Delay in seconds before next attempt
            
        Example:
            Calculate delays for multiple attempts::

                for attempt in range(1, 4):
                    delay = retry_handler.calculate_delay(attempt)
                    print(f"Attempt {attempt}: wait {delay:.2f}s")
                
                # Output might be:
                # Attempt 1: wait 1.23s
                # Attempt 2: wait 2.87s  
                # Attempt 3: wait 5.41s
        """
        # Check for rate limit retry-after header
        if isinstance(exception, APIRateLimitError):
            retry_after = getattr(exception, 'retry_after', None)
            if retry_after:
                logger.debug(f"Using rate limit retry-after: {retry_after}s")
                return min(retry_after, self.max_delay)
        
        # Handle HTTP responses with Retry-After header
        if (hasattr(exception, 'response') and 
            exception.response is not None and
            'Retry-After' in exception.response.headers):
            try:
                retry_after = float(exception.response.headers['Retry-After'])
                logger.debug(f"Using HTTP Retry-After header: {retry_after}s")
                return min(retry_after, self.max_delay)
            except (ValueError, TypeError):
                pass
        
        # Calculate exponential backoff delay
        delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            jitter_offset = random.uniform(-jitter_range, jitter_range)
            delay += jitter_offset
        
        # Cap at maximum delay
        delay = min(delay, self.max_delay)
        
        logger.debug(f"Calculated delay for attempt {attempt}: {delay:.2f}s")
        return delay
    
    def retry_on_failure(self, func: Callable) -> Callable:
        """
        Decorator to add retry logic to a function.
        
        Args:
            func (Callable): Function to wrap with retry logic
            
        Returns:
            Callable: Wrapped function with retry behavior
            
        Example:
            Add retry logic to API function::

                @retry_handler.retry_on_failure
                def get_models():
                    response = requests.get("https://api.echogrid.dev/v1/models")
                    response.raise_for_status()
                    return response.json()
                
                # Function will automatically retry on transient failures
                models = get_models()
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func (Callable): Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Any: Result of the function call
            
        Raises:
            Exception: The last exception if all retries are exhausted
            
        Example:
            Execute function with retries::

                def risky_operation():
                    # Potentially failing operation
                    response = requests.get("https://api.echogrid.dev/v1/models")
                    response.raise_for_status()
                    return response.json()
                
                result = retry_handler.execute_with_retry(risky_operation)
        """
        self._request_count += 1
        last_exception = None
        
        for attempt in range(1, self.max_retries + 2):  # +1 for initial attempt, +1 for range
            try:
                logger.debug(f"Executing function attempt {attempt}/{self.max_retries + 1}")
                result = func(*args, **kwargs)
                
                # Success - update statistics
                if attempt > 1:
                    self.successful_retries += 1
                    logger.info(f"Function succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if attempt <= self.max_retries and self.should_retry(e, attempt):
                    self.total_retries += 1
                    
                    # Calculate and apply delay
                    delay = self.calculate_delay(attempt, e)
                    
                    logger.warning(
                        f"Attempt {attempt} failed ({type(e).__name__}: {e}), "
                        f"retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
                    continue
                else:
                    # Don't retry - either max attempts reached or non-retryable error
                    if attempt > self.max_retries:
                        self.failed_retries += 1
                        logger.error(f"All {self.max_retries} retry attempts exhausted")
                    else:
                        logger.debug(f"Not retrying {type(e).__name__}")
                    
                    break
        
        # All retries exhausted, raise the last exception
        logger.error(f"Function failed after {self.max_retries} retries")
        raise last_exception
    
    def retry_request(
        self,
        method: str,
        url: str,
        session: requests.Session,
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method (str): HTTP method (GET, POST, etc.)
            url (str): Request URL
            session (requests.Session): HTTP session to use
            **kwargs: Additional request parameters
            
        Returns:
            requests.Response: HTTP response object
            
        Raises:
            Exception: Request exception if all retries fail
            
        Example:
            Make a retried HTTP request::

                session = requests.Session()
                response = retry_handler.retry_request(
                    "GET",
                    "https://api.echogrid.dev/v1/models",
                    session,
                    timeout=30
                )
        """
        def make_request():
            response = session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        
        return self.execute_with_retry(make_request)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retry handler statistics.
        
        Returns:
            Dict[str, Any]: Statistics including success rates and retry counts
            
        Example:
            Monitor retry performance::

                stats = retry_handler.get_statistics()
                print(f"Success rate: {stats['success_rate']:.1%}")
                print(f"Average retries per request: {stats['avg_retries_per_request']:.2f}")
        """
        total_requests = self._request_count
        success_rate = 1.0 - (self.failed_retries / total_requests) if total_requests > 0 else 0.0
        avg_retries = self.total_retries / total_requests if total_requests > 0 else 0.0
        
        return {
            "total_requests": total_requests,
            "total_retries": self.total_retries,
            "successful_retries": self.successful_retries,
            "failed_retries": self.failed_retries,
            "success_rate": success_rate,
            "avg_retries_per_request": avg_retries,
            "max_retries": self.max_retries,
            "retryable_status_codes": list(self.retryable_status_codes),
            "retryable_exceptions": [exc.__name__ for exc in self.retryable_exceptions]
        }
    
    def reset_statistics(self):
        """Reset retry statistics counters."""
        self.total_retries = 0
        self.successful_retries = 0
        self.failed_retries = 0
        self._request_count = 0
        logger.debug("Retry statistics reset")
    
    def update_configuration(
        self,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        **kwargs
    ):
        """
        Update retry handler configuration.
        
        Args:
            max_retries (int, optional): New maximum retry count
            base_delay (float, optional): New base delay
            max_delay (float, optional): New maximum delay
            **kwargs: Additional configuration parameters
        """
        if max_retries is not None:
            self.max_retries = max_retries
        if base_delay is not None:
            self.base_delay = base_delay
        if max_delay is not None:
            self.max_delay = max_delay
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.debug("Retry handler configuration updated")
    
    def __repr__(self) -> str:
        """String representation of retry handler."""
        return (
            f"RetryHandler(max_retries={self.max_retries}, "
            f"base_delay={self.base_delay}s, "
            f"max_delay={self.max_delay}s, "
            f"backoff_multiplier={self.backoff_multiplier})"
        )