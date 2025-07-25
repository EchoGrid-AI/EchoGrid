# echogrid/inference/remote.py
"""
Remote API inference engine for EchoGrid.

This module handles inference through the EchoGrid hosted API.
"""

import logging
import requests
import asyncio
import numpy as np
from typing import Dict, Any, Optional, Union
import time
import json

from ..config import get_config
from ..exceptions import NetworkError, APIRateLimitError, AuthenticationError, ValidationError
from ..utils.validation import validate_text_input

logger = logging.getLogger(__name__)


class APIInferenceEngine:
    """
    Remote API inference engine for hosted model inference.
    
    This class handles communication with the EchoGrid API for
    text-to-speech generation using hosted models.
    """
    
    def __init__(
        self,
        api_endpoint: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        region: str = "auto",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize API inference engine.
        
        Args:
            api_endpoint: API base URL
            api_key: API authentication key
            timeout: Request timeout in seconds
            region: Preferred region for processing
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
        """
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.region = region
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'echogrid-python/0.1.0',
            'Content-Type': 'application/json'
        })
        
        if self.api_key:
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
        
        # Performance tracking
        self._request_times = []
        self._total_requests = 0
        self._failed_requests = 0
        
        logger.debug(f"Initialized APIInferenceEngine for {api_endpoint}")
    
    def is_model_available(self, model_name: str, version: str = "latest") -> bool:
        """
        Check if model is available via API.
        
        Args:
            model_name: Model identifier
            version: Model version
        
        Returns:
            bool: True if model is available
        """
        try:
            response = self._make_request(
                'GET',
                f'/models/{model_name}/info',
                params={'version': version},
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(
        self,
        model,
        text: str,
        language: str = "auto",
        emotion: str = "neutral",
        emotion_strength: float = 1.0,
        speed: float = 1.0,
        pitch: float = 0.0,
        temperature: float = 0.75,
        priority: str = "balanced",
        region: Optional[str] = None,
        stream: bool = False,
        callback_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate speech from text using API.
        
        Args:
            model: Voice model instance
            text: Input text
            language: Target language
            emotion: Emotion type
            emotion_strength: Emotion intensity
            speed: Speech speed
            pitch: Pitch adjustment
            temperature: Generation randomness
            priority: Processing priority ("speed", "quality", "cost", "balanced")
            region: Processing region preference
            stream: Enable streaming response
            callback_url: Webhook URL for async processing
            **kwargs: Additional generation parameters
        
        Returns:
            Dict[str, Any]: Generated audio and metadata
        """
        try:
            start_time = time.time()
            
            # Validate inputs
            validate_text_input(text, max_length=2000)  # API limit
            
            # Prepare request payload
            payload = {
                "model": model.name,
                "text": text,
                "parameters": {
                    "language": language,
                    "emotion": emotion,
                    "emotion_strength": emotion_strength,
                    "speed": speed,
                    "pitch": pitch,
                    "temperature": temperature,
                    "sample_rate": kwargs.get("sample_rate", model.sample_rate),
                    "format": kwargs.get("format", "wav"),
                    "normalize_text": kwargs.get("normalize_text", True),
                    "enhance_quality": kwargs.get("enhance_quality", False),
                    "noise_reduction": kwargs.get("noise_reduction", False)
                },
                "options": {
                    "priority": priority,
                    "region": region or self.region,
                    "stream": stream,
                    "callback_url": callback_url,
                    "return_metadata": kwargs.get("return_metadata", False)
                }
            }
            
            # Remove None values
            payload = self._clean_payload(payload)
            
            if stream:
                return self._generate_streaming(payload, **kwargs)
            elif callback_url:
                return self._generate_async(payload, **kwargs)
            else:
                return self._generate_sync(payload, start_time, **kwargs)
                
        except Exception as e:
            self._failed_requests += 1
            logger.error(f"API generation failed: {e}")
            raise NetworkError(
                f"API generation failed: {e}",
                operation="generate",
                original_error=e
            )
    
    def _generate_sync(self, payload: Dict, start_time: float, **kwargs) -> Dict[str, Any]:
        """Synchronous generation."""
        response = self._make_request(
            'POST',
            '/generate',
            json=payload,
            timeout=kwargs.get('timeout', self.timeout)
        )
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            raise APIRateLimitError(retry_after=retry_after)
        elif response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code != 200:
            raise NetworkError(
                f"API request failed with status {response.status_code}: {response.text}",
                status_code=response.status_code
            )
        
        result = response.json()
        
        # Decode audio data
        audio_data = result.get('audio_data')
        if isinstance(audio_data, str):
            # Base64 encoded
            import base64
            audio_bytes = base64.b64decode(audio_data)
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
        elif isinstance(audio_data, list):
            # Direct array
            audio = np.array(audio_data, dtype=np.float32)
        else:
            raise NetworkError("Invalid audio data format from API")
        
        generation_time = time.time() - start_time
        self._request_times.append(generation_time)
        self._total_requests += 1
        
        # Prepare result
        api_result = {
            "audio": audio,
            "sample_rate": result.get("sample_rate", 22050),
            "metadata": result.get("metadata", {}) if kwargs.get("return_metadata", False) else {},
            "performance": {
                "api_time": generation_time,
                "server_time": result.get("generation_time", 0),
                "queue_time": result.get("queue_time", 0),
                "characters_per_second": len(payload["text"]) / generation_time
            } if kwargs.get("profile_performance", False) else {}
        }
        
        logger.debug(f"API generation completed in {generation_time:.2f}s")
        return api_result
    
    def _generate_streaming(self, payload: Dict, **kwargs) -> Dict[str, Any]:
        """Streaming generation."""
        # For streaming, we'd use Server-Sent Events or WebSocket
        # This is a simplified implementation
        response = self._make_request(
            'POST',
            '/generate/stream',
            json=payload,
            stream=True,
            timeout=kwargs.get('timeout', 300)  # Longer timeout for streaming
        )
        
        if response.status_code != 200:
            raise NetworkError(f"Streaming request failed: {response.status_code}")
        
        # Collect streamed audio chunks
        audio_chunks = []
        metadata = {}
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk_data = json.loads(line.decode('utf-8'))
                    
                    if chunk_data.get('type') == 'audio_chunk':
                        import base64
                        chunk_audio = base64.b64decode(chunk_data['data'])
                        audio_chunks.append(np.frombuffer(chunk_audio, dtype=np.float32))
                    elif chunk_data.get('type') == 'metadata':
                        metadata = chunk_data.get('data', {})
                    elif chunk_data.get('type') == 'error':
                        raise NetworkError(f"Streaming error: {chunk_data.get('message')}")
                        
                except json.JSONDecodeError:
                    continue
        
        # Concatenate audio chunks
        if audio_chunks:
            audio = np.concatenate(audio_chunks)
        else:
            raise NetworkError("No audio data received from streaming")
        
        return {
            "audio": audio,
            "sample_rate": metadata.get("sample_rate", 22050),
            "metadata": metadata if kwargs.get("return_metadata", False) else {}
        }
    
    def _generate_async(self, payload: Dict, **kwargs) -> Dict[str, Any]:
        """Asynchronous generation with callback."""
        response = self._make_request(
            'POST',
            '/generate/async',
            json=payload,
            timeout=10  # Quick timeout for async submission
        )
        
        if response.status_code != 202:
            raise NetworkError(f"Async request failed: {response.status_code}")
        
        result = response.json()
        
        return {
            "job_id": result.get("job_id"),
            "status": "queued",
            "callback_url": payload["options"]["callback_url"],
            "estimated_completion": result.get("estimated_completion")
        }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of async job.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Dict[str, Any]: Job status and result if complete
        """
        try:
            response = self._make_request('GET', f'/jobs/{job_id}')
            
            if response.status_code == 404:
                raise NetworkError(f"Job not found: {job_id}")
            elif response.status_code != 200:
                raise NetworkError(f"Status check failed: {response.status_code}")
            
            return response.json()
            
        except Exception as e:
            raise NetworkError(f"Failed to get job status: {e}")
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel async job.
        
        Args:
            job_id: Job identifier
        
        Returns:
            bool: True if cancelled successfully
        """
        try:
            response = self._make_request('DELETE', f'/jobs/{job_id}')
            return response.status_code == 200
        except Exception:
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Dict[str, Any]: Usage statistics
        """
        try:
            response = self._make_request('GET', '/usage')
            
            if response.status_code != 200:
                raise NetworkError(f"Usage request failed: {response.status_code}")
            
            api_stats = response.json()
            
            # Add client-side stats
            client_stats = {
                "client_requests": self._total_requests,
                "client_failures": self._failed_requests,
                "avg_request_time": np.mean(self._request_times) if self._request_times else 0,
                "success_rate": (self._total_requests - self._failed_requests) / max(self._total_requests, 1)
            }
            
            return {**api_stats, **client_stats}
            
        except Exception as e:
            raise NetworkError(f"Failed to get usage stats: {e}")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List models available via API.
        
        Returns:
            List[Dict[str, Any]]: Available models
        """
        try:
            response = self._make_request('GET', '/models')
            
            if response.status_code != 200:
                raise NetworkError(f"Model list request failed: {response.status_code}")
            
            return response.json().get('models', [])
            
        except Exception as e:
            raise NetworkError(f"Failed to list models: {e}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed model information.
        
        Args:
            model_name: Model identifier
        
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            response = self._make_request('GET', f'/models/{model_name}/info')
            
            if response.status_code == 404:
                raise NetworkError(f"Model not found: {model_name}")
            elif response.status_code != 200:
                raise NetworkError(f"Model info request failed: {response.status_code}")
            
            return response.json()
            
        except Exception as e:
            raise NetworkError(f"Failed to get model info: {e}")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection and authentication.
        
        Returns:
            Dict[str, Any]: Connection test results
        """
        try:
            start_time = time.time()
            response = self._make_request('GET', '/health', timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "connected": True,
                    "response_time": response_time,
                    "api_version": result.get("version"),
                    "region": result.get("region"),
                    "authenticated": bool(self.api_key)
                }
            else:
                return {
                    "connected": False,
                    "error": f"HTTP {response.status_code}",
                    "response_time": response_time
                }
                
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "response_time": None
            }
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        timeout: Optional[int] = None,
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            timeout: Request timeout
            **kwargs: Additional request parameters
        
        Returns:
            requests.Response: HTTP response
        """
        url = f"{self.api_endpoint}{endpoint}"
        timeout = timeout or self.timeout
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=timeout,
                    **kwargs
                )
                
                # Don't retry on client errors (4xx) except 429
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    return response
                
                # Don't retry on success
                if response.status_code < 400:
                    return response
                
                # Retry on server errors (5xx) and 429
                if attempt < self.max_retries:
                    retry_delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    
                    # Check for Retry-After header
                    if response.status_code == 429:
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            retry_delay = max(retry_delay, int(retry_after))
                    
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {retry_delay}s: {response.status_code}"
                    )
                    time.sleep(retry_delay)
                    continue
                
                return response
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    retry_delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Request exception (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {retry_delay}s: {e}"
                    )
                    time.sleep(retry_delay)
                    continue
        
        # All retries exhausted
        raise NetworkError(f"Request failed after {self.max_retries + 1} attempts: {last_exception}")
    
    def _clean_payload(self, payload: Dict) -> Dict:
        """Remove None values from payload recursively."""
        if isinstance(payload, dict):
            return {k: self._clean_payload(v) for k, v in payload.items() if v is not None}
        elif isinstance(payload, list):
            return [self._clean_payload(item) for item in payload]
        else:
            return payload
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client-side performance statistics."""
        if not self._request_times:
            return {}
        
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (self._total_requests - self._failed_requests) / self._total_requests,
            "avg_response_time": np.mean(self._request_times),
            "min_response_time": np.min(self._request_times),
            "max_response_time": np.max(self._request_times),
            "p50_response_time": np.percentile(self._request_times, 50),
            "p95_response_time": np.percentile(self._request_times, 95),
            "p99_response_time": np.percentile(self._request_times, 99)
        }
    
    def close(self):
        """Close HTTP session."""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncAPIInferenceEngine:
    """
    Async version of API inference engine.
    
    Provides the same functionality as APIInferenceEngine but with
    async/await support for better concurrency.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as APIInferenceEngine."""
        self._sync_engine = APIInferenceEngine(*args, **kwargs)
    
    async def generate(self, model, text: str, **kwargs) -> Dict[str, Any]:
        """
        Async generate speech from text.
        
        Args:
            model: Voice model instance
            text: Input text
            **kwargs: Generation parameters
        
        Returns:
            Dict[str, Any]: Generated audio and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self._sync_engine.generate(model, text, **kwargs)
        )
    
    async def is_model_available(self, model_name: str, version: str = "latest") -> bool:
        """Async check if model is available."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_engine.is_model_available(model_name, version)
        )
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Async get job status."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_engine.get_job_status(job_id)
        )
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Async get usage statistics."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_engine.get_usage_stats()
        )
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Async list available models."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_engine.list_available_models()
        )
    
    async def test_connection(self) -> Dict[str, Any]:
        """Async test connection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_engine.test_connection()
        )
    
    def close(self):
        """Close underlying session."""
        self._sync_engine.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()