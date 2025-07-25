# echogrid/api/streaming.py
"""
Streaming client for real-time text-to-speech generation.

This module provides streaming capabilities for EchoGrid API, enabling
real-time audio generation and playback for interactive applications.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, Iterator, Callable, Union, List
from threading import Thread, Event
import requests
import sseclient

from ..exceptions import NetworkError, APIRateLimitError, ValidationError

logger = logging.getLogger(__name__)


class StreamingClient:
    """
    Streaming client for real-time text-to-speech generation.
    
    This class provides streaming capabilities that allow applications to
    receive audio data as it's being generated, enabling real-time playback
    and reducing perceived latency for end users.
    
    Args:
        api_client: Parent API client instance
        
    Attributes:
        api_client: Reference to parent API client
        active_streams (int): Number of currently active streams
        
    Example:
        Stream speech generation::

            from echogrid.api import APIClient
            
            client = APIClient(api_key="your-key")
            
            # Stream audio chunks as they're generated
            for chunk in client.streaming.stream_speech(
                text="This is a long text that will be streamed...",
                model="alice/voice",
                format="wav"
            ):
                # Process audio chunk immediately
                if chunk["type"] == "audio":
                    audio_data = chunk["data"]
                    play_audio_chunk(audio_data)
                elif chunk["type"] == "metadata":
                    print(f"Generation progress: {chunk['progress']:.1%}")

        Advanced streaming with callbacks::

            def on_audio_chunk(chunk_data):
                # Handle audio chunk
                audio_buffer.extend(chunk_data)
            
            def on_progress(progress_info):
                print(f"Progress: {progress_info['progress']:.1%}")
            
            def on_complete(final_stats):
                print(f"Generation complete: {final_stats['duration']:.2f}s")
            
            streaming_client.stream_speech_async(
                text="Hello world",
                model="alice/voice",
                on_audio_chunk=on_audio_chunk,
                on_progress=on_progress,
                on_complete=on_complete
            )

    Note:
        Streaming is particularly useful for long-form content where
        you want to start playback before the entire audio is generated.
        It can significantly improve perceived performance.
    """
    
    def __init__(self, api_client):
        """Initialize streaming client with parent API client."""
        self.api_client = api_client
        self.active_streams = 0
        self._stream_counter = 0
        
        logger.debug("Initialized streaming client")
    
    def stream_speech(
        self,
        text: str,
        model_name: str,
        chunk_size: int = 4096,
        format: str = "wav",
        sample_rate: int = 22050,
        **generation_params
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream text-to-speech generation in real-time.
        
        Generates audio chunks as soon as they're available, allowing
        for immediate playback and reduced latency.
        
        Args:
            text (str): Text to synthesize
            model_name (str): Voice model identifier
            chunk_size (int): Audio chunk size in bytes. Defaults to 4096.
            format (str): Audio format ("wav", "mp3", "flac"). Defaults to "wav".
            sample_rate (int): Audio sample rate in Hz. Defaults to 22050.
            **generation_params: Additional generation parameters such as:
                - speed: Speech speed multiplier (0.5-2.0)
                - pitch: Pitch shift in semitones (-12 to +12)
                - emotion: Emotion intensity (0.0-1.0)
                - temperature: Generation randomness (0.0-1.0)
                
        Yields:
            Dict[str, Any]: Stream chunks containing:
                - type: Chunk type ("audio", "metadata", "progress", "complete")
                - data: Chunk data (audio bytes for audio chunks)
                - timestamp: Generation timestamp
                - sequence: Chunk sequence number
                - metadata: Additional chunk metadata
                
        Raises:
            ValidationError: If parameters are invalid
            NetworkError: If streaming connection fails
            APIRateLimitError: If rate limit is exceeded
            
        Example:
            Stream and save audio::

                audio_chunks = []
                
                for chunk in client.streaming.stream_speech(
                    text="Hello, this is a streaming test.",
                    model="alice/voice",
                    format="wav",
                    chunk_size=8192
                ):
                    if chunk["type"] == "audio":
                        audio_chunks.append(chunk["data"])
                        print(f"Received chunk {chunk['sequence']}")
                    elif chunk["type"] == "complete":
                        print(f"Stream complete: {chunk['total_chunks']} chunks")
                
                # Combine all chunks
                full_audio = b"".join(audio_chunks)
                with open("output.wav", "wb") as f:
                    f.write(full_audio)
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty")
        
        if len(text) > 50000:  # Streaming limit
            raise ValidationError("Text too long for streaming (max 50,000 characters)")
        
        if chunk_size < 1024 or chunk_size > 65536:
            raise ValidationError("Chunk size must be between 1024 and 65536 bytes")
        
        # Prepare streaming request
        stream_id = self._generate_stream_id()
        payload = {
            "text": text,
            "model": model_name,
            "stream": True,
            "stream_id": stream_id,
            "chunk_size": chunk_size,
            "format": format,
            "sample_rate": sample_rate,
            "parameters": generation_params
        }
        
        try:
            # Increment active stream counter
            self.active_streams += 1
            
            # Make streaming request
            response = self.api_client.session.post(
                self.api_client.endpoints.generate_stream,
                json=payload,
                headers={
                    **self.api_client.auth.get_auth_headers(),
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
                stream=True,
                timeout=None  # No timeout for streaming
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                raise APIRateLimitError(f"Rate limit exceeded", retry_after=retry_after)
            
            response.raise_for_status()
            
            # Process server-sent events
            client = sseclient.SSEClient(response)
            sequence = 0
            
            logger.info(f"Started streaming for text length {len(text)}")
            
            for event in client.events():
                if event.event == "error":
                    error_data = json.loads(event.data)
                    raise NetworkError(f"Streaming error: {error_data.get('message', 'Unknown error')}")
                
                if event.event in ["audio", "metadata", "progress", "complete"]:
                    chunk_data = json.loads(event.data)
                    
                    # Add standard fields
                    chunk = {
                        "type": event.event,
                        "stream_id": stream_id,
                        "sequence": sequence,
                        "timestamp": time.time(),
                        **chunk_data
                    }
                    
                    sequence += 1
                    yield chunk
                    
                    # End stream on completion
                    if event.event == "complete":
                        logger.info(f"Streaming completed: {sequence} chunks")
                        break
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Streaming request failed: {e}")
            raise NetworkError(f"Streaming failed: {e}") from e
        
        finally:
            # Decrement active stream counter
            self.active_streams = max(0, self.active_streams - 1)
    
    def stream_speech_async(
        self,
        text: str,
        model_name: str,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_metadata: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        **stream_params
    ) -> str:
        """
        Start asynchronous streaming with callback functions.
        
        Starts a background thread to handle streaming and calls the provided
        callback functions as events occur.
        
        Args:
            text (str): Text to synthesize
            model_name (str): Voice model identifier
            on_audio_chunk (Callable, optional): Called for each audio chunk
            on_progress (Callable, optional): Called for progress updates
            on_metadata (Callable, optional): Called for metadata events
            on_complete (Callable, optional): Called when streaming completes
            on_error (Callable, optional): Called if an error occurs
            **stream_params: Additional streaming parameters
            
        Returns:
            str: Stream ID for tracking and control
            
        Example:
            Async streaming with audio playback::

                import pyaudio
                
                # Setup audio output
                audio = pyaudio.PyAudio()
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=22050,
                    output=True
                )
                
                def play_chunk(audio_data):
                    stream.write(audio_data)
                
                def on_done(stats):
                    print(f"Playback complete: {stats['duration']:.2f}s")
                    stream.close()
                    audio.terminate()
                
                # Start async streaming
                stream_id = client.streaming.stream_speech_async(
                    text="Hello world",
                    model="alice/voice",
                    on_audio_chunk=play_chunk,
                    on_complete=on_done
                )
        """
        stream_id = self._generate_stream_id()
        
        def streaming_worker():
            """Background worker for async streaming."""
            try:
                for chunk in self.stream_speech(
                    text=text,
                    model_name=model_name,
                    **stream_params
                ):
                    chunk_type = chunk["type"]
                    
                    if chunk_type == "audio" and on_audio_chunk:
                        on_audio_chunk(chunk["data"])
                    elif chunk_type == "progress" and on_progress:
                        on_progress(chunk)
                    elif chunk_type == "metadata" and on_metadata:
                        on_metadata(chunk)
                    elif chunk_type == "complete" and on_complete:
                        on_complete(chunk)
                        
            except Exception as e:
                logger.error(f"Async streaming error: {e}")
                if on_error:
                    on_error(e)
                else:
                    raise
        
        # Start background thread
        thread = Thread(target=streaming_worker, daemon=True)
        thread.start()
        
        logger.debug(f"Started async streaming thread for stream {stream_id}")
        return stream_id
    
    def stream_batch(
        self,
        texts: List[str],
        model_name: str,
        **stream_params
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream multiple texts in sequence.
        
        Args:
            texts (List[str]): List of texts to synthesize
            model_name (str): Voice model identifier
            **stream_params: Streaming parameters
            
        Yields:
            Dict[str, Any]: Stream chunks with batch information
            
        Example:
            Stream multiple sentences::

                sentences = [
                    "Hello there!",
                    "How are you today?",
                    "This is batch streaming."
                ]
                
                for chunk in client.streaming.stream_batch(
                    texts=sentences,
                    model="alice/voice"
                ):
                    if chunk["type"] == "audio":
                        print(f"Batch {chunk['batch_index']}, Chunk {chunk['sequence']}")
                        play_audio(chunk["data"])
        """
        for batch_index, text in enumerate(texts):
            logger.debug(f"Starting batch item {batch_index + 1}/{len(texts)}")
            
            for chunk in self.stream_speech(
                text=text,
                model_name=model_name,
                **stream_params
            ):
                # Add batch information
                chunk["batch_index"] = batch_index
                chunk["batch_total"] = len(texts)
                chunk["batch_text"] = text
                
                yield chunk
    
    def create_stream_session(
        self,
        model_name: str,
        session_config: Optional[Dict[str, Any]] = None
    ) -> 'StreamSession':
        """
        Create a persistent streaming session for multiple requests.
        
        Args:
            model_name (str): Voice model to use for the session
            session_config (Dict, optional): Session configuration
            
        Returns:
            StreamSession: Persistent streaming session
            
        Example:
            Use persistent session for multiple streams::

                with client.streaming.create_stream_session("alice/voice") as session:
                    # Stream multiple texts using same model
                    for text in text_list:
                        for chunk in session.stream(text):
                            process_chunk(chunk)
        """
        return StreamSession(self, model_name, session_config or {})
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """
        Get streaming statistics.
        
        Returns:
            Dict[str, Any]: Streaming performance statistics
        """
        return {
            "active_streams": self.active_streams,
            "total_streams": self._stream_counter,
            "streaming_enabled": True,
            "supported_formats": ["wav", "mp3", "flac", "ogg"],
            "max_text_length": 50000,
            "chunk_size_range": [1024, 65536],
            "sample_rates": [16000, 22050, 44100, 48000]
        }
    
    def _generate_stream_id(self) -> str:
        """Generate unique stream identifier."""
        self._stream_counter += 1
        timestamp = int(time.time() * 1000)
        return f"stream_{timestamp}_{self._stream_counter}"
    
    def __repr__(self) -> str:
        """String representation of streaming client."""
        return f"StreamingClient(active_streams={self.active_streams})"


class StreamSession:
    """
    Persistent streaming session for efficient multiple requests.
    
    This class maintains a persistent connection and model context
    for multiple streaming requests, reducing overhead and latency.
    
    Args:
        streaming_client (StreamingClient): Parent streaming client
        model_name (str): Voice model for this session
        config (Dict[str, Any]): Session configuration
        
    Example:
        Use streaming session::

            with client.streaming.create_stream_session("alice/voice") as session:
                # Configure session
                session.set_voice_config(speed=1.2, emotion="happy")
                
                # Stream multiple texts
                texts = ["Hello", "How are you?", "Goodbye"]
                for text in texts:
                    for chunk in session.stream(text):
                        if chunk["type"] == "audio":
                            play_audio(chunk["data"])
    """
    
    def __init__(
        self,
        streaming_client: StreamingClient,
        model_name: str,
        config: Dict[str, Any]
    ):
        """Initialize streaming session."""
        self.streaming_client = streaming_client
        self.model_name = model_name
        self.config = config
        self.session_id = self._generate_session_id()
        self._voice_config = {}
        self._is_active = False
        
        logger.debug(f"Created streaming session {self.session_id}")
    
    def stream(
        self,
        text: str,
        override_params: Optional[Dict[str, Any]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream text using this session's configuration.
        
        Args:
            text (str): Text to synthesize
            override_params (Dict, optional): Parameters to override for this request
            
        Yields:
            Dict[str, Any]: Stream chunks
        """
        # Merge session config with override params
        stream_params = {**self._voice_config}
        if override_params:
            stream_params.update(override_params)
        
        # Add session information
        stream_params["session_id"] = self.session_id
        
        for chunk in self.streaming_client.stream_speech(
            text=text,
            model_name=self.model_name,
            **stream_params
        ):
            # Add session context to chunk
            chunk["session_id"] = self.session_id
            yield chunk
    
    def set_voice_config(self, **params):
        """
        Set voice configuration for this session.
        
        Args:
            **params: Voice parameters (speed, pitch, emotion, etc.)
        """
        self._voice_config.update(params)
        logger.debug(f"Updated voice config for session {self.session_id}")
    
    def get_voice_config(self) -> Dict[str, Any]:
        """Get current voice configuration."""
        return self._voice_config.copy()
    
    def reset_config(self):
        """Reset voice configuration to defaults."""
        self._voice_config.clear()
        logger.debug(f"Reset config for session {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = int(time.time() * 1000)
        return f"session_{timestamp}_{id(self)}"
    
    def __enter__(self):
        """Context manager entry."""
        self._is_active = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._is_active = False
        logger.debug(f"Closed streaming session {self.session_id}")
    
    def __repr__(self) -> str:
        """String representation of stream session."""
        return (
            f"StreamSession(id={self.session_id}, "
            f"model={self.model_name}, "
            f"active={self._is_active})"
        )


class StreamBuffer:
    """
    Buffer for managing streaming audio chunks.
    
    This utility class helps manage audio chunks from streaming,
    providing buffering, format conversion, and playback coordination.
    
    Args:
        buffer_size (int): Maximum buffer size in bytes
        format (str): Audio format
        sample_rate (int): Audio sample rate
        
    Example:
        Buffer streaming audio::

            buffer = StreamBuffer(buffer_size=8192, format="wav", sample_rate=22050)
            
            for chunk in stream:
                if chunk["type"] == "audio":
                    buffer.add_chunk(chunk["data"])
                    
                    # Get buffered audio when ready
                    if buffer.has_complete_frames():
                        audio_data = buffer.get_audio(duration=1.0)
                        play_audio(audio_data)
    """
    
    def __init__(
        self,
        buffer_size: int = 16384,
        format: str = "wav",
        sample_rate: int = 22050,
        channels: int = 1
    ):
        """Initialize stream buffer."""
        self.buffer_size = buffer_size
        self.format = format
        self.sample_rate = sample_rate
        self.channels = channels
        
        self._buffer = bytearray()
        self._total_bytes = 0
        self._chunks_received = 0
        
        # Calculate bytes per frame based on format
        self._bytes_per_sample = 2 if format in ["wav", "pcm"] else 1
        self._bytes_per_frame = self._bytes_per_sample * channels
        
        logger.debug(f"Initialized stream buffer: {buffer_size} bytes, {format} format")
    
    def add_chunk(self, chunk_data: bytes):
        """
        Add audio chunk to buffer.
        
        Args:
            chunk_data (bytes): Audio chunk data
        """
        self._buffer.extend(chunk_data)
        self._total_bytes += len(chunk_data)
        self._chunks_received += 1
        
        # Trim buffer if it exceeds maximum size
        if len(self._buffer) > self.buffer_size:
            excess = len(self._buffer) - self.buffer_size
            self._buffer = self._buffer[excess:]
    
    def get_audio(self, duration: Optional[float] = None) -> bytes:
        """
        Get audio data from buffer.
        
        Args:
            duration (float, optional): Duration in seconds to extract
            
        Returns:
            bytes: Audio data
        """
        if duration is None:
            # Return all buffered audio
            audio_data = bytes(self._buffer)
            self._buffer.clear()
            return audio_data
        
        # Calculate bytes needed for duration
        bytes_needed = int(duration * self.sample_rate * self._bytes_per_frame)
        bytes_available = min(bytes_needed, len(self._buffer))
        
        if bytes_available > 0:
            audio_data = bytes(self._buffer[:bytes_available])
            self._buffer = self._buffer[bytes_available:]
            return audio_data
        
        return b""
    
    def has_complete_frames(self, min_frames: int = 1024) -> bool:
        """
        Check if buffer has complete audio frames.
        
        Args:
            min_frames (int): Minimum frames required
            
        Returns:
            bool: True if buffer has enough complete frames
        """
        min_bytes = min_frames * self._bytes_per_frame
        return len(self._buffer) >= min_bytes
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dict[str, Any]: Buffer information
        """
        buffered_duration = (
            len(self._buffer) / (self.sample_rate * self._bytes_per_frame)
            if self._bytes_per_frame > 0 else 0
        )
        
        return {
            "buffered_bytes": len(self._buffer),
            "buffered_duration": buffered_duration,
            "total_bytes_received": self._total_bytes,
            "chunks_received": self._chunks_received,
            "buffer_utilization": len(self._buffer) / self.buffer_size,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "channels": self.channels
        }
    
    def clear(self):
        """Clear buffer contents."""
        self._buffer.clear()
        logger.debug("Stream buffer cleared")
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)
    
    def __repr__(self) -> str:
        """String representation of stream buffer."""
        return (
            f"StreamBuffer(size={len(self._buffer)}/{self.buffer_size}, "
            f"format={self.format}, "
            f"sample_rate={self.sample_rate}Hz)"
        )


class StreamController:
    """
    Advanced stream controller for managing multiple concurrent streams.
    
    This class provides centralized management for multiple streaming sessions,
    including load balancing, priority queuing, and resource management.
    
    Args:
        max_concurrent_streams (int): Maximum concurrent streams allowed
        priority_levels (int): Number of priority levels (1-5)
        
    Example:
        Manage multiple streams with priorities::

            controller = StreamController(max_concurrent_streams=3)
            
            # High priority stream (news, alerts)
            high_priority_stream = controller.create_stream(
                text="Breaking news alert",
                model="news/anchor",
                priority=1
            )
            
            # Normal priority stream (general content)
            normal_stream = controller.create_stream(
                text="Regular content",
                model="alice/voice",
                priority=3
            )
            
            # Process all streams
            for stream_id, chunk in controller.process_all_streams():
                handle_chunk(stream_id, chunk)
    """
    
    def __init__(
        self,
        streaming_client: StreamingClient,
        max_concurrent_streams: int = 5,
        priority_levels: int = 5
    ):
        """Initialize stream controller."""
        self.streaming_client = streaming_client
        self.max_concurrent_streams = max_concurrent_streams
        self.priority_levels = priority_levels
        
        # Stream management
        self._active_streams: Dict[str, Dict[str, Any]] = {}
        self._stream_queue: List[Dict[str, Any]] = []
        self._completed_streams: List[str] = []
        
        # Statistics
        self._total_streams_created = 0
        self._total_streams_completed = 0
        
        logger.debug(f"Initialized stream controller: max_streams={max_concurrent_streams}")
    
    def create_stream(
        self,
        text: str,
        model_name: str,
        priority: int = 3,
        **stream_params
    ) -> str:
        """
        Create a new stream with priority queuing.
        
        Args:
            text (str): Text to synthesize
            model_name (str): Voice model identifier
            priority (int): Stream priority (1=highest, 5=lowest)
            **stream_params: Additional streaming parameters
            
        Returns:
            str: Stream identifier
        """
        if priority < 1 or priority > self.priority_levels:
            priority = 3  # Default to medium priority
        
        stream_id = self.streaming_client._generate_stream_id()
        
        stream_info = {
            "stream_id": stream_id,
            "text": text,
            "model_name": model_name,
            "priority": priority,
            "created_at": time.time(),
            "params": stream_params,
            "status": "queued"
        }
        
        # Add to queue (sorted by priority)
        self._stream_queue.append(stream_info)
        self._stream_queue.sort(key=lambda x: x["priority"])
        
        self._total_streams_created += 1
        
        logger.debug(f"Created stream {stream_id} with priority {priority}")
        return stream_id
    
    def process_all_streams(self) -> Iterator[tuple]:
        """
        Process all queued and active streams.
        
        Yields:
            tuple: (stream_id, chunk) for each stream chunk
        """
        while self._stream_queue or self._active_streams:
            # Start new streams if capacity allows
            while (len(self._active_streams) < self.max_concurrent_streams and 
                   self._stream_queue):
                
                stream_info = self._stream_queue.pop(0)
                self._start_stream(stream_info)
            
            # Process active streams
            completed_streams = []
            
            for stream_id, stream_data in self._active_streams.items():
                try:
                    chunk = next(stream_data["iterator"])
                    yield stream_id, chunk
                    
                    # Check if stream completed
                    if chunk.get("type") == "complete":
                        completed_streams.append(stream_id)
                        
                except StopIteration:
                    # Stream ended
                    completed_streams.append(stream_id)
            
            # Clean up completed streams
            for stream_id in completed_streams:
                self._complete_stream(stream_id)
            
            # Small delay to prevent busy waiting
            if not self._active_streams and not self._stream_queue:
                break
            
            time.sleep(0.01)
    
    def _start_stream(self, stream_info: Dict[str, Any]):
        """Start a queued stream."""
        stream_id = stream_info["stream_id"]
        
        try:
            iterator = self.streaming_client.stream_speech(
                text=stream_info["text"],
                model_name=stream_info["model_name"],
                **stream_info["params"]
            )
            
            self._active_streams[stream_id] = {
                **stream_info,
                "iterator": iterator,
                "status": "active",
                "started_at": time.time()
            }
            
            logger.debug(f"Started stream {stream_id}")
            
        except Exception as e:
            logger.error(f"Failed to start stream {stream_id}: {e}")
    
    def _complete_stream(self, stream_id: str):
        """Complete and clean up a stream."""
        if stream_id in self._active_streams:
            stream_data = self._active_streams.pop(stream_id)
            stream_data["completed_at"] = time.time()
            stream_data["status"] = "completed"
            
            self._completed_streams.append(stream_id)
            self._total_streams_completed += 1
            
            logger.debug(f"Completed stream {stream_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get controller status and statistics.
        
        Returns:
            Dict[str, Any]: Controller status information
        """
        return {
            "active_streams": len(self._active_streams),
            "queued_streams": len(self._stream_queue),
            "completed_streams": len(self._completed_streams),
            "total_created": self._total_streams_created,
            "total_completed": self._total_streams_completed,
            "max_concurrent": self.max_concurrent_streams,
            "priority_levels": self.priority_levels,
            "queue_details": [
                {
                    "stream_id": s["stream_id"],
                    "priority": s["priority"],
                    "queue_time": time.time() - s["created_at"]
                }
                for s in self._stream_queue[:5]  # Show first 5 in queue
            ]
        }
    
    def cancel_stream(self, stream_id: str) -> bool:
        """
        Cancel a queued or active stream.
        
        Args:
            stream_id (str): Stream identifier to cancel
            
        Returns:
            bool: True if stream was cancelled
        """
        # Check queued streams
        for i, stream_info in enumerate(self._stream_queue):
            if stream_info["stream_id"] == stream_id:
                self._stream_queue.pop(i)
                logger.debug(f"Cancelled queued stream {stream_id}")
                return True
        
        # Check active streams
        if stream_id in self._active_streams:
            self._complete_stream(stream_id)
            logger.debug(f"Cancelled active stream {stream_id}")
            return True
        
        return False
    
    def clear_completed(self):
        """Clear completed stream history."""
        self._completed_streams.clear()
        logger.debug("Cleared completed streams history")
    
    def __repr__(self) -> str:
        """String representation of stream controller."""
        return f"StreamController(active={len(self._active_streams)}, queued={len(self._stream_queue)})"