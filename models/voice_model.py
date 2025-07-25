# echogrid/models/voice_model.py
"""
Voice model interface for EchoGrid.

This module provides the main VoiceModel class that wraps inference engines
and provides a unified interface for text-to-speech generation.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union, Iterator, Callable
from pathlib import Path

from ..config import get_config
from ..constants import SUPPORTED_EMOTIONS, SUPPORTED_LANGUAGES
from ..exceptions import ValidationError, AudioProcessingError
from ..audio.file import AudioFile
from ..utils.validation import validate_text_input, validate_audio_parameters

logger = logging.getLogger(__name__)


class AudioResult:
    """
    Container for generated audio with metadata.
    
    This class wraps generated audio data and provides convenient methods
    for saving, playing, and converting the audio.
    
    Attributes:
        audio: Raw audio array (samples,)
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        metadata: Generation metadata
        performance: Performance statistics
    """
    
    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize audio result.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate in Hz
            metadata: Generation metadata
            performance: Performance statistics
        """
        self.audio = audio
        self.sample_rate = sample_rate
        self.metadata = metadata or {}
        self.performance = performance or {}
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.audio) / self.sample_rate
    
    def save(
        self,
        filepath: str,
        format: Optional[str] = None,
        quality: str = "high",
        sample_rate: Optional[int] = None,
        normalize: bool = False,
        fade_in: float = 0.0,
        fade_out: float = 0.0
    ):
        """
        Save audio to file.
        
        Args:
            filepath: Output file path
            format: Audio format (auto-detect from extension if None)
            quality: Encoding quality ("low", "medium", "high", "lossless")
            sample_rate: Resample to this rate if different
            normalize: Normalize volume
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
        
        Example:
            >>> audio = model.tts("Hello world")
            >>> audio.save("output.wav")
            >>> audio.save("output.mp3", quality="high", normalize=True)
        """
        try:
            # Create AudioFile from result
            audio_file = AudioFile(
                source=self.audio,
                sample_rate=self.sample_rate
            )
            
            # Apply processing
            if normalize:
                audio_file = audio_file.normalize()
            
            if fade_in > 0 or fade_out > 0:
                audio_file = audio_file.volume(
                    factor=1.0,
                    fade_in=fade_in,
                    fade_out=fade_out
                )
            
            # Save with specified parameters
            audio_file.save(
                filepath=filepath,
                format=format,
                quality=quality,
                sample_rate=sample_rate or self.sample_rate
            )
            
            logger.debug(f"Saved audio to: {filepath}")
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to save audio: {e}",
                operation="save",
                file_path=filepath,
                original_error=e
            )
    
    def play(
        self,
        device: Optional[str] = None,
        blocking: bool = True,
        volume: float = 1.0
    ):
        """
        Play audio through speakers.
        
        Args:
            device: Audio output device
            blocking: Wait for playback to finish
            volume: Playback volume (0.0-2.0)
        
        Example:
            >>> audio = model.tts("Hello world")
            >>> audio.play()  # Play audio
            >>> audio.play(blocking=False)  # Play in background
        """
        try:
            audio_file = AudioFile(
                source=self.audio,
                sample_rate=self.sample_rate
            )
            
            if volume != 1.0:
                audio_file = audio_file.volume(volume)
            
            audio_file.play(
                device=device,
                blocking=blocking
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to play audio: {e}",
                operation="play",
                original_error=e
            )
    
    def numpy(self) -> np.ndarray:
        """
        Get raw numpy array.
        
        Returns:
            np.ndarray: Audio data as numpy array
        """
        return self.audio
    
    def bytes(self, format: str = "wav", quality: str = "high") -> bytes:
        """
        Get audio as bytes.
        
        Args:
            format: Output format ("wav", "mp3", "flac", "ogg")
            quality: Encoding quality
        
        Returns:
            bytes: Audio data as bytes
        """
        try:
            audio_file = AudioFile(
                source=self.audio,
                sample_rate=self.sample_rate
            )
            return audio_file.to_bytes(format=format, quality=quality)
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to convert to bytes: {e}",
                operation="bytes_conversion",
                original_error=e
            )
    
    def base64(self, format: str = "wav", quality: str = "high") -> str:
        """
        Get audio as base64 string.
        
        Args:
            format: Output format
            quality: Encoding quality
        
        Returns:
            str: Audio data as base64 string
        """
        try:
            audio_file = AudioFile(
                source=self.audio,
                sample_rate=self.sample_rate
            )
            return audio_file.to_base64(format=format, quality=quality)
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to convert to base64: {e}",
                operation="base64_conversion",
                original_error=e
            )
    
    def upload_temp(self, expire_minutes: int = 60) -> str:
        """
        Upload to temporary hosting and return URL.
        
        Args:
            expire_minutes: URL expiration time in minutes
        
        Returns:
            str: Temporary URL to audio file
        """
        # This would integrate with the EchoGrid temporary file hosting
        # For now, this is a placeholder
        raise NotImplementedError("Temporary upload not yet implemented")


class AudioChunk:
    """
    Container for streaming audio chunks.
    
    Used for streaming TTS generation where audio is generated
    and delivered in chunks rather than all at once.
    
    Attributes:
        audio: Chunk audio data
        sample_rate: Sample rate
        chunk_index: Chunk number in sequence
        is_final: Whether this is the last chunk
        text_segment: Original text segment for this chunk
        start_time: Cumulative start time
        duration: Chunk duration
    """
    
    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        chunk_index: int,
        is_final: bool = False,
        text_segment: str = "",
        start_time: float = 0.0
    ):
        """Initialize audio chunk."""
        self.audio = audio
        self.sample_rate = sample_rate
        self.chunk_index = chunk_index
        self.is_final = is_final
        self.text_segment = text_segment
        self.start_time = start_time
    
    @property
    def duration(self) -> float:
        """Chunk duration in seconds."""
        return len(self.audio) / self.sample_rate
    
    def play(self, blocking: bool = False):
        """Play this audio chunk."""
        audio_file = AudioFile(source=self.audio, sample_rate=self.sample_rate)
        audio_file.play(blocking=blocking)
    
    def bytes(self, format: str = "wav") -> bytes:
        """Get chunk as bytes."""
        audio_file = AudioFile(source=self.audio, sample_rate=self.sample_rate)
        return audio_file.to_bytes(format=format)


class VoiceModel:
    """
    Main interface for voice model inference.
    
    This class provides a unified interface for text-to-speech generation
    regardless of the underlying backend (local or API).
    """
    
    def __init__(
        self,
        name: str,
        backend,
        inference_engine,
        model_info: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize voice model.
        
        Args:
            name: Model name
            backend: Backend type
            inference_engine: Inference engine instance
            model_info: Model metadata
            **kwargs: Additional model parameters
        """
        self.name = name
        self.backend = backend
        self._inference_engine = inference_engine
        self.model_info = model_info
        self._kwargs = kwargs
        
        # Model properties
        self._sample_rate = getattr(model_info, 'sample_rate', 22050) if model_info else 22050
        self._supports_emotions = getattr(model_info, 'supports_emotions', False) if model_info else False
        self._supports_multilingual = getattr(model_info, 'supports_multilingual', False) if model_info else False
        self._supported_languages = getattr(model_info, 'supported_languages', ['en']) if model_info else ['en']
        self._max_text_length = 2000  # Default limit
        
        logger.debug(f"Initialized VoiceModel: {name}")
    
    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self._sample_rate
    
    @property
    def supports_emotions(self) -> bool:
        """Whether model supports emotion control."""
        return self._supports_emotions
    
    @property
    def supports_multilingual(self) -> bool:
        """Whether model supports multiple languages."""
        return self._supports_multilingual
    
    @property
    def supported_languages(self) -> List[str]:
        """List of supported language codes."""
        return self._supported_languages
    
    @property
    def max_text_length(self) -> int:
        """Maximum input text length."""
        return self._max_text_length
    
    @property
    def device(self) -> Optional[str]:
        """Device the model is running on."""
        if hasattr(self._inference_engine, 'device'):
            return self._inference_engine.device
        return None
    
    def tts(
        self,
        text: str,
        language: str = "auto",
        emotion: str = "neutral",
        emotion_strength: float = 1.0,
        speed: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        pause_between_sentences: float = 0.5,
        pause_between_paragraphs: float = 1.0,
        temperature: float = 0.75,
        repetition_penalty: float = 1.1,
        length_penalty: float = 1.0,
        seed: Optional[int] = None,
        top_k: int = 50,
        top_p: float = 0.9,
        guidance_scale: float = 3.0,
        num_inference_steps: Optional[int] = None,
        voice_conditioning: Optional[str] = None,
        speaker_embedding: Optional[np.ndarray] = None,
        style_vector: Optional[np.ndarray] = None,
        prosody_control: Optional[Dict] = None,
        phoneme_control: Optional[Dict] = None,
        normalize_text: bool = True,
        split_sentences: bool = True,
        remove_silence: bool = True,
        enhance_quality: bool = False,
        noise_reduction: bool = False,
        sample_rate: Optional[int] = None,
        format: str = "wav",
        bit_depth: int = 16,
        priority: str = "balanced",
        region: Optional[str] = None,
        stream: bool = False,
        callback_url: Optional[str] = None,
        return_metadata: bool = False,
        profile_performance: bool = False,
        log_level: str = "INFO"
    ) -> AudioResult:
        """
        Generate speech from text.
        
        This is the main method for text-to-speech generation. It handles
        validation, preprocessing, generation, and post-processing.
        
        Args:
            text: Input text to synthesize
            language: Target language ("auto", "en", "es", etc.)
            emotion: Emotion ("neutral", "happy", "sad", "angry", "excited", "whisper")
            emotion_strength: Emotion intensity (0.0-2.0)
            speed: Speech rate (0.25-4.0)
            pitch: Pitch shift in semitones (-12 to +12)
            volume: Volume multiplier (0.0-2.0)
            pause_between_sentences: Pause duration in seconds (0.0-3.0)
            pause_between_paragraphs: Paragraph pause (0.0-5.0)
            temperature: Randomness/creativity (0.1-1.5)
            repetition_penalty: Avoid repetition (0.5-2.0)
            length_penalty: Length preference (0.5-2.0)
            seed: Random seed for reproducibility
            top_k: Top-k sampling (1-100)
            top_p: Nucleus sampling (0.1-1.0)
            guidance_scale: Classifier-free guidance (1.0-10.0)
            num_inference_steps: Diffusion steps (model-dependent)
            voice_conditioning: Reference audio file for style
            speaker_embedding: Custom speaker embedding
            style_vector: Style control vector
            prosody_control: Fine-grained prosody control
            phoneme_control: Phoneme-level control
            normalize_text: Text normalization
            split_sentences: Auto-split long text
            remove_silence: Remove leading/trailing silence
            enhance_quality: Post-processing enhancement
            noise_reduction: Background noise removal
            sample_rate: Override default sample rate
            format: Output format ("wav", "mp3", "ogg", "flac")
            bit_depth: Audio bit depth (16, 24, 32)
            priority: API priority ("speed", "quality", "cost", "balanced")
            region: API region preference
            stream: Streaming generation (not used in this method)
            callback_url: Webhook for async processing
            return_metadata: Include generation metadata
            profile_performance: Include timing information
            log_level: Logging level for this generation
        
        Returns:
            AudioResult: Generated audio with metadata
        
        Raises:
            ValidationError: If invalid parameters provided
            AudioProcessingError: If generation fails
        
        Example:
            >>> model = eg.load("alice/uk-teen-female")
            >>> 
            >>> # Basic generation
            >>> audio = model.tts("Hello, world!")
            >>> audio.save("hello.wav")
            >>> 
            >>> # Emotional speech
            >>> excited_audio = model.tts(
            ...     "I'm so excited!",
            ...     emotion="excited",
            ...     emotion_strength=1.5,
            ...     speed=1.1
            ... )
            >>> 
            >>> # High-quality generation
            >>> quality_audio = model.tts(
            ...     "Welcome to our podcast",
            ...     temperature=0.6,
            ...     enhance_quality=True,
            ...     sample_rate=48000,
            ...     bit_depth=24
            ... )
        """
        # Validate inputs
        validate_text_input(text, max_length=self.max_text_length)
        validate_audio_parameters(
            speed=speed,
            pitch=pitch,
            volume=volume,
            temperature=temperature,
            emotion=emotion,
            language=language,
            supported_languages=self.supported_languages,
            supports_emotions=self.supports_emotions
        )
        
        # Prepare generation parameters
        generation_params = {
            "text": text,
            "language": language,
            "emotion": emotion,
            "emotion_strength": emotion_strength,
            "speed": speed,
            "pitch": pitch,
            "volume": volume,
            "pause_between_sentences": pause_between_sentences,
            "pause_between_paragraphs": pause_between_paragraphs,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "seed": seed,
            "top_k": top_k,
            "top_p": top_p,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "voice_conditioning": voice_conditioning,
            "speaker_embedding": speaker_embedding,
            "style_vector": style_vector,
            "prosody_control": prosody_control,
            "phoneme_control": phoneme_control,
            "normalize_text": normalize_text,
            "split_sentences": split_sentences,
            "remove_silence": remove_silence,
            "enhance_quality": enhance_quality,
            "noise_reduction": noise_reduction,
            "sample_rate": sample_rate or self.sample_rate,
            "format": format,
            "bit_depth": bit_depth,
            "priority": priority,
            "region": region,
            "callback_url": callback_url,
            "return_metadata": return_metadata,
            "profile_performance": profile_performance
        }
        
        try:
            # Generate audio using inference engine
            result = self._inference_engine.generate(
                model=self,
                **generation_params
            )
            
            # Create AudioResult
            audio_result = AudioResult(
                audio=result["audio"],
                sample_rate=result.get("sample_rate", self.sample_rate),
                metadata=result.get("metadata", {}) if return_metadata else {},
                performance=result.get("performance", {}) if profile_performance else {}
            )
            
            logger.debug(f"Generated audio: {audio_result.duration:.2f}s")
            return audio_result
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise AudioProcessingError(
                f"TTS generation failed: {e}",
                operation="generation",
                original_error=e
            )
    
    async def atts(self, text: str, **kwargs) -> AudioResult:
        """
        Asynchronous text-to-speech generation.
        
        Args:
            text: Input text
            **kwargs: Same arguments as tts()
        
        Returns:
            AudioResult: Generated audio
        
        Example:
            >>> import asyncio
            >>> 
            >>> async def main():
            ...     model = await eg.aload("alice/uk-teen-female")
            ...     audio = await model.atts("Hello world")
            ...     return audio
            >>> 
            >>> audio = asyncio.run(main())
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.tts(text, **kwargs))
    
    def tts_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[AudioResult]:
        """
        Generate multiple audio files efficiently.
        
        Args:
            texts: List of texts to synthesize
            batch_size: Process N texts simultaneously
            **kwargs: Same parameters as tts()
        
        Returns:
            List[AudioResult]: List of generated audio results
        
        Example:
            >>> texts = [
            ...     "Welcome to our service.",
            ...     "Please hold while we connect you.",
            ...     "Thank you for your patience."
            ... ]
            >>> audios = model.tts_batch(texts, batch_size=2)
            >>> for i, audio in enumerate(audios):
            ...     audio.save(f"message_{i}.wav")
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            # Generate each text in the batch
            for text in batch_texts:
                try:
                    audio = self.tts(text, **kwargs)
                    batch_results.append(audio)
                except Exception as e:
                    logger.error(f"Failed to generate audio for text '{text[:50]}...': {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return results
    
    async def tts_batch_async(
        self,
        texts: List[str],
        max_concurrent: int = 8,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> List[AudioResult]:
        """
        Asynchronous batch processing.
        
        Args:
            texts: List of texts to synthesize
            max_concurrent: Maximum concurrent requests
            progress_callback: Progress update function
            **kwargs: Same parameters as tts()
        
        Returns:
            List[AudioResult]: List of generated audio results
        
        Example:
            >>> async def process_batch():
            ...     def progress(completed, total):
            ...         print(f"Progress: {completed}/{total}")
            ...     
            ...     audios = await model.tts_batch_async(
            ...         texts,
            ...         max_concurrent=4,
            ...         progress_callback=progress
            ...     )
            ...     return audios
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        total = len(texts)
        
        async def generate_one(text: str) -> Optional[AudioResult]:
            nonlocal completed
            async with semaphore:
                try:
                    result = await self.atts(text, **kwargs)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
                    return result
                except Exception as e:
                    logger.error(f"Failed to generate audio for text '{text[:50]}...': {e}")
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
                    return None
        
        # Create tasks for all texts
        tasks = [generate_one(text) for text in texts]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        return results
    
    def tts_stream(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
        buffer_size: int = 3,
        yield_metadata: bool = False,
        **kwargs
    ) -> Iterator[AudioChunk]:
        """
        Stream audio generation for long texts.
        
        Args:
            text: Input text
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            buffer_size: Audio buffer size
            yield_metadata: Include chunk metadata
            **kwargs: Same TTS parameters
        
        Yields:
            AudioChunk: Audio chunks as they're generated
        
        Example:
            >>> long_text = "This is a very long piece of text..."
            >>> 
            >>> # Stream and save
            >>> with open("streaming_output.wav", "wb") as f:
            ...     for chunk in model.tts_stream(long_text):
            ...         f.write(chunk.bytes())
            >>> 
            >>> # Stream and play real-time
            >>> for chunk in model.tts_stream(long_text):
            ...     chunk.play(blocking=False)
        """
        # Split text into chunks
        text_chunks = self._split_text_for_streaming(text, chunk_size, overlap)
        
        cumulative_time = 0.0
        
        for i, text_chunk in enumerate(text_chunks):
            try:
                # Generate audio for this chunk
                audio_result = self.tts(text_chunk, **kwargs)
                
                # Create audio chunk
                chunk = AudioChunk(
                    audio=audio_result.audio,
                    sample_rate=audio_result.sample_rate,
                    chunk_index=i,
                    is_final=(i == len(text_chunks) - 1),
                    text_segment=text_chunk,
                    start_time=cumulative_time
                )
                
                cumulative_time += chunk.duration
                
                if yield_metadata:
                    chunk.metadata = audio_result.metadata
                
                yield chunk
                
            except Exception as e:
                logger.error(f"Failed to generate chunk {i}: {e}")
                continue
    
    def get_voice_info(self) -> Dict[str, Any]:
        """
        Get information about this voice model.
        
        Returns:
            Dict[str, Any]: Voice model information
        
        Example:
            >>> model = eg.load("alice/uk-teen-female")
            >>> info = model.get_voice_info()
            >>> print(f"Sample rate: {info['sample_rate']}")
            >>> print(f"Supported languages: {info['supported_languages']}")
        """
        return {
            "name": self.name,
            "backend": self.backend.value if hasattr(self.backend, 'value') else str(self.backend),
            "sample_rate": self.sample_rate,
            "supports_emotions": self.supports_emotions,
            "supports_multilingual": self.supports_multilingual,
            "supported_languages": self.supported_languages,
            "max_text_length": self.max_text_length,
            "device": self.device,
            "model_info": self.model_info.__dict__ if self.model_info else None
        }
    
    def clone_voice(
        self,
        reference_audio: Union[str, AudioFile, List[Union[str, AudioFile]]],
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Clone a new voice from reference audio.
        
        Args:
            reference_audio: Reference audio file(s)
            name: Name for the cloned voice
            **kwargs: Additional cloning parameters
        
        Returns:
            VoiceModel: New cloned voice model
        
        Example:
            >>> model = eg.load("base/xtts-v2")
            >>> cloned = model.clone_voice(
            ...     reference_audio="my_voice.wav",
            ...     name="my-personal-voice"
            ... )
            >>> audio = cloned.tts("This is my cloned voice!")
        """
        from ..training.clone import clone_voice_from_model
        
        return clone_voice_from_model(
            base_model=self,
            reference_audio=reference_audio,
            name=name,
            **kwargs
        )
    
    def fine_tune(
        self,
        dataset,
        name: str,
        **kwargs
    ):
        """
        Fine-tune this model with new data.
        
        Args:
            dataset: Training dataset
            name: Name for fine-tuned model
            **kwargs: Training parameters
        
        Returns:
            VoiceModel: Fine-tuned model
        
        Example:
            >>> from echogrid.training import AudioDataset
            >>> 
            >>> dataset = AudioDataset("./training_audio/")
            >>> fine_tuned = model.fine_tune(
            ...     dataset=dataset,
            ...     name="my-fine-tuned-model",
            ...     epochs=50
            ... )
        """
        from ..training.fine_tune import fine_tune_model
        
        return fine_tune_model(
            base_model=self,
            training_data=dataset,
            name=name,
            **kwargs
        )
    
    def get_memory_usage(self) -> Optional[float]:
        """
        Get estimated memory usage in MB.
        
        Returns:
            float: Memory usage in MB or None if unknown
        """
        if hasattr(self._inference_engine, 'get_memory_usage'):
            return self._inference_engine.get_memory_usage()
        return None
    
    def unload(self):
        """
        Unload the model to free memory.
        
        Example:
            >>> model = eg.load("alice/uk-teen-female")
            >>> # Use model...
            >>> model.unload()  # Free memory
        """
        try:
            if hasattr(self._inference_engine, 'unload'):
                self._inference_engine.unload()
            logger.info(f"Unloaded model: {self.name}")
        except Exception as e:
            logger.warning(f"Error unloading model: {e}")
    
    def _split_text_for_streaming(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks for streaming."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 20% of the chunk
                search_start = max(start + int(chunk_size * 0.8), start + 1)
                for i in range(end - 1, search_start - 1, -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
                else:
                    # No sentence boundary found, try word boundaries
                    for i in range(end - 1, search_start - 1, -1):
                        if text[i].isspace():
                            end = i
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(end - overlap, start + 1)
            
            if start >= len(text):
                break
        
        return chunks
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"VoiceModel(name='{self.name}', backend='{self.backend}', device='{self.device}')"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"EchoGrid Voice Model: {self.name}"