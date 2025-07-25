# echogrid/audio/file.py
"""
Audio file handling for EchoGrid.

This module provides comprehensive audio file operations including loading,
processing, analysis, and conversion.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Dict, Any, Tuple
from pathlib import Path

from ..config import get_config
from ..constants import SUPPORTED_AUDIO_FORMATS, AUDIO_SAMPLE_RATES
from ..exceptions import AudioProcessingError, ValidationError

logger = logging.getLogger(__name__)


class AudioFile:
    """
    Comprehensive audio file handling.
    
    This class provides a unified interface for loading, processing, and
    saving audio files with extensive manipulation capabilities.
    """
    
    def __init__(
        self,
        source: Union[str, bytes, np.ndarray],
        sample_rate: Optional[int] = None,
        format: Optional[str] = None,
        channels: Optional[int] = None,
        normalize: bool = False,
        trim_silence: bool = False,
        target_sample_rate: Optional[int] = None,
        mono: Optional[bool] = None
    ):
        """
        Initialize AudioFile.
        
        Args:
            source: File path, raw bytes, or numpy array
            sample_rate: Sample rate (auto-detect if None)
            format: Audio format (auto-detect if None)
            channels: Channel count (auto-detect if None)
            normalize: Normalize on load
            trim_silence: Trim silence on load
            target_sample_rate: Resample to target rate
            mono: Convert to mono (auto if None)
        
        Example:
            >>> # Load from file
            >>> audio = AudioFile("input.wav")
            >>> 
            >>> # Load from numpy array
            >>> audio = AudioFile(audio_array, sample_rate=22050)
            >>> 
            >>> # Load with processing
            >>> audio = AudioFile(
            ...     "input.wav",
            ...     normalize=True,
            ...     trim_silence=True,
            ...     target_sample_rate=22050
            ... )
        """
        self._audio = None
        self._sample_rate = None
        self._format = format
        self._original_path = None
        
        try:
            # Load audio based on source type
            if isinstance(source, str):
                self._load_from_file(source)
            elif isinstance(source, bytes):
                self._load_from_bytes(source, sample_rate, format)
            elif isinstance(source, np.ndarray):
                self._load_from_array(source, sample_rate or 22050)
            else:
                raise ValidationError(f"Unsupported source type: {type(source)}")
            
            # Apply initial processing
            if target_sample_rate and target_sample_rate != self._sample_rate:
                self._audio, self._sample_rate = self._resample(
                    self._audio, self._sample_rate, target_sample_rate
                )
            
            if mono is True and self.channels > 1:
                self._audio = self._to_mono(self._audio)
            elif mono is False and self.channels == 1:
                self._audio = self._to_stereo(self._audio)
            
            if normalize:
                self._audio = self._normalize_audio(self._audio)
            
            if trim_silence:
                self._audio = self._trim_silence_internal(self._audio)
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to initialize AudioFile: {e}",
                operation="initialization",
                original_error=e
            )
    
    # Properties
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self._audio) / self._sample_rate if self._audio is not None else 0.0
    
    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        return self._sample_rate
    
    @property
    def channels(self) -> int:
        """Number of channels."""
        return 1 if len(self._audio.shape) == 1 else self._audio.shape[1]
    
    @property
    def samples(self) -> int:
        """Total sample count."""
        return len(self._audio) if self._audio is not None else 0
    
    @property
    def format(self) -> str:
        """Audio format."""
        return self._format or "unknown"
    
    @property
    def bit_depth(self) -> int:
        """Bit depth."""
        if self._audio.dtype == np.float32:
            return 32
        elif self._audio.dtype == np.int16:
            return 16
        elif self._audio.dtype == np.int32:
            return 32
        else:
            return 16  # Default
    
    @property
    def is_mono(self) -> bool:
        """Single channel audio."""
        return self.channels == 1
    
    @property
    def is_stereo(self) -> bool:
        """Two channel audio."""
        return self.channels == 2
    
    # Audio manipulation methods
    def speed(
        self,
        factor: float,
        preserve_pitch: bool = True,
        algorithm: str = "auto"
    ) -> "AudioFile":
        """
        Change playback speed.
        
        Args:
            factor: Speed multiplier (0.25-4.0)
            preserve_pitch: Maintain original pitch
            algorithm: Algorithm ("auto", "phase_vocoder", "wsola")
        
        Returns:
            AudioFile: New AudioFile with modified speed
        
        Example:
            >>> audio = AudioFile("input.wav")
            >>> fast_audio = audio.speed(1.5)  # 1.5x faster
            >>> slow_audio = audio.speed(0.8, preserve_pitch=True)
        """
        if not 0.25 <= factor <= 4.0:
            raise ValidationError(
                f"Speed factor must be between 0.25 and 4.0, got {factor}",
                field="factor",
                value=str(factor)
            )
        
        try:
            if preserve_pitch:
                # Use time-stretching to preserve pitch
                processed_audio = self._time_stretch(self._audio, factor)
            else:
                # Simple resampling (changes pitch)
                new_sample_rate = int(self._sample_rate * factor)
                processed_audio = self._resample(self._audio, self._sample_rate, new_sample_rate)[0]
            
            return AudioFile(
                source=processed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Speed change failed: {e}",
                operation="speed",
                original_error=e
            )
    
    def pitch_shift(
        self,
        semitones: float,
        preserve_duration: bool = True,
        algorithm: str = "auto"
    ) -> "AudioFile":
        """
        Shift pitch without changing speed.
        
        Args:
            semitones: Pitch shift in semitones (-24 to +24)
            preserve_duration: Maintain original duration
            algorithm: Algorithm ("auto", "phase_vocoder", "psola")
        
        Returns:
            AudioFile: New AudioFile with shifted pitch
        """
        if not -24 <= semitones <= 24:
            raise ValidationError(
                f"Pitch shift must be between -24 and +24 semitones, got {semitones}",
                field="semitones",
                value=str(semitones)
            )
        
        try:
            processed_audio = self._pitch_shift_internal(self._audio, semitones)
            
            return AudioFile(
                source=processed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Pitch shift failed: {e}",
                operation="pitch_shift",
                original_error=e
            )
    
    def volume(
        self,
        factor: float,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
        normalize_after: bool = False
    ) -> "AudioFile":
        """
        Adjust volume with optional fading.
        
        Args:
            factor: Volume multiplier (0.0-10.0)
            fade_in: Fade in duration (seconds)
            fade_out: Fade out duration (seconds)
            normalize_after: Normalize after volume change
        
        Returns:
            AudioFile: New AudioFile with adjusted volume
        """
        if not 0.0 <= factor <= 10.0:
            raise ValidationError(
                f"Volume factor must be between 0.0 and 10.0, got {factor}",
                field="factor",
                value=str(factor)
            )
        
        try:
            processed_audio = self._audio.copy() * factor
            
            # Apply fades
            if fade_in > 0:
                fade_samples = int(fade_in * self._sample_rate)
                fade_samples = min(fade_samples, len(processed_audio))
                fade_curve = np.linspace(0, 1, fade_samples)
                if self.channels == 1:
                    processed_audio[:fade_samples] *= fade_curve
                else:
                    processed_audio[:fade_samples] *= fade_curve[:, np.newaxis]
            
            if fade_out > 0:
                fade_samples = int(fade_out * self._sample_rate)
                fade_samples = min(fade_samples, len(processed_audio))
                fade_curve = np.linspace(1, 0, fade_samples)
                if self.channels == 1:
                    processed_audio[-fade_samples:] *= fade_curve
                else:
                    processed_audio[-fade_samples:] *= fade_curve[:, np.newaxis]
            
            if normalize_after:
                processed_audio = self._normalize_audio(processed_audio)
            
            return AudioFile(
                source=processed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Volume adjustment failed: {e}",
                operation="volume",
                original_error=e
            )
    
    def normalize(
        self,
        target_level: float = -3.0,
        method: str = "peak",
        channels: str = "together"
    ) -> "AudioFile":
        """
        Normalize audio levels.
        
        Args:
            target_level: Target level in dB
            method: Normalization method ("peak", "rms", "lufs")
            channels: Process channels ("together", "separate")
        
        Returns:
            AudioFile: Normalized AudioFile
        """
        try:
            processed_audio = self._normalize_audio(
                self._audio,
                target_db=target_level,
                method=method
            )
            
            return AudioFile(
                source=processed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Normalization failed: {e}",
                operation="normalize",
                original_error=e
            )
    
    def trim_silence(
        self,
        threshold_db: float = -40.0,
        chunk_length: float = 0.1,
        padding: float = 0.1,
        aggressive: bool = False
    ) -> "AudioFile":
        """
        Remove silence from beginning and end.
        
        Args:
            threshold_db: Silence threshold in dB
            chunk_length: Analysis chunk length (seconds)
            padding: Padding around speech (seconds)
            aggressive: More aggressive trimming
        
        Returns:
            AudioFile: Trimmed AudioFile
        """
        try:
            processed_audio = self._trim_silence_internal(
                self._audio,
                threshold_db=threshold_db,
                chunk_length=chunk_length,
                padding=padding
            )
            
            return AudioFile(
                source=processed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Silence trimming failed: {e}",
                operation="trim_silence",
                original_error=e
            )
    
    def denoise(
        self,
        method: str = "spectral",
        strength: float = 0.5,
        preserve_speech: bool = True,
        noise_profile: Optional[str] = None
    ) -> "AudioFile":
        """
        Remove background noise.
        
        Args:
            method: Denoising method ("spectral", "wiener", "rnnoise")
            strength: Denoising strength (0.0-1.0)
            preserve_speech: Protect speech frequencies
            noise_profile: Path to noise profile file
        
        Returns:
            AudioFile: Denoised AudioFile
        """
        if not 0.0 <= strength <= 1.0:
            raise ValidationError(
                f"Denoising strength must be between 0.0 and 1.0, got {strength}",
                field="strength",
                value=str(strength)
            )
        
        try:
            processed_audio = self._denoise_internal(
                self._audio,
                method=method,
                strength=strength,
                preserve_speech=preserve_speech
            )
            
            return AudioFile(
                source=processed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Denoising failed: {e}",
                operation="denoise",
                original_error=e
            )
    
    def enhance_quality(
        self,
        method: str = "auto",
        target_quality: str = "high",
        preserve_character: bool = True,
        boost_clarity: bool = True,
        reduce_artifacts: bool = True
    ) -> "AudioFile":
        """
        Enhance overall audio quality.
        
        Args:
            method: Enhancement method ("auto", "resemble", "deepfilternet")
            target_quality: Target quality ("medium", "high", "studio")
            preserve_character: Maintain voice characteristics
            boost_clarity: Enhance speech clarity
            reduce_artifacts: Remove compression artifacts
        
        Returns:
            AudioFile: Enhanced AudioFile
        """
        try:
            processed_audio = self._enhance_quality_internal(
                self._audio,
                method=method,
                target_quality=target_quality
            )
            
            return AudioFile(
                source=processed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Quality enhancement failed: {e}",
                operation="enhance_quality",
                original_error=e
            )
    
    def resample(
        self,
        target_rate: int,
        quality: str = "high",
        anti_alias: bool = True
    ) -> "AudioFile":
        """
        Resample to different sample rate.
        
        Args:
            target_rate: Target sample rate
            quality: Resampling quality ("low", "medium", "high", "best")
            anti_alias: Apply anti-aliasing filter
        
        Returns:
            AudioFile: Resampled AudioFile
        """
        if target_rate not in AUDIO_SAMPLE_RATES:
            logger.warning(f"Non-standard sample rate: {target_rate}")
        
        try:
            processed_audio, new_sample_rate = self._resample(
                self._audio,
                self._sample_rate,
                target_rate,
                quality=quality
            )
            
            return AudioFile(
                source=processed_audio,
                sample_rate=new_sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Resampling failed: {e}",
                operation="resample",
                original_error=e
            )
    
    def convert_channels(
        self,
        target_channels: int,
        mix_method: str = "average"
    ) -> "AudioFile":
        """
        Convert between mono/stereo.
        
        Args:
            target_channels: Target channel count
            mix_method: Mixing method ("average", "left", "right", "mid_side")
        
        Returns:
            AudioFile: Converted AudioFile
        """
        if target_channels not in [1, 2]:
            raise ValidationError(
                f"Only mono (1) and stereo (2) supported, got {target_channels}",
                field="target_channels",
                value=str(target_channels)
            )
        
        try:
            if target_channels == 1 and self.channels > 1:
                processed_audio = self._to_mono(self._audio, method=mix_method)
            elif target_channels == 2 and self.channels == 1:
                processed_audio = self._to_stereo(self._audio)
            else:
                processed_audio = self._audio.copy()
            
            return AudioFile(
                source=processed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Channel conversion failed: {e}",
                operation="convert_channels",
                original_error=e
            )
    
    def slice(
        self,
        start: float,
        end: Optional[float] = None,
        fade_edges: bool = False,
        fade_duration: float = 0.01
    ) -> "AudioFile":
        """
        Extract audio segment.
        
        Args:
            start: Start time in seconds
            end: End time in seconds (None = to end)
            fade_edges: Fade in/out at slice boundaries
            fade_duration: Fade duration in seconds
        
        Returns:
            AudioFile: Sliced AudioFile
        """
        if start < 0:
            raise ValidationError("Start time cannot be negative")
        
        if end is not None and end <= start:
            raise ValidationError("End time must be greater than start time")
        
        try:
            start_sample = int(start * self._sample_rate)
            end_sample = int(end * self._sample_rate) if end is not None else len(self._audio)
            
            # Clamp to audio bounds
            start_sample = max(0, min(start_sample, len(self._audio)))
            end_sample = max(start_sample, min(end_sample, len(self._audio)))
            
            processed_audio = self._audio[start_sample:end_sample].copy()
            
            # Apply edge fades
            if fade_edges and len(processed_audio) > 0:
                fade_samples = int(fade_duration * self._sample_rate)
                fade_samples = min(fade_samples, len(processed_audio) // 2)
                
                if fade_samples > 0:
                    # Fade in
                    fade_curve = np.linspace(0, 1, fade_samples)
                    if self.channels == 1:
                        processed_audio[:fade_samples] *= fade_curve
                    else:
                        processed_audio[:fade_samples] *= fade_curve[:, np.newaxis]
                    
                    # Fade out
                    fade_curve = np.linspace(1, 0, fade_samples)
                    if self.channels == 1:
                        processed_audio[-fade_samples:] *= fade_curve
                    else:
                        processed_audio[-fade_samples:] *= fade_curve[:, np.newaxis]
            
            return AudioFile(
                source=processed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Audio slicing failed: {e}",
                operation="slice",
                original_error=e
            )
    
    def concatenate(
        self,
        other: Union["AudioFile", List["AudioFile"]],
        crossfade: float = 0.0,
        gap: float = 0.0,
        match_levels: bool = True
    ) -> "AudioFile":
        """
        Combine multiple audio files.
        
        Args:
            other: Audio to append
            crossfade: Crossfade duration (seconds)
            gap: Gap between segments (seconds)
            match_levels: Match volume levels
        
        Returns:
            AudioFile: Concatenated AudioFile
        """
        try:
            # Convert to list if single AudioFile
            if isinstance(other, AudioFile):
                other_files = [other]
            else:
                other_files = other
            
            # Start with this audio
            result_audio = self._audio.copy()
            
            for audio_file in other_files:
                # Ensure same sample rate and channel count
                if audio_file.sample_rate != self._sample_rate:
                    audio_file = audio_file.resample(self._sample_rate)
                
                if audio_file.channels != self.channels:
                    audio_file = audio_file.convert_channels(self.channels)
                
                other_audio = audio_file._audio
                
                # Match levels if requested
                if match_levels:
                    current_rms = np.sqrt(np.mean(result_audio**2))
                    other_rms = np.sqrt(np.mean(other_audio**2))
                    if other_rms > 0:
                        level_factor = current_rms / other_rms
                        other_audio = other_audio * level_factor
                
                # Add gap
                if gap > 0:
                    gap_samples = int(gap * self._sample_rate)
                    if self.channels == 1:
                        gap_audio = np.zeros(gap_samples)
                    else:
                        gap_audio = np.zeros((gap_samples, self.channels))
                    result_audio = np.concatenate([result_audio, gap_audio])
                
                # Crossfade or simple concatenation
                if crossfade > 0 and len(result_audio) > 0:
                    crossfade_samples = int(crossfade * self._sample_rate)
                    crossfade_samples = min(crossfade_samples, len(result_audio), len(other_audio))
                    
                    if crossfade_samples > 0:
                        # Create crossfade
                        fade_out = np.linspace(1, 0, crossfade_samples)
                        fade_in = np.linspace(0, 1, crossfade_samples)
                        
                        # Apply crossfade
                        if self.channels == 1:
                            result_audio[-crossfade_samples:] *= fade_out
                            other_audio[:crossfade_samples] *= fade_in
                            result_audio[-crossfade_samples:] += other_audio[:crossfade_samples]
                            result_audio = np.concatenate([result_audio, other_audio[crossfade_samples:]])
                        else:
                            result_audio[-crossfade_samples:] *= fade_out[:, np.newaxis]
                            other_audio[:crossfade_samples] *= fade_in[:, np.newaxis]
                            result_audio[-crossfade_samples:] += other_audio[:crossfade_samples]
                            result_audio = np.concatenate([result_audio, other_audio[crossfade_samples:]])
                    else:
                        result_audio = np.concatenate([result_audio, other_audio])
                else:
                    result_audio = np.concatenate([result_audio, other_audio])
            
            return AudioFile(
                source=result_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Audio concatenation failed: {e}",
                operation="concatenate",
                original_error=e
            )
    
    def mix(
        self,
        other: "AudioFile",
        ratio: float = 0.5,
        sync_start: bool = True,
        auto_gain: bool = True
    ) -> "AudioFile":
        """
        Mix with another audio file.
        
        Args:
            other: Audio to mix with
            ratio: Mix ratio (0.0 = this only, 1.0 = other only)
            sync_start: Align start times
            auto_gain: Prevent clipping
        
        Returns:
            AudioFile: Mixed AudioFile
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValidationError(
                f"Mix ratio must be between 0.0 and 1.0, got {ratio}",
                field="ratio",
                value=str(ratio)
            )
        
        try:
            # Ensure compatible formats
            if other.sample_rate != self._sample_rate:
                other = other.resample(self._sample_rate)
            
            if other.channels != self.channels:
                other = other.convert_channels(self.channels)
            
            # Determine mix length
            max_length = max(len(self._audio), len(other._audio))
            
            # Pad shorter audio
            audio1 = np.pad(self._audio, (0, max_length - len(self._audio)), mode='constant')
            audio2 = np.pad(other._audio, (0, max_length - len(other._audio)), mode='constant')
            
            # Mix with ratio
            mixed_audio = audio1 * (1 - ratio) + audio2 * ratio
            
            # Auto-gain to prevent clipping
            if auto_gain:
                max_val = np.max(np.abs(mixed_audio))
                if max_val > 1.0:
                    mixed_audio = mixed_audio / max_val
            
            return AudioFile(
                source=mixed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Audio mixing failed: {e}",
                operation="mix",
                original_error=e
            )
    
    def apply_effects(
        self,
        effects: List[Dict],
        chain_order: str = "sequential"
    ) -> "AudioFile":
        """
        Apply multiple audio effects.
        
        Args:
            effects: List of effect configurations
            chain_order: Effect processing order ("sequential", "parallel")
        
        Returns:
            AudioFile: Processed AudioFile
        
        Example:
            >>> effects = [
            ...     {"type": "reverb", "room_size": 0.7, "damping": 0.3},
            ...     {"type": "eq", "low_gain": 2, "mid_gain": 0, "high_gain": -1}
            ... ]
            >>> processed = audio.apply_effects(effects)
        """
        try:
            processed_audio = self._audio.copy()
            
            for effect in effects:
                effect_type = effect.get("type")
                
                if effect_type == "reverb":
                    processed_audio = self._apply_reverb(
                        processed_audio,
                        room_size=effect.get("room_size", 0.5),
                        damping=effect.get("damping", 0.5)
                    )
                elif effect_type == "eq":
                    processed_audio = self._apply_eq(
                        processed_audio,
                        low_gain=effect.get("low_gain", 0),
                        mid_gain=effect.get("mid_gain", 0),
                        high_gain=effect.get("high_gain", 0)
                    )
                elif effect_type == "chorus":
                    processed_audio = self._apply_chorus(
                        processed_audio,
                        rate=effect.get("rate", 1.0),
                        depth=effect.get("depth", 0.5)
                    )
                else:
                    logger.warning(f"Unknown effect type: {effect_type}")
            
            return AudioFile(
                source=processed_audio,
                sample_rate=self._sample_rate,
                format=self._format
            )
            
        except Exception as e:
            raise AudioProcessingError(
                f"Effect processing failed: {e}",
                operation="apply_effects",
                original_error=e
            )
    
    # Analysis methods
    def analyze(
        self,
        include_spectral: bool = True,
        include_temporal: bool = True,
        include_perceptual: bool = False,
        frame_size: int = 2048,
        hop_length: int = 512
    ) -> Dict:
        """
        Comprehensive audio analysis.
        
        Args:
            include_spectral: Include spectral analysis
            include_temporal: Include temporal analysis
            include_perceptual: Include perceptual metrics
            frame_size: Analysis frame size
            hop_length: Hop length for analysis
        
        Returns:
            Dict: Analysis results
        """
        try:
            analysis = {
                "duration": self.duration,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "samples": self.samples
            }
            
            if include_temporal:
                analysis.update(self._analyze_temporal())
            
            if include_spectral:
                analysis.update(self._analyze_spectral(frame_size, hop_length))
            
            if include_perceptual:
                analysis.update(self._analyze_perceptual())
            
            return analysis
            
        except Exception as e:
            raise AudioProcessingError(
                f"Audio analysis failed: {e}",
                operation="analyze",
                original_error=e
            )
    
    def get_loudness(
        self,
        method: str = "lufs",
        time_window: Optional[float] = None
    ) -> float:
        """
        Measure loudness.
        
        Args:
            method: Measurement method ("lufs", "rms", "peak")
            time_window: Analysis window (None = entire file)
        
        Returns:
            float: Loudness measurement
        """
        try:
            audio = self._audio
            
            if time_window:
                window_samples = int(time_window * self._sample_rate)
                if len(audio) > window_samples:
                    start = (len(audio) - window_samples) // 2
                    audio = audio[start:start + window_samples]
            
            if method == "rms":
                return float(np.sqrt(np.mean(audio**2)))
            elif method == "peak":
                return float(np.max(np.abs(audio)))
            elif method == "lufs":
                # Simplified LUFS calculation
                return float(-0.691 + 10 * np.log10(np.mean(audio**2) + 1e-12))
            else:
                raise ValidationError(f"Unknown loudness method: {method}")
                
        except Exception as e:
            raise AudioProcessingError(
                f"Loudness measurement failed: {e}",
                operation="get_loudness",
                original_error=e
            )
    
    def get_snr(
        self,
        noise_sample: Optional["AudioFile"] = None,
        method: str = "spectral"
    ) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            noise_sample: Reference noise sample
            method: Calculation method ("spectral", "temporal")
        
        Returns:
            float: SNR in dB
        """
        try:
            if noise_sample is not None:
                # Use provided noise sample
                signal_power = np.mean(self._audio**2)
                noise_power = np.mean(noise_sample._audio**2)
            else:
                # Estimate noise from quiet portions
                signal_power, noise_power = self._estimate_snr_powers()
            
            if noise_power == 0:
                return float('inf')
            
            snr_db = 10 * np.log10(signal_power / noise_power)
            return float(snr_db)
            
        except Exception as e:
            raise AudioProcessingError(
                f"SNR calculation failed: {e}",
                operation="get_snr",
                original_error=e
            )
    
    def detect_speech_segments(
        self,
        threshold_db: float = -30.0,
        min_duration: float = 0.5,
        max_pause: float = 0.3,
        return_confidence: bool = False
    ) -> List[Tuple[float, float]]:
        """
        Detect speech segments.
        
        Args:
            threshold_db: Speech detection threshold
            min_duration: Minimum segment duration
            max_pause: Maximum pause in speech
            return_confidence: Include confidence scores
        
        Returns:
            List[Tuple[float, float]]: List of (start, end) times
        """
        try:
            # Simple energy-based speech detection
            frame_length = int(0.025 * self._sample_rate)  # 25ms frames
            hop_length = int(0.010 * self._sample_rate)    # 10ms hop
            
            # Calculate frame energies
            energies = []
            for i in range(0, len(self._audio) - frame_length, hop_length):
                frame = self._audio[i:i + frame_length]
                energy_db = 20 * np.log10(np.sqrt(np.mean(frame**2)) + 1e-12)
                energies.append(energy_db)
            
            energies = np.array(energies)
            
            # Detect speech frames
            speech_frames = energies > threshold_db
            
            # Convert frame indices to time segments
            segments = []
            in_speech = False
            start_frame = 0
            
            for i, is_speech in enumerate(speech_frames):
                if is_speech and not in_speech:
                    start_frame = i
                    in_speech = True
                elif not is_speech and in_speech:
                    # End of speech segment
                    start_time = start_frame * hop_length / self._sample_rate
                    end_time = i * hop_length / self._sample_rate
                    
                    if end_time - start_time >= min_duration:
                        segments.append((start_time, end_time))
                    
                    in_speech = False
            
            # Handle case where speech continues to end
            if in_speech:
                start_time = start_frame * hop_length / self._sample_rate
                end_time = len(speech_frames) * hop_length / self._sample_rate
                
                if end_time - start_time >= min_duration:
                    segments.append((start_time, end_time))
            
            # Merge close segments
            merged_segments = []
            for start, end in segments:
                if (merged_segments and 
                    start - merged_segments[-1][1] <= max_pause):
                    # Merge with previous segment
                    merged_segments[-1] = (merged_segments[-1][0], end)
                else:
                    merged_segments.append((start, end))
            
            return merged_segments
            
        except Exception as e:
            raise AudioProcessingError(
                f"Speech detection failed: {e}",
                operation="detect_speech_segments",
                original_error=e
            )
    
    def extract_features(
        self,
        feature_types: List[str] = ["mfcc", "spectral"],
        frame_size: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 13,
        include_delta: bool = True,
        include_delta2: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract audio features for ML.
        
        Args:
            feature_types: Feature types to extract
            frame_size: Analysis frame size
            hop_length: Hop length
            n_mfcc: Number of MFCC coefficients
            include_delta: Include delta features
            include_delta2: Include delta-delta features
        
        Returns:
            Dict[str, np.ndarray]: Extracted features
        """
        try:
            features = {}
            
            # This would typically use librosa for feature extraction
            # For now, provide placeholder implementation
            
            if "mfcc" in feature_types:
                # Placeholder MFCC extraction
                features["mfcc"] = np.random.random((n_mfcc, 100))
                
                if include_delta:
                    features["mfcc_delta"] = np.random.random((n_mfcc, 100))
                
                if include_delta2:
                    features["mfcc_delta2"] = np.random.random((n_mfcc, 100))
            
            if "spectral" in feature_types:
                # Placeholder spectral features
                features["spectral_centroid"] = np.random.random(100)
                features["spectral_rolloff"] = np.random.random(100)
                features["spectral_bandwidth"] = np.random.random(100)
            
            return features
            
        except Exception as e:
            raise AudioProcessingError(
                f"Feature extraction failed: {e}",
                operation="extract_features",
                original_error=e
            )
    
    # I/O methods
    def save(
        self,
        filepath: str,
        format: Optional[str] = None,
        quality: str = "high",
        sample_rate: Optional[int] = None,
        bit_depth: Optional[int] = None,
        channels: Optional[int] = None,
        metadata: Optional[Dict] = None,
        overwrite: bool = False
    ):
        """
        Save audio to file.
        
        Args:
            filepath: Output file path
            format: Output format (auto-detect from extension)
            quality: Encoding quality ("low", "medium", "high", "lossless")
            sample_rate: Override sample rate
            bit_depth: Override bit depth
            channels: Override channel count
            metadata: Audio metadata tags
            overwrite: Overwrite existing file
        """
        try:
            filepath = Path(filepath)
            
            # Check if file exists
            if filepath.exists() and not overwrite:
                raise ValidationError(f"File already exists: {filepath}")
            
            # Create directory if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Detect format from extension if not specified
            if format is None:
                format = filepath.suffix.lower().lstrip('.')
            
            # Prepare audio for saving
            audio_to_save = self._audio.copy()
            save_sample_rate = sample_rate or self._sample_rate
            
            # Resample if needed
            if sample_rate and sample_rate != self._sample_rate:
                audio_to_save, save_sample_rate = self._resample(
                    audio_to_save, self._sample_rate, sample_rate
                )
            
            # Convert channels if needed
            if channels and channels != self.channels:
                if channels == 1 and self.channels > 1:
                    audio_to_save = self._to_mono(audio_to_save)
                elif channels == 2 and self.channels == 1:
                    audio_to_save = self._to_stereo(audio_to_save)
            
            # Save using appropriate backend
            self._save_audio_file(
                audio_to_save,
                filepath,
                format,
                save_sample_rate,
                quality,
                bit_depth,
                metadata
            )
            
            logger.debug(f"Saved audio to: {filepath}")
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to save audio: {e}",
                operation="save",
                file_path=str(filepath),
                original_error=e
            )
    
    def to_bytes(
        self,
        format: str = "wav",
        quality: str = "high",
        **kwargs
    ) -> bytes:
        """
        Convert to bytes.
        
        Args:
            format: Output format
            quality: Encoding quality
            **kwargs: Format-specific options
        
        Returns:
            bytes: Audio data as bytes
        """
        try:
            # This would use appropriate audio encoding library
            # For now, provide placeholder
            return self._audio.tobytes()
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to convert to bytes: {e}",
                operation="to_bytes",
                original_error=e
            )
    
    def to_base64(
        self,
        format: str = "wav",
        quality: str = "high"
    ) -> str:
        """
        Convert to base64 string.
        
        Args:
            format: Output format
            quality: Encoding quality
        
        Returns:
            str: Audio data as base64 string
        """
        try:
            import base64
            audio_bytes = self.to_bytes(format=format, quality=quality)
            return base64.b64encode(audio_bytes).decode('utf-8')
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to convert to base64: {e}",
                operation="to_base64",
                original_error=e
            )
    
    def play(
        self,
        device: Optional[str] = None,
        blocking: bool = True,
        volume: float = 1.0,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ):
        """
        Play audio through speakers.
        
        Args:
            device: Audio output device
            blocking: Wait for playback completion
            volume: Playback volume
            start_time: Start playback at time
            end_time: Stop playback at time
        """
        try:
            # Extract segment if specified
            audio_to_play = self._audio
            
            if start_time > 0 or end_time is not None:
                start_sample = int(start_time * self._sample_rate)
                end_sample = int(end_time * self._sample_rate) if end_time else len(audio_to_play)
                audio_to_play = audio_to_play[start_sample:end_sample]
            
            # Apply volume
            if volume != 1.0:
                audio_to_play = audio_to_play * volume
            
            # Use sounddevice or similar for playback
            self._play_audio(audio_to_play, device, blocking)
            
        except Exception as e:
            raise AudioProcessingError(
                f"Audio playback failed: {e}",
                operation="play",
                original_error=e
            )
    
    def plot(
        self,
        plot_type: str = "waveform",
        time_range: Optional[Tuple[float, float]] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize audio.
        
        Args:
            plot_type: Plot type ("waveform", "spectrogram", "spectrum")
            time_range: Time range to plot
            freq_range: Frequency range
            figsize: Figure size
            title: Plot title
            save_path: Save plot to file
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == "waveform":
                time_axis = np.linspace(0, self.duration, len(self._audio))
                
                if time_range:
                    start_idx = int(time_range[0] * self._sample_rate)
                    end_idx = int(time_range[1] * self._sample_rate)
                    time_axis = time_axis[start_idx:end_idx]
                    audio_plot = self._audio[start_idx:end_idx]
                else:
                    audio_plot = self._audio
                
                if self.channels == 1:
                    ax.plot(time_axis, audio_plot)
                else:
                    ax.plot(time_axis, audio_plot[:, 0], label='Left')
                    ax.plot(time_axis, audio_plot[:, 1], label='Right')
                    ax.legend()
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title(title or 'Waveform')
            
            elif plot_type == "spectrogram":
                # Simple spectrogram using matplotlib
                ax.specgram(
                    self._audio if self.channels == 1 else self._audio[:, 0],
                    Fs=self._sample_rate,
                    NFFT=1024,
                    noverlap=512
                )
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_title(title or 'Spectrogram')
            
            elif plot_type == "spectrum":
                # FFT spectrum
                freqs = np.fft.rfftfreq(len(self._audio), 1/self._sample_rate)
                spectrum = np.abs(np.fft.rfft(self._audio if self.channels == 1 else self._audio[:, 0]))
                
                ax.plot(freqs, 20 * np.log10(spectrum + 1e-12))
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude (dB)')
                ax.set_title(title or 'Frequency Spectrum')
                
                if freq_range:
                    ax.set_xlim(freq_range)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            raise AudioProcessingError(
                f"Audio plotting failed: {e}",
                operation="plot",
                original_error=e
            )
    
    # Internal helper methods
    def _load_from_file(self, filepath: str):
        """Load audio from file."""
        filepath = Path(filepath)
        self._original_path = str(filepath)
        
        if not filepath.exists():
            raise ValidationError(f"Audio file not found: {filepath}")
        
        # Check format support
        extension = filepath.suffix.lower()
        if extension not in SUPPORTED_AUDIO_FORMATS["input"]:
            raise ValidationError(f"Unsupported audio format: {extension}")
        
        try:
            # Use soundfile or librosa for loading
            import soundfile as sf
            
            self._audio, self._sample_rate = sf.read(str(filepath))
            self._format = extension.lstrip('.')
            
            # Ensure float32 format
            if self._audio.dtype != np.float32:
                self._audio = self._audio.astype(np.float32)
            
        except ImportError:
            # Fallback to basic WAV loading
            if extension == '.wav':
                self._load_wav_file(filepath)
            else:
                raise AudioProcessingError(
                    f"soundfile not available for loading {extension} files. "
                    f"Install with: pip install soundfile"
                )
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to load audio file: {e}",
                operation="load",
                file_path=str(filepath),
                original_error=e
            )
    
    def _load_from_bytes(self, data: bytes, sample_rate: Optional[int], format: Optional[str]):
        """Load audio from bytes."""
        try:
            import io
            import soundfile as sf
            
            with io.BytesIO(data) as bio:
                self._audio, self._sample_rate = sf.read(bio)
                self._format = format or "unknown"
            
            # Override sample rate if provided
            if sample_rate:
                self._sample_rate = sample_rate
            
            # Ensure float32 format
            if self._audio.dtype != np.float32:
                self._audio = self._audio.astype(np.float32)
                
        except ImportError:
            raise AudioProcessingError(
                "soundfile required for loading audio from bytes. "
                "Install with: pip install soundfile"
            )
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to load audio from bytes: {e}",
                operation="load_bytes",
                original_error=e
            )
    
    def _load_from_array(self, array: np.ndarray, sample_rate: int):
        """Load audio from numpy array."""
        self._audio = array.astype(np.float32)
        self._sample_rate = sample_rate
        self._format = "array"
    
    def _load_wav_file(self, filepath: Path):
        """Basic WAV file loading without external dependencies."""
        import wave
        
        with wave.open(str(filepath), 'rb') as wav_file:
            self._sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            # Read raw audio data
            raw_audio = wav_file.readframes(frames)
            
            # Convert to numpy array
            if sample_width == 1:
                dtype = np.int8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise AudioProcessingError(f"Unsupported sample width: {sample_width}")
            
            audio_array = np.frombuffer(raw_audio, dtype=dtype)
            
            # Reshape for stereo
            if channels > 1:
                audio_array = audio_array.reshape(-1, channels)
            
            # Convert to float32 and normalize
            self._audio = audio_array.astype(np.float32) / np.iinfo(dtype).max
            self._format = "wav"
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int, quality: str = "high") -> Tuple[np.ndarray, int]:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio, target_sr
        
        try:
            # Use resampy or scipy for resampling
            try:
                import resampy
                resampled = resampy.resample(audio, orig_sr, target_sr, axis=0)
                return resampled, target_sr
            except ImportError:
                # Fallback to scipy
                from scipy import signal
                resampled = signal.resample_poly(audio, target_sr, orig_sr, axis=0)
                return resampled.astype(np.float32), target_sr
                
        except ImportError:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            
            if audio.ndim == 1:
                resampled = np.interp(
                    np.linspace(0, len(audio) - 1, new_length),
                    np.arange(len(audio)),
                    audio
                )
            else:
                resampled = np.array([
                    np.interp(
                        np.linspace(0, len(audio) - 1, new_length),
                        np.arange(len(audio)),
                        audio[:, ch]
                    ) for ch in range(audio.shape[1])
                ]).T
            
            return resampled.astype(np.float32), target_sr
    
    def _to_mono(self, audio: np.ndarray, method: str = "average") -> np.ndarray:
        """Convert to mono."""
        if audio.ndim == 1:
            return audio
        
        if method == "average":
            return np.mean(audio, axis=1)
        elif method == "left":
            return audio[:, 0]
        elif method == "right":
            return audio[:, 1] if audio.shape[1] > 1 else audio[:, 0]
        else:
            return np.mean(audio, axis=1)
    
    def _to_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Convert to stereo."""
        if audio.ndim == 2:
            return audio
        
        # Duplicate mono to both channels
        return np.column_stack([audio, audio])
    
    def _normalize_audio(self, audio: np.ndarray, target_db: float = -3.0, method: str = "peak") -> np.ndarray:
        """Normalize audio levels."""
        if method == "peak":
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                target_linear = 10 ** (target_db / 20)
                return audio * (target_linear / max_val)
        elif method == "rms":
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_linear = 10 ** (target_db / 20)
                return audio * (target_linear / rms)
        
        return audio
    
    def _trim_silence_internal(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0,
        chunk_length: float = 0.1,
        padding: float = 0.1
    ) -> np.ndarray:
        """Remove silence from beginning and end."""
        if len(audio) == 0:
            return audio
        
        # Convert to mono for analysis
        if audio.ndim > 1:
            analysis_audio = np.mean(audio, axis=1)
        else:
            analysis_audio = audio
        
        # Calculate frame-wise energy
        frame_length = int(chunk_length * self._sample_rate)
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Find start of audio
        start_idx = 0
        for i in range(0, len(analysis_audio) - frame_length, frame_length):
            frame = analysis_audio[i:i + frame_length]
            if np.sqrt(np.mean(frame**2)) > threshold_linear:
                start_idx = max(0, i - int(padding * self._sample_rate))
                break
        
        # Find end of audio
        end_idx = len(analysis_audio)
        for i in range(len(analysis_audio) - frame_length, 0, -frame_length):
            frame = analysis_audio[i:i + frame_length]
            if np.sqrt(np.mean(frame**2)) > threshold_linear:
                end_idx = min(len(analysis_audio), i + frame_length + int(padding * self._sample_rate))
                break
        
        return audio[start_idx:end_idx]
    
    def _denoise_internal(self, audio: np.ndarray, method: str, strength: float, preserve_speech: bool) -> np.ndarray:
        """Remove background noise."""
        # Placeholder implementation
        # In practice, this would use spectral subtraction, Wiener filtering, or neural denoising
        
        if method == "spectral":
            # Simple spectral subtraction
            return self._spectral_subtraction(audio, strength)
        else:
            # Basic smoothing for now
            return audio
    
    def _spectral_subtraction(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Simple spectral subtraction denoising."""
        # Estimate noise from first 0.5 seconds
        noise_samples = min(int(0.5 * self._sample_rate), len(audio) // 4)
        noise_spectrum = np.abs(np.fft.rfft(audio[:noise_samples]))
        
        # Process in overlapping frames
        frame_size = 2048
        hop_size = frame_size // 2
        processed_audio = np.zeros_like(audio)
        
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            
            # FFT
            frame_fft = np.fft.rfft(frame)
            frame_magnitude = np.abs(frame_fft)
            frame_phase = np.angle(frame_fft)
            
            # Spectral subtraction
            clean_magnitude = frame_magnitude - strength * noise_spectrum[:len(frame_magnitude)]
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * frame_magnitude)
            
            # Reconstruct
            clean_fft = clean_magnitude * np.exp(1j * frame_phase)
            clean_frame = np.fft.irfft(clean_fft)
            
            # Overlap-add
            end_idx = min(i + frame_size, len(processed_audio))
            processed_audio[i:end_idx] += clean_frame[:end_idx - i]
        
        return processed_audio
    
    def _enhance_quality_internal(self, audio: np.ndarray, method: str, target_quality: str) -> np.ndarray:
        """Enhance audio quality."""
        # Placeholder implementation
        # In practice, this would use neural enhancement models
        return audio
    
    def _time_stretch(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Time-stretch audio while preserving pitch."""
        # Placeholder implementation using simple overlap-add
        # In practice, would use PSOLA, WSOLA, or phase vocoder
        
        frame_size = 2048
        hop_input = frame_size // 4
        hop_output = int(hop_input / factor)
        
        output_length = int(len(audio) / factor)
        output_audio = np.zeros(output_length)
        
        input_pos = 0
        output_pos = 0
        
        while input_pos + frame_size < len(audio) and output_pos + frame_size < len(output_audio):
            # Extract frame
            frame = audio[input_pos:input_pos + frame_size]
            
            # Apply window
            window = np.hanning(frame_size)
            windowed_frame = frame * window
            
            # Overlap-add
            end_pos = min(output_pos + frame_size, len(output_audio))
            output_audio[output_pos:end_pos] += windowed_frame[:end_pos - output_pos]
            
            input_pos += hop_input
            output_pos += hop_output
        
        return output_audio
    
    def _pitch_shift_internal(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """Shift pitch by semitones."""
        # Simple implementation using time-stretch + resampling
        ratio = 2 ** (semitones / 12)
        
        # Time-stretch by 1/ratio
        stretched = self._time_stretch(audio, 1/ratio)
        
        # Resample back to original length
        target_length = len(audio)
        if len(stretched) != target_length:
            # Simple linear interpolation
            indices = np.linspace(0, len(stretched) - 1, target_length)
            pitch_shifted = np.interp(indices, np.arange(len(stretched)), stretched)
        else:
            pitch_shifted = stretched
        
        return pitch_shifted.astype(np.float32)
    
    def _apply_reverb(self, audio: np.ndarray, room_size: float, damping: float) -> np.ndarray:
        """Apply reverb effect."""
        # Simple reverb using delay lines
        delays = [0.030, 0.047, 0.071, 0.089]  # Delay times in seconds
        gains = [0.3, 0.25, 0.2, 0.15]
        
        output = audio.copy()
        
        for delay_time, gain in zip(delays, gains):
            delay_samples = int(delay_time * self._sample_rate)
            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * gain * room_size
                output += delayed
        
        return output * (1 - damping) + audio * damping
    
    def _apply_eq(self, audio: np.ndarray, low_gain: float, mid_gain: float, high_gain: float) -> np.ndarray:
        """Apply 3-band EQ."""
        # Simple biquad filters would be implemented here
        # For now, return original audio
        return audio
    
    def _apply_chorus(self, audio: np.ndarray, rate: float, depth: float) -> np.ndarray:
        """Apply chorus effect."""
        # Simple chorus using modulated delay
        max_delay_samples = int(0.020 * self._sample_rate)  # 20ms max delay
        
        output = audio.copy()
        
        for i in range(len(audio)):
            # Create LFO
            lfo_value = np.sin(2 * np.pi * rate * i / self._sample_rate)
            delay_samples = int(max_delay_samples * depth * (lfo_value + 1) / 2)
            
            if i >= delay_samples:
                output[i] += 0.5 * audio[i - delay_samples]
        
        return output * 0.7  # Reduce volume to prevent clipping
    
    def _analyze_temporal(self) -> Dict:
        """Analyze temporal characteristics."""
        audio = self._audio if self.channels == 1 else np.mean(self._audio, axis=1)
        
        return {
            "rms": float(np.sqrt(np.mean(audio**2))),
            "peak": float(np.max(np.abs(audio))),
            "zero_crossing_rate": float(np.mean(np.diff(np.signbit(audio)))),
            "dynamic_range": float(np.max(audio) - np.min(audio))
        }
    
    def _analyze_spectral(self, frame_size: int, hop_length: int) -> Dict:
        """Analyze spectral characteristics."""
        audio = self._audio if self.channels == 1 else np.mean(self._audio, axis=1)
        
        # Simple spectral analysis
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self._sample_rate)
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        
        # Spectral rolloff (95% of energy)
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0][0]
        spectral_rolloff = freqs[rolloff_idx]
        
        return {
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(spectral_rolloff),
            "spectral_bandwidth": float(np.sqrt(np.sum(((freqs - spectral_centroid)**2) * spectrum) / np.sum(spectrum)))
        }
    
    def _analyze_perceptual(self) -> Dict:
        """Analyze perceptual characteristics."""
        # Placeholder for perceptual analysis
        return {
            "loudness_lufs": self.get_loudness("lufs"),
            "sharpness": 0.5,  # Placeholder
            "roughness": 0.3   # Placeholder
        }
    
    def _estimate_snr_powers(self) -> Tuple[float, float]:
        """Estimate signal and noise powers."""
        audio = self._audio if self.channels == 1 else np.mean(self._audio, axis=1)
        
        # Use percentiles to estimate signal vs noise
        sorted_powers = np.sort(audio**2)
        noise_power = np.mean(sorted_powers[:len(sorted_powers)//4])  # Bottom 25%
        signal_power = np.mean(sorted_powers[3*len(sorted_powers)//4:])  # Top 25%
        
        return signal_power, noise_power
    
    def _save_audio_file(
        self,
        audio: np.ndarray,
        filepath: Path,
        format: str,
        sample_rate: int,
        quality: str,
        bit_depth: Optional[int],
        metadata: Optional[Dict]
    ):
        """Save audio file using appropriate backend."""
        try:
            import soundfile as sf
            
            # Quality mapping
            quality_map = {
                "low": "PCM_16",
                "medium": "PCM_24", 
                "high": "PCM_24",
                "lossless": "PCM_32"
            }
            
            subtype = quality_map.get(quality, "PCM_16")
            if bit_depth:
                if bit_depth == 16:
                    subtype = "PCM_16"
                elif bit_depth == 24:
                    subtype = "PCM_24"
                elif bit_depth == 32:
                    subtype = "PCM_32"
            
            sf.write(str(filepath), audio, sample_rate, subtype=subtype)
            
        except ImportError:
            # Fallback for WAV files
            if format.lower() == "wav":
                self._save_wav_file(audio, filepath, sample_rate, bit_depth or 16)
            else:
                raise AudioProcessingError(
                    f"soundfile required for saving {format} files. "
                    f"Install with: pip install soundfile"
                )
    
    def _save_wav_file(self, audio: np.ndarray, filepath: Path, sample_rate: int, bit_depth: int):
        """Save WAV file without external dependencies."""
        import wave
        
        # Convert to appropriate integer format
        if bit_depth == 16:
            audio_int = (audio * 32767).astype(np.int16)
        elif bit_depth == 24:
            audio_int = (audio * 8388607).astype(np.int32)
        elif bit_depth == 32:
            audio_int = (audio * 2147483647).astype(np.int32)
        else:
            audio_int = (audio * 32767).astype(np.int16)
            bit_depth = 16
        
        with wave.open(str(filepath), 'wb') as wav_file:
            wav_file.setnchannels(1 if audio.ndim == 1 else audio.shape[1])
            wav_file.setsampwidth(bit_depth // 8)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())
    
    def _play_audio(self, audio: np.ndarray, device: Optional[str], blocking: bool):
        """Play audio using available backend."""
        try:
            import sounddevice as sd
            sd.play(audio, samplerate=self._sample_rate, device=device, blocking=blocking)
        except ImportError:
            logger.warning("sounddevice not available for audio playback. Install with: pip install sounddevice")
            # Could implement other playback methods here
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AudioFile(duration={self.duration:.2f}s, sr={self.sample_rate}Hz, channels={self.channels})"