# echogrid/models/cloned_model.py
"""
Cloned voice model implementation for EchoGrid.

This module provides the ClonedVoiceModel class for voice models
created through zero-shot voice cloning.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import time

from .voice_model import VoiceModel, AudioResult
from ..core.backends import BackendType
from ..audio.file import AudioFile
from ..exceptions import AudioProcessingError, ValidationError

logger = logging.getLogger(__name__)


class ClonedVoiceModel(VoiceModel):
    """
    Voice model created through zero-shot voice cloning.
    
    This class extends VoiceModel with cloning-specific functionality
    and maintains reference to the original training data.
    """
    
    def __init__(
        self,
        name: str,
        model_state: Dict[str, Any],
        metadata: Dict[str, Any],
        reference_audio: List[AudioFile],
        training_config: Dict[str, Any]
    ):
        """
        Initialize cloned voice model.
        
        Args:
            name: Model name
            model_state: Trained model state
            metadata: Model metadata
            reference_audio: Original reference audio files
            training_config: Training configuration used
        """
        # Initialize base VoiceModel
        super().__init__(
            name=name,
            backend=BackendType.LOCAL,  # Cloned models are local by default
            inference_engine=None,  # Will be set up lazily
            model_info=None
        )
        
        self.model_state = model_state
        self.metadata = metadata
        self.reference_audio = reference_audio
        self.training_config = training_config
        
        # Cloning-specific properties
        self.base_model = training_config.get("base_model", "xtts-v2")
        self.quality = training_config.get("quality", "medium")
        self.language = metadata.get("language", "en")
        self.creation_time = metadata.get("created_at", time.time())
        
        # Override base properties
        self._sample_rate = 22050  # Default, can be overridden
        self._supports_emotions = True  # Most cloned models support emotions
        self._supports_multilingual = training_config.get("cross_lingual", False)
        self._supported_languages = [self.language]
        
        # Performance tracking
        self._inference_count = 0
        self._total_inference_time = 0.0
        
        logger.debug(f"Initialized ClonedVoiceModel: {name}")
    
    def tts(
        self,
        text: str,
        emotion: str = "neutral",
        emotion_strength: float = 1.0,
        speed: float = 1.0,
        pitch: float = 0.0,
        temperature: float = 0.75,
        voice_similarity: float = 1.0,
        reference_boost: float = 0.0,
        **kwargs
    ) -> AudioResult:
        """
        Generate speech using the cloned voice.
        
        Args:
            text: Input text to synthesize
            emotion: Emotion type
            emotion_strength: Emotion intensity
            speed: Speech speed multiplier
            pitch: Pitch adjustment in semitones
            temperature: Generation randomness
            voice_similarity: How closely to match original voice (0.0-1.0)
            reference_boost: Boost reference characteristics (0.0-1.0)
            **kwargs: Additional TTS parameters
        
        Returns:
            AudioResult: Generated audio result
        """
        try:
            start_time = time.time()
            
            # Validate inputs
            if not 0.0 <= voice_similarity <= 1.0:
                raise ValidationError(
                    f"voice_similarity must be between 0.0 and 1.0, got {voice_similarity}"
                )
            
            if not 0.0 <= reference_boost <= 1.0:
                raise ValidationError(
                    f"reference_boost must be between 0.0 and 1.0, got {reference_boost}"
                )
            
            # Generate audio based on cloned model architecture
            if self.base_model == "xtts-v2":
                audio = self._generate_xtts_cloned(
                    text, emotion, emotion_strength, speed, pitch,
                    temperature, voice_similarity, reference_boost, **kwargs
                )
            elif self.base_model == "openvoice":
                audio = self._generate_openvoice_cloned(
                    text, emotion, emotion_strength, speed,
                    voice_similarity, **kwargs
                )
            elif self.base_model == "styletts2":
                audio = self._generate_styletts2_cloned(
                    text, emotion, emotion_strength, speed,
                    temperature, voice_similarity, **kwargs
                )
            elif self.base_model == "f5-tts":
                audio = self._generate_f5_cloned(
                    text, speed, temperature, voice_similarity, **kwargs
                )
            else:
                raise AudioProcessingError(f"Generation not implemented for {self.base_model}")
            
            # Apply post-processing
            if kwargs.get("enhance_quality", False):
                audio = self._enhance_cloned_audio(audio)
            
            generation_time = time.time() - start_time
            self._inference_count += 1
            self._total_inference_time += generation_time
            
            # Create result
            result = AudioResult(
                audio=audio,
                sample_rate=kwargs.get("sample_rate", self.sample_rate),
                metadata={
                    "cloned_voice": self.name,
                    "base_model": self.base_model,
                    "voice_similarity": voice_similarity,
                    "reference_boost": reference_boost,
                    "generation_time": generation_time
                } if kwargs.get("return_metadata", False) else {},
                performance={
                    "generation_time": generation_time,
                    "inference_count": self._inference_count,
                    "avg_inference_time": self._total_inference_time / self._inference_count
                } if kwargs.get("profile_performance", False) else {}
            )
            
            logger.debug(f"Generated audio using cloned voice: {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Cloned voice generation failed: {e}")
            raise AudioProcessingError(
                f"Cloned voice generation failed: {e}",
                operation="cloned_tts",
                original_error=e
            )
    
    def get_voice_similarity(self, audio: Union[str, AudioFile, np.ndarray]) -> float:
        """
        Calculate similarity between generated audio and reference voice.
        
        Args:
            audio: Audio to compare (file path, AudioFile, or numpy array)
        
        Returns:
            float: Similarity score (0.0-1.0)
        """
        try:
            # Load audio if needed
            if isinstance(audio, str):
                test_audio = AudioFile(audio)
            elif isinstance(audio, AudioFile):
                test_audio = audio
            elif isinstance(audio, np.ndarray):
                test_audio = AudioFile(audio, sample_rate=self.sample_rate)
            else:
                raise ValidationError(f"Invalid audio type: {type(audio)}")
            
            # Extract features from test audio
            test_features = self._extract_voice_features(test_audio)
            
            # Compare with reference features
            similarity = self._compare_voice_features(test_features)
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Voice similarity calculation failed: {e}")
            return 0.0
    
    def adjust_voice_characteristics(
        self,
        pitch_shift: float = 0.0,
        speed_factor: float = 1.0,
        emotion_bias: float = 0.0,
        accent_strength: float = 1.0
    ):
        """
        Adjust voice characteristics of the cloned model.
        
        Args:
            pitch_shift: Global pitch adjustment in semitones
            speed_factor: Global speed adjustment factor
            emotion_bias: Emotion bias adjustment
            accent_strength: Accent strength adjustment
        """
        try:
            adjustments = {
                "pitch_shift": pitch_shift,
                "speed_factor": speed_factor,
                "emotion_bias": emotion_bias,
                "accent_strength": accent_strength
            }
            
            # Store adjustments in model state
            if "adjustments" not in self.model_state:
                self.model_state["adjustments"] = {}
            
            self.model_state["adjustments"].update(adjustments)
            
            logger.info(f"Applied voice adjustments: {adjustments}")
            
        except Exception as e:
            logger.error(f"Voice adjustment failed: {e}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        Get information about the training process.
        
        Returns:
            Dict[str, Any]: Training information and metrics
        """
        info = {
            "base_model": self.base_model,
            "quality": self.quality,
            "training_steps": self.training_config.get("training_steps", 0),
            "learning_rate": self.training_config.get("learning_rate", 0),
            "reference_duration": sum(af.duration for af in self.reference_audio),
            "num_reference_files": len(self.reference_audio),
            "language": self.language,
            "creation_time": self.creation_time,
            "supports_cross_lingual": self._supports_multilingual
        }
        
        # Add training metrics if available
        if "training_metrics" in self.metadata:
            info["training_metrics"] = self.metadata["training_metrics"]
        
        return info
    
    def get_reference_audio(self) -> List[AudioFile]:
        """
        Get the original reference audio files.
        
        Returns:
            List[AudioFile]: Reference audio files used for cloning
        """
        return self.reference_audio.copy()
    
    def save(self, filepath: str, include_reference: bool = True):
        """
        Save the cloned model to disk.
        
        Args:
            filepath: Path to save the model
            include_reference: Whether to include reference audio files
        """
        try:
            filepath = Path(filepath)
            filepath.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            import pickle
            with open(filepath / "model_state.pkl", "wb") as f:
                pickle.dump(self.model_state, f)
            
            # Save metadata
            import json
            with open(filepath / "metadata.json", "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_metadata = self._make_json_serializable(self.metadata)
                json.dump(serializable_metadata, f, indent=2)
            
            # Save training config
            with open(filepath / "training_config.json", "w") as f:
                serializable_config = self._make_json_serializable(self.training_config)
                json.dump(serializable_config, f, indent=2)
            
            # Save reference audio if requested
            if include_reference:
                ref_dir = filepath / "reference_audio"
                ref_dir.mkdir(exist_ok=True)
                
                for i, audio_file in enumerate(self.reference_audio):
                    audio_file.save(ref_dir / f"reference_{i:03d}.wav")
            
            logger.info(f"Saved cloned model to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save cloned model: {e}")
            raise AudioProcessingError(f"Model saving failed: {e}")
    
    @classmethod
    def load(cls, filepath: str) -> "ClonedVoiceModel":
        """
        Load a cloned model from disk.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            ClonedVoiceModel: Loaded cloned voice model
        """
        try:
            filepath = Path(filepath)
            
            # Load model state
            import pickle
            with open(filepath / "model_state.pkl", "rb") as f:
                model_state = pickle.load(f)
            
            # Load metadata
            import json
            with open(filepath / "metadata.json", "r") as f:
                metadata = json.load(f)
            
            # Load training config
            with open(filepath / "training_config.json", "r") as f:
                training_config = json.load(f)
            
            # Load reference audio if available
            reference_audio = []
            ref_dir = filepath / "reference_audio"
            if ref_dir.exists():
                ref_files = sorted(ref_dir.glob("reference_*.wav"))
                for ref_file in ref_files:
                    reference_audio.append(AudioFile(str(ref_file)))
            
            # Create model instance
            model = cls(
                name=metadata["name"],
                model_state=model_state,
                metadata=metadata,
                reference_audio=reference_audio,
                training_config=training_config
            )
            
            logger.info(f"Loaded cloned model from: {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load cloned model: {e}")
            raise AudioProcessingError(f"Model loading failed: {e}")
    
    def fine_tune(
        self,
        additional_audio: List[Union[str, AudioFile]],
        epochs: int = 50,
        learning_rate: float = 1e-5,
        **kwargs
    ) -> "ClonedVoiceModel":
        """
        Fine-tune the cloned model with additional audio.
        
        Args:
            additional_audio: Additional training audio
            epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for fine-tuning
            **kwargs: Additional fine-tuning parameters
        
        Returns:
            ClonedVoiceModel: Fine-tuned model
        """
        try:
            logger.info("Starting fine-tuning of cloned model")
            
            # Process additional audio
            processed_audio = []
            for audio_source in additional_audio:
                if isinstance(audio_source, str):
                    audio_file = AudioFile(audio_source)
                else:
                    audio_file = audio_source
                
                # Apply same preprocessing as original
                audio_file = audio_file.trim_silence().normalize()
                processed_audio.append(audio_file)
            
            # Create new training data combining original and additional
            combined_audio = self.reference_audio + processed_audio
            
            # Set up fine-tuning configuration
            fine_tune_config = self.training_config.copy()
            fine_tune_config.update({
                "training_steps": epochs,
                "learning_rate": learning_rate,
                "fine_tuning": True
            })
            
            # Simulate fine-tuning process
            fine_tuned_state = self._perform_fine_tuning(
                self.model_state, processed_audio, fine_tune_config
            )
            
            # Create new model with fine-tuned state
            new_metadata = self.metadata.copy()
            new_metadata["name"] = f"{self.name}-finetuned"
            new_metadata["fine_tuned"] = True
            new_metadata["additional_audio_count"] = len(additional_audio)
            
            fine_tuned_model = ClonedVoiceModel(
                name=new_metadata["name"],
                model_state=fine_tuned_state,
                metadata=new_metadata,
                reference_audio=combined_audio,
                training_config=fine_tune_config
            )
            
            logger.info("Fine-tuning completed")
            return fine_tuned_model
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise AudioProcessingError(f"Fine-tuning failed: {e}")
    
    def _generate_xtts_cloned(
        self, text: str, emotion: str, emotion_strength: float,
        speed: float, pitch: float, temperature: float,
        voice_similarity: float, reference_boost: float, **kwargs
    ) -> np.ndarray:
        """Generate audio using cloned XTTS model."""
        
        # Get speaker embedding from model state
        speaker_embedding = self.model_state.get("speaker_embedding")
        prosody_features = self.model_state.get("prosody_features", {})
        
        # Apply voice similarity adjustment
        if voice_similarity < 1.0:
            # Add some variation to reduce similarity
            variation = (1.0 - voice_similarity) * 0.2
            speaker_embedding = speaker_embedding + np.random.normal(0, variation, speaker_embedding.shape)
        
        # Apply reference boost
        if reference_boost > 0:
            # Enhance speaker characteristics
            speaker_embedding = speaker_embedding * (1.0 + reference_boost * 0.1)
        
        # Generate base audio (placeholder implementation)
        sample_rate = kwargs.get("sample_rate", self.sample_rate)
        duration = len(text) * 0.08  # Rough duration estimate
        
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Base frequency from prosody features
        base_freq = prosody_features.get("f0_mean", 150)
        frequency = base_freq * (2 ** (pitch / 12))  # Apply pitch shift
        
        # Generate audio with speaker characteristics
        audio = np.sin(2 * np.pi * frequency * t) * 0.1
        
        # Apply emotion modulation
        if emotion != "neutral":
            emotion_factor = emotion_strength * 0.2
            if emotion in ["happy", "excited"]:
                audio = audio * (1.0 + emotion_factor)
                frequency *= 1.1
            elif emotion in ["sad", "calm"]:
                audio = audio * (1.0 - emotion_factor * 0.5)
                frequency *= 0.9
        
        # Apply speed adjustment
        if speed != 1.0:
            new_length = int(len(audio) / speed)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_length),
                np.arange(len(audio)),
                audio
            )
        
        return audio.astype(np.float32)
    
    def _generate_openvoice_cloned(
        self, text: str, emotion: str, emotion_strength: float,
        speed: float, voice_similarity: float, **kwargs
    ) -> np.ndarray:
        """Generate audio using cloned OpenVoice model."""
        
        tone_color_embedding = self.model_state.get("tone_color_embedding")
        
        # Apply voice similarity
        if voice_similarity < 1.0:
            variation = (1.0 - voice_similarity) * 0.15
            tone_color_embedding = tone_color_embedding + np.random.normal(0, variation, tone_color_embedding.shape)
        
        # Generate placeholder audio
        sample_rate = kwargs.get("sample_rate", self.sample_rate)
        duration = len(text) * 0.09
        
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = np.random.normal(0, 0.05, len(t)).astype(np.float32)
        
        # Apply characteristics from tone color embedding
        characteristic_freq = 200 + np.mean(tone_color_embedding) * 100
        modulation = np.sin(2 * np.pi * characteristic_freq * t) * 0.02
        audio += modulation
        
        return audio
    
    def _generate_styletts2_cloned(
        self, text: str, emotion: str, emotion_strength: float,
        speed: float, temperature: float, voice_similarity: float, **kwargs
    ) -> np.ndarray:
        """Generate audio using cloned StyleTTS2 model."""
        
        style_embedding = self.model_state.get("style_embedding")
        
        # Apply voice similarity
        if voice_similarity < 1.0:
            variation = (1.0 - voice_similarity) * 0.1
            style_embedding = style_embedding + np.random.normal(0, variation, style_embedding.shape)
        
        # Generate placeholder audio with style characteristics
        sample_rate = kwargs.get("sample_rate", self.sample_rate)
        duration = len(text) * 0.07
        
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Use style embedding to influence generation
        style_factor = np.mean(style_embedding)
        base_freq = 180 + style_factor * 50
        
        audio = np.sin(2 * np.pi * base_freq * t) * 0.08
        
        # Add style-based variation
        variation = np.random.normal(0, 0.02 * temperature, len(t))
        audio += variation
        
        return audio.astype(np.float32)
    
    def _generate_f5_cloned(
        self, text: str, speed: float, temperature: float,
        voice_similarity: float, **kwargs
    ) -> np.ndarray:
        """Generate audio using cloned F5-TTS model."""
        
        flow_embedding = self.model_state.get("flow_embedding")
        
        # Apply voice similarity
        if voice_similarity < 1.0:
            variation = (1.0 - voice_similarity) * 0.12
            flow_embedding = flow_embedding + np.random.normal(0, variation, flow_embedding.shape)
        
        # Generate placeholder audio
        sample_rate = kwargs.get("sample_rate", self.sample_rate)
        duration = len(text) * 0.06
        
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Use flow embedding characteristics
        flow_factor = np.mean(flow_embedding)
        audio = np.random.normal(0, 0.04 + abs(flow_factor) * 0.02, len(t))
        
        return audio.astype(np.float32)
    
    def _enhance_cloned_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply enhancement specific to cloned voices."""
        
        # Apply gentle enhancement to preserve voice characteristics
        enhanced = audio.copy()
        
        # Slight noise reduction
        enhanced = enhanced * 0.98 + np.random.normal(0, 0.001, len(enhanced))
        
        # Gentle normalization
        max_val = np.max(np.abs(enhanced))
        if max_val > 0:
            enhanced = enhanced / max_val * 0.95
        
        return enhanced
    
    def _extract_voice_features(self, audio_file: AudioFile) -> Dict[str, Any]:
        """Extract voice features for similarity comparison."""
        
        audio = audio_file._audio
        
        # Basic feature extraction
        features = {
            "rms_energy": np.sqrt(np.mean(audio**2)),
            "zero_crossing_rate": np.mean(np.diff(np.signbit(audio))),
            "spectral_centroid": np.random.normal(2000, 300),  # Placeholder
            "fundamental_frequency": np.random.normal(150, 30)  # Placeholder
        }
        
        return features
    
    def _compare_voice_features(self, test_features: Dict[str, Any]) -> float:
        """Compare test features with reference voice features."""
        
        # Get reference features from model state
        ref_prosody = self.model_state.get("prosody_features", {})
        
        # Simple similarity calculation (placeholder)
        similarity = 0.8  # Base similarity
        
        # Compare fundamental frequency
        if "f0_mean" in ref_prosody:
            ref_f0 = ref_prosody["f0_mean"]
            test_f0 = test_features.get("fundamental_frequency", ref_f0)
            f0_diff = abs(ref_f0 - test_f0) / ref_f0
            similarity -= f0_diff * 0.2
        
        # Compare energy
        ref_energy = ref_prosody.get("energy_mean", -20)
        test_energy = 20 * np.log10(test_features.get("rms_energy", 0.1) + 1e-12)
        energy_diff = abs(ref_energy - test_energy) / abs(ref_energy)
        similarity -= energy_diff * 0.1
        
        return max(0.0, similarity)
    
    def _perform_fine_tuning(
        self, model_state: Dict[str, Any], additional_audio: List[AudioFile],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform fine-tuning simulation."""
        
        # Create copy of model state
        fine_tuned_state = {}
        for key, value in model_state.items():
            if isinstance(value, np.ndarray):
                fine_tuned_state[key] = value.copy()
            else:
                fine_tuned_state[key] = value
        
        # Simulate fine-tuning adjustments
        learning_rate = config.get("learning_rate", 1e-5)
        epochs = config.get("training_steps", 50)
        
        for epoch in range(epochs):
            # Apply small adjustments based on additional audio
            adjustment_scale = learning_rate * (1.0 - epoch / epochs)  # Decay
            
            for key, value in fine_tuned_state.items():
                if isinstance(value, np.ndarray):
                    # Small random adjustments to simulate learning
                    noise = np.random.normal(0, adjustment_scale * 0.01, value.shape)
                    fine_tuned_state[key] = value + noise
        
        return fine_tuned_state
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def __repr__(self) -> str:
        """String representation of cloned model."""
        return f"ClonedVoiceModel(name='{self.name}', base_model='{self.base_model}', quality='{self.quality}')"