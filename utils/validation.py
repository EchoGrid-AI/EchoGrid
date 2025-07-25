# echogrid/utils/validation.py
"""
Validation utilities for EchoGrid.

This module provides validation functions for various inputs and parameters.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

from ..constants import (
    SUPPORTED_LANGUAGES, SUPPORTED_EMOTIONS, VOICE_GENDERS, 
    AGE_RANGES, VOICE_CATEGORIES, MAX_TEXT_LENGTH, MIN_AUDIO_DURATION
)
from ..exceptions import ValidationError

logger = logging.getLogger(__name__)


def validate_model_name(name: str) -> bool:
    """
    Validate model name format.
    
    Args:
        name: Model name to validate
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid format
    """
    if not name:
        raise ValidationError("Model name cannot be empty")
    
    if "/" not in name:
        raise ValidationError(
            "Model name must be in 'username/model-name' format",
            field="model_name",
            value=name
        )
    
    parts = name.split("/")
    if len(parts) != 2:
        raise ValidationError(
            "Model name must have exactly one '/' separator",
            field="model_name", 
            value=name
        )
    
    username, model_name = parts
    
    # Validate username
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        raise ValidationError(
            "Username can only contain letters, numbers, hyphens, and underscores",
            field="username",
            value=username
        )
    
    if len(username) < 2 or len(username) > 39:
        raise ValidationError(
            "Username must be between 2 and 39 characters",
            field="username",
            value=username
        )
    
    # Validate model name
    if not re.match(r'^[a-zA-Z0-9_.-]+$', model_name):
        raise ValidationError(
            "Model name can only contain letters, numbers, hyphens, underscores, and dots",
            field="model_name",
            value=model_name
        )
    
    if len(model_name) < 1 or len(model_name) > 100:
        raise ValidationError(
            "Model name must be between 1 and 100 characters",
            field="model_name",
            value=model_name
        )
    
    return True


def validate_text_input(text: str, max_length: int = MAX_TEXT_LENGTH) -> bool:
    """
    Validate text input for TTS generation.
    
    Args:
        text: Input text to validate
        max_length: Maximum allowed length
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid text
    """
    if not text:
        raise ValidationError("Text cannot be empty")
    
    if not isinstance(text, str):
        raise ValidationError(
            f"Text must be a string, got {type(text).__name__}",
            field="text",
            value=str(text)
        )
    
    if len(text) > max_length:
        raise ValidationError(
            f"Text too long: {len(text)} characters (max: {max_length})",
            field="text",
            constraint=f"Maximum {max_length} characters"
        )
    
    # Check for potentially problematic characters
    if '\x00' in text:
        raise ValidationError(
            "Text contains null bytes",
            field="text"
        )
    
    # Warn about very long sentences
    sentences = text.split('.')
    max_sentence_length = max(len(s.strip()) for s in sentences if s.strip())
    if max_sentence_length > 500:
        logger.warning(f"Very long sentence detected: {max_sentence_length} characters")
    
    return True


def validate_audio_parameters(
    speed: float = 1.0,
    pitch: float = 0.0,
    volume: float = 1.0,
    temperature: float = 0.75,
    emotion: str = "neutral",
    language: str = "auto",
    supported_languages: Optional[List[str]] = None,
    supports_emotions: bool = True,
    **kwargs
) -> bool:
    """
    Validate audio generation parameters.
    
    Args:
        speed: Speech speed multiplier
        pitch: Pitch shift in semitones
        volume: Volume multiplier
        temperature: Generation temperature
        emotion: Emotion type
        language: Language code
        supported_languages: List of supported languages
        supports_emotions: Whether emotions are supported
        **kwargs: Additional parameters
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid parameters
    """
    # Validate speed
    if not 0.25 <= speed <= 4.0:
        raise ValidationError(
            f"Speed must be between 0.25 and 4.0, got {speed}",
            field="speed",
            value=str(speed),
            constraint="0.25 ≤ speed ≤ 4.0"
        )
    
    # Validate pitch
    if not -24.0 <= pitch <= 24.0:
        raise ValidationError(
            f"Pitch must be between -24 and +24 semitones, got {pitch}",
            field="pitch",
            value=str(pitch),
            constraint="-24 ≤ pitch ≤ +24 semitones"
        )
    
    # Validate volume
    if not 0.0 <= volume <= 10.0:
        raise ValidationError(
            f"Volume must be between 0.0 and 10.0, got {volume}",
            field="volume",
            value=str(volume),
            constraint="0.0 ≤ volume ≤ 10.0"
        )
    
    # Validate temperature
    if not 0.1 <= temperature <= 2.0:
        raise ValidationError(
            f"Temperature must be between 0.1 and 2.0, got {temperature}",
            field="temperature",
            value=str(temperature),
            constraint="0.1 ≤ temperature ≤ 2.0"
        )
    
    # Validate emotion
    if emotion not in SUPPORTED_EMOTIONS:
        if supports_emotions:
            raise ValidationError(
                f"Unsupported emotion: {emotion}",
                field="emotion",
                value=emotion,
                constraint=f"Must be one of: {', '.join(SUPPORTED_EMOTIONS)}"
            )
        else:
            logger.warning(f"Model does not support emotions, ignoring: {emotion}")
    
    # Validate language
    if language != "auto":
        # Check against global supported languages
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Language {language} not in global supported list")
        
        # Check against model-specific supported languages
        if supported_languages and language not in supported_languages:
            raise ValidationError(
                f"Language {language} not supported by this model",
                field="language",
                value=language,
                constraint=f"Must be one of: {', '.join(supported_languages)}"
            )
    
    return True


def validate_audio_quality(audio_file) -> List[str]:
    """
    Validate audio quality for voice cloning.
    
    Args:
        audio_file: AudioFile instance to validate
    
    Returns:
        List[str]: List of quality issues found
    """
    issues = []
    
    try:
        # Check duration
        if audio_file.duration < MIN_AUDIO_DURATION:
            issues.append(f"Audio too short: {audio_file.duration:.1f}s (minimum: {MIN_AUDIO_DURATION}s)")
        
        # Check sample rate
        if audio_file.sample_rate < 16000:
            issues.append(f"Low sample rate: {audio_file.sample_rate}Hz (recommended: ≥22050Hz)")
        
        # Check if audio is too quiet
        rms = np.sqrt(np.mean(audio_file._audio**2))
        if rms < 0.01:
            issues.append("Audio is very quiet (low RMS energy)")
        
        # Check for clipping
        max_amplitude = np.max(np.abs(audio_file._audio))
        if max_amplitude > 0.95:
            issues.append("Audio may be clipped (amplitude > 0.95)")
        
        # Check for silence
        silence_threshold = 0.001
        silent_ratio = np.mean(np.abs(audio_file._audio) < silence_threshold)
        if silent_ratio > 0.5:
            issues.append(f"Audio contains too much silence ({silent_ratio*100:.1f}%)")
        
        # Check for mono/stereo
        if audio_file.channels > 2:
            issues.append(f"Unsupported channel count: {audio_file.channels} (use mono or stereo)")
        
        # Estimate SNR
        try:
            snr = estimate_snr(audio_file._audio)
            if snr < 15:
                issues.append(f"Low signal-to-noise ratio: {snr:.1f}dB (recommended: >20dB)")
        except Exception:
            issues.append("Could not estimate signal-to-noise ratio")
        
    except Exception as e:
        issues.append(f"Audio analysis failed: {e}")
    
    return issues


def validate_model_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate model metadata.
    
    Args:
        metadata: Model metadata dictionary
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid metadata
    """
    required_fields = ["name", "description", "language"]
    
    for field in required_fields:
        if field not in metadata:
            raise ValidationError(f"Missing required field: {field}")
        
        if not metadata[field]:
            raise ValidationError(f"Field cannot be empty: {field}")
    
    # Validate specific fields
    if "language" in metadata:
        language = metadata["language"]
        if language not in SUPPORTED_LANGUAGES and language != "auto":
            raise ValidationError(
                f"Unsupported language: {language}",
                field="language",
                value=language
            )
    
    if "gender" in metadata:
        gender = metadata["gender"]
        if gender and gender not in VOICE_GENDERS:
            raise ValidationError(
                f"Invalid gender: {gender}",
                field="gender",
                value=gender,
                constraint=f"Must be one of: {', '.join(VOICE_GENDERS)}"
            )
    
    if "age_range" in metadata:
        age_range = metadata["age_range"]
        if age_range and age_range not in AGE_RANGES:
            raise ValidationError(
                f"Invalid age range: {age_range}",
                field="age_range",
                value=age_range,
                constraint=f"Must be one of: {', '.join(AGE_RANGES)}"
            )
    
    if "category" in metadata:
        category = metadata["category"]
        if category and category not in VOICE_CATEGORIES:
            raise ValidationError(
                f"Invalid category: {category}",
                field="category",
                value=category,
                constraint=f"Must be one of: {', '.join(VOICE_CATEGORIES)}"
            )
    
    if "tags" in metadata:
        tags = metadata["tags"]
        if not isinstance(tags, list):
            raise ValidationError("Tags must be a list")
        
        for tag in tags:
            if not isinstance(tag, str):
                raise ValidationError("All tags must be strings")
            
            if len(tag) > 50:
                raise ValidationError(f"Tag too long: '{tag}' (max: 50 characters)")
            
            if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                raise ValidationError(f"Invalid tag format: '{tag}' (use only letters, numbers, hyphens, underscores)")
    
    return True


def validate_training_parameters(
    learning_rate: float,
    batch_size: int,
    epochs: int,
    **kwargs
) -> bool:
    """
    Validate training parameters.
    
    Args:
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
        **kwargs: Additional parameters
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid parameters
    """
    # Validate learning rate
    if not 1e-6 <= learning_rate <= 1e-1:
        raise ValidationError(
            f"Learning rate must be between 1e-6 and 1e-1, got {learning_rate}",
            field="learning_rate",
            value=str(learning_rate)
        )
    
    # Validate batch size
    if not 1 <= batch_size <= 128:
        raise ValidationError(
            f"Batch size must be between 1 and 128, got {batch_size}",
            field="batch_size",
            value=str(batch_size)
        )
    
    # Validate epochs
    if not 1 <= epochs <= 10000:
        raise ValidationError(
            f"Epochs must be between 1 and 10000, got {epochs}",
            field="epochs",
            value=str(epochs)
        )
    
    # Validate additional parameters
    if "weight_decay" in kwargs:
        weight_decay = kwargs["weight_decay"]
        if not 0.0 <= weight_decay <= 1.0:
            raise ValidationError(
                f"Weight decay must be between 0.0 and 1.0, got {weight_decay}",
                field="weight_decay",
                value=str(weight_decay)
            )
    
    if "temperature" in kwargs:
        temperature = kwargs["temperature"]
        if not 0.1 <= temperature <= 2.0:
            raise ValidationError(
                f"Temperature must be between 0.1 and 2.0, got {temperature}",
                field="temperature",
                value=str(temperature)
            )
    
    return True


def validate_device_string(device: str) -> bool:
    """
    Validate device string format.
    
    Args:
        device: Device string (e.g., "cpu", "cuda", "cuda:0", "mps")
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid device format
    """
    valid_devices = ["auto", "cpu", "mps"]
    
    if device in valid_devices:
        return True
    
    # Check CUDA format
    if device.startswith("cuda"):
        if device == "cuda":
            return True
        
        # Check cuda:N format
        if re.match(r'^cuda:\d+$', device):
            return True
        
        raise ValidationError(
            f"Invalid CUDA device format: {device}",
            field="device",
            value=device,
            constraint="Use 'cuda' or 'cuda:N' where N is device index"
        )
    
    raise ValidationError(
        f"Unsupported device: {device}",
        field="device",
        value=device,
        constraint=f"Must be one of: {', '.join(valid_devices)} or cuda[:N]"
    )


def validate_precision_string(precision: str) -> bool:
    """
    Validate precision string.
    
    Args:
        precision: Precision string (e.g., "fp32", "fp16", "int8")
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid precision
    """
    valid_precisions = ["auto", "fp32", "fp16", "bf16", "int8", "int4"]
    
    if precision not in valid_precisions:
        raise ValidationError(
            f"Invalid precision: {precision}",
            field="precision",
            value=precision,
            constraint=f"Must be one of: {', '.join(valid_precisions)}"
        )
    
    return True


def validate_memory_string(memory_str: str) -> bool:
    """
    Validate memory limit string.
    
    Args:
        memory_str: Memory string (e.g., "4GB", "512MB")
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid memory format
    """
    if not re.match(r'^\d+(\.\d+)?(GB|MB|gb|mb)$', memory_str):
        raise ValidationError(
            f"Invalid memory format: {memory_str}",
            field="memory",
            value=memory_str,
            constraint="Use format like '4GB', '512MB', '1.5GB'"
        )
    
    # Convert to MB for validation
    memory_str_upper = memory_str.upper()
    if memory_str_upper.endswith('GB'):
        value = float(memory_str_upper[:-2]) * 1024
    else:  # MB
        value = float(memory_str_upper[:-2])
    
    if value < 128:  # Minimum 128MB
        raise ValidationError(
            f"Memory limit too small: {memory_str} (minimum: 128MB)",
            field="memory",
            value=memory_str
        )
    
    if value > 1024 * 1024:  # Maximum 1TB
        raise ValidationError(
            f"Memory limit too large: {memory_str} (maximum: 1TB)",
            field="memory", 
            value=memory_str
        )
    
    return True


def validate_file_path(file_path: str, extensions: Optional[List[str]] = None) -> bool:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        extensions: Allowed file extensions (e.g., ['.wav', '.mp3'])
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid path
    """
    from pathlib import Path
    
    if not file_path:
        raise ValidationError("File path cannot be empty")
    
    path = Path(file_path)
    
    # Check if path exists
    if not path.exists():
        raise ValidationError(
            f"File does not exist: {file_path}",
            field="file_path",
            value=file_path
        )
    
    # Check if it's a file (not directory)
    if not path.is_file():
        raise ValidationError(
            f"Path is not a file: {file_path}",
            field="file_path",
            value=file_path
        )
    
    # Check file extension
    if extensions:
        file_ext = path.suffix.lower()
        if file_ext not in [ext.lower() for ext in extensions]:
            raise ValidationError(
                f"Invalid file extension: {file_ext}",
                field="file_path",
                value=file_path,
                constraint=f"Must be one of: {', '.join(extensions)}"
            )
    
    # Check file size (max 100MB)
    try:
        file_size = path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB
            raise ValidationError(
                f"File too large: {file_size / (1024*1024):.1f}MB (max: 100MB)",
                field="file_path",
                value=file_path
            )
    except OSError:
        raise ValidationError(f"Cannot access file: {file_path}")
    
    return True


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid URL
    """
    import re
    
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)', re.IGNORECASE)
    
    if not url_pattern.match(url):
        raise ValidationError(
            f"Invalid URL format: {url}",
            field="url",
            value=url
        )
    
    return True


def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email to validate
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If invalid email
    """
    import re
    
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    if not email_pattern.match(email):
        raise ValidationError(
            f"Invalid email format: {email}",
            field="email",
            value=email
        )
    
    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
    
    Returns:
        str: Sanitized filename
    """
    import re
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)
    
    # Trim whitespace and dots
    filename = filename.strip('. ')
    
    # Ensure not empty
    if not filename:
        filename = "untitled"
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_len = 255 - len(ext) - 1 if ext else 255
        filename = name[:max_name_len] + ('.' + ext if ext else '')
    
    return filename


def estimate_snr(audio: np.ndarray, percentile_threshold: float = 25.0) -> float:
    """
    Estimate signal-to-noise ratio of audio.
    
    Args:
        audio: Audio signal
        percentile_threshold: Percentile for noise estimation
    
    Returns:
        float: Estimated SNR in dB
    """
    try:
        # Calculate frame energies
        frame_size = 2048
        hop_size = 1024
        
        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = np.mean(frame**2)
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Estimate noise as lower percentile of energies
        noise_power = np.percentile(energies, percentile_threshold)
        
        # Estimate signal as upper percentile
        signal_power = np.percentile(energies, 100 - percentile_threshold)
        
        # Calculate SNR
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
        
        return float(snr_db)
        
    except Exception:
        return 0.0  # Return 0 if estimation fails


def validate_json_serializable(obj: Any) -> bool:
    """
    Check if object is JSON serializable.
    
    Args:
        obj: Object to check
    
    Returns:
        bool: True if serializable
    
    Raises:
        ValidationError: If not serializable
    """
    import json
    
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"Object is not JSON serializable: {e}",
            field="json_data"
        )


def validate_range(value: Union[int, float], min_val: Union[int, float], 
                  max_val: Union[int, float], field_name: str = "value") -> bool:
    """
    Validate that value is within specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of field for error messages
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If value out of range
    """
    if not min_val <= value <= max_val:
        raise ValidationError(
            f"{field_name} must be between {min_val} and {max_val}, got {value}",
            field=field_name,
            value=str(value),
            constraint=f"{min_val} ≤ {field_name} ≤ {max_val}"
        )
    
    return True


def validate_choice(value: Any, choices: List[Any], field_name: str = "value") -> bool:
    """
    Validate that value is one of allowed choices.
    
    Args:
        value: Value to validate
        choices: List of allowed values
        field_name: Name of field for error messages
    
    Returns:
        bool: True if valid
    
    Raises:
        ValidationError: If value not in choices
    """
    if value not in choices:
        raise ValidationError(
            f"Invalid {field_name}: {value}",
            field=field_name,
            value=str(value),
            constraint=f"Must be one of: {', '.join(str(c) for c in choices)}"
        )
    
    return True