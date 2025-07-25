# echogrid/constants.py
"""
Constants and configuration values for EchoGrid.

This module contains various constants used throughout the EchoGrid library,
including API endpoints, model architectures, supported formats, and default values.
"""

import os
from pathlib import Path

# API Configuration
DEFAULT_API_ENDPOINT = "https://api.echogrid.dev/v1"
DEFAULT_HUB_ENDPOINT = "https://hub.echogrid.dev"
API_VERSION = "v1"
USER_AGENT = f"echogrid-python/{__version__}"

# Model Architectures
SUPPORTED_ARCHITECTURES = [
    "xtts-v2",
    "styletts2", 
    "bark",
    "openvoice",
    "f5-tts",
    "kokoro",
    "mms-tts",
    "vits"
]

# Audio Formats
SUPPORTED_AUDIO_FORMATS = {
    "input": [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"],
    "output": [".wav", ".mp3", ".flac", ".ogg"]
}

AUDIO_SAMPLE_RATES = [8000, 16000, 22050, 24000, 44100, 48000]
DEFAULT_SAMPLE_RATE = 22050

# Model Format Types
MODEL_FORMATS = ["pytorch", "onnx", "gguf", "tensorrt"]
QUANTIZATION_TYPES = ["fp32", "fp16", "bf16", "int8", "int4"]

# Languages (ISO 639-1 codes)
SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
    "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "he",
    "th", "vi", "uk", "cs", "hu", "ro", "bg", "hr", "sk", "sl"
]

# Voice Categories
VOICE_CATEGORIES = [
    "conversational",
    "narrator", 
    "character",
    "singing",
    "professional",
    "casual",
    "emotional"
]

# Emotions
SUPPORTED_EMOTIONS = [
    "neutral",
    "happy",
    "sad", 
    "angry",
    "excited",
    "calm",
    "whisper",
    "shouting",
    "storytelling",
    "professional"
]

# Genders
VOICE_GENDERS = ["male", "female", "neutral", "child"]

# Age Ranges  
AGE_RANGES = ["child", "young", "adult", "elderly"]

# Accents (English)
ENGLISH_ACCENTS = [
    "american",
    "british", 
    "australian",
    "canadian",
    "indian",
    "irish",
    "scottish",
    "south_african"
]

# File Paths
DEFAULT_CACHE_DIR = Path.home() / ".echogrid"
DEFAULT_CONFIG_DIR = DEFAULT_CACHE_DIR / "config"
DEFAULT_MODELS_DIR = DEFAULT_CACHE_DIR / "models"
DEFAULT_TEMP_DIR = DEFAULT_CACHE_DIR / "temp"

# Memory and Performance
DEFAULT_MAX_MEMORY_GB = 8
MIN_GPU_MEMORY_GB = 2
DEFAULT_BATCH_SIZE = 4
MAX_TEXT_LENGTH = 2000
MIN_AUDIO_DURATION = 0.5
MAX_AUDIO_DURATION = 30.0

# Training Constants
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 100
DEFAULT_VALIDATION_SPLIT = 0.1
MIN_TRAINING_SAMPLES = 10

# API Limits
DEFAULT_RATE_LIMIT = 100  # requests per minute
MAX_API_RETRIES = 3
DEFAULT_TIMEOUT = 30  # seconds

# Quality Thresholds
MIN_AUDIO_QUALITY_SNR = 15  # dB
MIN_MODEL_RATING = 1.0
DEFAULT_QUALITY_THRESHOLD = 0.7

# Licenses
SUPPORTED_LICENSES = [
    "cc0",
    "cc-by",
    "cc-by-sa", 
    "cc-by-nc",
    "cc-by-nc-sa",
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "custom"
]

# Environment Variables
ENV_VAR_PREFIX = "ECHOGRID_"
ENV_VARS = {
    "API_KEY": f"{ENV_VAR_PREFIX}API_KEY",
    "API_ENDPOINT": f"{ENV_VAR_PREFIX}API_ENDPOINT", 
    "CACHE_DIR": f"{ENV_VAR_PREFIX}CACHE_DIR",
    "DEVICE": f"{ENV_VAR_PREFIX}DEVICE",
    "DEFAULT_BACKEND": f"{ENV_VAR_PREFIX}DEFAULT_BACKEND",
    "LOG_LEVEL": f"{ENV_VAR_PREFIX}LOG_LEVEL",
}

# File Extensions
MODEL_FILE_EXTENSIONS = {
    "pytorch": [".pth", ".pt", ".bin"],
    "onnx": [".onnx"],
    "gguf": [".gguf"],
    "tensorrt": [".trt", ".engine"]
}

CONFIG_FILE_EXTENSIONS = [".yaml", ".yml", ".json"]

# URLs and Endpoints
ENDPOINTS = {
    "models": "/models",
    "generate": "/generate", 
    "upload": "/upload",
    "download": "/download",
    "search": "/search",
    "info": "/info",
    "usage": "/usage",
    "auth": "/auth"
}

# Error Messages
ERROR_MESSAGES = {
    "model_not_found": "Model '{model_name}' not found. Available models: {suggestions}",
    "insufficient_memory": "Insufficient memory: {required}GB required, {available}GB available",
    "api_rate_limit": "API rate limit exceeded. Retry after {retry_after} seconds",
    "invalid_audio": "Invalid audio file: {details}",
    "training_failed": "Training failed: {error}",
    "network_error": "Network error: {details}"
}