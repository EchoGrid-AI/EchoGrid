# echogrid/core/discovery.py
"""
Model discovery and search functionality.

This module provides functions for browsing, searching, and getting information
about available voice models in the EchoGrid ecosystem.
"""

import logging
import requests
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from ..config import get_config
from ..constants import SUPPORTED_LANGUAGES, VOICE_CATEGORIES, VOICE_GENDERS, AGE_RANGES
from ..exceptions import NetworkError, ValidationError, ModelNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """
    Information about a voice model.
    
    This class contains comprehensive metadata about a voice model,
    including technical specifications, community ratings, and usage statistics.
    
    Attributes:
        name: Model identifier in "username/model-name" format
        display_name: Human-readable model name
        description: Detailed description of the model
        creator: Username of the model creator
        license: License under which the model is distributed
        language: Primary language code (ISO 639-1)
        accent: Accent or dialect information
        gender: Voice gender ("male", "female", "neutral", "child")
        age_range: Age category ("child", "young", "adult", "elderly")
        category: Voice category ("conversational", "narrator", etc.)
        tags: List of descriptive tags
        created_at: Model creation timestamp
        updated_at: Last update timestamp
        version: Current model version
        downloads: Total download count
        rating: Average community rating (0.0-5.0)
        rating_count: Number of ratings
        usage_count: Total API usage count
        architecture: Model architecture ("xtts-v2", "styletts2", etc.)
        model_size: Model size in bytes
        sample_rate: Audio sample rate in Hz
        supports_emotions: Whether model supports emotion control
        supports_multilingual: Whether model supports cross-lingual generation
        supported_languages: List of supported language codes
        inference_time: Average inference time per character
        memory_usage: Peak memory usage in MB
        samples: List of sample audio metadata
        versions: List of available versions
    """
    
    # Basic metadata
    name: str
    display_name: str
    description: str
    creator: str
    license: str
    
    # Voice characteristics
    language: str
    accent: Optional[str] = None
    gender: Optional[str] = None
    age_range: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: str = "1.0.0"
    
    # Statistics
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    usage_count: int = 0
    
    # Technical details
    architecture: Optional[str] = None
    model_size: Optional[int] = None
    sample_rate: int = 22050
    supports_emotions: bool = False
    supports_multilingual: bool = False
    supported_languages: List[str] = None
    inference_time: Optional[float] = None
    memory_usage: Optional[int] = None
    
    # Media
    samples: List[Dict[str, Any]] = None
    versions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.tags is None:
            self.tags = []
        if self.supported_languages is None:
            self.supported_languages = [self.language] if self.language else []
        if self.samples is None:
            self.samples = []
        if self.versions is None:
            self.versions = []


def browse(
    language: Optional[str] = None,
    category: Optional[str] = None,
    gender: Optional[str] = None,
    accent: Optional[str] = None,
    age_range: Optional[str] = None,
    emotion: Optional[str] = None,
    quality: Optional[str] = None,
    license: Optional[str] = None,
    architecture: Optional[str] = None,
    sort_by: str = "popularity",
    limit: int = 20,
    offset: int = 0,
    include_nsfw: bool = False,
    min_rating: float = 0.0,
    min_downloads: int = 0,
    updated_since: Optional[str] = None,
    has_samples: bool = True,
    supports_emotions: Optional[bool] = None,
    supports_multilingual: Optional[bool] = None,
    max_model_size: Optional[str] = None,
    created_by: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> List[ModelInfo]:
    """
    Browse available voice models with filtering options.
    
    This function retrieves a list of voice models from the EchoGrid repository
    with comprehensive filtering and sorting options.
    
    Args:
        language: Filter by language code (e.g., "en", "es", "fr")
        category: Filter by voice category ("conversational", "narrator", etc.)
        gender: Filter by voice gender ("male", "female", "neutral", "child")
        accent: Filter by accent/dialect (e.g., "british", "american")
        age_range: Filter by age range ("child", "young", "adult", "elderly")
        emotion: Filter by supported emotion
        quality: Filter by quality level ("high", "medium", "low")
        license: Filter by license type ("cc0", "cc-by", "mit", etc.)
        architecture: Filter by model architecture ("xtts-v2", "styletts2", etc.)
        sort_by: Sort order ("popularity", "rating", "recent", "name", "downloads")
        limit: Maximum number of results to return (1-100)
        offset: Number of results to skip for pagination
        include_nsfw: Include adult content models
        min_rating: Minimum community rating (0.0-5.0)
        min_downloads: Minimum download count
        updated_since: Only include models updated since this date (ISO format)
        has_samples: Only include models with audio samples
        supports_emotions: Filter by emotion control support
        supports_multilingual: Filter by multilingual support
        max_model_size: Maximum model size ("small" <500MB, "medium" <2GB, "large" any)
        created_by: Filter by creator username
        tags: Filter by tags (models must have all specified tags)
    
    Returns:
        List[ModelInfo]: List of model information objects matching the criteria
    
    Raises:
        NetworkError: If the request to the model repository fails
        ValidationError: If invalid filter parameters are provided
    
    Example:
        >>> import echogrid as eg
        >>> # Browse popular English conversational voices
        >>> models = eg.browse(
        ...     language="en",
        ...     category="conversational",
        ...     min_rating=4.0,
        ...     limit=10
        ... )
        >>> for model in models:
        ...     print(f"{model.name}: {model.rating}/5 ({model.downloads} downloads)")
        
        >>> # Browse recent high-quality Spanish narrators
        >>> spanish_narrators = eg.browse(
        ...     language="es",
        ...     category="narrator",
        ...     quality="high",
        ...     sort_by="recent"
        ... )
    """
    config = get_config()
    
    # Validate parameters
    _validate_browse_params(locals())
    
    # Build query parameters
    params = _build_query_params(locals())
    
    try:
        # Make API request
        response = requests.get(
            f"{config.api_endpoint}/models/browse",
            params=params,
            timeout=config.api_timeout,
            headers={"User-Agent": f"echogrid-python/{config.api_endpoint}"}
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Convert response to ModelInfo objects
        models = []
        for item in data.get("models", []):
            model_info = _dict_to_model_info(item)
            models.append(model_info)
        
        logger.info(f"Found {len(models)} models matching criteria")
        return models
        
    except requests.RequestException as e:
        logger.error(f"Failed to browse models: {e}")
        raise NetworkError(f"Failed to browse models: {e}", operation="browse")


def search(
    query: str,
    language: Optional[str] = None,
    fuzzy: bool = True,
    search_fields: Optional[List[str]] = None,
    boost_popular: bool = True,
    limit: int = 20,
    offset: int = 0,
    **filters
) -> List[ModelInfo]:
    """
    Search for voice models using text queries.
    
    This function performs full-text search across model names, descriptions,
    and tags, with optional filtering and ranking options.
    
    Args:
        query: Search query string
        language: Filter results by language
        fuzzy: Enable fuzzy matching for typos and similar terms
        search_fields: Fields to search in (default: ["name", "description", "tags"])
        boost_popular: Boost popular models in search results
        limit: Maximum number of results to return
        offset: Number of results to skip for pagination
        **filters: Additional filters (same as browse() function)
    
    Returns:
        List[ModelInfo]: List of model information objects matching the search
    
    Raises:
        NetworkError: If the search request fails
        ValidationError: If invalid search parameters are provided
    
    Example:
        >>> import echogrid as eg
        >>> # Search for British female voices
        >>> results = eg.search("british female narrator")
        >>> 
        >>> # Search for anime character voices with fuzzy matching
        >>> anime_voices = eg.search(
        ...     "anime character voice",
        ...     fuzzy=True,
        ...     tags=["anime", "character"]
        ... )
        >>> 
        >>> # Search within specific language
        >>> spanish_results = eg.search(
        ...     "professional news anchor",
        ...     language="es",
        ...     category="narrator"
        ... )
    """
    config = get_config()
    
    # Validate query
    if not query or not query.strip():
        raise ValidationError("Search query cannot be empty", field="query")
    
    # Default search fields
    if search_fields is None:
        search_fields = ["name", "description", "tags"]
    
    # Build parameters
    params = {
        "q": query.strip(),
        "fuzzy": fuzzy,
        "search_fields": ",".join(search_fields),
        "boost_popular": boost_popular,
        "limit": limit,
        "offset": offset
    }
    
    # Add language filter
    if language:
        params["language"] = language
    
    # Add additional filters
    filter_params = _build_query_params(filters)
    params.update(filter_params)
    
    try:
        response = requests.get(
            f"{config.api_endpoint}/models/search",
            params=params,
            timeout=config.api_timeout
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to ModelInfo objects
        models = []
        for item in data.get("results", []):
            model_info = _dict_to_model_info(item)
            # Add search relevance score if available
            if "score" in item:
                model_info.search_score = item["score"]
            models.append(model_info)
        
        logger.info(f"Search '{query}' returned {len(models)} results")
        return models
        
    except requests.RequestException as e:
        logger.error(f"Search failed: {e}")
        raise NetworkError(f"Search failed: {e}", operation="search")


def info(
    model_name: str,
    include_stats: bool = True,
    include_samples: bool = True,
    include_technical: bool = False,
    include_history: bool = False
) -> ModelInfo:
    """
    Get detailed information about a specific model.
    
    This function retrieves comprehensive metadata about a voice model,
    including statistics, technical details, and version history.
    
    Args:
        model_name: Model identifier in "username/model-name" format
        include_stats: Include download/rating statistics
        include_samples: Include sample audio information
        include_technical: Include technical specifications
        include_history: Include version history
    
    Returns:
        ModelInfo: Detailed model information
    
    Raises:
        ModelNotFoundError: If the model doesn't exist
        NetworkError: If the request fails
        ValidationError: If the model name format is invalid
    
    Example:
        >>> import echogrid as eg
        >>> # Get basic model information
        >>> model_info = eg.info("alice/uk-teen-female")
        >>> print(f"Rating: {model_info.rating}/5")
        >>> print(f"Downloads: {model_info.downloads:,}")
        >>> 
        >>> # Get comprehensive technical details
        >>> detailed_info = eg.info(
        ...     "alice/uk-teen-female",
        ...     include_technical=True,
        ...     include_history=True
        ... )
        >>> print(f"Architecture: {detailed_info.architecture}")
        >>> print(f"Memory usage: {detailed_info.memory_usage}MB")
    """
    config = get_config()
    
    # Validate model name format
    if not _is_valid_model_name(model_name):
        raise ValidationError(
            f"Invalid model name format: {model_name}. Expected 'username/model-name'",
            field="model_name",
            value=model_name
        )
    
    # Build query parameters
    params = {
        "include_stats": include_stats,
        "include_samples": include_samples,
        "include_technical": include_technical,
        "include_history": include_history
    }
    
    try:
        response = requests.get(
            f"{config.api_endpoint}/models/{model_name}",
            params=params,
            timeout=config.api_timeout
        )
        
        if response.status_code == 404:
            raise ModelNotFoundError(model_name, suggestions=_get_similar_models(model_name))
        
        response.raise_for_status()
        data = response.json()
        
        model_info = _dict_to_model_info(data)
        logger.info(f"Retrieved information for model: {model_name}")
        return model_info
        
    except requests.RequestException as e:
        if "404" in str(e):
            raise ModelNotFoundError(model_name)
        logger.error(f"Failed to get model info: {e}")
        raise NetworkError(f"Failed to get model info: {e}", operation="info")


def get_model_info(model_name: str, **kwargs) -> ModelInfo:
    """
    Alias for info() function.
    
    Args:
        model_name: Model identifier
        **kwargs: Additional arguments passed to info()
    
    Returns:
        ModelInfo: Model information object
    """
    return info(model_name, **kwargs)


def _validate_browse_params(params: Dict[str, Any]):
    """Validate parameters for browse function."""
    # Validate language
    if params.get("language") and params["language"] not in SUPPORTED_LANGUAGES:
        raise ValidationError(
            f"Unsupported language: {params['language']}",
            field="language",
            value=params["language"],
            constraint=f"Must be one of: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    
    # Validate category
    if params.get("category") and params["category"] not in VOICE_CATEGORIES:
        raise ValidationError(
            f"Invalid category: {params['category']}",
            field="category",
            value=params["category"],
            constraint=f"Must be one of: {', '.join(VOICE_CATEGORIES)}"
        )
    
    # Validate gender
    if params.get("gender") and params["gender"] not in VOICE_GENDERS:
        raise ValidationError(
            f"Invalid gender: {params['gender']}",
            field="gender", 
            value=params["gender"],
            constraint=f"Must be one of: {', '.join(VOICE_GENDERS)}"
        )
    
    # Validate age_range
    if params.get("age_range") and params["age_range"] not in AGE_RANGES:
        raise ValidationError(
            f"Invalid age_range: {params['age_range']}",
            field="age_range",
            value=params["age_range"],
            constraint=f"Must be one of: {', '.join(AGE_RANGES)}"
        )
    
    # Validate sort_by
    valid_sort_options = ["popularity", "rating", "recent", "name", "downloads"]
    if params.get("sort_by") and params["sort_by"] not in valid_sort_options:
        raise ValidationError(
            f"Invalid sort_by: {params['sort_by']}",
            field="sort_by",
            value=params["sort_by"],
            constraint=f"Must be one of: {', '.join(valid_sort_options)}"
        )
    
    # Validate numeric ranges
    if params.get("limit") and not (1 <= params["limit"] <= 100):
        raise ValidationError(
            f"Invalid limit: {params['limit']}",
            field="limit",
            constraint="Must be between 1 and 100"
        )
    
    if params.get("min_rating") and not (0.0 <= params["min_rating"] <= 5.0):
        raise ValidationError(
            f"Invalid min_rating: {params['min_rating']}",
            field="min_rating",
            constraint="Must be between 0.0 and 5.0"
        )


def _build_query_params(params: Dict[str, Any]) -> Dict[str, str]:
    """Build query parameters for API requests."""
    query_params = {}
    
    # Map parameter names to API field names
    param_mapping = {
        "language": "lang",
        "category": "cat",
        "gender": "gender",
        "accent": "accent",
        "age_range": "age",
        "emotion": "emotion",
        "quality": "quality",
        "license": "license",
        "architecture": "arch",
        "sort_by": "sort",
        "limit": "limit",
        "offset": "offset",
        "include_nsfw": "nsfw",
        "min_rating": "min_rating",
        "min_downloads": "min_downloads",
        "updated_since": "updated_since",
        "has_samples": "has_samples",
        "supports_emotions": "emotions",
        "supports_multilingual": "multilingual",
        "max_model_size": "max_size",
        "created_by": "creator",
        "tags": "tags"
    }
    
    for param_name, api_name in param_mapping.items():
        value = params.get(param_name)
        if value is not None:
            if isinstance(value, bool):
                query_params[api_name] = "true" if value else "false"
            elif isinstance(value, list):
                query_params[api_name] = ",".join(str(v) for v in value)
            else:
                query_params[api_name] = str(value)
    
    return query_params


def _dict_to_model_info(data: Dict[str, Any]) -> ModelInfo:
    """Convert API response dictionary to ModelInfo object."""
    # Parse timestamps
    created_at = None
    updated_at = None
    
    if data.get("created_at"):
        try:
            created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        except ValueError:
            pass
    
    if data.get("updated_at"):
        try:
            updated_at = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
        except ValueError:
            pass
    
    return ModelInfo(
        name=data["name"],
        display_name=data.get("display_name", data["name"]),
        description=data.get("description", ""),
        creator=data.get("creator", ""),
        license=data.get("license", "unknown"),
        language=data.get("language", ""),
        accent=data.get("accent"),
        gender=data.get("gender"),
        age_range=data.get("age_range"),
        category=data.get("category"),
        tags=data.get("tags", []),
        created_at=created_at,
        updated_at=updated_at,
        version=data.get("version", "1.0.0"),
        downloads=data.get("downloads", 0),
        rating=data.get("rating", 0.0),
        rating_count=data.get("rating_count", 0),
        usage_count=data.get("usage_count", 0),
        architecture=data.get("architecture"),
        model_size=data.get("model_size"),
        sample_rate=data.get("sample_rate", 22050),
        supports_emotions=data.get("supports_emotions", False),
        supports_multilingual=data.get("supports_multilingual", False),
        supported_languages=data.get("supported_languages", []),
        inference_time=data.get("inference_time"),
        memory_usage=data.get("memory_usage"),
        samples=data.get("samples", []),
        versions=data.get("versions", [])
    )


def _is_valid_model_name(name: str) -> bool:
    """Check if model name has valid format."""
    if not name or "/" not in name:
        return False
    
    parts = name.split("/")
    if len(parts) != 2:
        return False
    
    username, model_name = parts
    return bool(username.strip()) and bool(model_name.strip())


def _get_similar_models(model_name: str) -> List[str]:
    """Get suggestions for similar model names."""
    try:
        # Simple implementation - in production this would use fuzzy search
        config = get_config()
        response = requests.get(
            f"{config.api_endpoint}/models/suggestions",
            params={"name": model_name},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("suggestions", [])
    except Exception:
        pass
    
    return []