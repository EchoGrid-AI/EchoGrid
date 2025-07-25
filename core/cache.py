# echogrid/core/cache.py
"""
Model cache management for EchoGrid.

This module handles caching of downloaded models, including storage management,
cleanup operations, and cache statistics.
"""

import os
import json
import shutil
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from ..config import get_config
from ..constants import MODEL_FILE_EXTENSIONS
from ..exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """
    Information about a cached model.
    
    Attributes:
        name: Model identifier ("username/model-name")
        version: Model version
        path: Local path to cached model
        size_bytes: Size in bytes
        size_mb: Size in megabytes
        download_date: When model was downloaded
        last_used: When model was last accessed
        checksum: File checksum for integrity verification
        format: Model format ("pytorch", "onnx", etc.)
        architecture: Model architecture
        metadata: Additional metadata
    """
    name: str
    version: str
    path: str
    size_bytes: int
    download_date: datetime
    last_used: datetime
    checksum: str
    format: str = "pytorch"
    architecture: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        return self.size_bytes / (1024 * 1024 * 1024)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CacheStats:
    """
    Cache statistics.
    
    Attributes:
        total_size_bytes: Total cache size in bytes
        total_size_mb: Total cache size in megabytes
        total_size_gb: Total cache size in gigabytes
        model_count: Number of cached models
        free_space_bytes: Available disk space in bytes
        free_space_gb: Available disk space in gigabytes
        oldest_model: Oldest cached model
        newest_model: Most recently cached model
        most_used: Most frequently used model
        disk_usage_percent: Cache disk usage percentage
    """
    total_size_bytes: int
    model_count: int
    free_space_bytes: int
    oldest_model: Optional[str] = None
    newest_model: Optional[str] = None
    most_used: Optional[str] = None
    
    @property
    def total_size_mb(self) -> float:
        """Total size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)
    
    @property
    def total_size_gb(self) -> float:
        """Total size in gigabytes."""
        return self.total_size_bytes / (1024 * 1024 * 1024)
    
    @property
    def free_space_gb(self) -> float:
        """Free space in gigabytes."""
        return self.free_space_bytes / (1024 * 1024 * 1024)
    
    @property
    def disk_usage_percent(self) -> float:
        """Cache disk usage as percentage."""
        total_space = self.total_size_bytes + self.free_space_bytes
        if total_space == 0:
            return 0.0
        return (self.total_size_bytes / total_space) * 100


class CacheManager:
    """
    Manages the local model cache.
    
    This class handles all operations related to caching models locally,
    including storage, retrieval, cleanup, and statistics.
    """
    
    def __init__(self):
        """Initialize cache manager."""
        self.config = get_config()
        self.cache_dir = Path(self.config.cache_dir)
        self.models_dir = self.cache_dir / "models"
        self.index_file = self.cache_dir / "cache_index.json"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create cache index
        self._index = self._load_index()
    
    def list(
        self,
        sort_by: str = "last_used",
        include_size: bool = True,
        include_metadata: bool = True
    ) -> List[CachedModel]:
        """
        List all cached models.
        
        Args:
            sort_by: Sort criteria ("last_used", "size", "name", "download_date")
            include_size: Include file size information
            include_metadata: Include model metadata
        
        Returns:
            List[CachedModel]: List of cached model information
        
        Example:
            >>> cache = CacheManager()
            >>> models = cache.list(sort_by="size")
            >>> for model in models:
            ...     print(f"{model.name}: {model.size_mb:.1f}MB")
        """
        models = []
        
        for model_data in self._index.get("models", {}).values():
            try:
                # Convert timestamps
                download_date = datetime.fromisoformat(model_data["download_date"])
                last_used = datetime.fromisoformat(model_data["last_used"])
                
                # Calculate size if requested
                size_bytes = model_data.get("size_bytes", 0)
                if include_size and not size_bytes:
                    model_path = Path(model_data["path"])
                    if model_path.exists():
                        size_bytes = self._get_directory_size(model_path)
                        # Update index with calculated size
                        model_data["size_bytes"] = size_bytes
                
                cached_model = CachedModel(
                    name=model_data["name"],
                    version=model_data["version"],
                    path=model_data["path"],
                    size_bytes=size_bytes,
                    download_date=download_date,
                    last_used=last_used,
                    checksum=model_data.get("checksum", ""),
                    format=model_data.get("format", "pytorch"),
                    architecture=model_data.get("architecture"),
                    metadata=model_data.get("metadata", {}) if include_metadata else {}
                )
                models.append(cached_model)
                
            except Exception as e:
                logger.warning(f"Error loading cached model info: {e}")
                continue
        
        # Sort models
        if sort_by == "last_used":
            models.sort(key=lambda m: m.last_used, reverse=True)
        elif sort_by == "size":
            models.sort(key=lambda m: m.size_bytes, reverse=True)
        elif sort_by == "name":
            models.sort(key=lambda m: m.name)
        elif sort_by == "download_date":
            models.sort(key=lambda m: m.download_date, reverse=True)
        
        # Save updated index
        self._save_index()
        
        return models
    
    def get(self, model_name: str, version: str = "latest") -> Optional[CachedModel]:
        """
        Get information about a specific cached model.
        
        Args:
            model_name: Model identifier
            version: Model version
        
        Returns:
            CachedModel: Cached model information or None if not found
        """
        models = self.list()
        
        for model in models:
            if model.name == model_name:
                if version == "latest" or model.version == version:
                    return model
        
        return None
    
    def is_cached(self, model_name: str, version: str = "latest") -> bool:
        """
        Check if a model is cached locally.
        
        Args:
            model_name: Model identifier
            version: Model version
        
        Returns:
            bool: True if model is cached
        """
        return self.get(model_name, version) is not None
    
    def get_path(self, model_name: str, version: str = "latest") -> Optional[str]:
        """
        Get the local path to a cached model.
        
        Args:
            model_name: Model identifier
            version: Model version
        
        Returns:
            str: Local path to model or None if not cached
        """
        cached_model = self.get(model_name, version)
        if cached_model and Path(cached_model.path).exists():
            # Update last used timestamp
            self._update_last_used(model_name, version)
            return cached_model.path
        return None
    
    def add(
        self,
        model_name: str,
        version: str,
        model_path: str,
        format: str = "pytorch",
        architecture: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a model to the cache.
        
        Args:
            model_name: Model identifier
            version: Model version
            model_path: Path to model files
            format: Model format
            architecture: Model architecture
            metadata: Additional metadata
        
        Returns:
            str: Path to cached model
        
        Raises:
            ValidationError: If model path doesn't exist
        """
        source_path = Path(model_path)
        if not source_path.exists():
            raise ValidationError(f"Model path does not exist: {model_path}")
        
        # Create cache path
        safe_name = self._sanitize_name(model_name)
        cache_path = self.models_dir / safe_name / version
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        if source_path.is_file():
            shutil.copy2(source_path, cache_path / source_path.name)
        else:
            shutil.copytree(source_path, cache_path, dirs_exist_ok=True)
        
        # Calculate checksum and size
        checksum = self._calculate_checksum(cache_path)
        size_bytes = self._get_directory_size(cache_path)
        
        # Add to index
        now = datetime.now()
        model_id = f"{model_name}:{version}"
        
        self._index["models"][model_id] = {
            "name": model_name,
            "version": version,
            "path": str(cache_path),
            "size_bytes": size_bytes,
            "download_date": now.isoformat(),
            "last_used": now.isoformat(),
            "checksum": checksum,
            "format": format,
            "architecture": architecture,
            "metadata": metadata or {}
        }
        
        self._save_index()
        logger.info(f"Added model to cache: {model_name}:{version}")
        
        return str(cache_path)
    
    def remove(self, model_name: str, version: Optional[str] = None) -> List[str]:
        """
        Remove model(s) from cache.
        
        Args:
            model_name: Model identifier
            version: Specific version to remove (None = remove all versions)
        
        Returns:
            List[str]: List of removed model identifiers
        """
        removed = []
        models_to_remove = []
        
        # Find models to remove
        for model_id, model_data in self._index["models"].items():
            if model_data["name"] == model_name:
                if version is None or model_data["version"] == version:
                    models_to_remove.append(model_id)
        
        # Remove models
        for model_id in models_to_remove:
            model_data = self._index["models"][model_id]
            model_path = Path(model_data["path"])
            
            # Remove files
            if model_path.exists():
                shutil.rmtree(model_path, ignore_errors=True)
            
            # Remove from index
            del self._index["models"][model_id]
            removed.append(model_id)
            
            logger.info(f"Removed cached model: {model_id}")
        
        self._save_index()
        return removed
    
    def clear(
        self,
        model_name: Optional[str] = None,
        older_than: Optional[int] = None,
        larger_than: Optional[str] = None,
        unused: bool = False,
        dry_run: bool = False
    ) -> List[str]:
        """
        Clear cache with various criteria.
        
        Args:
            model_name: Specific model to clear (None = all models)
            older_than: Remove models unused for N days
            larger_than: Remove models larger than size ("1GB", "500MB")
            unused: Only remove models that haven't been used recently
            dry_run: Show what would be removed without actually removing
        
        Returns:
            List[str]: List of model identifiers that were (or would be) removed
        """
        models = self.list()
        to_remove = []
        
        for model in models:
            should_remove = False
            
            # Check specific model name
            if model_name and model.name != model_name:
                continue
            
            # Check age criteria
            if older_than is not None:
                cutoff_date = datetime.now() - timedelta(days=older_than)
                if model.last_used < cutoff_date:
                    should_remove = True
            
            # Check size criteria
            if larger_than is not None:
                size_limit = self._parse_size_string(larger_than)
                if model.size_bytes > size_limit:
                    should_remove = True
            
            # Check unused criteria
            if unused:
                # Consider unused if not accessed in last 7 days
                cutoff_date = datetime.now() - timedelta(days=7)
                if model.last_used < cutoff_date:
                    should_remove = True
            
            if should_remove or (model_name and not older_than and not larger_than and not unused):
                to_remove.append(f"{model.name}:{model.version}")
        
        if dry_run:
            logger.info(f"Would remove {len(to_remove)} models")
            return to_remove
        
        # Actually remove models
        removed = []
        for model_id in to_remove:
            model_name_part, version = model_id.split(":", 1)
            removed.extend(self.remove(model_name_part, version))
        
        logger.info(f"Removed {len(removed)} models from cache")
        return removed
    
    def stats(self) -> CacheStats:
        """
        Get cache statistics.
        
        Returns:
            CacheStats: Cache statistics object
        """
        models = self.list()
        
        total_size = sum(model.size_bytes for model in models)
        model_count = len(models)
        
        # Get disk space
        try:
            stat = shutil.disk_usage(self.cache_dir)
            free_space = stat.free
        except Exception:
            free_space = 0
        
        # Find oldest, newest, most used
        oldest_model = None
        newest_model = None
        most_used = None
        
        if models:
            oldest_model = min(models, key=lambda m: m.download_date).name
            newest_model = max(models, key=lambda m: m.download_date).name
            most_used = max(models, key=lambda m: m.last_used).name
        
        return CacheStats(
            total_size_bytes=total_size,
            model_count=model_count,
            free_space_bytes=free_space,
            oldest_model=oldest_model,
            newest_model=newest_model,
            most_used=most_used
        )
    
    def move(self, new_path: str):
        """
        Move cache to new location.
        
        Args:
            new_path: New cache directory path
        
        Raises:
            ConfigurationError: If move operation fails
        """
        new_path = Path(new_path).expanduser()
        
        try:
            # Create new directory
            new_path.mkdir(parents=True, exist_ok=True)
            
            # Move contents
            if self.cache_dir.exists():
                for item in self.cache_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, new_path / item.name)
                    else:
                        shutil.copytree(item, new_path / item.name, dirs_exist_ok=True)
                
                # Remove old cache
                shutil.rmtree(self.cache_dir)
            
            # Update paths
            self.cache_dir = new_path
            self.models_dir = new_path / "models"
            self.index_file = new_path / "cache_index.json"
            
            # Update index paths
            for model_data in self._index["models"].values():
                old_path = Path(model_data["path"])
                relative_path = old_path.relative_to(Path(self.config.cache_dir))
                model_data["path"] = str(new_path / relative_path)
            
            self._save_index()
            
            # Update config
            self.config.cache_dir = str(new_path)
            
            logger.info(f"Moved cache to: {new_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to move cache: {e}")
    
    def rebuild(self):
        """
        Rebuild cache index by scanning directory.
        
        This function scans the cache directory and rebuilds the index
        from the actual files present on disk.
        """
        logger.info("Rebuilding cache index...")
        
        new_index = {"version": 1, "models": {}}
        
        if not self.models_dir.exists():
            self._index = new_index
            self._save_index()
            return
        
        # Scan cache directory
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = self._desanitize_name(model_dir.name)
            
            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                
                version = version_dir.name
                
                try:
                    # Calculate properties
                    size_bytes = self._get_directory_size(version_dir)
                    checksum = self._calculate_checksum(version_dir)
                    
                    # Get timestamps
                    stat = version_dir.stat()
                    download_date = datetime.fromtimestamp(stat.st_ctime)
                    last_used = datetime.fromtimestamp(stat.st_atime)
                    
                    # Detect format
                    format_type = self._detect_format(version_dir)
                    
                    model_id = f"{model_name}:{version}"
                    new_index["models"][model_id] = {
                        "name": model_name,
                        "version": version,
                        "path": str(version_dir),
                        "size_bytes": size_bytes,
                        "download_date": download_date.isoformat(),
                        "last_used": last_used.isoformat(),
                        "checksum": checksum,
                        "format": format_type,
                        "architecture": None,
                        "metadata": {}
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing {version_dir}: {e}")
                    continue
        
        self._index = new_index
        self._save_index()
        
        logger.info(f"Rebuilt cache index with {len(new_index['models'])} models")
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify cache integrity.
        
        Returns:
            Dict[str, Any]: Verification results with any issues found
        """
        results = {
            "valid_models": [],
            "corrupted_models": [],
            "missing_models": [],
            "orphaned_files": []
        }
        
        # Check each model in index
        for model_id, model_data in self._index["models"].items():
            model_path = Path(model_data["path"])
            
            if not model_path.exists():
                results["missing_models"].append(model_id)
                continue
            
            # Verify checksum
            current_checksum = self._calculate_checksum(model_path)
            if current_checksum != model_data.get("checksum", ""):
                results["corrupted_models"].append(model_id)
            else:
                results["valid_models"].append(model_id)
        
        # Check for orphaned files
        if self.models_dir.exists():
            indexed_paths = {Path(data["path"]) for data in self._index["models"].values()}
            
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    for version_dir in model_dir.iterdir():
                        if version_dir.is_dir() and version_dir not in indexed_paths:
                            results["orphaned_files"].append(str(version_dir))
        
        return results
    
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        
        # Return empty index
        return {"version": 1, "models": {}}
    
    def _save_index(self):
        """Save cache index to file."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _update_last_used(self, model_name: str, version: str):
        """Update last used timestamp for a model."""
        model_id = f"{model_name}:{version}"
        if model_id in self._index["models"]:
            self._index["models"][model_id]["last_used"] = datetime.now().isoformat()
            self._save_index()
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize model name for filesystem."""
        return name.replace("/", "_").replace("\\", "_")
    
    def _desanitize_name(self, name: str) -> str:
        """Convert sanitized name back to original."""
        # Simple heuristic - in practice you'd want more robust mapping
        if "_" in name and "/" not in name:
            parts = name.split("_", 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"
        return name
    
    def _get_directory_size(self, path: Path) -> int:
        """Calculate total size of directory."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = Path(dirpath) / filename
                    if filepath.exists():
                        total_size += filepath.stat().st_size
        except Exception:
            pass
        return total_size
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate checksum for model files."""
        hasher = hashlib.sha256()
        
        try:
            if path.is_file():
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
            else:
                # For directories, hash all files in sorted order
                for file_path in sorted(path.rglob("*")):
                    if file_path.is_file():
                        with open(file_path, 'rb') as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hasher.update(chunk)
        except Exception:
            pass
        
        return hasher.hexdigest()
    
    def _parse_size_string(self, size_str: str) -> int:
        """Parse size string like '1GB' or '500MB' to bytes."""
        size_str = size_str.upper().strip()
        
        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
            'TB': 1024 ** 4
        }
        
        for unit, multiplier in multipliers.items():
            if size_str.endswith(unit):
                try:
                    value = float(size_str[:-len(unit)])
                    return int(value * multiplier)
                except ValueError:
                    break
        
        raise ValidationError(f"Invalid size format: {size_str}")
    
    def _detect_format(self, path: Path) -> str:
        """Detect model format from files."""
        for format_name, extensions in MODEL_FILE_EXTENSIONS.items():
            for ext in extensions:
                if any(path.glob(f"*{ext}")):
                    return format_name
        return "unknown"


# Global cache manager instance
cache = CacheManager()