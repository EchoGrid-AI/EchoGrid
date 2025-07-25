# echogrid/core/__init__.py
"""
Core functionality for EchoGrid.

This module contains the core functionality for model discovery, loading,
caching, and backend management.
"""

from .discovery import browse, search, info, get_model_info
from .models import load, load_model
from .cache import cache, CacheManager
from .backends import get_backend, BackendManager
from .session import Session, get_session

__all__ = [
    "browse",
    "search", 
    "info",
    "get_model_info",
    "load",
    "load_model",
    "cache",
    "CacheManager",
    "get_backend",
    "BackendManager", 
    "Session",
    "get_session"
]