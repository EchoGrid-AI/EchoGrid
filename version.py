# echogrid/version.py
"""
Version information for EchoGrid.

This module contains version metadata and build information for the EchoGrid package.
"""

__version__ = "0.1.0"
__version_info__ = tuple(int(part) for part in __version__.split('.'))

# Build metadata (populated by CI/CD)
__build__ = None
__commit__ = None
__build_date__ = None


def get_version_info():
    """
    Get comprehensive version information.
    
    Returns:
        dict: Dictionary containing version information including:
            - version: Version string
            - version_info: Version tuple
            - build: Build number (if available)
            - commit: Git commit hash (if available)  
            - build_date: Build timestamp (if available)
    
    Example:
        >>> from echogrid.version import get_version_info
        >>> info = get_version_info()
        >>> print(f"EchoGrid v{info['version']}")
        EchoGrid v0.1.0
    """
    info = {
        "version": __version__,
        "version_info": __version_info__,
    }
    
    if __build__:
        info["build"] = __build__
    if __commit__:
        info["commit"] = __commit__
    if __build_date__:
        info["build_date"] = __build_date__
        
    return info