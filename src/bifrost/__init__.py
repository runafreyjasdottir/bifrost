"""
Bifrǫst Bridge — Composite Memory Provider
==============================

Named for the rainbow bridge that connects the Nine Realms.
"""

from .config import BifrostConfig, MemoryBackend, get_config, set_config
from .core import BifrostBridge

__all__ = [
    "BifrostBridge",
    "BifrostConfig",
    "MemoryBackend",
    "get_config",
    "set_config",
]

__version__ = "1.0.0"