"""
Bifrǫst Configuration
======================
Named for the rainbow bridge that connects the Nine Realms.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class MemoryBackend(Enum):
    """Available memory backends — the Nine Realms of memory."""
    MIMIR = "mimir"       # SQLite + FTS5 (structured + search)
    HUGINN = "huginn"     # Qdrant (semantic vector search)
    MUNINN = "muninn"     # Hebbian (associative reinforcement)
    ALL = "all"            # Query all backends


@dataclass
class BifrostConfig:
    """Configuration for the Bifrost Bridge composite memory provider.
    
    The Bifrǫst connects Asgard (structured), Midgard (semantic),
    and the realms beyond (associative) into a single bridge.
    """
    # ─── Backend Paths ─────────────────────────────────────────────────
    mimir_db_path: str = "~/.hermes/memory/runa_memory.db"
    muninn_db_path: str = "~/.hermes/memory/muninn_hebbian.db"
    
    # ─── Qdrant Settings ───────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    
    # ─── Search Behavior ────────────────────────────────────────────────
    default_backend: MemoryBackend = MemoryBackend.ALL
    max_results_per_backend: int = 10
    enable_hebbian_reinforcement: bool = True  # Auto-strengthen co-activated memories
    
    # ─── Scoring Weights ────────────────────────────────────────────────
    # How much weight each backend gets in composite scoring
    mimir_weight: float = 0.4     # Structured keyword search
    huginn_weight: float = 0.4    # Semantic similarity
    muninn_weight: float = 0.2    # Associative reinforcement
    
    # ─── Consolidation ─────────────────────────────────────────────────
    auto_consolidate: bool = True
    consolidation_threshold: float = 0.8
    
    # ─── Decay ─────────────────────────────────────────────────────────
    auto_decay: bool = True
    decay_interval_hours: float = 24.0


# ─── Singleton Config ──────────────────────────────────────────────────────
_config: Optional[BifrostConfig] = None


def get_config() -> BifrostConfig:
    """Get or create the global Bifrost configuration."""
    global _config
    if _config is None:
        _config = BifrostConfig()
    return _config


def set_config(config: BifrostConfig):
    """Override the global configuration."""
    global _config
    _config = config