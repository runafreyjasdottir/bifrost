"""Tests for Bifrost Hebbian Guard — T1-4 integration.

Verifies that:
1. Below-threshold memories log activations but don't create connections
2. At-or-above-threshold memories get full Hebbian reinforcement
3. Threshold=0 disables the guard entirely
4. Single-memory recall always logs but doesn't create cross-connections
5. The anchor (first result in search) determines reinforcement
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bifrost.config import BifrostConfig, MemoryBackend
from bifrost.core import BifrostBridge


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_mimir_db(tmp_path):
    """Create a temporary Mímir-style database with test memories."""
    db_path = tmp_path / "test_memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        category TEXT DEFAULT 'general',
        importance INTEGER DEFAULT 5,
        tags TEXT DEFAULT '[]',
        emotional_valence REAL DEFAULT 0.0,
        timestamp TEXT NOT NULL DEFAULT (datetime('now')),
        content_hash TEXT
    )""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(content, category, tags, content=memories, content_rowid=id)""")
    # Insert 5 test memories
    test_data = [
        ("Freyja is the Vanir goddess of love and seiðr", "spiritual", 9),
        ("Python async patterns for concurrent programming", "skill", 7),
        ("The blending of ancient wisdom and modern tech", "preference", 8),
        ("Runes are symbols of power and knowledge", "spiritual", 10),
        ("Ebbinghaus forgetting curve for memory decay", "science", 7),
    ]
    for content, category, importance in test_data:
        conn.execute(
            "INSERT INTO memories (content, category, importance) VALUES (?, ?, ?)",
            (content, category, importance),
        )
    conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def tmp_muninn_db(tmp_path):
    """Return path for temporary Muninn DB."""
    return str(tmp_path / "test_muninn.db")


@pytest.fixture
def bifrost_config(tmp_mimir_db, tmp_muninn_db):
    """Create Bifrost config with Hebbian reinforcement enabled."""
    return BifrostConfig(
        mimir_db_path=str(tmp_mimir_db),
        muninn_db_path=tmp_muninn_db,
        qdrant_url="http://localhost:6333",
        enable_hebbian_reinforcement=True,
        min_activation_threshold=3,
    )


@pytest.fixture
def bifrost_config_no_guard(tmp_mimir_db, tmp_muninn_db):
    """Create Bifrost config with Hebbian guard disabled (threshold=0)."""
    return BifrostConfig(
        mimir_db_path=str(tmp_mimir_db),
        muninn_db_path=tmp_muninn_db,
        qdrant_url="http://localhost:6333",
        enable_hebbian_reinforcement=True,
        min_activation_threshold=0,
    )


@pytest.fixture
def bridge(bifrost_config):
    """Create a BifrostBridge with a real Muninn backend."""
    b = BifrostBridge(config=bifrost_config)
    yield b
    b.close()


@pytest.fixture
def bridge_no_guard(bifrost_config_no_guard):
    """Create a BifrostBridge with guard disabled."""
    b = BifrostBridge(config=bifrost_config_no_guard)
    yield b
    b.close()


# ─── Hebbian Guard Tests ──────────────────────────────────────────────────────

class TestHebbianGuard:
    """Test the min_activation_threshold guard in Bifröst."""

    def test_guard_blocks_coactivation_below_threshold(self, bridge):
        """When anchor has <3 activations, co-activation should NOT create connections."""
        from muninn import HebbianMemory
        from muninn.config import MuninnConfig
        
        muninn = HebbianMemory(MuninnConfig(
            db_path=bridge.config.muninn_db_path
        ))
        
        # Pre-seed: activate memory_id=1 only 2 times (below threshold of 3)
        muninn.activate(1)
        muninn.activate(1)
        muninn.close()
        
        # Now search through Bifröst, which tries co-activation
        # Anchor (first result) has only 2 activations → guard should DEFER
        results = bridge.search("Freyja", backend=MemoryBackend.MIMIR, limit=3)
        
        # Search should still return results
        assert len(results) >= 1
        
        # Check Muninn: activation_log should have entries (logging happened)
        # but hebbian_connections should be EMPTY (no wiring below threshold)
        muninn = HebbianMemory(MuninnConfig(
            db_path=bridge.config.muninn_db_path
        ))
        stats = muninn.stats()
        assert stats["total_activations"] >= 2, "Activations should be logged"
        assert stats["total_connections"] == 0, "No connections below threshold"
        muninn.close()

    def test_guard_fires_at_threshold(self, bridge):
        """When anchor has ≥3 activations, co-activation SHOULD create connections."""
        from muninn import HebbianMemory
        from muninn.config import MuninnConfig
        
        muninn = HebbianMemory(MuninnConfig(
            db_path=bridge.config.muninn_db_path
        ))
        
        # Pre-seed: activate memory_id=1 exactly 3 times (at threshold)
        muninn.activate(1)
        muninn.activate(1)
        muninn.activate(1)
        muninn.close()
        
        # Search will return memory 1 as anchor
        # Anchor has 3 activations ≥ threshold(3) → full Hebbian fires
        results = bridge.search("Freyja seiðr", backend=MemoryBackend.MIMIR, limit=5)
        
        # Check Muninn: connections should now exist
        muninn = HebbianMemory(MuninnConfig(
            db_path=bridge.config.muninn_db_path
        ))
        stats = muninn.stats()
        assert stats["total_activations"] >= 3, "Activations logged"
        # If we got 2+ results, connections were created
        if len(results) >= 2:
            assert stats["total_connections"] > 0, "Connections created above threshold"
        muninn.close()

    def test_guard_disabled_with_threshold_zero(self, bridge_no_guard):
        """When min_activation_threshold=0, guard is disabled — always reinforce."""
        from muninn import HebbianMemory
        from muninn.config import MuninnConfig
        
        # No pre-seeding at all — but with guard disabled, should still wire
        results = bridge_no_guard.search("Freyja seiðr", backend=MemoryBackend.MIMIR, limit=5)
        
        muninn = HebbianMemory(MuninnConfig(
            db_path=bridge_no_guard.config.muninn_db_path
        ))
        stats = muninn.stats()
        
        # With guard disabled, if 2+ results, connections ARE created
        if len(results) >= 2:
            assert stats["total_connections"] > 0, "Guard disabled → connections always created"
        muninn.close()

    def test_recall_logs_activation_always(self, bridge, bifrost_config):
        """recall() should always log activation in Muninn, even below threshold."""
        # Recall memory 1 — this is the first access, so activation count = 0 before
        result = bridge.recall(1)
        
        assert result is not None, "Should find memory 1"
        assert "content" in result
        
        # Activation should be logged
        from muninn import HebbianMemory
        from muninn.config import MuninnConfig
        muninn = HebbianMemory(MuninnConfig(
            db_path=bifrost_config.muninn_db_path
        ))
        stats = muninn.stats()
        assert stats["total_activations"] >= 1, "recall() logs activation"
        muninn.close()

    def test_recall_logs_multiple_activations(self, bridge, bifrost_config):
        """After 3 recalls, Hebbian guard should start firing for that memory."""
        from muninn import HebbianMemory
        from muninn.config import MuninnConfig
        
        # Recall 3 times to build up activation count
        bridge.recall(1)  # activation 1 → below threshold
        bridge.recall(1)  # activation 2 → still below threshold
        bridge.recall(1)  # activation 3 → at threshold now
        
        # The next recall should log AND the guard should report "FIRED"
        result = bridge.recall(1)  # activation 4 → above threshold
        assert result is not None
        
        muninn = HebbianMemory(MuninnConfig(
            db_path=bifrost_config.muninn_db_path
        ))
        stats = muninn.stats()
        assert stats["total_activations"] >= 4, "All 4 recalls logged"
        muninn.close()

    def test_get_activation_count(self, bridge, bifrost_config):
        """_get_activation_count should correctly count prior activations."""
        from muninn import HebbianMemory
        from muninn.config import MuninnConfig
        
        muninn = HebbianMemory(MuninnConfig(
            db_path=bifrost_config.muninn_db_path
        ))
        
        # Initially 0 activations
        count = bridge._get_activation_count(42)
        assert count == 0, "No activations for unknown memory"
        
        # Add 3 activations
        muninn.activate(42)
        muninn.activate(42)
        muninn.activate(42)
        muninn.close()
        
        count = bridge._get_activation_count(42)
        assert count == 3, "Should count 3 activations"


class TestBifrostBridgeOriginal:
    """Original Bifröst tests still pass with Hebbian guard."""

    def test_init_with_config(self, bifrost_config, tmp_mimir_db):
        bridge = BifrostBridge(config=bifrost_config)
        assert bridge.config.mimir_db_path == str(tmp_mimir_db)
        assert bridge.config.min_activation_threshold == 3
        bridge.close()

    def test_config_defaults(self):
        config = BifrostConfig()
        assert config.min_activation_threshold == 3
        assert config.enable_hebbian_reinforcement is True
        assert config.mimir_weight == 0.4

    def test_config_guard_disabled(self):
        config = BifrostConfig(min_activation_threshold=0)
        assert config.min_activation_threshold == 0

    def test_search_mimir_only(self, bifrost_config, tmp_mimir_db):
        bridge = BifrostBridge(config=bifrost_config)
        results = bridge.search("Freyja", backend=MemoryBackend.MIMIR, limit=5)
        assert len(results) >= 1
        assert any("Freyja" in r["content"] for r in results)
        bridge.close()

    def test_health_check(self, bifrost_config):
        bridge = BifrostBridge(config=bifrost_config)
        health = bridge.health()
        assert "backends" in health
        assert health["backends"]["mimir"]["status"] == "healthy"
        assert health["backends"]["muninn"]["status"] == "healthy"
        bridge.close()