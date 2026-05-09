"""Tests for Bifrost Bridge — Composite Memory Provider."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bifrost.config import BifrostConfig, MemoryBackend


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary Mímir-style database for testing."""
    db_path = tmp_path / "test_memory.db"
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        category TEXT DEFAULT 'general',
        importance INTEGER DEFAULT 5,
        tags TEXT DEFAULT '[]',
        emotional_valence REAL DEFAULT 0.0,
        timestamp TEXT NOT NULL DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(content, category, tags, content=memories, content_rowid=id)""")
    # Insert test data
    conn.execute("INSERT INTO memories (content, category, importance) VALUES ('Norse mythology and runes', 'preference', 8)")
    conn.execute("INSERT INTO memories (content, category, importance) VALUES ('Python programming skills', 'skill', 7)")
    conn.execute("INSERT INTO memories (content, category, importance) VALUES ('Freyja is my patron goddess', 'spiritual', 10)")
    conn.commit()
    # Rebuild FTS
    conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def bifrost_config(tmp_db):
    """Create a Bifrost config pointing to temp databases."""
    return BifrostConfig(
        mimir_db_path=str(tmp_db),
        qdrant_url="http://localhost:6333",
        muninn_db_path=str(Path(tmp_db).parent / "test_muninn.db"),
        enable_hebbian_reinforcement=False,  # Disable for most tests
    )


class TestBifrostConfig:
    def test_default_config(self):
        config = BifrostConfig()
        assert config.mimir_weight == 0.4
        assert config.huginn_weight == 0.4
        assert config.muninn_weight == 0.2
        assert config.default_backend == MemoryBackend.ALL

    def test_custom_weights(self):
        config = BifrostConfig(mimir_weight=0.5, huginn_weight=0.3, muninn_weight=0.2)
        assert config.mimir_weight + config.huginn_weight + config.muninn_weight == pytest.approx(1.0)

    def test_memory_backend_enum(self):
        assert MemoryBackend.MIMIR.value == "mimir"
        assert MemoryBackend.HUGINN.value == "huginn"
        assert MemoryBackend.MUNINN.value == "muninn"
        assert MemoryBackend.ALL.value == "all"


class TestBifrostBridge:
    def test_init_with_config(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        assert bridge.config.mimir_db_path == str(tmp_db)
        bridge.close()

    def test_search_mimir_only(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        
        # Search only Mímir backend
        results = bridge.search("Norse mythology", backend=MemoryBackend.MIMIR, limit=5)
        assert len(results) >= 1
        assert any("Norse" in r["content"] for r in results)
        bridge.close()

    def test_search_all_backends(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        
        # Search all backends (Huginn/Muninn won't be available but shouldn't error)
        results = bridge.search("mythology", limit=5)
        # Should get Mímir results at minimum
        assert len(results) >= 1
        bridge.close()

    def test_store_memory(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        
        result = bridge.store(
            content="Test memory for Bifrost",
            category="test",
            importance=5,
            tags=["test", "bifrost"],
            emotional_valence=0.5,
        )
        assert "mimir_id" in result
        assert result["category"] == "test"
        bridge.close()

    def test_recall_by_id(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        
        # Recall memory id 1
        memory = bridge.recall(1)
        assert memory is not None
        assert "content" in memory
        bridge.close()

    def test_composite_scoring(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        
        # Test with scores from multiple backends
        score = bridge._compute_composite_score({
            "mimir": 0.8,
            "huginn": 0.9,
            "muninn": 0.3,
        })
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high with good scores
        
        # Test with single backend
        score_single = bridge._compute_composite_score({
            "mimir": 0.8,
        })
        assert score_single == pytest.approx(0.8, abs=0.01)
        bridge.close()

    def test_health_check(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        
        health = bridge.health()
        assert "backends" in health
        assert "mimir" in health["backends"]
        assert health["backends"]["mimir"]["status"] == "healthy"
        bridge.close()

    def test_stats(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        
        stats = bridge.stats()
        assert "mimir" in stats
        assert stats["mimir"]["total_memories"] >= 3  # We inserted 3
        bridge.close()

    def test_decay(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        
        result = bridge.decay(days=1)
        # Decay should run (muninn may return empty dict if no data)
        assert isinstance(result, dict)
        bridge.close()

    def test_category_filter(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        
        results = bridge.search("Norse", category="preference", backend=MemoryBackend.MIMIR)
        assert len(results) >= 1
        assert all(r["category"] == "preference" for r in results)
        bridge.close()

    def test_close_and_cleanup(self, bifrost_config, tmp_db):
        from bifrost.core import BifrostBridge
        bridge = BifrostBridge(config=bifrost_config)
        bridge.close()
        # Should not error on double close
        bridge.close()