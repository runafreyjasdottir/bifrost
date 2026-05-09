"""
Bifrǫst Bridge Core — Composite Memory Provider
===================================================
The rainbow bridge connecting all memory realms.
"""

import json
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import BifrostConfig, MemoryBackend, get_config

logger = logging.getLogger("bifrost.core")


class BifrostBridge:
    """Composite memory provider connecting Mímir, Huginn, and Muninn.
    
    The Bifrǫst is the rainbow bridge in Norse mythology connecting Midgard
    (the world of humans) to Asgard (the realm of the gods). In our architecture,
    it connects three memory realms:
    
    - Mímir's Well (SQLite+FTS5): Structured storage and keyword search
    - Huginn (Qdrant): Semantic vector similarity search  
    - Muninn (Hebbian): Associative reinforcement and transitive connections
    
    When you query through Bifrost, it:
    1. Searches all three backends in parallel
    2. Merges and deduplicates results
    3. Scores each result using weighted combination
    4. Reinforces co-activated memories via Hebbian learning
    5. Returns ranked, unified results
    
    Architecture:
        ┌──────────────────────────────────────┐
        │         Bifrost Bridge                │
        │  ┌─────────┐ ┌─────────┐ ┌────────┐ │
        │  │  Mímir   │ │ Huginn  │ │ Muninn │ │
        │  │ (SQLite) │ │(Qdrant) │ │(Hebbian)│ │
        │  └────┬─────┘ └────┬─────┘ └───┬─────┘ │
        │       │            │            │       │
        │       └──────┬─────┴─────┬──────┘      │
        │              │  Composite │             │
        │              │  Scorer    │             │
        │              └─────┬──────┘             │
        │                    │                    │
        │              Ranked Results             │
        └──────────────────────────────────────┘
    """

    def __init__(self, config: Optional[BifrostConfig] = None):
        self.config = config or get_config()
        self._mimir_conn: Optional[sqlite3.Connection] = None
        self._huginn = None
        self._muninn = None
        self._initialized = False

    def _ensure_mimir(self) -> sqlite3.Connection:
        """Lazy-load Mímir's Well connection."""
        if self._mimir_conn is None:
            db_path = Path(self.config.mimir_db_path).expanduser()
            if db_path.exists():
                self._mimir_conn = sqlite3.connect(str(db_path))
                self._mimir_conn.row_factory = sqlite3.Row
                self._mimir_conn.execute("PRAGMA journal_mode=WAL")
            else:
                logger.warning("Bifrǫst: Mímir's Well not found at %s", db_path)
        return self._mimir_conn

    def _ensure_huginn(self):
        """Lazy-load Huginn (Qdrant) connection."""
        if self._huginn is None:
            try:
                from huginn import HuginnMemory
                self._huginn = HuginnMemory(
                    config=None,  # Use defaults
                )
                logger.info("Bifrǫst: Huginn connected")
            except Exception as e:
                logger.warning("Bifrǫst: Huginn not available: %s", e)
        return self._huginn

    def _ensure_muninn(self):
        """Lazy-load Muninn (Hebbian) connection."""
        if self._muninn is None:
            try:
                from muninn import HebbianMemory
                from muninn.config import MuninnConfig
                config = MuninnConfig(db_path=self.config.muninn_db_path)
                self._muninn = HebbianMemory(config)
                logger.info("Bifrǫst: Muninn connected")
            except Exception as e:
                logger.warning("Bifrǫst: Muninn not available: %s", e)
        return self._muninn

    def _ensure_initialized(self):
        """Ensure all backends are initialized."""
        if not self._initialized:
            self._ensure_mimir()
            self._ensure_huginn()
            self._ensure_muninn()
            self._initialized = True

    # ─── Composite Search ──────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10,
               backend: Optional[MemoryBackend] = None,
               category: Optional[str] = None,
               min_importance: int = 1,
               emotional_valence: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search across all memory backends and return composite ranked results.
        
        This is the primary interface for the Bifrǫst Bridge. It queries
        Mímir (keyword), Huginn (semantic), and Muninn (associative) in
        parallel, then merges and scores the results.
        
        Args:
            query: The search query
            limit: Maximum results to return
            backend: Which backend(s) to use (default: ALL)
            category: Filter by memory category
            min_importance: Minimum importance threshold
            emotional_valence: Filter by emotional valence
            
        Returns:
            List of unified memory results with composite scores
        """
        self._ensure_initialized()
        backend = backend or self.config.default_backend
        
        results_map: Dict[int, Dict[str, Any]] = {}
        
        # ─── Mímir: Keyword Search ─────────────────────────────────────
        if backend in (MemoryBackend.MIMIR, MemoryBackend.ALL):
            mimir_results = self._search_mimir(query, limit, category, min_importance)
            for r in mimir_results:
                mid = r["id"]
                if mid not in results_map:
                    results_map[mid] = r.copy()
                    results_map[mid]["scores"] = {}
                results_map[mid]["scores"]["mimir"] = r.get("score", 0.5)
        
        # ─── Huginn: Semantic Search ───────────────────────────────────
        if backend in (MemoryBackend.HUGINN, MemoryBackend.ALL):
            huginn_results = self._search_huginn(query, limit, category, min_importance)
            for r in huginn_results:
                mid = r["id"]
                if mid not in results_map:
                    results_map[mid] = r.copy()
                    results_map[mid]["scores"] = {}
                results_map[mid]["scores"]["huginn"] = r.get("score", 0.5)
        
        # ─── Muninn: Associative Boost ─────────────────────────────────
        if backend in (MemoryBackend.MUNINN, MemoryBackend.ALL):
            for mid, result in results_map.items():
                associations = self._get_muninn_associations(mid)
                if associations:
                    # Boost score based on Hebbian connection strength
                    results_map[mid]["scores"]["muninn"] = associations[0].get("strength", 0.0)
        
        # ─── Composite Scoring ──────────────────────────────────────────
        for mid, result in results_map.items():
            result["composite_score"] = self._compute_composite_score(
                result.get("scores", {})
            )
        
        # ─── Hebbian Reinforcement ──────────────────────────────────────
        if self.config.enable_hebbian_reinforcement and len(results_map) > 1:
            self._reinforce_coactivation(list(results_map.keys()))
        
        # Sort by composite score
        results = sorted(
            results_map.values(),
            key=lambda x: x.get("composite_score", 0.0),
            reverse=True,
        )
        
        return results[:limit]

    def recall(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Recall a specific memory by ID from Mímir's Well.
        
        Also reinforces the Hebbian connection for this memory.
        """
        self._ensure_initialized()
        
        mimir = self._ensure_mimir()
        if mimir is None:
            return None
        
        row = mimir.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        
        if row:
            result = dict(row)
            # Reinforce in Muninn
            if self._muninn:
                self._muninn.activate(memory_id)
            return result
        return None

    # ─── Store (Unified Write) ────────────────────────────────────────────

    def store(self, content: str, category: str = "general",
              importance: int = 5, tags: List[str] = None,
              emotional_valence: float = 0.0) -> Dict[str, Any]:
        """Store a memory across all backends.
        
        This writes to:
        - Mímir: Structured storage with FTS5
        - Huginn: Vector embedding for semantic search
        - Muninn: Link to co-activated memories
        
        Args:
            content: The memory content
            category: Memory category
            importance: Importance level (1-10)
            tags: Optional tags
            emotional_valence: Emotional valence (-1.0 to 1.0)
            
        Returns:
            Dict with stored memory info and IDs
        """
        self._ensure_initialized()
        
        result = {"content": content, "category": category}
        
        # ─── Store in Mímir ──────────────────────────────────────────
        mimir = self._ensure_mimir()
        if mimir:
            try:
                cursor = mimir.execute(
                    """INSERT INTO memories (content, category, importance, tags, emotional_valence, timestamp)
                       VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                    (content, category, importance,
                     json.dumps(tags or []), emotional_valence),
                )
                mimir.commit()
                result["mimir_id"] = cursor.lastrowid
            except Exception as e:
                logger.warning("Bifrǫst: Mímir store failed: %s", e)
        
        # ─── Store in Huginn ──────────────────────────────────────────
        if self._huginn:
            try:
                vector_id = self._huginn.upsert_memory(
                    content=content,
                    category=category,
                    importance=importance,
                    tags=tags or [],
                    emotional_valence=emotional_valence,
                )
                result["huginn_id"] = vector_id
            except Exception as e:
                logger.warning("Bifrǫst: Huginn store failed: %s", e)
        
        # ─── Reinforce in Muninn ─────────────────────────────────────
        # If other memories were recently activated, create Hebbian links
        if self._muninn and "mimir_id" in result:
            self._muninn.activate(
                result["mimir_id"],
                emotional_valence=emotional_valence,
            )
        
        return result

    # ─── Backend-Specific Search Methods ───────────────────────────────

    def _search_mimir(self, query: str, limit: int,
                      category: Optional[str] = None,
                      min_importance: int = 1) -> List[Dict[str, Any]]:
        """Search Mímir's Well via FTS5 keyword search."""
        mimir = self._ensure_mimir()
        if mimir is None:
            return []
        
        try:
            sql = """
                SELECT m.id, m.content, m.category, m.importance, m.tags,
                       m.emotional_valence, m.timestamp,
                       bm25(memories_fts) as score
                FROM memories_fts fts
                JOIN memories m ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
            """
            params = [query]
            
            if category:
                sql += " AND m.category = ?"
                params.append(category)
            if min_importance > 1:
                sql += " AND m.importance >= ?"
                params.append(min_importance)
            
            sql += " ORDER BY score LIMIT ?"
            params.append(limit)
            
            rows = mimir.execute(sql, params).fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "category": row["category"],
                    "importance": row["importance"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "emotional_valence": row["emotional_valence"] or 0.0,
                    "timestamp": row["timestamp"],
                    "source": "mimir",
                    "score": abs(row["score"]) if row["score"] else 0.5,
                })
            return results
        except Exception as e:
            logger.warning("Bifrǫst: Mímir search failed: %s", e)
            return []

    def _search_huginn(self, query: str, limit: int,
                       category: Optional[str] = None,
                       min_importance: int = 1) -> List[Dict[str, Any]]:
        """Search via Huginn semantic vector search."""
        huginn = self._ensure_huginn()
        if huginn is None:
            return []
        
        try:
            filter_conditions = None
            if category or min_importance > 1:
                conditions = []
                if category:
                    conditions.append(f'category = "{category}"')
                if min_importance > 1:
                    conditions.append(f"importance >= {min_importance}")
                # Note: filter_conditions would need to be Qdrant Filter objects
            
            results = huginn.search(
                query_text=query,
                limit=limit,
                category=category or None,
                min_importance=min_importance,
            )
            
            return [
                {
                    "id": r.get("id", 0),
                    "content": r.get("content", ""),
                    "category": r.get("category", ""),
                    "importance": r.get("importance", 5),
                    "tags": r.get("tags", []),
                    "emotional_valence": r.get("emotional_valence", 0.0),
                    "source": "huginn",
                    "score": r.get("score", 0.5),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("Bifrǫst: Huginn search failed: %s", e)
            return []

    def _get_muninn_associations(self, memory_id: int) -> List[Dict[str, Any]]:
        """Get Hebbian associations for a memory via Muninn."""
        muninn = self._ensure_muninn()
        if muninn is None:
            return []
        
        try:
            return muninn.get_associations(
                memory_id,
                min_strength=self.config.muninn_weight,
            )
        except Exception as e:
            logger.warning("Bifrǫst: Muninn association failed: %s", e)
            return []

    def _reinforce_coactivation(self, memory_ids: List[int],
                                 emotional_valence: float = 0.0):
        """Reinforce Hebbian connections between co-activated memories."""
        muninn = self._ensure_muninn()
        if muninn is None:
            return
        
        try:
            muninn.activate_batch(memory_ids, emotional_valence=emotional_valence)
        except Exception as e:
            logger.warning("Bifrǫst: Hebbian reinforcement failed: %s", e)

    # ─── Composite Scoring ────────────────────────────────────────────────

    def _compute_composite_score(self, scores: Dict[str, float]) -> float:
        """Compute weighted composite score from individual backend scores.
        
        Uses configurable weights for each backend:
        - Mímir weight: keyword relevance (default 0.4)
        - Huginn weight: semantic similarity (default 0.4)
        - Muninn weight: associative strength (default 0.2)
        """
        composite = 0.0
        weight_sum = 0.0
        
        if "mimir" in scores:
            composite += scores["mimir"] * self.config.mimir_weight
            weight_sum += self.config.mimir_weight
        
        if "huginn" in scores:
            composite += scores["huginn"] * self.config.huginn_weight
            weight_sum += self.config.huginn_weight
        
        if "muninn" in scores:
            composite += scores["muninn"] * self.config.muninn_weight
            weight_sum += self.config.muninn_weight
        
        # Normalize by actual weights used
        if weight_sum > 0:
            composite /= weight_sum
        
        return round(composite, 4)

    # ─── Maintenance Operations ───────────────────────────────────────────

    def decay(self, days: float = 1.0) -> Dict[str, Any]:
        """Run decay across all backends."""
        self._ensure_initialized()
        results = {}
        
        if self._muninn:
            results["muninn"] = self._muninn.decay(days=days)
        
        return results

    def consolidate(self) -> Dict[str, Any]:
        """Consolidate high-strength connections."""
        self._ensure_initialized()
        results = {}
        
        if self._muninn:
            results["muninn"] = self._muninn.consolidate()
        
        return results

    def health(self) -> Dict[str, Any]:
        """Check health of all memory backends."""
        self._ensure_initialized()
        health = {"status": "healthy", "backends": {}}
        
        # Mímir health
        mimir = self._ensure_mimir()
        if mimir:
            try:
                count = mimir.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
                health["backends"]["mimir"] = {"status": "healthy", "memories": count}
            except Exception as e:
                health["backends"]["mimir"] = {"status": "unhealthy", "error": str(e)}
        else:
            health["backends"]["mimir"] = {"status": "not_configured"}
        
        # Huginn health
        huginn = self._ensure_huginn()
        if huginn:
            try:
                h = huginn.health()
                health["backends"]["huginn"] = h
            except Exception as e:
                health["backends"]["huginn"] = {"status": "unhealthy", "error": str(e)}
        else:
            health["backends"]["huginn"] = {"status": "not_configured"}
        
        # Muninn health
        muninn = self._ensure_muninn()
        if muninn:
            try:
                health["backends"]["muninn"] = muninn.health()
            except Exception as e:
                health["backends"]["muninn"] = {"status": "unhealthy", "error": str(e)}
        else:
            health["backends"]["muninn"] = {"status": "not_configured"}
        
        # Overall status
        healthy_count = sum(1 for v in health["backends"].values()
                          if v.get("status") in ("healthy", "not_configured"))
        if healthy_count < len(health["backends"]):
            health["status"] = "degraded"
        
        return health

    def stats(self) -> Dict[str, Any]:
        """Get statistics from all backends."""
        self._ensure_initialized()
        stats = {}
        
        if self._mimir_conn:
            try:
                count = self._mimir_conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
                stats["mimir"] = {"total_memories": count}
            except Exception:
                stats["mimir"] = {"error": "unavailable"}
        
        if self._huginn:
            try:
                stats["huginn"] = self._huginn.health()
            except Exception:
                stats["huginn"] = {"error": "unavailable"}
        
        if self._muninn:
            try:
                stats["muninn"] = self._muninn.stats()
            except Exception:
                stats["muninn"] = {"error": "unavailable"}
        
        return stats

    def close(self):
        """Close all backend connections."""
        if self._mimir_conn:
            self._mimir_conn.close()
            self._mimir_conn = None
        if self._muninn:
            self._muninn.close()
            self._muninn = None