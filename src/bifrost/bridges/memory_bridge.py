"""BifrostMemory — Bridge between NSE dict-shim Muninn and real Hebbian + Mímir DBs.

The NorseSagaEngine's MemoryQueryEngine expects a Muninn-like object with
a retrieve(memory_type, top_k) method returning node-like objects (.content).
The existing NSE Muninn is a dict-shim that loses ALL memories on restart.

BifrostMemory replaces that shim with a bridge to two real databases:

1. Mímir Well (runa_memory.db) — 4300+ memories with FTS5 search,
   emotional valence, categories, importance scoring.
2. Muninn Hebbian (muninn_hebbian.db) — associative connections between
   memories, co-activation counts, and consolidated paths.

When MemoryQueryEngine calls retrieve(), BifrostMemory:
  - Queries Mímir for matching memories (by category, tags, FTS5)
  - Boosts results that have strong Hebbian connections to recently
    activated memories
  - Returns _NodeShim-compatible objects with .content dicts

Architecture:
    ┌───────────────────────┐
    │  MemoryQueryEngine    │  ← NSE reads from this
    │  (unchanged)           │
    └──────────┬─────────────┘
               │ retrieve()
    ┌──────────▼─────────────┐
    │  BifrostMemory         │  ← THIS CLASS (drop-in replacement)
    ├─────────────────────────┤
    │  Mímir Well DB          │  ← 4300+ memories, FTS5, categories
    │  Muninn Hebbian DB      │  ← associative connections, activation
    └─────────────────────────┘

Usage in engine.py::

    from bifrost.bridges import BifrostMemory

    # Replace:
    # self.muninn = Muninn(dispatcher=self.dispatcher)
    # With:
    self.muninn = BifrostMemory()

    # MemoryQueryEngine continues to work unchanged:
    self.memory_query_engine = MemoryQueryEngine(self.muninn)

Author: Runa Gridweaver Freyjasdottir
Created: 2026-05-14 (T2-6)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("bifrost.memory_bridge")

# ─── Default Paths ────────────────────────────────────────────────────
HERMES_STATE_DIR = Path.home() / ".hermes" / "state"
HERMES_MEMORY_DIR = Path.home() / ".hermes" / "memory"
DEFAULT_MIMIR_DB = HERMES_MEMORY_DIR / "runa_memory.db"
DEFAULT_HEBBIAN_DB = HERMES_MEMORY_DIR / "muninn_hebbian.db"


class _MemoryNode:
    """Drop-in replacement for Muninn._NodeShim — node with .content dict.

    MemoryQueryEngine accesses .content on each result from retrieve().
    """

    __slots__ = ("content",)

    def __init__(self, content: Dict[str, Any]) -> None:
        self.content = content

    def __repr__(self) -> str:
        return f"_MemoryNode(keys={list(self.content.keys())})"


class BifrostMemory:
    """Bridge from NSE dict-shim Muninn to real Hebbian + Mímir databases.

    Drop-in replacement for the NSE Muninn class. Exposes the same
    retrieve(memory_type, top_k) interface that MemoryQueryEngine expects,
    but backed by persistent SQLite databases instead of an in-memory dict.

    Also preserves the dict-shim API (store_memory, recall_memory,
    store_subjective_memory, recall_subjective_memories) so existing
    NSE engine code continues to work without modification.

    Args:
        mimir_db_path:   Path to the Mímir Well SQLite database.
        hebbian_db_path:  Path to the Muninn Hebbian SQLite database.
    """

    def __init__(
        self,
        mimir_db_path: str | Path | None = None,
        hebbian_db_path: str | Path | None = None,
    ) -> None:
        self._mimir_path = Path(mimir_db_path or DEFAULT_MIMIR_DB)
        self._hebbian_path = Path(hebbian_db_path or DEFAULT_HEBBIAN_DB)

        # Dict-shim state (for NSE engine compatibility)
        self.memory_tree: Dict[str, Dict[str, Any]] = {}
        self.dispatcher = None  # NSE engine sets this, we ignore it

        # Recent activation log for Hebbian boost
        self._recent_activations: List[str] = []
        self._max_recent = 100

        # Verify DBs exist
        self._mimir_available = self._mimir_path.exists()
        self._hebbian_available = self._hebbian_path.exists()

        if not self._mimir_available:
            logger.warning("Mímir DB not found at %s — falling back to dict-shim", self._mimir_path)
        if not self._hebbian_available:
            logger.warning("Hebbian DB not found at %s — associative boost disabled", self._hebbian_path)

        logger.info(
            "BifrostMemory initialized (mímir=%s, hebbian=%s)",
            "available" if self._mimir_available else "fallback",
            "available" if self._hebbian_available else "disabled",
        )

    # ─── Connection helpers ──────────────────────────────────────────

    def _mimir_conn(self) -> sqlite3.Connection:
        """Open a read connection to Mímir."""
        conn = sqlite3.connect(str(self._mimir_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _hebbian_conn(self) -> sqlite3.Connection:
        """Open a read connection to Hebbian DB."""
        conn = sqlite3.connect(str(self._hebbian_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # ─── Primary interface: retrieve() ─────────────────────────────────

    def retrieve(
        self,
        query: str | None = None,
        path: str | None = None,
        tags: List[str] | None = None,
        memory_type: str | None = None,
        top_k: int = 5,
    ) -> List[_MemoryNode]:
        """Retrieve memories from Mímir + Hebbian associative boost.

        This is the primary interface that MemoryQueryEngine calls.
        Maps memory_type to Mímir categories, performs FTS5 search
        when query is provided, and boosts results with Hebbian
        co-activation data.

        Args:
            query:         Free-text search query (uses FTS5).
            path:          Key prefix filter (mapped to content hash).
            tags:          Tag filter (matches against Mímir tags).
            memory_type:  Category filter (mapped to Mímir categories).
            top_k:         Maximum results to return.

        Returns:
            List of _MemoryNode objects with .content dicts.
        """
        top_k = max(1, min(int(top_k), 100))

        # If Mímir is not available, fall back to dict-shim
        if not self._mimir_available:
            return self._retrieve_from_dict(
                query=query, path=path, tags=tags,
                memory_type=memory_type, top_k=top_k,
            )

        try:
            return self._retrieve_from_mimir(
                query=query, path=path, tags=tags,
                memory_type=memory_type, top_k=top_k,
            )
        except Exception:
            logger.exception("Mímir query failed — falling back to dict-shim")
            return self._retrieve_from_dict(
                query=query, path=path, tags=tags,
                memory_type=memory_type, top_k=top_k,
            )

    def _retrieve_from_mimir(
        self,
        query: str | None,
        path: str | None,
        tags: List[str] | None,
        memory_type: str | None,
        top_k: int,
    ) -> List[_MemoryNode]:
        """Query Mímir Well for memories, applying Hebbian boost."""
        results: List[Dict[str, Any]] = []

        with self._mimir_conn() as conn:
            # Build the query
            conditions = []
            params: List[Any] = []

            # Map memory_type to category
            if memory_type:
                category_map = {
                    "turn_event": "saga_moment",
                    "character_development": "lesson",
                    "world_state": "preference",
                    "narrative_state": "saga_moment",
                    "relationship": "relationship",
                    "emotional_memory": "saga_moment",
                    "milestone": "saga_moment",
                    "location": "preference",
                    "fact": "knowledge",
                }
                category = category_map.get(memory_type, memory_type)
                conditions.append("category = ?")
                params.append(category)

            # FTS5 full-text search if query provided
            if query:
                if conditions:
                    # Filtered + FTS
                    where = " AND ".join(conditions)
                    sql = f"""
                        SELECT m.*, rank AS fts_rank
                        FROM memories m
                        JOIN memories_fts fts ON m.id = fts.rowid
                        WHERE {where} AND memories_fts MATCH ?
                        ORDER BY importance DESC, fts_rank
                        LIMIT ?
                    """
                    params.extend([query, top_k * 3])
                else:
                    # Pure FTS
                    sql = """
                        SELECT m.*, rank AS fts_rank
                        FROM memories m
                        JOIN memories_fts fts ON m.id = fts.rowid
                        WHERE memories_fts MATCH ?
                        ORDER BY importance DESC, fts_rank
                        LIMIT ?
                    """
                    params.extend([query, top_k * 3])

                rows = conn.execute(sql, params).fetchall()
            elif conditions:
                # Category filter only
                where = " AND ".join(conditions)
                sql = f"""
                    SELECT * FROM memories
                    WHERE {where}
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT ?
                """
                params.append(top_k * 3)
                rows = conn.execute(sql, params).fetchall()
            else:
                # No filters — top memories by importance
                sql = """
                    SELECT * FROM memories
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT ?
                """
                rows = conn.execute(sql, [top_k * 3]).fetchall()

            # Convert rows to dicts
            for row in rows:
                content = {
                    "id": row["id"],
                    "category": row["category"],
                    "content": row["content"],
                    "tags": row["tags"].split(",") if row["tags"] else [],
                    "importance": row["importance"],
                    "emotional_valence": row["emotional_valence"],
                    "timestamp": row["timestamp"],
                    "memory_type": row["category"],  # Map back for MQE compatibility
                }

                # Parse tags filter if specified
                if tags:
                    row_tags = set(row["tags"].split(",")) if row["tags"] else set()
                    if not any(t in row_tags for t in tags):
                        continue

                results.append(content)

        # Apply Hebbian boost if available
        if self._hebbian_available and self._recent_activations:
            results = self._apply_hebbian_boost(results)

        # Sort by importance (already primary sort) and return top_k
        results.sort(key=lambda x: x.get("importance", 5), reverse=True)
        results = results[:top_k]

        return [_MemoryNode(r) for r in results]

    def _apply_hebbian_boost(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Boost results that have Hebbian connections to recently activated memories.

        Co-activated memories get a small importance boost, making associative
        recall more organic and human-like — memories that frequently fire
        together surface more readily.
        """
        if not self._recent_activations or not results:
            return results

        try:
            with self._hebbian_conn() as conn:
                # Get Hebbian connections to recently activated memories
                placeholders = ",".join("?" * len(self._recent_activations))
                rows = conn.execute(
                    f"""
                    SELECT memory_b_id, strength, co_activation_count
                    FROM hebbian_connections
                    WHERE memory_a_id IN ({placeholders})
                    ORDER BY strength DESC
                    LIMIT 50
                    """,
                    self._recent_activations,
                ).fetchall()

                # Build boost map
                boost_map: Dict[str, float] = {}
                for row in rows:
                    boost_map[str(row["memory_b_id"])] = min(
                        row["strength"] * 0.1, 0.5
                    )

                # Apply boost
                for result in results:
                    result_id = str(result.get("id", ""))
                    if result_id in boost_map:
                        result["importance"] = result.get("importance", 5) + boost_map[result_id]

        except Exception:
            logger.debug("Hebbian boost failed — skipping")

        return results

    def _retrieve_from_dict(
        self,
        query: str | None,
        path: str | None,
        tags: List[str] | None,
        memory_type: str | None,
        top_k: int,
    ) -> List[_MemoryNode]:
        """Fallback: retrieve from the in-memory dict-shim (legacy NSE compatibility)."""
        results: List[_MemoryNode] = []

        for key, node_data in self.memory_tree.items():
            if not isinstance(node_data, dict):
                continue

            payload = node_data.get("data") or node_data

            # Filter by memory_type
            if memory_type:
                stored_type = (
                    node_data.get("memory_type")
                    or (payload.get("memory_type") if isinstance(payload, dict) else None)
                )
                if stored_type != memory_type:
                    continue

            # Filter by tags
            if tags:
                stored_tags = node_data.get("tags") or []
                if isinstance(payload, dict):
                    stored_tags = stored_tags or payload.get("tags", [])
                if not any(t in stored_tags for t in tags):
                    continue

            # Filter by path prefix
            if path and not key.startswith(path):
                continue

            content = payload if isinstance(payload, dict) else node_data
            results.append(_MemoryNode(content))

            if len(results) >= top_k:
                break

        return results

    # ─── Dict-shim compatibility (NSE engine writes to these) ──────────

    def store_memory(
        self, key: str, data: Dict[str, Any], emotion_vector: Sequence[float]
    ) -> None:
        """Store a memory in the dict-shim.

        NSE engine calls this. We store locally AND to Mímir DB if available.
        """
        # Always store in dict-shim for backward compatibility
        self.memory_tree[key] = {
            "data": data,
            "rune": "",  # EmotionService not available in bridge
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Also persist to Mímir DB if available
        if self._mimir_available:
            try:
                self._persist_to_mimir(key, data, emotion_vector)
            except Exception:
                logger.debug("Failed to persist to Mímir: %s", key)

    def _persist_to_mimir(
        self, key: str, data: Dict[str, Any], emotion_vector: Sequence[float]
    ) -> None:
        """Write a memory to Mímir Well DB."""
        with self._mimir_conn() as conn:
            content = data.get("content", str(data))
            category = data.get("memory_type", "saga_moment")
            tags = ",".join(data.get("tags", []))
            importance = int(data.get("importance", 5))
            emotional_valence = float(emotion_vector[0]) if emotion_vector else 0.0

            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (category, content, tags, importance, emotional_valence)
                VALUES (?, ?, ?, ?, ?)
                """,
                [category, content, tags, importance, emotional_valence],
            )
            conn.commit()

    def recall_memory(self, key: str) -> Dict[str, Any]:
        """Recall a memory by key from the dict-shim."""
        return self.memory_tree.get(key, {})

    def store_subjective_memory(
        self,
        character_id: str,
        event_key: str,
        data: Dict[str, Any],
        emotional_context: Dict[str, Any],
    ) -> str:
        """Store character-scoped memory with emotional tags."""
        scoped_key = f"{character_id}:{event_key}"
        payload = {
            "character_id": character_id,
            "event": data,
            "emotional_context": emotional_context,
        }
        self.memory_tree[scoped_key] = {
            "data": payload,
            "rune": emotional_context.get("emotion_rune", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Log activation for Hebbian boost
        self._log_activation(scoped_key)

        return scoped_key

    def recall_subjective_memories(
        self,
        character_id: str,
        dominant_emotion: str = "",
        query_text: str = "",
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return top emotionally resonant memories for one character.

        Queries Mímir DB for character-scoped memories, falling back
        to dict-shim if unavailable.
        """
        # Try Mímir first
        if self._mimir_available:
            try:
                return self._recall_subjective_from_mimir(
                    character_id, dominant_emotion, query_text, limit
                )
            except Exception:
                logger.debug("Mímir subjective recall failed — falling back to dict-shim")

        # Fallback to dict-shim
        target_emotion = (dominant_emotion or "").strip().lower()
        query = (query_text or "").strip().lower()
        scored: List[tuple[float, Dict[str, Any]]] = []

        for key, memory in self.memory_tree.items():
            if not key.startswith(f"{character_id}:"):
                continue

            payload = memory.get("data", {})
            emo_ctx = payload.get("emotional_context", {})
            event = payload.get("event", {})
            score = float(emo_ctx.get("emotional_charge", 0.0))

            if target_emotion and emo_ctx.get("dominant_emotion", "") == target_emotion:
                score += 0.35

            haystack = f"{event} {emo_ctx}".lower()
            if query and query in haystack:
                score += 0.2

            scored.append((score, {"id": key, **memory}))

        scored.sort(key=lambda item: item[0], reverse=True)
        bounded_limit = max(1, min(int(limit or 3), 8))
        return [item[1] for item in scored[:bounded_limit]]

    def _recall_subjective_from_mimir(
        self,
        character_id: str,
        dominant_emotion: str,
        query_text: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Query Mímir for character-scoped memories with emotional boosting."""
        results = []

        with self._mimir_conn() as conn:
            search_query = f"{character_id}"
            if query_text:
                search_query = f"{character_id} {query_text}"

            sql = """
                SELECT m.*, rank AS fts_rank
                FROM memories m
                JOIN memories_fts fts ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY importance DESC, fts_rank
                LIMIT ?
            """
            rows = conn.execute(sql, [search_query, limit * 2]).fetchall()

            for row in rows:
                content = {
                    "id": row["id"],
                    "character_id": character_id,
                    "event": row["content"],
                    "emotional_context": {
                        "dominant_emotion": dominant_emotion,
                        "emotional_charge": row["emotional_valence"],
                    },
                    "data": row["content"],
                    "rune": "",
                    "timestamp": row["timestamp"],
                }
                results.append(content)

        # Apply Hebbian boost
        if self._hebbian_available:
            results = self._apply_hebbian_boost(results)

        results.sort(key=lambda x: x.get("emotional_context", {}).get("emotional_charge", 0), reverse=True)
        return results[:max(1, min(limit, 8))]

    # ─── Hebbian activation tracking ────────────────────────────────────

    def _log_activation(self, memory_id: str) -> None:
        """Log a memory activation for Hebbian boost tracking."""
        self._recent_activations.append(memory_id)
        if len(self._recent_activations) > self._max_recent:
            self._recent_activations = self._recent_activations[-self._max_recent:]

        # Also persist to Hebbian activation_log
        if self._hebbian_available:
            try:
                with self._hebbian_conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO activation_log
                        (memory_id, activated_at, emotional_valence, source)
                        VALUES (?, ?, ?, ?)
                        """,
                        [memory_id, datetime.now(timezone.utc).isoformat(), 0.0, "bifrost"],
                    )
                    conn.commit()
            except Exception:
                logger.debug("Failed to log activation for %s", memory_id)

    # ─── Status ─────────────────────────────────────────────────────────

    @property
    def is_persistent(self) -> bool:
        """Whether backed by persistent databases (vs dict-shim only)."""
        return self._mimir_available

    @property
    def is_hebbian(self) -> bool:
        """Whether Hebbian associative recall is available."""
        return self._hebbian_available

    def stats(self) -> Dict[str, Any]:
        """Return health statistics for monitoring."""
        stats = {
            "mímir_available": self._mimir_available,
            "hebbian_available": self._hebbian_available,
            "dict_shim_entries": len(self.memory_tree),
            "recent_activations": len(self._recent_activations),
            "persistent": self.is_persistent,
        }

        if self._mimir_available:
            try:
                with self._mimir_conn() as conn:
                    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                    stats["mímir_memories"] = count
            except Exception:
                stats["mímir_memories"] = "error"

        if self._hebbian_available:
            try:
                with self._hebbian_conn() as conn:
                    conns = conn.execute("SELECT COUNT(*) FROM hebbian_connections").fetchone()[0]
                    acts = conn.execute("SELECT COUNT(*) FROM activation_log").fetchone()[0]
                    stats["hebbian_connections"] = conns
                    stats["hebbian_activations"] = acts

            except Exception:
                pass

        return stats