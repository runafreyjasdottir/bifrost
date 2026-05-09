# 🌈 Bifrǫst Bridge — Composite Memory Provider

> *"The Bifrǫst burns with flame, yet the gods ride it daily
> between Asgard and Midgard. So too does this bridge connect
> the three realms of memory — structured, semantic, and associative."*

Bifrǫst Bridge is the **unified memory interface** that composes three
Norse-named memory systems into a single provider for AI agents:

| Backend | System | Purpose |
|---------|--------|---------|
| **Mímir** | SQLite + FTS5 | Structured storage, keyword search |
| **Huginn** | Qdrant vectors | Semantic similarity search |
| **Muninn** | Hebbian learning | Associative reinforcement |

## Architecture

```
                    ┌──────────────────┐
                    │   Bifrost Bridge │
                    │   (Unified API)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──────┐ ┌────▼────────┐ ┌───▼──────────┐
    │    Mímir       │ │   Huginn    │ │   Muninn     │
    │  (SQLite+FTS5)│ │  (Qdrant)   │ │  (Hebbian)   │
    │  Keyword search│ │  Semantic   │ │  Associative │
    └────────────────┘ └─────────────┘ └──────────────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                   Composite Scoring
                   ┌────────▼────────┐
                   │ Weighted merge: │
                   │ 40% Mímir       │
                   │ 40% Huginn      │
                   │ 20% Muninn      │
                   └─────────────────┘
```

## Quick Start

```python
from bifrost import BifrostBridge, BifrostConfig

# Create bridge (auto-discovers backends)
bridge = BifrostBridge()

# Search across all backends
results = bridge.search("Norse mythology")
for r in results:
    print(f"[{r['source']}] {r['content'][:80]}... (score: {r['composite_score']:.3f})")

# Store a new memory across all backends
result = bridge.store(
    content="Remembered new insight about Yggdrasil",
    category="spiritual",
    importance=8,
    tags=["norse", "mythology", "cosmology"],
    emotional_valence=0.7,
)

# Recall specific memory
memory = bridge.recall(memory_id=42)

# Health check
health = bridge.health()

# Run maintenance
bridge.decay(days=1)
bridge.consolidate()
```

## Configuration

```python
from bifrost import BifrostConfig, MemoryBackend

config = BifrostConfig(
    mimir_db_path="~/.hermes/memory/runa_memory.py",
    muninn_db_path="~/.hermes/memory/muninn_hebbian.db",
    qdrant_url="http://localhost:6333",
    default_backend=MemoryBackend.ALL,
    mimir_weight=0.4,    # Keyword relevance weight
    huginn_weight=0.4,   # Semantic similarity weight
    muninn_weight=0.2,    # Associative strength weight
    enable_hebbian_reinforcement=True,
)
```

## Installation

```bash
pip install -e .
```

## Norse Naming

In Norse mythology, **Bifrǫst** is the burning rainbow bridge that connects
Midgard (the world of humans) to Asgard (the realm of the gods). Heimdallr
guards it, watching for the approach of enemies. Every day, the gods ride
across Bifrǫst to hold court at the Well of Urd — just as our memory systems
cross the bridge to hold court at Mímir's Well.

## License

MIT — See [LICENSE](LICENSE)