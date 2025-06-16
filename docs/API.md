# SpiralDeltaDB API Reference

Complete API documentation for SpiralDeltaDB vector database.

## Table of Contents

- [SpiralDeltaDB Class](#spiraldeltadb-class)
- [Configuration Parameters](#configuration-parameters)
- [Data Types](#data-types)
- [Search Results](#search-results)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Examples](#examples)

## SpiralDeltaDB Class

### Constructor

```python
SpiralDeltaDB(
    dimensions: int,
    compression_ratio: float = 0.5,
    spiral_constant: float = 1.618,
    storage_path: str = "./spiraldelta.db",
    # Compression settings
    quantization_levels: int = 4,
    n_subspaces: int = 8,
    n_bits: int = 8,
    anchor_stride: int = 64,
    # Search settings
    max_layers: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
    distance_metric: str = "cosine",
    # Storage settings
    cache_size_mb: int = 512,
    enable_mmap: bool = True,
    batch_size: int = 1000,
    # Advanced options
    enable_spiral_optimization: bool = True,
    adaptive_reference: bool = True,
    auto_train_threshold: int = 1000,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimensions` | `int` | Required | Vector dimensionality (e.g., 300, 768, 1536) |
| `compression_ratio` | `float` | `0.5` | Target compression ratio (0.0-1.0) |
| `spiral_constant` | `float` | `1.618` | Spiral transformation parameter (golden ratio) |
| `storage_path` | `str` | `"./spiraldelta.db"` | Database file location |
| `quantization_levels` | `int` | `4` | Number of delta encoding tiers |
| `n_subspaces` | `int` | `8` | Product quantization subspaces |
| `n_bits` | `int` | `8` | Bits per PQ subspace |
| `anchor_stride` | `int` | `64` | Distance between anchor points |
| `max_layers` | `int` | `16` | HNSW graph layers |
| `ef_construction` | `int` | `200` | Index construction parameter |
| `ef_search` | `int` | `50` | Search parameter |
| `distance_metric` | `str` | `"cosine"` | Distance metric ("cosine", "l2", "ip") |
| `cache_size_mb` | `int` | `512` | Storage cache size in MB |
| `enable_mmap` | `bool` | `True` | Enable memory-mapped files |
| `batch_size` | `int` | `1000` | Processing batch size |
| `enable_spiral_optimization` | `bool` | `True` | Enable spiral-aware optimizations |
| `adaptive_reference` | `bool` | `True` | Adaptive reference vector updates |
| `auto_train_threshold` | `int` | `1000` | Auto-train encoder after N insertions |

### Core Methods

#### insert()

```python
def insert(
    self, 
    vectors: Union[np.ndarray, List[List[float]]], 
    metadata: Optional[List[Dict[str, Any]]] = None,
    batch_size: Optional[int] = None,
    auto_optimize: bool = True,
) -> List[int]
```

Insert vectors into the database.

**Parameters:**
- `vectors`: Input vectors (shape: [n, dimensions])
- `metadata`: Optional metadata for each vector
- `batch_size`: Processing batch size override
- `auto_optimize`: Automatically optimize after insertion

**Returns:**
- `List[int]`: List of assigned vector IDs

**Example:**
```python
import numpy as np

# Single vector
vector = np.random.randn(768)
vector_id = db.insert([vector], [{"doc_id": "doc1"}])

# Multiple vectors
vectors = np.random.randn(1000, 768)
metadata = [{"id": i, "category": f"cat_{i%5}"} for i in range(1000)]
vector_ids = db.insert(vectors, metadata)
```

#### search()

```python
def search(
    self, 
    query: np.ndarray, 
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    ef_search: Optional[int] = None,
    return_vectors: bool = True,
) -> List[SearchResult]
```

Search for similar vectors.

**Parameters:**
- `query`: Query vector (shape: [dimensions])
- `k`: Number of results to return
- `filters`: Optional metadata filters
- `ef_search`: Search parameter override
- `return_vectors`: Whether to include vectors in results

**Returns:**
- `List[SearchResult]`: List of search results with similarities

**Example:**
```python
# Basic search
query = np.random.randn(768)
results = db.search(query, k=5)

# Search with filters
results = db.search(
    query, 
    k=10, 
    filters={"category": "tech", "score": {"$gte": 0.8}}
)

# High-quality search
results = db.search(query, k=5, ef_search=200)
```

#### get_stats()

```python
def get_stats(self) -> DatabaseStats
```

Get comprehensive database statistics.

**Returns:**
- `DatabaseStats`: Database statistics object

**Example:**
```python
stats = db.get_stats()
print(f"Compression: {stats.compression_ratio:.1%}")
print(f"Storage: {stats.storage_size_mb:.1f} MB")
print(f"Vectors: {stats.vector_count}")
print(f"Query time: {stats.avg_query_time_ms:.1f}ms")
```

### Utility Methods

#### get_vector_by_id()

```python
def get_vector_by_id(self, vector_id: int) -> Tuple[np.ndarray, Dict[str, Any]]
```

Retrieve specific vector by ID.

#### delete()

```python
def delete(self, vector_ids: List[int]) -> int
```

Delete vectors by IDs.

#### update()

```python
def update(self, vector_id: int, new_vector: np.ndarray, metadata: Optional[Dict] = None) -> bool
```

Update existing vector.

#### optimize()

```python
def optimize(self) -> None
```

Perform comprehensive database optimization.

#### save() / load()

```python
def save(self, path: str) -> None
@classmethod
def load(cls, path: str, storage_path: Optional[str] = None) -> "SpiralDeltaDB"
```

Save/load database state.

#### Context Manager

```python
with SpiralDeltaDB(dimensions=768) as db:
    db.insert(vectors)
    results = db.search(query)
# Automatically closed
```

## Configuration Parameters

### Compression Settings

For different use cases, optimize these parameters:

#### High Compression (Storage-Constrained)
```python
db = SpiralDeltaDB(
    dimensions=768,
    compression_ratio=0.8,        # Aggressive compression
    quantization_levels=6,        # Deep delta encoding
    n_subspaces=16,              # More PQ subspaces
    n_bits=6,                    # Fewer bits per subspace
    anchor_stride=32,            # More anchors
)
```

#### High Performance (Speed-Optimized)
```python
db = SpiralDeltaDB(
    dimensions=768,
    compression_ratio=0.4,        # Moderate compression
    quantization_levels=2,        # Shallow encoding
    max_layers=20,               # Larger HNSW graph
    ef_construction=400,         # High-quality index
    ef_search=200,               # Thorough search
    cache_size_mb=2048,          # Large cache
)
```

#### Balanced (Production Default)
```python
db = SpiralDeltaDB(
    dimensions=768,
    compression_ratio=0.6,        # Good compression
    quantization_levels=4,        # Standard encoding
    ef_construction=200,         # Standard index
    ef_search=100,               # Quality search
    cache_size_mb=512,           # Reasonable cache
)
```

### Optimal Parameters by Dimension

| Dimensions | n_subspaces | Recommended Settings |
|------------|-------------|---------------------|
| 256-384 | 8-12 | `n_bits=8, anchor_stride=64` |
| 512-768 | 12-16 | `n_bits=7, anchor_stride=48` |
| 1024-1536 | 16-24 | `n_bits=6, anchor_stride=32` |
| 2048+ | 24-32 | `n_bits=6, anchor_stride=24` |

## Data Types

### SearchResult

```python
@dataclass
class SearchResult:
    id: int                    # Vector ID
    similarity: float          # Similarity score (0.0-1.0)
    vector: np.ndarray        # Reconstructed vector
    metadata: Dict[str, Any]  # Associated metadata
```

### DatabaseStats

```python
@dataclass
class DatabaseStats:
    vector_count: int         # Total vectors stored
    storage_size_mb: float    # Storage size in MB
    compression_ratio: float  # Achieved compression ratio
    avg_query_time_ms: float  # Average query time
    index_size_mb: float      # Index size in MB
    memory_usage_mb: float    # Memory usage in MB
    dimensions: int           # Vector dimensions
```

### SpiralCoordinate

```python
@dataclass
class SpiralCoordinate:
    theta: float              # Spiral angle
    radius: float             # Distance from center
    vector: np.ndarray        # Original vector
    metadata: Optional[Dict[str, Any]] = None
```

### CompressedSequence

```python
@dataclass
class CompressedSequence:
    anchors: List[np.ndarray]           # Anchor points (full precision)
    delta_codes: List[np.ndarray]       # Quantized delta codes
    metadata: Dict[str, Any]            # Compression metadata
    compression_ratio: float            # Achieved compression ratio
```

## Metadata Filtering

### Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"status": {"$eq": "active"}}` |
| `$ne` | Not equals | `{"status": {"$ne": "deleted"}}` |
| `$gt` | Greater than | `{"score": {"$gt": 0.5}}` |
| `$gte` | Greater than or equal | `{"score": {"$gte": 0.8}}` |
| `$lt` | Less than | `{"priority": {"$lt": 5}}` |
| `$lte` | Less than or equal | `{"priority": {"$lte": 3}}` |
| `$in` | In list | `{"category": {"$in": ["tech", "ai"]}}` |
| `$nin` | Not in list | `{"category": {"$nin": ["spam"]}}` |

### Filter Examples

```python
# Simple equality
filters = {"category": "technology"}

# Range queries
filters = {
    "score": {"$gte": 0.7, "$lt": 0.9},
    "timestamp": {"$gte": "2024-01-01"}
}

# List membership
filters = {
    "tags": {"$in": ["ml", "ai", "nlp"]},
    "status": {"$ne": "archived"}
}

# Complex combinations
filters = {
    "user_id": {"$in": [1, 2, 3, 4, 5]},
    "rating": {"$gte": 4.0},
    "category": {"$ne": "spam"},
    "created_at": {"$gte": "2024-01-01", "$lt": "2024-12-31"}
}
```

## Error Handling

### Common Exceptions

```python
from spiraldelta import SpiralDeltaDB

try:
    db = SpiralDeltaDB(dimensions=768)
    vectors = np.random.randn(100, 512)  # Wrong dimensions!
    db.insert(vectors)
except ValueError as e:
    print(f"Dimension mismatch: {e}")

try:
    query = np.random.randn(768)
    results = db.search(query, k=10)
except RuntimeError as e:
    print(f"Search failed: {e}")
```

### Error Types

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError` | Dimension mismatch, invalid parameters | Check vector dimensions and parameter ranges |
| `RuntimeError` | Search/insert failures | Check database initialization and training status |
| `FileNotFoundError` | Missing storage files | Verify storage path and permissions |
| `MemoryError` | Insufficient memory | Reduce batch size or cache size |

## Performance Optimization

### Memory Management

```python
# Memory-efficient insertion
def insert_large_dataset(db, vectors, batch_size=1000):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        db.insert(batch, auto_optimize=False)
    
    # Optimize once at the end
    db.optimize()
```

### Search Optimization

```python
# Tune search parameters based on use case
def configure_for_use_case(use_case: str) -> dict:
    configs = {
        "realtime": {
            "ef_search": 50,
            "max_layers": 8,
            "cache_size_mb": 1024
        },
        "batch": {
            "ef_search": 200,
            "max_layers": 16,
            "cache_size_mb": 256
        },
        "high_precision": {
            "ef_search": 500,
            "max_layers": 20,
            "cache_size_mb": 2048
        }
    }
    return configs.get(use_case, configs["realtime"])
```

### Compression Tuning

```python
# Adaptive compression based on data characteristics
def tune_compression(vector_correlation: float) -> dict:
    if vector_correlation > 0.8:  # Highly correlated
        return {
            "compression_ratio": 0.8,
            "quantization_levels": 6,
            "anchor_stride": 24
        }
    elif vector_correlation > 0.5:  # Moderately correlated
        return {
            "compression_ratio": 0.6,
            "quantization_levels": 4,
            "anchor_stride": 48
        }
    else:  # Low correlation
        return {
            "compression_ratio": 0.4,
            "quantization_levels": 2,
            "anchor_stride": 96
        }
```

## Examples

### Complete RAG System

```python
from spiraldelta import SpiralDeltaDB
import numpy as np
from typing import List, Dict

class RAGSystem:
    def __init__(self, embedding_dim: int = 768):
        self.db = SpiralDeltaDB(
            dimensions=embedding_dim,
            compression_ratio=0.7,
            distance_metric="cosine",
            ef_search=100
        )
        self.embedding_model = self._load_embedding_model()
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base."""
        embeddings = self.embedding_model.encode(documents)
        
        if metadata is None:
            metadata = [{"doc_id": i, "text": doc} for i, doc in enumerate(documents)]
        
        return self.db.insert(embeddings, metadata)
    
    def search(self, query: str, k: int = 5, filters: Dict = None) -> List[Dict]:
        """Search for relevant documents."""
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.db.search(query_embedding, k=k, filters=filters)
        
        return [
            {
                "text": result.metadata.get("text", ""),
                "similarity": result.similarity,
                "metadata": result.metadata
            }
            for result in results
        ]
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        stats = self.db.get_stats()
        return {
            "documents": stats.vector_count,
            "storage_mb": stats.storage_size_mb,
            "compression": f"{stats.compression_ratio:.1%}",
            "avg_query_ms": stats.avg_query_time_ms
        }

# Usage
rag = RAGSystem(embedding_dim=384)

documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Vector databases store high-dimensional embeddings."
]

rag.add_documents(documents)
results = rag.search("What is machine learning?", k=2)
print(rag.get_stats())
```

### Recommendation System

```python
class RecommendationSystem:
    def __init__(self, user_embedding_dim: int = 512):
        self.user_db = SpiralDeltaDB(
            dimensions=user_embedding_dim,
            compression_ratio=0.6,
            distance_metric="cosine"
        )
        
        self.item_db = SpiralDeltaDB(
            dimensions=user_embedding_dim,
            compression_ratio=0.6,
            distance_metric="cosine"
        )
    
    def add_users(self, user_embeddings: np.ndarray, user_metadata: List[Dict]):
        """Add user profiles."""
        return self.user_db.insert(user_embeddings, user_metadata)
    
    def add_items(self, item_embeddings: np.ndarray, item_metadata: List[Dict]):
        """Add item profiles."""
        return self.item_db.insert(item_embeddings, item_metadata)
    
    def recommend_items(self, user_id: int, k: int = 10, filters: Dict = None) -> List[Dict]:
        """Get item recommendations for a user."""
        # Get user embedding
        user_vector, user_meta = self.user_db.get_vector_by_id(user_id)
        
        # Find similar items
        results = self.item_db.search(user_vector, k=k, filters=filters)
        
        return [
            {
                "item_id": result.metadata.get("item_id"),
                "title": result.metadata.get("title"),
                "similarity": result.similarity,
                "metadata": result.metadata
            }
            for result in results
        ]
    
    def find_similar_users(self, user_id: int, k: int = 20) -> List[Dict]:
        """Find users with similar preferences."""
        user_vector, _ = self.user_db.get_vector_by_id(user_id)
        results = self.user_db.search(user_vector, k=k+1)  # +1 to exclude self
        
        # Filter out the user themselves
        return [r for r in results if r.id != user_id][:k]
```

### Multimodal Search

```python
class MultimodalSearch:
    def __init__(self):
        # Separate databases for different modalities
        self.text_db = SpiralDeltaDB(dimensions=768, compression_ratio=0.7)
        self.image_db = SpiralDeltaDB(dimensions=512, compression_ratio=0.6)
        self.audio_db = SpiralDeltaDB(dimensions=256, compression_ratio=0.5)
    
    def add_content(self, content_type: str, embeddings: np.ndarray, metadata: List[Dict]):
        """Add content of specific type."""
        db_map = {
            "text": self.text_db,
            "image": self.image_db,
            "audio": self.audio_db
        }
        
        if content_type not in db_map:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        return db_map[content_type].insert(embeddings, metadata)
    
    def search_multimodal(self, query_embeddings: Dict[str, np.ndarray], k: int = 10) -> Dict[str, List]:
        """Search across multiple modalities."""
        results = {}
        
        for modality, embedding in query_embeddings.items():
            if modality == "text":
                results["text"] = self.text_db.search(embedding, k=k)
            elif modality == "image":
                results["image"] = self.image_db.search(embedding, k=k)
            elif modality == "audio":
                results["audio"] = self.audio_db.search(embedding, k=k)
        
        return results
    
    def cross_modal_search(self, query_embedding: np.ndarray, source_modality: str, target_modality: str, k: int = 10):
        """Search across modalities (e.g., text query for images)."""
        # This would require a trained cross-modal mapping
        # For demonstration, we'll use the same embedding
        target_db = {
            "text": self.text_db,
            "image": self.image_db, 
            "audio": self.audio_db
        }[target_modality]
        
        return target_db.search(query_embedding, k=k)
```

This comprehensive API documentation provides all the details needed to effectively use SpiralDeltaDB in production applications.