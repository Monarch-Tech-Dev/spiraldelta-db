# Getting Started with SpiralDeltaDB

Welcome to SpiralDeltaDB! This guide will help you get up and running quickly with the geometric vector database that achieves 30-70% compression without quality loss.

## 🚀 Quick Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/monarch-ai/spiraldelta-db.git
cd spiraldelta-db

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
python -c "from spiraldelta import SpiralDeltaDB; print('✅ Installation successful!')"
```

## 📝 Your First Vector Database

Let's create your first SpiralDeltaDB database with just a few lines of code:

```python
from spiraldelta import SpiralDeltaDB
import numpy as np

# 1. Initialize database
db = SpiralDeltaDB(
    dimensions=384,           # Vector size (e.g., sentence-transformers)
    compression_ratio=0.6,    # Target 60% compression
    storage_path="./my_vectors.db"
)

# 2. Create some sample vectors
vectors = np.random.randn(1000, 384).astype(np.float32)
metadata = [{"doc_id": f"doc_{i}", "category": f"cat_{i%5}"} for i in range(1000)]

# 3. Insert vectors
vector_ids = db.insert(vectors, metadata)
print(f"✅ Inserted {len(vector_ids)} vectors")

# 4. Search for similar vectors
query = np.random.randn(384).astype(np.float32)
results = db.search(query, k=5)

# 5. View results
for result in results:
    print(f"ID: {result.id}, Similarity: {result.similarity:.3f}")

# 6. Get database statistics
stats = db.get_stats()
print(f"Compression: {stats.compression_ratio:.1%}")
print(f"Storage: {stats.storage_size_mb:.2f} MB")

# 7. Close database
db.close()
```

## 🎯 Choose Your Configuration

SpiralDeltaDB can be optimized for different use cases:

### 📦 **High Compression** (Storage-Constrained)
Perfect for mobile apps, edge devices, or cost-sensitive deployments:

```python
storage_optimized_db = SpiralDeltaDB(
    dimensions=768,
    compression_ratio=0.8,        # Aggressive 80% compression
    quantization_levels=6,        # Deep compression
    n_subspaces=16,              # More compression stages
    cache_size_mb=128,           # Small memory footprint
)
```

### ⚡ **High Performance** (Speed-Optimized)
Ideal for real-time applications and high-throughput services:

```python
performance_db = SpiralDeltaDB(
    dimensions=768,
    compression_ratio=0.4,        # Moderate compression for speed
    max_layers=20,               # Large search graph
    ef_construction=400,         # High-quality index
    ef_search=200,               # Thorough search
    cache_size_mb=2048,          # Large memory cache
)
```

### ⚖️ **Balanced** (Production Default)
Great starting point for most applications:

```python
balanced_db = SpiralDeltaDB(
    dimensions=768,
    compression_ratio=0.6,        # Good compression
    ef_construction=200,         # Standard index quality
    ef_search=100,               # Quality search
    cache_size_mb=512,           # Reasonable memory
)
```

## 🔍 Metadata Filtering

SpiralDeltaDB supports powerful metadata filtering for precise search:

```python
# Insert vectors with rich metadata
metadata = [
    {
        "doc_id": f"doc_{i}",
        "category": ["tech", "ai", "ml"][i % 3],
        "score": np.random.uniform(0, 1),
        "timestamp": f"2024-{(i%12)+1:02d}-01",
        "active": i % 2 == 0,
        "tags": ["python", "vectors", "database"]
    }
    for i in range(1000)
]

db.insert(vectors, metadata)

# Search with filters
results = db.search(
    query_vector,
    k=10,
    filters={
        "category": "ai",                    # Exact match
        "score": {"$gte": 0.7},            # Range filter
        "active": True,                     # Boolean filter
        "tags": {"$in": ["vectors", "ai"]} # List membership
    }
)
```

### Available Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"status": {"$eq": "active"}}` |
| `$ne` | Not equals | `{"status": {"$ne": "deleted"}}` |
| `$gt`, `$gte` | Greater than (or equal) | `{"score": {"$gte": 0.8}}` |
| `$lt`, `$lte` | Less than (or equal) | `{"priority": {"$lt": 5}}` |
| `$in` | In list | `{"category": {"$in": ["tech", "ai"]}}` |
| `$nin` | Not in list | `{"category": {"$nin": ["spam"]}}` |

## 📊 Performance Monitoring

Track your database performance:

```python
# Get comprehensive statistics
stats = db.get_stats()

print(f"📊 Database Statistics:")
print(f"  Vectors: {stats.vector_count:,}")
print(f"  Storage: {stats.storage_size_mb:.2f} MB")
print(f"  Compression: {stats.compression_ratio:.1%}")
print(f"  Memory: {stats.memory_usage_mb:.2f} MB")
print(f"  Avg Query: {stats.avg_query_time_ms:.2f}ms")

# Performance benchmarking
import time

n_queries = 100
query_vectors = np.random.randn(n_queries, 384).astype(np.float32)

start_time = time.time()
for query in query_vectors:
    results = db.search(query, k=10)
total_time = time.time() - start_time

print(f"⚡ Performance Metrics:")
print(f"  {n_queries} queries in {total_time:.2f}s")
print(f"  {total_time/n_queries*1000:.2f}ms per query")
print(f"  {n_queries/total_time:.0f} QPS throughput")
```

## 🛠️ Common Patterns

### Document Search System

```python
class DocumentSearchSystem:
    def __init__(self, embedding_dim=384):
        self.db = SpiralDeltaDB(
            dimensions=embedding_dim,
            compression_ratio=0.7,
            storage_path="./documents.db"
        )
        # Initialize your embedding model here
        self.embedder = YourEmbeddingModel()
    
    def add_document(self, doc_id: str, title: str, content: str, tags: List[str]):
        """Add a document to the search system."""
        # Generate embedding
        embedding = self.embedder.encode(content)
        
        # Store with metadata
        metadata = {
            "doc_id": doc_id,
            "title": title,
            "content": content[:500],  # Store preview
            "tags": tags,
            "indexed_at": time.time()
        }
        
        return self.db.insert([embedding], [metadata])
    
    def search_documents(self, query: str, k: int = 5, tag_filter: List[str] = None):
        """Search for relevant documents."""
        # Generate query embedding
        query_embedding = self.embedder.encode(query)
        
        # Build filters
        filters = {}
        if tag_filter:
            filters["tags"] = {"$in": tag_filter}
        
        # Search
        results = self.db.search(query_embedding, k=k, filters=filters)
        
        return [
            {
                "doc_id": r.metadata["doc_id"],
                "title": r.metadata["title"],
                "content": r.metadata["content"],
                "similarity": r.similarity
            }
            for r in results
        ]
```

### Recommendation System

```python
class RecommendationSystem:
    def __init__(self, user_dim=256, item_dim=256):
        self.user_db = SpiralDeltaDB(dimensions=user_dim, storage_path="./users.db")
        self.item_db = SpiralDeltaDB(dimensions=item_dim, storage_path="./items.db")
    
    def add_user_profile(self, user_id: str, preferences: np.ndarray, metadata: dict):
        """Add user profile."""
        metadata.update({"user_id": user_id, "type": "user"})
        return self.user_db.insert([preferences], [metadata])
    
    def add_item(self, item_id: str, features: np.ndarray, metadata: dict):
        """Add item to catalog."""
        metadata.update({"item_id": item_id, "type": "item"})
        return self.item_db.insert([features], [metadata])
    
    def recommend_items(self, user_id: str, k: int = 10, category: str = None):
        """Get recommendations for user."""
        # Get user vector (simplified - would use ID lookup)
        user_vector, user_meta = self.user_db.get_vector_by_id(user_id)
        
        # Filter by category if specified
        filters = {"category": category} if category else {}
        
        # Find similar items
        return self.item_db.search(user_vector, k=k, filters=filters)
```

## 🔧 Advanced Features

### Batch Operations

```python
# Efficient batch insertion
def insert_large_dataset(db, vectors, metadata, batch_size=1000):
    """Insert large datasets efficiently."""
    total_inserted = 0
    
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i+batch_size]
        batch_metadata = metadata[i:i+batch_size] if metadata else None
        
        ids = db.insert(batch_vectors, batch_metadata, auto_optimize=False)
        total_inserted += len(ids)
        
        print(f"Inserted {total_inserted:,}/{len(vectors):,} vectors")
    
    # Optimize once at the end
    db.optimize()
    return total_inserted

# Usage
large_vectors = np.random.randn(50000, 768)
large_metadata = [{"id": i} for i in range(50000)]
insert_large_dataset(db, large_vectors, large_metadata)
```

### Database Persistence

```python
# Save database state
db.save("./backup.json")

# Load database from saved state
loaded_db = SpiralDeltaDB.load("./backup.json", storage_path="./restored.db")

# Context manager for automatic cleanup
with SpiralDeltaDB(dimensions=768, storage_path="./temp.db") as db:
    db.insert(vectors, metadata)
    results = db.search(query, k=10)
# Database automatically closed
```

### Custom Distance Metrics

```python
# Cosine similarity (default)
cosine_db = SpiralDeltaDB(dimensions=768, distance_metric="cosine")

# L2 (Euclidean) distance
l2_db = SpiralDeltaDB(dimensions=768, distance_metric="l2")

# Inner product
ip_db = SpiralDeltaDB(dimensions=768, distance_metric="ip")
```

## 🚨 Common Issues & Solutions

### Issue: Poor Compression Ratio

**Symptoms:** Compression ratio < 30%
**Solutions:**
- Increase `quantization_levels` (try 4-6)
- Decrease `anchor_stride` (try 24-48)
- Ensure vectors have semantic similarity
- Check if encoder is trained (`db._is_trained`)

```python
# Force encoder training
if not db._is_trained:
    db._auto_train_encoder()
```

### Issue: Slow Search Performance

**Symptoms:** Query time > 10ms
**Solutions:**
- Increase `cache_size_mb`
- Tune `ef_search` (lower = faster, higher = more accurate)
- Reduce `quantization_levels` for speed
- Use `distance_metric="ip"` if appropriate

```python
# Performance-optimized search
results = db.search(query, k=10, ef_search=50)  # Faster
results = db.search(query, k=10, ef_search=200) # More accurate
```

### Issue: High Memory Usage

**Symptoms:** Memory usage > expected
**Solutions:**
- Reduce `cache_size_mb`
- Set `enable_mmap=True`
- Use smaller `batch_size`
- Optimize after large insertions

```python
# Memory-efficient configuration
memory_efficient_db = SpiralDeltaDB(
    dimensions=768,
    cache_size_mb=128,
    enable_mmap=True,
    batch_size=500
)
```

## 📚 Next Steps

1. **📖 Read the [API Documentation](API.md)** for detailed method references
2. **🏃 Run the [Examples](../examples/)** to see real implementations
3. **🧪 Try the [Benchmarks](../scripts/)** to measure performance
4. **🤝 Check the [Contributing Guide](../CONTRIBUTING.md)** to contribute

## 🆘 Getting Help

- **📋 GitHub Issues**: [Report bugs or request features](https://github.com/monarch-ai/spiraldelta-db/issues)
- **💬 Discussions**: [Ask questions and share ideas](https://github.com/monarch-ai/spiraldelta-db/discussions)
- **📧 Enterprise Support**: [enterprise@monarchai.com](mailto:enterprise@monarchai.com)

## 🎉 You're Ready!

You now have everything you need to build powerful vector applications with SpiralDeltaDB. The geometric approach to vector compression will help you:

- **Save 30-70% storage costs** compared to traditional vector databases
- **Achieve sub-millisecond search** performance through spiral locality
- **Deploy to edge devices** with compressed vector representations
- **Scale to millions of vectors** with efficient memory usage

Happy building! 🚀