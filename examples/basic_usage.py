#!/usr/bin/env python3
"""
Basic usage example for SpiralDeltaDB.

This example demonstrates the core functionality of SpiralDeltaDB including
vector insertion, search, and basic database operations.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spiraldelta import SpiralDeltaDB


def generate_sample_data(n_vectors=1000, dimensions=128):
    """Generate sample vector data for demonstration."""
    print(f"Generating {n_vectors} random vectors with {dimensions} dimensions...")
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Generate base embeddings (simulating real embeddings)
    vectors = np.random.randn(n_vectors, dimensions).astype(np.float32)
    
    # Normalize vectors (common in embedding systems)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    # Generate metadata
    metadata = []
    categories = ["technology", "science", "art", "sports", "music"]
    
    for i in range(n_vectors):
        meta = {
            "id": i,
            "category": categories[i % len(categories)],
            "score": np.random.uniform(0, 1),
            "created_at": f"2024-01-{(i % 28) + 1:02d}",
            "active": i % 3 == 0,
        }
        metadata.append(meta)
    
    return vectors, metadata


def basic_operations_demo():
    """Demonstrate basic SpiralDeltaDB operations."""
    print("=" * 60)
    print("SpiralDeltaDB Basic Operations Demo")
    print("=" * 60)
    
    # Initialize database
    print("\n1. Initializing SpiralDeltaDB...")
    db = SpiralDeltaDB(
        dimensions=128,
        compression_ratio=0.6,  # Target 60% compression
        storage_path="./demo.db",
        auto_train_threshold=500,  # Train after 500 vectors
    )
    
    print(f"✓ Database initialized: {db}")
    
    # Generate sample data
    vectors, metadata = generate_sample_data(n_vectors=800, dimensions=128)
    print(f"✓ Generated {len(vectors)} sample vectors")
    
    # Insert vectors
    print("\n2. Inserting vectors...")
    start_time = np.time.time()
    vector_ids = db.insert(vectors, metadata, batch_size=100)
    insert_time = np.time.time() - start_time
    
    print(f"✓ Inserted {len(vector_ids)} vectors in {insert_time:.2f}s")
    print(f"✓ Average insertion speed: {len(vector_ids)/insert_time:.1f} vectors/sec")
    print(f"✓ Database now contains {len(db)} vectors")
    
    # Get database statistics
    print("\n3. Database Statistics...")
    stats = db.get_stats()
    print(f"✓ Vector count: {stats.vector_count}")
    print(f"✓ Storage size: {stats.storage_size_mb:.2f} MB")
    print(f"✓ Compression ratio: {stats.compression_ratio:.1%}")
    print(f"✓ Index size: {stats.index_size_mb:.2f} MB")
    print(f"✓ Memory usage: {stats.memory_usage_mb:.2f} MB")
    
    # Search demonstrations
    print("\n4. Search Operations...")
    
    # Basic search
    query_vector = vectors[0]  # Use first vector as query
    
    start_time = np.time.time()
    results = db.search(query_vector, k=5)
    search_time = np.time.time() - start_time
    
    print(f"✓ Basic search completed in {search_time*1000:.2f}ms")
    print(f"✓ Found {len(results)} results:")
    
    for i, result in enumerate(results):
        print(f"   {i+1}. ID: {result.id}, Similarity: {result.similarity:.3f}, "
              f"Category: {result.metadata.get('category', 'N/A')}")
    
    # Filtered search
    print("\n5. Filtered Search...")
    filters = {"category": "technology"}
    
    filtered_results = db.search(query_vector, k=5, filters=filters)
    print(f"✓ Technology category search: {len(filtered_results)} results")
    
    for i, result in enumerate(filtered_results):
        print(f"   {i+1}. ID: {result.id}, Similarity: {result.similarity:.3f}, "
              f"Category: {result.metadata['category']}")
    
    # Range filters
    print("\n6. Range Filtered Search...")
    range_filters = {"score": {"$gte": 0.7}}
    
    range_results = db.search(query_vector, k=5, filters=range_filters)
    print(f"✓ High score search (≥0.7): {len(range_results)} results")
    
    for i, result in enumerate(range_results):
        print(f"   {i+1}. ID: {result.id}, Similarity: {result.similarity:.3f}, "
              f"Score: {result.metadata['score']:.3f}")
    
    # Performance comparison
    print("\n7. Performance Comparison...")
    
    # Multiple search queries
    n_queries = 100
    query_vectors = vectors[:n_queries]
    
    start_time = np.time.time()
    for query in query_vectors:
        results = db.search(query, k=10)
    total_time = np.time.time() - start_time
    
    print(f"✓ Performed {n_queries} searches in {total_time:.2f}s")
    print(f"✓ Average search time: {total_time/n_queries*1000:.2f}ms per query")
    print(f"✓ Search throughput: {n_queries/total_time:.1f} queries/sec")
    
    # Cleanup
    print("\n8. Cleanup...")
    db.close()
    print("✓ Database closed")
    
    return True


def compression_demo():
    """Demonstrate compression effectiveness."""
    print("\n" + "=" * 60)
    print("Compression Effectiveness Demo")
    print("=" * 60)
    
    # Test different types of vector data
    test_cases = [
        ("Random vectors", lambda: np.random.randn(500, 128)),
        ("Correlated vectors", lambda: generate_correlated_vectors(500, 128)),
        ("Clustered vectors", lambda: generate_clustered_vectors(500, 128)),
    ]
    
    for name, generator in test_cases:
        print(f"\n{name}:")
        
        # Generate test data
        vectors = generator().astype(np.float32)
        
        # Initialize database with high compression target
        db = SpiralDeltaDB(
            dimensions=128,
            compression_ratio=0.7,
            storage_path=f"./compression_test_{name.lower().replace(' ', '_')}.db",
            auto_train_threshold=100,
        )
        
        # Insert vectors
        vector_ids = db.insert(vectors)
        
        # Get compression statistics
        stats = db.get_stats()
        
        # Calculate storage efficiency
        original_size_mb = len(vectors) * 128 * 4 / (1024 * 1024)  # float32
        
        print(f"  Original size: {original_size_mb:.2f} MB")
        print(f"  Compressed size: {stats.storage_size_mb:.2f} MB")
        print(f"  Compression ratio: {stats.compression_ratio:.1%}")
        print(f"  Space saved: {original_size_mb - stats.storage_size_mb:.2f} MB")
        
        # Test search quality
        query = vectors[0]
        results = db.search(query, k=5)
        
        if results:
            print(f"  Search quality: {results[0].similarity:.3f} (top result)")
        
        db.close()


def generate_correlated_vectors(n_vectors, dimensions):
    """Generate correlated vectors for compression testing."""
    base = np.random.randn(dimensions)
    vectors = []
    
    for i in range(n_vectors):
        # Add small random perturbation
        noise = np.random.randn(dimensions) * 0.1
        vector = base + noise * (i / n_vectors)
        vectors.append(vector)
    
    return np.array(vectors)


def generate_clustered_vectors(n_vectors, dimensions, n_clusters=5):
    """Generate clustered vectors for compression testing."""
    vectors = []
    cluster_centers = [np.random.randn(dimensions) for _ in range(n_clusters)]
    
    for i in range(n_vectors):
        # Choose cluster
        cluster_id = i % n_clusters
        center = cluster_centers[cluster_id]
        
        # Add noise around cluster center
        noise = np.random.randn(dimensions) * 0.3
        vector = center + noise
        vectors.append(vector)
    
    return np.array(vectors)


def advanced_features_demo():
    """Demonstrate advanced SpiralDeltaDB features."""
    print("\n" + "=" * 60)
    print("Advanced Features Demo")
    print("=" * 60)
    
    # Initialize database with custom settings
    db = SpiralDeltaDB(
        dimensions=256,
        compression_ratio=0.5,
        storage_path="./advanced_demo.db",
        # Advanced settings
        quantization_levels=6,  # More compression levels
        n_subspaces=16,         # More PQ subspaces
        max_layers=20,          # Larger HNSW graph
        ef_construction=400,    # Higher quality index
        enable_spiral_optimization=True,
    )
    
    print(f"✓ Advanced database initialized: {db}")
    
    # Generate high-dimensional data
    vectors, metadata = generate_sample_data(n_vectors=1000, dimensions=256)
    
    # Insert with custom batch size
    print("\nInserting vectors with custom batch size...")
    vector_ids = db.insert(vectors, metadata, batch_size=50)
    print(f"✓ Inserted {len(vector_ids)} vectors")
    
    # Advanced search options
    print("\nAdvanced search options...")
    
    query = vectors[0]
    
    # Search with custom ef_search
    results_standard = db.search(query, k=10)
    results_thorough = db.search(query, k=10, ef_search=200)
    
    print(f"✓ Standard search: {len(results_standard)} results")
    print(f"✓ Thorough search: {len(results_thorough)} results")
    
    # Complex filters
    print("\nComplex filtering...")
    
    complex_filters = {
        "category": ["technology", "science"],
        "score": {"$gte": 0.3, "$lt": 0.8},
        "active": True
    }
    
    complex_results = db.search(query, k=10, filters=complex_filters)
    print(f"✓ Complex filter search: {len(complex_results)} results")
    
    # Database optimization
    print("\nDatabase optimization...")
    db.optimize()
    print("✓ Database optimized")
    
    # Final statistics
    final_stats = db.get_stats()
    print(f"\nFinal Statistics:")
    print(f"✓ Vectors: {final_stats.vector_count}")
    print(f"✓ Compression: {final_stats.compression_ratio:.1%}")
    print(f"✓ Avg query time: {final_stats.avg_query_time_ms:.2f}ms")
    
    db.close()


def main():
    """Run all demonstrations."""
    try:
        print("SpiralDeltaDB Examples")
        print("=" * 60)
        print("This demo showcases the key features of SpiralDeltaDB:")
        print("• Geometric spiral ordering for better compression")
        print("• Multi-tier delta encoding")
        print("• High-performance similarity search")
        print("• Flexible metadata filtering")
        print("• Memory-efficient storage")
        
        # Run demonstrations
        basic_operations_demo()
        compression_demo()
        advanced_features_demo()
        
        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("=" * 60)
        
        print("\nKey Takeaways:")
        print("• SpiralDeltaDB achieves significant compression while maintaining search quality")
        print("• Spiral ordering improves compression effectiveness")
        print("• Fast search performance with sub-millisecond queries")
        print("• Flexible filtering and metadata support")
        print("• Suitable for large-scale vector applications")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())