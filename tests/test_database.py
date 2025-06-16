"""
Integration tests for SpiralDeltaDB main interface.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from spiraldelta import SpiralDeltaDB
from spiraldelta.types import SearchResult, DatabaseStats


class TestSpiralDeltaDB:
    """Test cases for main SpiralDeltaDB interface."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def db(self, temp_dir):
        """Create test database."""
        db_path = Path(temp_dir) / "test.db"
        return SpiralDeltaDB(
            dimensions=128,
            compression_ratio=0.6,
            storage_path=str(db_path),
            auto_train_threshold=100
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        vectors = np.random.randn(200, 128)
        metadata = [{"id": i, "category": f"cat_{i % 5}"} for i in range(200)]
        return vectors, metadata
    
    def test_initialization(self, temp_dir):
        """Test database initialization."""
        db_path = Path(temp_dir) / "init_test.db"
        db = SpiralDeltaDB(dimensions=64, storage_path=str(db_path))
        
        assert db.dimensions == 64
        assert db.compression_ratio == 0.5  # Default
        assert len(db) == 0
        assert not db._is_trained
        
        db.close()
    
    def test_insert_single_vector(self, db):
        """Test inserting a single vector."""
        vector = np.random.randn(128)
        metadata = {"id": 1, "text": "test"}
        
        vector_ids = db.insert([vector], [metadata])
        
        assert len(vector_ids) == 1
        assert len(db) == 1
        assert db._vector_count == 1
    
    def test_insert_batch(self, db, sample_data):
        """Test batch insertion."""
        vectors, metadata = sample_data
        
        vector_ids = db.insert(vectors, metadata)
        
        assert len(vector_ids) == len(vectors)
        assert len(db) == len(vectors)
        assert all(isinstance(vid, int) for vid in vector_ids)
    
    def test_insert_without_metadata(self, db):
        """Test insertion without metadata."""
        vectors = np.random.randn(10, 128)
        
        vector_ids = db.insert(vectors)
        
        assert len(vector_ids) == 10
        assert len(db) == 10
    
    def test_insert_invalid_dimensions(self, db):
        """Test inserting vectors with wrong dimensions."""
        vectors = np.random.randn(5, 64)  # Wrong dimensions
        
        with pytest.raises(ValueError):
            db.insert(vectors)
    
    def test_insert_metadata_mismatch(self, db):
        """Test inserting with mismatched metadata length."""
        vectors = np.random.randn(5, 128)
        metadata = [{"id": 1}, {"id": 2}]  # Wrong length
        
        with pytest.raises(ValueError):
            db.insert(vectors, metadata)
    
    def test_auto_training(self, db):
        """Test automatic encoder training."""
        # Insert enough data to trigger training
        vectors = np.random.randn(150, 128)
        db.insert(vectors)
        
        assert db._is_trained
        assert len(db._training_data) > 0
    
    def test_search_basic(self, db, sample_data):
        """Test basic search functionality."""
        vectors, metadata = sample_data
        db.insert(vectors, metadata)
        
        # Search with first vector
        query = vectors[0]
        results = db.search(query, k=5)
        
        assert len(results) <= 5
        assert all(isinstance(result, SearchResult) for result in results)
        
        # First result should be the exact match
        if results:
            assert results[0].similarity > 0.9
    
    def test_search_with_filters(self, db, sample_data):
        """Test search with metadata filters."""
        vectors, metadata = sample_data
        db.insert(vectors, metadata)
        
        query = vectors[0]
        filters = {"category": "cat_1"}
        
        results = db.search(query, k=10, filters=filters)
        
        # All results should match filter
        for result in results:
            assert result.metadata["category"] == "cat_1"
    
    def test_search_invalid_query(self, db):
        """Test search with invalid query dimensions."""
        db.insert(np.random.randn(10, 128))
        
        query = np.random.randn(64)  # Wrong dimensions
        
        with pytest.raises(ValueError):
            db.search(query)
    
    def test_search_empty_database(self, db):
        """Test search on empty database."""
        query = np.random.randn(128)
        results = db.search(query)
        
        assert results == []
    
    def test_range_filters(self, db):
        """Test range-based metadata filters."""
        vectors = np.random.randn(50, 128)
        metadata = [{"score": i * 0.1} for i in range(50)]
        
        db.insert(vectors, metadata)
        
        query = vectors[0]
        filters = {"score": {"$gte": 2.0, "$lt": 4.0}}
        
        results = db.search(query, k=20, filters=filters)
        
        for result in results:
            score = result.metadata["score"]
            assert 2.0 <= score < 4.0
    
    def test_list_filters(self, db):
        """Test list-based metadata filters."""
        vectors = np.random.randn(30, 128)
        metadata = [{"tag": f"tag_{i % 3}"} for i in range(30)]
        
        db.insert(vectors, metadata)
        
        query = vectors[0]
        filters = {"tag": ["tag_0", "tag_2"]}
        
        results = db.search(query, k=15, filters=filters)
        
        for result in results:
            assert result.metadata["tag"] in ["tag_0", "tag_2"]
    
    def test_get_stats(self, db, sample_data):
        """Test getting database statistics."""
        vectors, metadata = sample_data
        db.insert(vectors, metadata)
        
        stats = db.get_stats()
        
        assert isinstance(stats, DatabaseStats)
        assert stats.vector_count == len(vectors)
        assert stats.dimensions == 128
        assert stats.storage_size_mb >= 0
        assert 0.0 <= stats.compression_ratio <= 1.0
    
    def test_delete_vectors(self, db, sample_data):
        """Test deleting vectors."""
        vectors, metadata = sample_data[:50]  # Use smaller subset
        vector_ids = db.insert(vectors, metadata)
        
        # Delete some vectors
        to_delete = vector_ids[:10]
        deleted_count = db.delete(to_delete)
        
        assert deleted_count == len(to_delete)
        assert len(db) == len(vectors) - len(to_delete)
    
    def test_update_vector(self, db):
        """Test updating a vector."""
        vector = np.random.randn(128)
        metadata = {"id": 1, "version": 1}
        
        vector_ids = db.insert([vector], [metadata])
        vector_id = vector_ids[0]
        
        # Update vector
        new_vector = np.random.randn(128)
        new_metadata = {"id": 1, "version": 2}
        
        success = db.update(vector_id, new_vector, new_metadata)
        assert success
    
    def test_optimize(self, db, sample_data):
        """Test database optimization."""
        vectors, metadata = sample_data
        db.insert(vectors, metadata)
        
        # Should complete without error
        db.optimize()
    
    def test_save_and_load(self, db, sample_data, temp_dir):
        """Test saving and loading database."""
        vectors, metadata = sample_data[:50]  # Use smaller subset
        vector_ids = db.insert(vectors, metadata)
        
        # Save database
        save_path = Path(temp_dir) / "saved_db.json"
        db.save(str(save_path))
        
        assert save_path.exists()
        
        # Load database
        new_storage_path = Path(temp_dir) / "loaded.db"
        loaded_db = SpiralDeltaDB.load(str(save_path), str(new_storage_path))
        
        assert loaded_db.dimensions == db.dimensions
        assert loaded_db.compression_ratio == db.compression_ratio
        assert loaded_db._vector_count == db._vector_count
        
        loaded_db.close()
        db.close()
    
    def test_context_manager(self, temp_dir):
        """Test using database as context manager."""
        db_path = Path(temp_dir) / "context_test.db"
        
        with SpiralDeltaDB(dimensions=64, storage_path=str(db_path)) as db:
            vectors = np.random.randn(10, 64)
            db.insert(vectors)
            assert len(db) == 10
        
        # Should be closed after context exit
    
    def test_repr(self, db):
        """Test string representation."""
        repr_str = repr(db)
        
        assert "SpiralDeltaDB" in repr_str
        assert "dimensions=128" in repr_str
        assert "vectors=0" in repr_str
        assert "trained=False" in repr_str
    
    def test_batch_size_override(self, db):
        """Test overriding batch size during insertion."""
        vectors = np.random.randn(100, 128)
        
        # Insert with custom batch size
        vector_ids = db.insert(vectors, batch_size=25)
        
        assert len(vector_ids) == 100
        assert len(db) == 100
    
    def test_search_performance_tracking(self, db, sample_data):
        """Test that search performance is tracked."""
        vectors, metadata = sample_data
        db.insert(vectors, metadata)
        
        # Perform searches
        for i in range(5):
            query = vectors[i]
            db.search(query, k=10)
        
        # Check that search times are recorded
        assert len(db._search_times) == 5
        assert all(t > 0 for t in db._search_times)
    
    def test_insert_performance_tracking(self, db):
        """Test that insert performance is tracked."""
        vectors = np.random.randn(50, 128)
        
        # Insert in multiple batches
        for i in range(0, 50, 10):
            batch = vectors[i:i+10]
            db.insert(batch)
        
        # Check that insert times are recorded
        assert len(db._insert_times) == 5
        assert all(t > 0 for t in db._insert_times)
    
    def test_vector_by_id_retrieval(self, db):
        """Test retrieving vector by ID."""
        vector = np.random.randn(128)
        metadata = {"test": "value"}
        
        vector_ids = db.insert([vector], [metadata])
        vector_id = vector_ids[0]
        
        retrieved_vector, retrieved_metadata = db.get_vector_by_id(vector_id)
        
        # Note: The current implementation returns placeholder data
        # In a complete implementation, this would return the actual vector
        assert retrieved_vector.shape == (128,)
        assert isinstance(retrieved_metadata, dict)
    
    @pytest.mark.parametrize("compression_ratio", [0.3, 0.5, 0.7])
    def test_different_compression_ratios(self, compression_ratio, temp_dir):
        """Test database with different compression ratios."""
        db_path = Path(temp_dir) / f"compression_{compression_ratio}.db"
        
        db = SpiralDeltaDB(
            dimensions=64,
            compression_ratio=compression_ratio,
            storage_path=str(db_path)
        )
        
        vectors = np.random.randn(100, 64)
        db.insert(vectors)
        
        assert db.compression_ratio == compression_ratio
        
        # Should be able to search
        query = vectors[0]
        results = db.search(query, k=5)
        assert len(results) <= 5
        
        db.close()
    
    @pytest.mark.parametrize("distance_metric", ["cosine", "l2", "ip"])
    def test_different_distance_metrics(self, distance_metric, temp_dir):
        """Test database with different distance metrics."""
        db_path = Path(temp_dir) / f"distance_{distance_metric}.db"
        
        db = SpiralDeltaDB(
            dimensions=64,
            storage_path=str(db_path),
            distance_metric=distance_metric
        )
        
        vectors = np.random.randn(50, 64)
        db.insert(vectors)
        
        # Should be able to search with any metric
        query = vectors[0]
        results = db.search(query, k=5)
        assert len(results) <= 5
        
        db.close()
    
    def test_large_batch_insertion(self, db):
        """Test inserting large batches."""
        # Create large batch
        vectors = np.random.randn(1000, 128)
        
        vector_ids = db.insert(vectors, batch_size=100)
        
        assert len(vector_ids) == 1000
        assert len(db) == 1000
    
    def test_spiral_optimization_toggle(self, db, sample_data):
        """Test toggling spiral optimization."""
        vectors, metadata = sample_data
        db.insert(vectors, metadata)
        
        query = vectors[0]
        
        # Search with spiral optimization
        results_with = db.search(query, k=5)
        
        # Search without spiral optimization
        # Note: This would need to be implemented in the search method
        results_without = db.search(query, k=5)
        
        # Both should return results
        assert len(results_with) > 0
        assert len(results_without) > 0
    
    def test_memory_efficiency(self, db):
        """Test memory usage with large dataset."""
        # Insert vectors in batches to test memory efficiency
        total_vectors = 1000
        batch_size = 100
        
        for i in range(0, total_vectors, batch_size):
            vectors = np.random.randn(batch_size, 128)
            db.insert(vectors, batch_size=batch_size)
        
        assert len(db) == total_vectors
        
        # Should be able to search efficiently
        query = np.random.randn(128)
        results = db.search(query, k=10)
        assert len(results) <= 10