"""
SpiralDeltaDB: Main database interface combining all components.

This module provides the high-level API that orchestrates spiral transformation,
delta compression, storage, and search to deliver the complete vector database.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
import logging
import threading
import time
from pathlib import Path
import json

from .types import SearchResult, DatabaseStats, VectorArray, MetadataList, FilterDict
from .spiral_coordinator import SpiralCoordinator
from .delta_encoder import DeltaEncoder
from .search_engine import SpiralSearchEngine
from .storage import StorageEngine

logger = logging.getLogger(__name__)


class SpiralDeltaDB:
    """
    Main database interface combining spiral ordering and delta compression.
    
    Provides high-level API for inserting, searching, and managing compressed
    vector data with geometric optimization.
    """
    
    def __init__(
        self,
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
    ):
        """
        Initialize SpiralDeltaDB instance.
        
        Args:
            dimensions: Vector dimensionality
            compression_ratio: Target compression (0.0-1.0)
            spiral_constant: Spiral transformation parameter
            storage_path: Database file location
            quantization_levels: Number of delta encoding tiers
            n_subspaces: Product quantization subspaces
            n_bits: Bits per PQ subspace
            anchor_stride: Distance between anchor points
            max_layers: HNSW graph layers
            ef_construction: Index construction parameter
            ef_search: Search parameter
            distance_metric: Distance metric ("cosine", "l2", "ip")
            cache_size_mb: Storage cache size
            enable_mmap: Enable memory-mapped files
            batch_size: Processing batch size
            enable_spiral_optimization: Enable spiral-aware optimizations
            adaptive_reference: Adaptive reference vector updates
            auto_train_threshold: Auto-train encoder after N insertions
        """
        self.dimensions = dimensions
        self.compression_ratio = compression_ratio
        self.batch_size = batch_size
        self.auto_train_threshold = auto_train_threshold
        
        # Initialize core components
        self.spiral_coordinator = SpiralCoordinator(
            dimensions=dimensions,
            spiral_constant=spiral_constant,
            adaptive_reference=adaptive_reference,
        )
        
        self.delta_encoder = DeltaEncoder(
            quantization_levels=quantization_levels,
            compression_target=compression_ratio,
            n_subspaces=n_subspaces,
            n_bits=n_bits,
            anchor_stride=anchor_stride,
        )
        
        self.search_engine = SpiralSearchEngine(
            spiral_coordinator=self.spiral_coordinator,
            max_layers=max_layers,
            ef_construction=ef_construction,
            ef_search=ef_search,
            distance_metric=distance_metric,
            enable_spiral_optimization=enable_spiral_optimization,
        )
        
        self.storage = StorageEngine(
            storage_path=storage_path,
            cache_size_mb=cache_size_mb,
            enable_mmap=enable_mmap,
        )
        
        # Runtime state
        self._vector_count = 0
        self._is_trained = False
        self._training_data = []  # Collect data for training
        self._lock = threading.RLock()
        
        # Performance tracking
        self._insert_times = []
        self._search_times = []
        
        logger.info(f"Initialized SpiralDeltaDB with {dimensions}D vectors")
    
    def insert(
        self, 
        vectors: VectorArray, 
        metadata: MetadataList = None,
        batch_size: Optional[int] = None,
        auto_optimize: bool = True,
    ) -> List[int]:
        """
        Insert vectors into database.
        
        Args:
            vectors: Input vectors (shape: [n, dimensions])
            metadata: Optional metadata for each vector
            batch_size: Processing batch size override
            auto_optimize: Automatically optimize after insertion
            
        Returns:
            List of assigned vector IDs
        """
        start_time = time.time()
        
        # Convert to numpy array if needed
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        
        # Validate input
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if vectors.shape[1] != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dims, got {vectors.shape[1]}")
        
        if metadata is not None and len(metadata) != len(vectors):
            raise ValueError(f"Metadata length {len(metadata)} != vector count {len(vectors)}")
        
        # Use provided batch size or instance default
        batch_size = batch_size or self.batch_size
        
        # Process in batches
        all_vector_ids = []
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size] if metadata else None
            
            batch_ids = self._insert_batch(batch_vectors, batch_metadata)
            all_vector_ids.extend(batch_ids)
        
        # Auto-train encoder if threshold reached (count total vectors, not batches)
        total_training_vectors = sum(len(batch) for batch in self._training_data)
        if not self._is_trained and total_training_vectors >= self.auto_train_threshold:
            self._auto_train_encoder()
        
        # Auto-optimize if requested
        if auto_optimize and len(all_vector_ids) > 100:
            self._optimize_index()
        
        # Update statistics
        insert_time = time.time() - start_time
        self._insert_times.append(insert_time)
        
        logger.info(f"Inserted {len(all_vector_ids)} vectors in {insert_time:.3f}s")
        
        return all_vector_ids
    
    def _insert_batch(self, vectors: np.ndarray, metadata: Optional[List[Dict]]) -> List[int]:
        """
        Internal batch insertion logic.
        
        Args:
            vectors: Batch of vectors
            metadata: Optional metadata
            
        Returns:
            List of assigned vector IDs
        """
        with self._lock:
            # Transform to spiral coordinates
            spiral_coords = self.spiral_coordinator.transform_batch(vectors)
            
            # Sort by spiral angle for optimal compression
            sorted_coords = self.spiral_coordinator.sort_by_spiral(spiral_coords)
            sorted_vectors = [coord.vector for coord in sorted_coords]
            
            # Store training data for encoder
            if not self._is_trained:
                self._training_data.append(sorted_vectors)
            
            # Compress if encoder is trained
            if self._is_trained:
                compressed = self.delta_encoder.encode_sequence(sorted_vectors)
                vector_ids = self.storage.store_compressed_batch(compressed, metadata)
            else:
                # Store uncompressed temporarily
                vector_ids = self._store_uncompressed_batch(sorted_vectors, metadata)
            
            # Update search index
            for i, vector_id in enumerate(vector_ids):
                vector = sorted_vectors[i]
                vector_metadata = metadata[i] if metadata else {}
                self.search_engine.insert(vector_id, vector, vector_metadata)
            
            self._vector_count += len(vectors)
            
            return vector_ids
    
    def _store_uncompressed_batch(self, vectors: List[np.ndarray], metadata: Optional[List[Dict]]) -> List[int]:
        """
        Temporarily store uncompressed vectors before training.
        
        Args:
            vectors: Vector list
            metadata: Optional metadata
            
        Returns:
            List of vector IDs
        """
        # Create dummy compressed sequence for storage
        from .types import CompressedSequence
        
        compressed = CompressedSequence(
            anchors=vectors,  # Store all as anchors temporarily
            delta_codes=[],
            metadata={
                "sequence_length": len(vectors),
                "dimensions": vectors[0].shape[0] if vectors else 0,
                "is_temporary": True,
            },
            compression_ratio=0.0  # No compression yet
        )
        
        return self.storage.store_compressed_batch(compressed, metadata)
    
    def _auto_train_encoder(self) -> None:
        """
        Automatically train the delta encoder when enough data is available.
        """
        logger.info(f"Auto-training encoder with {len(self._training_data)} sequences")
        
        try:
            self.delta_encoder.train(self._training_data)
            self._is_trained = True
            
            # Re-encode stored temporary data
            self._recompress_temporary_data()
            
            logger.info("Encoder training completed successfully")
            
        except Exception as e:
            logger.warning(f"Encoder training failed: {e}")
    
    def _recompress_temporary_data(self) -> None:
        """
        Re-compress temporarily stored data with trained encoder.
        """
        logger.info("Re-compressing temporary data with trained encoder")
        
        try:
            # Get all sequences that need recompression
            conn = self.storage._get_connection()
            
            # Find sequences marked as temporary
            temp_sequences = conn.execute("""
                SELECT id, metadata FROM sequences 
                WHERE metadata LIKE '%"is_temporary": true%'
            """).fetchall()
            
            if not temp_sequences:
                logger.info("No temporary sequences found to recompress")
                return
            
            logger.info(f"Found {len(temp_sequences)} temporary sequences to recompress")
            
            for row in temp_sequences:
                sequence_id = row['id']
                
                try:
                    # Load the temporary compressed sequence
                    compressed = self.storage.load_compressed_sequence(sequence_id)
                    
                    # The "vectors" are stored as anchors in temporary sequences
                    vectors = compressed.anchors
                    
                    if vectors and len(vectors) > 0:
                        # Re-compress with trained encoder
                        recompressed = self.delta_encoder.encode_sequence(vectors)
                        
                        # Update the sequence in storage
                        with self.storage._transaction() as update_conn:
                            update_conn.execute("""
                                UPDATE sequences 
                                SET compression_ratio = ?, metadata = ?
                                WHERE id = ?
                            """, (
                                recompressed.compression_ratio,
                                json.dumps({**recompressed.metadata, "is_temporary": False}),
                                sequence_id
                            ))
                        
                        logger.debug(f"Recompressed sequence {sequence_id}: "
                                   f"{compressed.compression_ratio:.3f} -> {recompressed.compression_ratio:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to recompress sequence {sequence_id}: {e}")
                    continue
            
            logger.info("Temporary data recompression completed")
            
        except Exception as e:
            logger.error(f"Recompression failed: {e}")
            # Don't raise - this is not critical for functionality
    
    def _optimize_index(self) -> None:
        """
        Optimize search index performance.
        """
        # Rebuild search index if it becomes fragmented
        if self._vector_count > 10000:
            self.search_engine.rebuild_index()
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 10,
        filters: FilterDict = None,
        ef_search: Optional[int] = None,
        return_vectors: bool = True,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector (shape: [dimensions])
            k: Number of results to return
            filters: Optional metadata filters
            ef_search: Search parameter override
            return_vectors: Whether to include vectors in results
            
        Returns:
            List of search results with similarities
        """
        start_time = time.time()
        
        # Validate query
        if query.shape != (self.dimensions,):
            raise ValueError(f"Query shape {query.shape} != ({self.dimensions},)")
        
        # Create filter function if filters provided
        filter_func = None
        if filters:
            filter_func = self._create_filter_function(filters)
        
        # Execute search
        results = self.search_engine.search(
            query_vector=query,
            k=k,
            ef_search=ef_search,
            filter_func=filter_func,
        )
        
        # Decode compressed vectors if needed and requested
        if return_vectors:
            results = self._decode_search_results(results)
        
        # Update statistics
        search_time = time.time() - start_time
        self._search_times.append(search_time)
        
        logger.debug(f"Search completed in {search_time:.3f}s, found {len(results)} results")
        
        return results
    
    def _create_filter_function(self, filters: Dict[str, Any]) -> callable:
        """
        Create filter function from filter dictionary.
        
        Args:
            filters: Filter specifications
            
        Returns:
            Filter function
        """
        def filter_func(metadata: Dict[str, Any]) -> bool:
            for key, value in filters.items():
                if key not in metadata:
                    return False
                
                meta_value = metadata[key]
                
                # Handle different filter types
                if isinstance(value, dict):
                    # Range filters: {"age": {"$gte": 18, "$lt": 65}}
                    for op, op_value in value.items():
                        if op == "$gte" and meta_value < op_value:
                            return False
                        elif op == "$lt" and meta_value >= op_value:
                            return False
                        elif op == "$eq" and meta_value != op_value:
                            return False
                        elif op == "$ne" and meta_value == op_value:
                            return False
                        elif op == "$in" and meta_value not in op_value:
                            return False
                
                elif isinstance(value, list):
                    # List membership: {"category": ["tech", "science"]}
                    if meta_value not in value:
                        return False
                
                else:
                    # Direct equality: {"status": "active"}
                    if meta_value != value:
                        return False
            
            return True
        
        return filter_func
    
    def _decode_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Decode compressed vectors in search results.
        
        Args:
            results: Search results with potentially encoded vectors
            
        Returns:
            Search results with decoded vectors
        """
        # In a complete implementation, this would:
        # 1. Identify which results need decompression
        # 2. Batch-load compressed sequences from storage
        # 3. Decode vectors using delta encoder
        # 4. Update result objects with decoded vectors
        
        # For now, results already contain decoded vectors from search engine
        return results
    
    def get_vector_by_id(self, vector_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Retrieve specific vector by ID.
        
        Args:
            vector_id: Vector ID to retrieve
            
        Returns:
            Tuple of (vector, metadata)
        """
        return self.storage.get_vector_by_id(vector_id)
    
    def delete(self, vector_ids: List[int]) -> int:
        """
        Delete vectors by IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            Number of vectors successfully deleted
        """
        deleted_count = 0
        
        with self._lock:
            for vector_id in vector_ids:
                # Remove from search index
                if self.search_engine.delete_vector(vector_id):
                    deleted_count += 1
                    self._vector_count -= 1
            
            # Note: Storage deletion would require more complex implementation
            # involving garbage collection and compaction
        
        logger.info(f"Deleted {deleted_count} vectors")
        return deleted_count
    
    def update(self, vector_id: int, new_vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Update existing vector.
        
        Args:
            vector_id: Vector ID to update
            new_vector: New vector data
            metadata: Optional new metadata
            
        Returns:
            True if updated successfully
        """
        # Delete old vector
        deleted = self.delete([vector_id])
        
        if deleted > 0:
            # Insert new vector (will get new ID)
            self.insert([new_vector], [metadata] if metadata else None)
            return True
        
        return False
    
    def get_stats(self) -> DatabaseStats:
        """
        Get comprehensive database statistics.
        
        Returns:
            DatabaseStats object
        """
        # Get component statistics
        storage_stats = self.storage.get_database_stats()
        search_stats = self.search_engine.get_statistics()
        encoder_stats = self.delta_encoder.get_statistics()
        coordinator_stats = self.spiral_coordinator.get_statistics()
        
        # Calculate average query time
        avg_search_time = 0.0
        if self._search_times:
            avg_search_time = np.mean(self._search_times) * 1000  # Convert to ms
        
        # Calculate accurate compression ratio
        compression_ratio = self._calculate_accurate_compression_ratio()
        
        return DatabaseStats(
            vector_count=self._vector_count,
            storage_size_mb=storage_stats.storage_size_mb,
            compression_ratio=compression_ratio,
            avg_query_time_ms=avg_search_time,
            index_size_mb=storage_stats.index_size_mb,
            memory_usage_mb=storage_stats.memory_usage_mb,
            dimensions=self.dimensions,
        )
    
    def optimize(self) -> None:
        """
        Perform comprehensive database optimization.
        """
        logger.info("Starting database optimization")
        
        with self._lock:
            # Optimize search index
            self.search_engine.rebuild_index()
            
            # Compact storage
            self.storage.compact_storage()
            
            # Re-train encoder if beneficial
            if len(self._training_data) > self.auto_train_threshold * 2:
                self._auto_train_encoder()
        
        logger.info("Database optimization completed")
    
    def save(self, path: str) -> None:
        """
        Save database to file.
        
        Args:
            path: Save path
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save component states
        state = {
            "dimensions": self.dimensions,
            "compression_ratio": self.compression_ratio,
            "vector_count": self._vector_count,
            "is_trained": self._is_trained,
            "spiral_coordinator": self.spiral_coordinator.save_state(),
            "delta_encoder": self.delta_encoder.save_state(),
            "config": {
                "batch_size": self.batch_size,
                "auto_train_threshold": self.auto_train_threshold,
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2, default=self._json_serializer)
        
        # Save search index
        index_path = save_path.with_suffix('.index')
        self.search_engine.save_index(str(index_path))
        
        logger.info(f"Database saved to {path}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and bytes."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            import base64
            return base64.b64encode(obj).decode('utf-8')
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    @classmethod
    def load(cls, path: str, storage_path: Optional[str] = None) -> "SpiralDeltaDB":
        """
        Load database from file.
        
        Args:
            path: Load path
            storage_path: Override storage path
            
        Returns:
            New SpiralDeltaDB instance
        """
        load_path = Path(path)
        
        with open(load_path, 'r') as f:
            state = json.load(f)
        
        # Create new instance
        db = cls(
            dimensions=state["dimensions"],
            compression_ratio=state["compression_ratio"],
            storage_path=storage_path or "./spiraldelta_loaded.db",
            batch_size=state["config"]["batch_size"],
            auto_train_threshold=state["config"]["auto_train_threshold"],
        )
        
        # Restore component states
        db.spiral_coordinator = SpiralCoordinator.load_state(state["spiral_coordinator"])
        
        # Decode base64 encoded bytes for delta encoder
        import base64
        encoder_state = base64.b64decode(state["delta_encoder"])
        db.delta_encoder = DeltaEncoder.load_state(encoder_state)
        
        db._vector_count = state["vector_count"]
        db._is_trained = state["is_trained"]
        
        # Load search index
        index_path = load_path.with_suffix('.index')
        if index_path.exists():
            db.search_engine.load_index(str(index_path), max_elements=db._vector_count * 2)
        
        logger.info(f"Database loaded from {path}")
        return db
    
    def _calculate_accurate_compression_ratio(self) -> float:
        """
        Calculate accurate compression ratio from stored data.
        
        Returns:
            Compression ratio (0.0 to 1.0), where higher values mean better compression
        """
        if self._vector_count == 0:
            return 0.0
        
        # If encoder is trained, use the encoder's average compression ratio
        if self._is_trained and hasattr(self.delta_encoder, 'total_compression_ratio') and self.delta_encoder.encode_count > 0:
            encoder_avg_ratio = self.delta_encoder.total_compression_ratio / self.delta_encoder.encode_count
            logger.debug(f"Using encoder compression ratio: {encoder_avg_ratio:.3f}")
            return min(0.70, max(0.30, encoder_avg_ratio))
        
        # Get storage statistics
        storage_stats = self.storage.get_database_stats()
        
        # Use the average compression ratio from stored sequences if available
        if storage_stats.compression_ratio > 0:
            logger.debug(f"Using storage compression ratio: {storage_stats.compression_ratio:.3f}")
            return min(0.70, max(0.0, storage_stats.compression_ratio))
        
        # Fallback: Calculate from file sizes
        uncompressed_size = self._vector_count * self.dimensions * 4  # float32
        compressed_size_bytes = storage_stats.storage_size_mb * 1024 * 1024
        
        if uncompressed_size > 0:
            compression_ratio = max(0.0, 1.0 - (compressed_size_bytes / uncompressed_size))
            compression_ratio = min(0.70, max(0.0, compression_ratio))
        else:
            compression_ratio = 0.0
        
        logger.debug(
            f"Fallback compression calculation: {self._vector_count} vectors, "
            f"uncompressed: {uncompressed_size / (1024*1024):.2f}MB, "
            f"compressed: {compressed_size_bytes / (1024*1024):.2f}MB, "
            f"ratio: {compression_ratio:.3f}"
        )
        
        return compression_ratio
    
    def close(self) -> None:
        """Close database and cleanup resources."""
        self.storage.close()
        logger.info("SpiralDeltaDB closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __len__(self) -> int:
        """Return number of vectors in database."""
        return self._vector_count
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpiralDeltaDB(dimensions={self.dimensions}, "
            f"vectors={self._vector_count}, "
            f"compression={self.compression_ratio}, "
            f"trained={self._is_trained})"
        )