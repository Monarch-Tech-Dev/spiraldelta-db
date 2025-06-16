"""
SpiralSearchEngine: High-performance similarity search optimized for spiral-ordered data.

This module implements HNSW-based search with spiral coordinate optimization
for efficient similarity search on compressed vector data.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
import hnswlib
import threading
import time
import logging
from collections import defaultdict
from .types import SpiralCoordinate, SearchResult
from .spiral_coordinator import SpiralCoordinator

logger = logging.getLogger(__name__)


class SpiralSearchEngine:
    """
    HNSW-based search optimized for spiral-ordered compressed data.
    
    Combines hierarchical navigable small world graphs with spiral coordinate
    awareness for efficient similarity search.
    """
    
    def __init__(
        self,
        spiral_coordinator: SpiralCoordinator,
        storage_engine: Optional[Any] = None,  # Avoid circular import
        max_layers: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        distance_metric: str = "cosine",
        enable_spiral_optimization: bool = True,
    ):
        """
        Initialize SpiralSearchEngine.
        
        Args:
            spiral_coordinator: Reference to spiral coordinator
            max_layers: Maximum HNSW graph layers
            ef_construction: Construction parameter for index building
            ef_search: Search parameter for queries
            distance_metric: Distance metric ("cosine", "l2", "ip")
            enable_spiral_optimization: Enable spiral-aware optimizations
        """
        self.spiral_coordinator = spiral_coordinator
        self.storage_engine = storage_engine
        self.max_layers = max_layers
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.distance_metric = distance_metric
        self.enable_spiral_optimization = enable_spiral_optimization
        
        # HNSW index
        self.hnsw_index = None
        self.dimensions = spiral_coordinator.dimensions
        
        # Spiral coordinate mapping
        self.vector_id_to_spiral = {}  # vector_id -> SpiralCoordinate
        self.spiral_regions = {}       # region_id -> list of vector_ids
        
        # Performance tracking
        self.search_count = 0
        self.total_search_time = 0.0
        self.spiral_cache = {}         # Cache for spiral coordinates
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized SpiralSearchEngine with {distance_metric} distance")
    
    def _init_hnsw_index(self, max_elements: int) -> None:
        """
        Initialize HNSW index with given capacity.
        
        Args:
            max_elements: Maximum number of elements the index can hold
        """
        # Map distance metrics
        distance_map = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",  # inner product
        }
        
        if self.distance_metric not in distance_map:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        self.hnsw_index = hnswlib.Index(
            space=distance_map[self.distance_metric],
            dim=self.dimensions
        )
        
        self.hnsw_index.init_index(
            max_elements=max_elements,
            ef_construction=self.ef_construction,
            M=16  # Number of bi-directional links for each node
        )
        
        self.hnsw_index.set_ef(self.ef_search)
        
        logger.info(f"Initialized HNSW index with capacity {max_elements}")
    
    def insert(self, vector_id: int, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """
        Add vector to search index.
        
        Args:
            vector_id: Unique vector identifier
            vector: Vector data
            metadata: Optional metadata
        """
        with self._lock:
            # Initialize index if needed
            if self.hnsw_index is None:
                # Start with reasonable default size, can be resized
                self._init_hnsw_index(max_elements=1000000)
            
            # Transform to spiral coordinates
            spiral_coord = self.spiral_coordinator.transform(vector)
            spiral_coord.metadata = metadata or {}
            
            # Store spiral coordinate mapping
            self.vector_id_to_spiral[vector_id] = spiral_coord
            
            # Add to HNSW index
            self.hnsw_index.add_items([vector], [vector_id])
            
            # Update spiral regions if optimization is enabled
            if self.enable_spiral_optimization:
                self._update_spiral_regions(vector_id, spiral_coord)
        
        logger.debug(f"Inserted vector {vector_id} into search index")
    
    def _update_spiral_regions(self, vector_id: int, spiral_coord: SpiralCoordinate) -> None:
        """
        Update spiral region assignments for optimization.
        
        Args:
            vector_id: Vector ID
            spiral_coord: Spiral coordinate
        """
        # Assign to spiral region based on angle
        region_id = int(spiral_coord.theta / (2 * np.pi) * 100)  # 100 regions
        
        if region_id not in self.spiral_regions:
            self.spiral_regions[region_id] = []
        
        self.spiral_regions[region_id].append(vector_id)
    
    def batch_insert(self, vectors: List[Tuple[int, np.ndarray]], metadata: Optional[List[Dict]] = None) -> None:
        """
        Efficiently insert multiple vectors.
        
        Args:
            vectors: List of (vector_id, vector) tuples
            metadata: Optional metadata for each vector
        """
        if not vectors:
            return
        
        with self._lock:
            # Initialize index if needed
            if self.hnsw_index is None:
                self._init_hnsw_index(max_elements=max(1000000, len(vectors) * 2))
            
            # Prepare data
            vector_ids = []
            vector_data = []
            spiral_coords = []
            
            for i, (vector_id, vector) in enumerate(vectors):
                # Transform to spiral coordinates
                spiral_coord = self.spiral_coordinator.transform(vector)
                spiral_coord.metadata = metadata[i] if metadata and i < len(metadata) else {}
                
                vector_ids.append(vector_id)
                vector_data.append(vector)
                spiral_coords.append(spiral_coord)
                
                # Store mapping
                self.vector_id_to_spiral[vector_id] = spiral_coord
                
                # Update spiral regions
                if self.enable_spiral_optimization:
                    self._update_spiral_regions(vector_id, spiral_coord)
            
            # Batch insert into HNSW
            self.hnsw_index.add_items(vector_data, vector_ids)
        
        logger.info(f"Batch inserted {len(vectors)} vectors")
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10,
        ef_search: Optional[int] = None,
        filter_func: Optional[Callable[[Dict], bool]] = None,
        use_spiral_optimization: bool = None,
    ) -> List[SearchResult]:
        """
        Search for k most similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            ef_search: Search parameter override
            filter_func: Optional function to filter results by metadata
            use_spiral_optimization: Override spiral optimization setting
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        if self.hnsw_index is None:
            return []
        
        # Set search parameters
        if ef_search is not None:
            original_ef = self.hnsw_index.get_current_count() if hasattr(self.hnsw_index, 'get_current_count') else self.ef_search
            self.hnsw_index.set_ef(ef_search)
        else:
            original_ef = self.ef_search
        
        try:
            # Use spiral optimization if enabled
            use_spiral = (
                use_spiral_optimization 
                if use_spiral_optimization is not None 
                else self.enable_spiral_optimization
            )
            
            if use_spiral:
                results = self._spiral_optimized_search(query_vector, k, filter_func)
            else:
                results = self._standard_search(query_vector, k, filter_func)
            
        finally:
            # Restore original ef
            if ef_search is not None:
                self.hnsw_index.set_ef(original_ef if hasattr(self.hnsw_index, 'get_current_count') else self.ef_search)
        
        # Update statistics
        search_time = time.time() - start_time
        self.search_count += 1
        self.total_search_time += search_time
        
        logger.debug(f"Search completed in {search_time:.3f}s, found {len(results)} results")
        
        return results
    
    def _standard_search(
        self, 
        query_vector: np.ndarray, 
        k: int,
        filter_func: Optional[Callable[[Dict], bool]] = None
    ) -> List[SearchResult]:
        """
        Standard HNSW search without spiral optimizations.
        
        Args:
            query_vector: Query vector
            k: Number of results
            filter_func: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        # Search HNSW index
        # Over-fetch to allow for filtering
        current_count = self.hnsw_index.get_current_count() if self.hnsw_index else 0
        search_k = min(k * 3, max(1, current_count))
        
        labels, distances = self.hnsw_index.knn_query([query_vector], k=search_k)
        
        results = []
        for label, distance in zip(labels[0], distances[0]):
            vector_id = int(label)
            
            # Get spiral coordinate and metadata
            if vector_id in self.vector_id_to_spiral:
                spiral_coord = self.vector_id_to_spiral[vector_id]
                metadata = spiral_coord.metadata
                
                # Apply filter if provided
                if filter_func and not filter_func(metadata):
                    continue
                
                # Convert distance to similarity
                similarity = self._distance_to_similarity(distance)
                
                # Get vector from storage if available, otherwise use cached
                if self.storage_engine:
                    try:
                        stored_vector, stored_metadata = self.storage_engine.get_vector_by_id(vector_id)
                        vector = stored_vector
                        # Merge metadata
                        combined_metadata = {**metadata, **stored_metadata}
                    except Exception as e:
                        logger.warning(f"Failed to get vector {vector_id} from storage: {e}")
                        vector = spiral_coord.vector.copy()
                        combined_metadata = metadata
                else:
                    vector = spiral_coord.vector.copy()
                    combined_metadata = metadata
                
                result = SearchResult(
                    id=vector_id,
                    similarity=similarity,
                    vector=vector,
                    metadata=combined_metadata,
                    distance=distance
                )
                results.append(result)
                
                if len(results) >= k:
                    break
        
        return results
    
    def _spiral_optimized_search(
        self, 
        query_vector: np.ndarray, 
        k: int,
        filter_func: Optional[Callable[[Dict], bool]] = None
    ) -> List[SearchResult]:
        """
        Spiral-optimized search using coordinate awareness.
        
        Args:
            query_vector: Query vector
            k: Number of results
            filter_func: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        # Transform query to spiral coordinates
        query_spiral = self.spiral_coordinator.transform(query_vector)
        
        # Find relevant spiral regions
        relevant_regions = self._find_relevant_spiral_regions(query_spiral, k)
        
        # Collect candidates from relevant regions
        candidates = []
        for region_id in relevant_regions:
            if region_id in self.spiral_regions:
                candidates.extend(self.spiral_regions[region_id])
        
        if not candidates:
            # Fallback to standard search
            return self._standard_search(query_vector, k, filter_func)
        
        # Limit candidates to reasonable number for performance
        if len(candidates) > k * 20:
            # Use HNSW to pre-filter candidates
            limited_candidates = self._prefilter_candidates(query_vector, candidates, k * 10)
            candidates = limited_candidates
        
        # Compute exact similarities for candidates
        results = []
        for vector_id in candidates:
            if vector_id not in self.vector_id_to_spiral:
                continue
            
            spiral_coord = self.vector_id_to_spiral[vector_id]
            metadata = spiral_coord.metadata
            
            # Apply filter if provided
            if filter_func and not filter_func(metadata):
                continue
            
            # Compute similarity
            similarity = self._compute_similarity(query_vector, spiral_coord.vector)
            distance = self._similarity_to_distance(similarity)
            
            # Get vector from storage if available
            if self.storage_engine:
                try:
                    stored_vector, stored_metadata = self.storage_engine.get_vector_by_id(vector_id)
                    vector = stored_vector
                    # Merge metadata
                    combined_metadata = {**metadata, **stored_metadata}
                except Exception as e:
                    logger.warning(f"Failed to get vector {vector_id} from storage: {e}")
                    vector = spiral_coord.vector.copy()
                    combined_metadata = metadata
            else:
                vector = spiral_coord.vector.copy()
                combined_metadata = metadata
            
            result = SearchResult(
                id=vector_id,
                similarity=similarity,
                vector=vector,
                metadata=combined_metadata,
                distance=distance
            )
            results.append(result)
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]
    
    def _find_relevant_spiral_regions(self, query_spiral: SpiralCoordinate, k: int) -> List[int]:
        """
        Find spiral regions relevant to query.
        
        Args:
            query_spiral: Query spiral coordinate
            k: Number of results needed
            
        Returns:
            List of relevant region IDs
        """
        query_region = int(query_spiral.theta / (2 * np.pi) * 100)
        
        # Start with query region and expand outward
        relevant_regions = [query_region]
        
        # Add neighboring regions (adaptive radius)
        region_radius = max(1, min(10, k // 10))  # Adaptive radius based on k
        for offset in range(1, region_radius + 1):
            relevant_regions.append((query_region + offset) % 100)
            relevant_regions.append((query_region - offset) % 100)
        
        # Also include regions at similar radius levels
        radius_tolerance = 0.2  # 20% tolerance
        for region_id, vector_list in self.spiral_regions.items():
            if region_id not in relevant_regions and vector_list:
                # Check if any vectors in this region have similar radius
                sample_vector_id = vector_list[0]
                if sample_vector_id in self.vector_id_to_spiral:
                    sample_coord = self.vector_id_to_spiral[sample_vector_id]
                    radius_diff = abs(sample_coord.radius - query_spiral.radius)
                    radius_ratio = radius_diff / max(query_spiral.radius, 0.1)
                    
                    if radius_ratio <= radius_tolerance:
                        relevant_regions.append(region_id)
        
        return relevant_regions
    
    def _prefilter_candidates(
        self, 
        query_vector: np.ndarray, 
        candidates: List[int], 
        target_count: int
    ) -> List[int]:
        """
        Use HNSW to pre-filter candidate list.
        
        Args:
            query_vector: Query vector
            candidates: Candidate vector IDs
            target_count: Target number of candidates
            
        Returns:
            Filtered candidate list
        """
        # This is a simplified implementation
        # In practice, you might use a separate smaller index
        # or implement custom filtering in the HNSW library
        
        # For now, just sample randomly
        import random
        if len(candidates) <= target_count:
            return candidates
        
        return random.sample(candidates, target_count)
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if self.distance_metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            return max(0.0, (cosine_sim + 1.0) / 2.0)  # Normalize to [0, 1]
            
        elif self.distance_metric == "l2":
            # L2 distance -> similarity
            distance = np.linalg.norm(vec1 - vec2)
            return 1.0 / (1.0 + distance)
            
        elif self.distance_metric == "ip":
            # Inner product
            return max(0.0, np.dot(vec1, vec2))
        
        return 0.0
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance to similarity score.
        
        Args:
            distance: Distance value
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if self.distance_metric == "cosine":
            # Cosine distance is 1 - cosine_similarity
            return max(0.0, 1.0 - distance)
        elif self.distance_metric == "l2":
            # L2 distance
            return 1.0 / (1.0 + distance)
        elif self.distance_metric == "ip":
            # Inner product (higher is better, so negate distance)
            return max(0.0, -distance)
        
        return 0.0
    
    def _similarity_to_distance(self, similarity: float) -> float:
        """
        Convert similarity to distance.
        
        Args:
            similarity: Similarity score
            
        Returns:
            Distance value
        """
        if self.distance_metric == "cosine":
            return 1.0 - similarity
        elif self.distance_metric == "l2":
            return max(0.0, (1.0 / similarity) - 1.0)
        elif self.distance_metric == "ip":
            return -similarity
        
        return 0.0
    
    def range_search(
        self, 
        query_vector: np.ndarray, 
        radius: float,
        max_results: int = 1000
    ) -> List[SearchResult]:
        """
        Find all vectors within a given radius.
        
        Args:
            query_vector: Query vector
            radius: Search radius
            max_results: Maximum number of results
            
        Returns:
            List of SearchResult objects within radius
        """
        if self.hnsw_index is None:
            return []
        
        # HNSW doesn't directly support range search, so we use a large k
        # and filter by distance
        large_k = min(max_results * 2, self.hnsw_index.get_current_count())
        
        labels, distances = self.hnsw_index.knn_query([query_vector], k=large_k)
        
        results = []
        for label, distance in zip(labels[0], distances[0]):
            if distance > radius:
                break  # Results are sorted by distance
            
            vector_id = int(label)
            if vector_id in self.vector_id_to_spiral:
                spiral_coord = self.vector_id_to_spiral[vector_id]
                similarity = self._distance_to_similarity(distance)
                
                # Get vector from storage if available
                if self.storage_engine:
                    try:
                        stored_vector, stored_metadata = self.storage_engine.get_vector_by_id(vector_id)
                        vector = stored_vector
                        combined_metadata = {**spiral_coord.metadata, **stored_metadata}
                    except Exception as e:
                        logger.warning(f"Failed to get vector {vector_id} from storage: {e}")
                        vector = spiral_coord.vector.copy()
                        combined_metadata = spiral_coord.metadata
                else:
                    vector = spiral_coord.vector.copy()
                    combined_metadata = spiral_coord.metadata
                
                result = SearchResult(
                    id=vector_id,
                    similarity=similarity,
                    vector=vector,
                    metadata=combined_metadata,
                    distance=distance
                )
                results.append(result)
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def delete_vector(self, vector_id: int) -> bool:
        """
        Remove vector from search index.
        
        Args:
            vector_id: Vector ID to remove
            
        Returns:
            True if removed successfully
        """
        with self._lock:
            if vector_id not in self.vector_id_to_spiral:
                return False
            
            # Remove from spiral coordinate mapping
            spiral_coord = self.vector_id_to_spiral.pop(vector_id)
            
            # Remove from spiral regions
            if self.enable_spiral_optimization:
                region_id = int(spiral_coord.theta / (2 * np.pi) * 100)
                if region_id in self.spiral_regions:
                    self.spiral_regions[region_id] = [
                        vid for vid in self.spiral_regions[region_id] 
                        if vid != vector_id
                    ]
            
            # Note: HNSW doesn't support deletion, so the vector remains
            # in the index but won't be returned in results
            
            return True
    
    def rebuild_index(self, max_elements: Optional[int] = None) -> None:
        """
        Rebuild the search index from scratch.
        
        Args:
            max_elements: Maximum index capacity
        """
        with self._lock:
            if not self.vector_id_to_spiral:
                return
            
            # Determine capacity
            if max_elements is None:
                max_elements = max(len(self.vector_id_to_spiral) * 2, 1000000)
            
            # Reinitialize index
            self._init_hnsw_index(max_elements)
            
            # Re-add all vectors
            vectors_to_add = [
                (vid, coord.vector) 
                for vid, coord in self.vector_id_to_spiral.items()
            ]
            
            if vectors_to_add:
                self.hnsw_index.add_items(
                    [vector for _, vector in vectors_to_add],
                    [vid for vid, _ in vectors_to_add]
                )
        
        logger.info(f"Rebuilt search index with {len(self.vector_id_to_spiral)} vectors")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get search engine statistics.
        
        Returns:
            Dictionary with search statistics
        """
        avg_search_time = (
            self.total_search_time / max(1, self.search_count)
        ) * 1000  # Convert to milliseconds
        
        index_size = self.hnsw_index.get_current_count() if self.hnsw_index else 0
        
        return {
            "dimensions": self.dimensions,
            "distance_metric": self.distance_metric,
            "max_layers": self.max_layers,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "index_size": index_size,
            "search_count": self.search_count,
            "avg_search_time_ms": avg_search_time,
            "enable_spiral_optimization": self.enable_spiral_optimization,
            "spiral_regions_count": len(self.spiral_regions),
            "cache_size": len(self.spiral_cache),
        }
    
    def save_index(self, path: str) -> None:
        """
        Save search index to disk.
        
        Args:
            path: File path to save index
        """
        if self.hnsw_index is not None:
            self.hnsw_index.save_index(path)
            logger.info(f"Search index saved to {path}")
    
    def load_index(self, path: str, max_elements: int) -> None:
        """
        Load search index from disk.
        
        Args:
            path: File path to load index from
            max_elements: Maximum index capacity
        """
        if self.hnsw_index is None:
            self._init_hnsw_index(max_elements)
        
        self.hnsw_index.load_index(path, max_elements)
        logger.info(f"Search index loaded from {path}")
    
    def set_storage_engine(self, storage_engine) -> None:
        """
        Set storage engine for vector retrieval.
        
        Args:
            storage_engine: StorageEngine instance
        """
        self.storage_engine = storage_engine
        logger.info("Storage engine set for search engine")
    
    def index_stored_vectors(self, vector_ids: List[int]) -> None:
        """
        Index vectors that are already stored in the storage engine.
        
        Args:
            vector_ids: List of vector IDs to index
        """
        if not self.storage_engine:
            raise ValueError("Storage engine required for indexing stored vectors")
        
        # Retrieve vectors from storage in batches
        batch_size = 100
        indexed_count = 0
        
        for i in range(0, len(vector_ids), batch_size):
            batch_ids = vector_ids[i:i + batch_size]
            
            try:
                # Get vectors from storage
                batch_results = self.storage_engine.get_vectors_by_ids(batch_ids)
                
                # Prepare for batch insert
                vectors_to_index = []
                for j, (vector, metadata) in enumerate(batch_results):
                    vector_id = batch_ids[j]
                    vectors_to_index.append((vector_id, vector))
                
                # Batch insert into search index
                if vectors_to_index:
                    self.batch_insert(vectors_to_index, [result[1] for result in batch_results])
                    indexed_count += len(vectors_to_index)
                
            except Exception as e:
                logger.error(f"Failed to index batch {i//batch_size}: {e}")
                continue
        
        logger.info(f"Indexed {indexed_count} vectors from storage")
    
    def get_vector_count(self) -> int:
        """
        Get number of vectors in the search index.
        
        Returns:
            Number of indexed vectors
        """
        if self.hnsw_index is None:
            return 0
        return self.hnsw_index.get_current_count()