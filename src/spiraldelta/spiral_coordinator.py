"""
SpiralCoordinator: Core spiral coordinate transformation logic.

This module implements the geometric spiral ordering that preserves semantic
relationships between vectors, enabling efficient delta compression.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from numba import jit
import logging
from .types import SpiralCoordinate

logger = logging.getLogger(__name__)


class SpiralCoordinator:
    """
    Manages spiral coordinate transformation and clustering.
    
    The spiral transformation maps high-dimensional vectors to spiral coordinates
    using a reference vector and spiral constant, preserving semantic relationships.
    """
    
    def __init__(
        self,
        dimensions: int,
        reference_vector: Optional[np.ndarray] = None,
        spiral_constant: float = 1.618,  # Golden ratio
        learning_rate: float = 0.01,
        adaptive_reference: bool = True,
    ):
        """
        Initialize SpiralCoordinator.
        
        Args:
            dimensions: Vector dimensionality
            reference_vector: Spiral center (learned if None)
            spiral_constant: Spiral transformation parameter (golden ratio default)
            learning_rate: Rate for adaptive reference vector updates
            adaptive_reference: Whether to adapt reference vector during operations
        """
        self.dimensions = dimensions
        self.spiral_constant = spiral_constant
        self.learning_rate = learning_rate
        self.adaptive_reference = adaptive_reference
        
        # Initialize or set reference vector
        if reference_vector is not None:
            if reference_vector.shape != (dimensions,):
                raise ValueError(f"Reference vector must have shape ({dimensions},)")
            self.reference_vector = reference_vector.astype(np.float32)
        else:
            # Initialize with random unit vector
            self.reference_vector = self._generate_random_reference()
        
        # Statistics tracking
        self.transform_count = 0
        self.reference_updates = 0
        
        logger.info(f"Initialized SpiralCoordinator with {dimensions}D vectors")
    
    def _generate_random_reference(self) -> np.ndarray:
        """Generate a random unit reference vector."""
        ref = np.random.randn(self.dimensions).astype(np.float32)
        return ref / np.linalg.norm(ref)
    
    def transform(self, vector: np.ndarray) -> SpiralCoordinate:
        """
        Convert vector to spiral coordinates.
        
        Args:
            vector: Input vector to transform
            
        Returns:
            SpiralCoordinate with theta (angle) and radius
        """
        if vector.shape != (self.dimensions,):
            raise ValueError(f"Vector must have shape ({self.dimensions},)")
        
        # Compute spiral angle using optimized function
        theta = float(self._compute_spiral_angle(vector, self.reference_vector))
        radius = float(np.linalg.norm(vector))
        
        # Update reference vector adaptively
        if self.adaptive_reference:
            self._update_reference_vector(vector)
        
        self.transform_count += 1
        
        return SpiralCoordinate(
            theta=theta,
            radius=radius,
            vector=vector.copy(),
            metadata={"transform_id": self.transform_count}
        )
    
    @staticmethod
    @jit(nopython=True)
    def _compute_spiral_angle(vector: np.ndarray, reference: np.ndarray) -> float:
        """
        Core spiral transformation math (optimized with Numba).
        
        Args:
            vector: Input vector
            reference: Reference vector for spiral center
            
        Returns:
            Spiral angle in radians
        """
        # Ensure both arrays have same dtype
        vector = vector.astype(np.float32)
        reference = reference.astype(np.float32)
        
        # Compute dot product and magnitudes
        dot_product = np.dot(vector, reference)
        vector_magnitude = np.sqrt(np.sum(vector * vector))
        ref_magnitude = np.sqrt(np.sum(reference * reference))
        
        # Normalize dot product
        if vector_magnitude > 0 and ref_magnitude > 0:
            cos_angle = dot_product / (vector_magnitude * ref_magnitude)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
            
            # Convert to spiral angle using arctan2 for better numerical stability
            sin_angle = np.sqrt(1.0 - cos_angle * cos_angle)
            spiral_angle = np.arctan2(sin_angle * vector_magnitude, cos_angle * vector_magnitude)
        else:
            spiral_angle = 0.0
        
        return spiral_angle
    
    def _update_reference_vector(self, vector: np.ndarray) -> None:
        """
        Adaptively update reference vector based on new vectors.
        
        Args:
            vector: New vector to incorporate into reference
        """
        if self.transform_count % 100 == 0:  # Update every 100 transforms
            # Exponential moving average update
            self.reference_vector = (
                (1 - self.learning_rate) * self.reference_vector +
                self.learning_rate * (vector / np.linalg.norm(vector))
            )
            # Renormalize
            self.reference_vector = self.reference_vector / np.linalg.norm(self.reference_vector)
            self.reference_updates += 1
    
    def transform_batch(self, vectors: np.ndarray) -> List[SpiralCoordinate]:
        """
        Transform multiple vectors efficiently.
        
        Args:
            vectors: Array of vectors with shape (n, dimensions)
            
        Returns:
            List of SpiralCoordinate objects
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dimensions:
            raise ValueError(f"Vectors must have shape (n, {self.dimensions})")
        
        coordinates = []
        for i, vector in enumerate(vectors):
            coord = self.transform(vector)
            coord.metadata["batch_index"] = i
            coordinates.append(coord)
        
        return coordinates
    
    def sort_by_spiral(
        self, 
        vectors: Union[np.ndarray, List[SpiralCoordinate]]
    ) -> List[SpiralCoordinate]:
        """
        Sort vectors in spiral order for optimal compression.
        
        Args:
            vectors: Either array of vectors or list of SpiralCoordinates
            
        Returns:
            List of SpiralCoordinates sorted by spiral angle
        """
        if isinstance(vectors, np.ndarray):
            coordinates = self.transform_batch(vectors)
        else:
            coordinates = vectors
        
        # Sort by spiral angle (theta)
        sorted_coords = sorted(coordinates, key=lambda coord: coord.theta)
        
        logger.debug(f"Sorted {len(sorted_coords)} vectors by spiral angle")
        return sorted_coords
    
    def cluster_spiral_regions(
        self, 
        coordinates: List[SpiralCoordinate], 
        n_clusters: int
    ) -> List[List[SpiralCoordinate]]:
        """
        Cluster vectors into spiral regions for hierarchical compression.
        
        Args:
            coordinates: List of spiral coordinates
            n_clusters: Number of clusters to create
            
        Returns:
            List of clusters, each containing SpiralCoordinates
        """
        if not coordinates:
            return []
        
        if n_clusters >= len(coordinates):
            return [[coord] for coord in coordinates]
        
        # Sort by spiral angle first
        sorted_coords = sorted(coordinates, key=lambda coord: coord.theta)
        
        # Divide into approximately equal clusters
        cluster_size = len(sorted_coords) // n_clusters
        clusters = []
        
        for i in range(n_clusters):
            start_idx = i * cluster_size
            if i == n_clusters - 1:  # Last cluster gets remaining vectors
                end_idx = len(sorted_coords)
            else:
                end_idx = (i + 1) * cluster_size
            
            clusters.append(sorted_coords[start_idx:end_idx])
        
        logger.info(f"Created {len(clusters)} spiral clusters")
        return clusters
    
    def get_spiral_neighbors(
        self, 
        target_coord: SpiralCoordinate, 
        coordinates: List[SpiralCoordinate], 
        k: int = 5
    ) -> List[SpiralCoordinate]:
        """
        Find k nearest neighbors in spiral space.
        
        Args:
            target_coord: Target spiral coordinate
            coordinates: List of candidate coordinates
            k: Number of neighbors to return
            
        Returns:
            List of k nearest spiral neighbors
        """
        if not coordinates:
            return []
        
        # Compute spiral distances
        distances = []
        for coord in coordinates:
            # Combine angular and radial distances
            angular_dist = abs(coord.theta - target_coord.theta)
            radial_dist = abs(coord.radius - target_coord.radius)
            
            # Weighted combination (favor angular similarity)
            spiral_dist = 0.7 * angular_dist + 0.3 * radial_dist
            distances.append((spiral_dist, coord))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[0])
        neighbors = [coord for _, coord in distances[:k]]
        
        return neighbors
    
    def estimate_compression_potential(
        self, 
        coordinates: List[SpiralCoordinate]
    ) -> float:
        """
        Estimate compression potential based on spiral locality.
        
        Args:
            coordinates: List of spiral coordinates
            
        Returns:
            Estimated compression ratio (higher = better compression)
        """
        if len(coordinates) < 2:
            return 0.0
        
        # Sort by spiral angle
        sorted_coords = sorted(coordinates, key=lambda coord: coord.theta)
        
        # Compute average angular difference between consecutive vectors
        angular_diffs = []
        for i in range(1, len(sorted_coords)):
            diff = abs(sorted_coords[i].theta - sorted_coords[i-1].theta)
            angular_diffs.append(diff)
        
        avg_angular_diff = np.mean(angular_diffs)
        
        # Smaller differences indicate better spiral locality
        # Map to compression ratio estimate
        max_diff = np.pi  # Maximum possible angular difference
        compression_potential = 1.0 - (avg_angular_diff / max_diff)
        
        return max(0.0, min(1.0, compression_potential))
    
    def get_statistics(self) -> dict:
        """
        Get coordinator statistics.
        
        Returns:
            Dictionary with transformation statistics
        """
        return {
            "dimensions": self.dimensions,
            "spiral_constant": self.spiral_constant,
            "transform_count": self.transform_count,
            "reference_updates": self.reference_updates,
            "reference_norm": np.linalg.norm(self.reference_vector),
            "adaptive_reference": self.adaptive_reference,
        }
    
    def save_state(self) -> dict:
        """
        Save coordinator state for persistence.
        
        Returns:
            Dictionary with state information
        """
        return {
            "dimensions": self.dimensions,
            "reference_vector": self.reference_vector.tolist(),
            "spiral_constant": self.spiral_constant,
            "learning_rate": self.learning_rate,
            "adaptive_reference": self.adaptive_reference,
            "transform_count": self.transform_count,
            "reference_updates": self.reference_updates,
        }
    
    @classmethod
    def load_state(cls, state: dict) -> "SpiralCoordinator":
        """
        Load coordinator from saved state.
        
        Args:
            state: State dictionary from save_state()
            
        Returns:
            New SpiralCoordinator instance
        """
        coordinator = cls(
            dimensions=state["dimensions"],
            reference_vector=np.array(state["reference_vector"]),
            spiral_constant=state["spiral_constant"],
            learning_rate=state["learning_rate"],
            adaptive_reference=state["adaptive_reference"],
        )
        
        coordinator.transform_count = state["transform_count"]
        coordinator.reference_updates = state["reference_updates"]
        
        return coordinator