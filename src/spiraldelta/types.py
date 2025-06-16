"""
Type definitions and data structures for SpiralDeltaDB.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import numpy as np


@dataclass
class SpiralCoordinate:
    """Represents a vector in spiral coordinate space."""
    theta: float  # Spiral angle
    radius: float  # Distance from center
    vector: np.ndarray  # Original vector
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate coordinate values."""
        if not isinstance(self.theta, (int, float)):
            raise ValueError("theta must be a numeric value")
        if not isinstance(self.radius, (int, float)) or self.radius < 0:
            raise ValueError("radius must be a non-negative numeric value")
        if not isinstance(self.vector, np.ndarray):
            raise ValueError("vector must be a numpy array")


@dataclass
class CompressedSequence:
    """Represents a compressed sequence of vectors."""
    anchors: List[np.ndarray]  # Anchor points (full precision)
    delta_codes: List[np.ndarray]  # Quantized delta codes
    metadata: Dict[str, Any]  # Compression metadata
    compression_ratio: float  # Achieved compression ratio
    
    def __post_init__(self):
        """Validate compressed sequence."""
        if not self.anchors:
            raise ValueError("Must have at least one anchor point")
        if self.compression_ratio < 0 or self.compression_ratio > 1:
            raise ValueError("Compression ratio must be between 0 and 1")


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    id: int  # Vector ID
    similarity: float  # Similarity score (0.0-1.0)
    vector: np.ndarray  # Reconstructed vector
    metadata: Dict[str, Any]  # Associated metadata
    distance: float  # Distance metric
    
    def __post_init__(self):
        """Validate search result."""
        if self.similarity < 0 or self.similarity > 1:
            raise ValueError("Similarity must be between 0 and 1")
        if self.distance < 0:
            raise ValueError("Distance must be non-negative")


@dataclass
class DatabaseStats:
    """Database statistics and performance metrics."""
    vector_count: int  # Total vectors stored
    storage_size_mb: float  # Storage size in MB
    compression_ratio: float  # Achieved compression ratio
    avg_query_time_ms: float  # Average query latency
    index_size_mb: float  # Search index size
    memory_usage_mb: float  # Current memory usage
    dimensions: int  # Vector dimensionality
    
    def __post_init__(self):
        """Validate database stats."""
        if self.vector_count < 0:
            raise ValueError("Vector count must be non-negative")
        if self.compression_ratio < 0 or self.compression_ratio > 1:
            raise ValueError("Compression ratio must be between 0 and 1")


# Type aliases for common operations
VectorId = int
VectorArray = Union[np.ndarray, List[np.ndarray]]
MetadataList = Optional[List[Dict[str, Any]]]
FilterDict = Optional[Dict[str, Any]]