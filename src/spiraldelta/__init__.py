"""
SpiralDeltaDB - A geometric approach to vector database optimization.

This package implements spiral ordering and multi-tier delta compression
to achieve 30-70% storage reduction while maintaining search quality.
"""

__version__ = "0.1.0"
__author__ = "Monarch AI"
__email__ = "info@monarch-ai.com"

# Core imports - will be implemented
from .spiral_coordinator import SpiralCoordinator
from .delta_encoder import DeltaEncoder
from .search_engine import SpiralSearchEngine
from .storage import StorageEngine
from .database import SpiralDeltaDB

# Data structures
from .types import (
    SpiralCoordinate,
    CompressedSequence,
    SearchResult,
    DatabaseStats,
)

__all__ = [
    # Main classes
    "SpiralDeltaDB",
    "SpiralCoordinator", 
    "DeltaEncoder",
    "SpiralSearchEngine",
    "StorageEngine",
    # Data types
    "SpiralCoordinate",
    "CompressedSequence", 
    "SearchResult",
    "DatabaseStats",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]