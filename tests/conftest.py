"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import tempfile
import shutil
import logging
from pathlib import Path


# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="spiraldelta_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def random_vectors():
    """Generate random test vectors."""
    np.random.seed(42)
    return np.random.randn(100, 128).astype(np.float32)


@pytest.fixture
def correlated_vectors():
    """Generate correlated vectors for compression testing."""
    np.random.seed(42)
    base = np.random.randn(128).astype(np.float32)
    vectors = []
    
    for i in range(100):
        # Add small random perturbation
        noise = np.random.randn(128).astype(np.float32) * 0.1
        vector = base + noise * (i / 100.0)
        vectors.append(vector)
    
    return np.array(vectors)


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for testing."""
    return [
        {
            "id": i,
            "category": f"cat_{i % 5}",
            "score": i * 0.1,
            "active": i % 2 == 0,
            "tags": [f"tag_{j}" for j in range(i % 3 + 1)]
        }
        for i in range(100)
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid or "test_database" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "large" in item.name or "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)