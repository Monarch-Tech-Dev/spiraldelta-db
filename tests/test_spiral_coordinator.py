"""
Unit tests for SpiralCoordinator class.
"""

import pytest
import numpy as np
from spiraldelta.spiral_coordinator import SpiralCoordinator
from spiraldelta.types import SpiralCoordinate


class TestSpiralCoordinator:
    """Test cases for SpiralCoordinator."""
    
    @pytest.fixture
    def coordinator(self):
        """Create a test coordinator."""
        return SpiralCoordinator(dimensions=128, spiral_constant=1.618)
    
    @pytest.fixture
    def sample_vectors(self):
        """Create sample test vectors."""
        np.random.seed(42)
        return np.random.randn(10, 128)
    
    def test_initialization(self):
        """Test coordinator initialization."""
        coord = SpiralCoordinator(dimensions=64)
        
        assert coord.dimensions == 64
        assert coord.spiral_constant == 1.618  # Default golden ratio
        assert coord.reference_vector.shape == (64,)
        assert np.allclose(np.linalg.norm(coord.reference_vector), 1.0)
        assert coord.transform_count == 0
    
    def test_initialization_with_reference(self):
        """Test initialization with custom reference vector."""
        ref_vector = np.array([1, 0, 0, 0])
        coord = SpiralCoordinator(dimensions=4, reference_vector=ref_vector)
        
        assert np.array_equal(coord.reference_vector, ref_vector)
    
    def test_invalid_reference_vector(self):
        """Test initialization with invalid reference vector."""
        with pytest.raises(ValueError):
            SpiralCoordinator(dimensions=4, reference_vector=np.array([1, 2]))
    
    def test_transform_single_vector(self, coordinator):
        """Test transforming a single vector."""
        vector = np.random.randn(128)
        spiral_coord = coordinator.transform(vector)
        
        assert isinstance(spiral_coord, SpiralCoordinate)
        assert isinstance(spiral_coord.theta, float)
        assert isinstance(spiral_coord.radius, float)
        assert spiral_coord.radius >= 0
        assert np.array_equal(spiral_coord.vector, vector)
        assert spiral_coord.metadata["transform_id"] == 1
    
    def test_transform_invalid_vector(self, coordinator):
        """Test transforming vector with wrong dimensions."""
        with pytest.raises(ValueError):
            coordinator.transform(np.array([1, 2, 3]))  # Wrong dimensions
    
    def test_transform_batch(self, coordinator, sample_vectors):
        """Test batch transformation."""
        coordinates = coordinator.transform_batch(sample_vectors)
        
        assert len(coordinates) == len(sample_vectors)
        assert all(isinstance(coord, SpiralCoordinate) for coord in coordinates)
        
        # Check batch indices
        for i, coord in enumerate(coordinates):
            assert coord.metadata["batch_index"] == i
    
    def test_spiral_angle_computation(self):
        """Test spiral angle computation."""
        # Test with known vectors
        vector = np.array([1, 0, 0])
        reference = np.array([1, 0, 0])
        
        angle = SpiralCoordinator._compute_spiral_angle(vector, reference)
        assert angle == 0.0  # Same direction should give 0 angle
        
        # Test orthogonal vectors
        vector = np.array([0, 1, 0])
        reference = np.array([1, 0, 0])
        
        angle = SpiralCoordinator._compute_spiral_angle(vector, reference)
        assert angle > 0  # Should have positive angle
    
    def test_sort_by_spiral(self, coordinator, sample_vectors):
        """Test sorting vectors by spiral order."""
        coordinates = coordinator.transform_batch(sample_vectors)
        sorted_coords = coordinator.sort_by_spiral(coordinates)
        
        # Check that angles are in ascending order
        angles = [coord.theta for coord in sorted_coords]
        assert angles == sorted(angles)
    
    def test_sort_by_spiral_with_vectors(self, coordinator, sample_vectors):
        """Test sorting with vector input."""
        sorted_coords = coordinator.sort_by_spiral(sample_vectors)
        
        assert len(sorted_coords) == len(sample_vectors)
        angles = [coord.theta for coord in sorted_coords]
        assert angles == sorted(angles)
    
    def test_cluster_spiral_regions(self, coordinator, sample_vectors):
        """Test clustering into spiral regions."""
        coordinates = coordinator.transform_batch(sample_vectors)
        clusters = coordinator.cluster_spiral_regions(coordinates, n_clusters=3)
        
        assert len(clusters) == 3
        assert sum(len(cluster) for cluster in clusters) == len(coordinates)
        
        # Check that clusters are sorted by angle
        for cluster in clusters:
            if len(cluster) > 1:
                angles = [coord.theta for coord in cluster]
                assert angles == sorted(angles)
    
    def test_cluster_edge_cases(self, coordinator):
        """Test clustering edge cases."""
        # Empty coordinates
        clusters = coordinator.cluster_spiral_regions([], 3)
        assert clusters == []
        
        # More clusters than coordinates
        coordinates = coordinator.transform_batch(np.random.randn(2, 128))
        clusters = coordinator.cluster_spiral_regions(coordinates, 5)
        assert len(clusters) == 2
        assert all(len(cluster) == 1 for cluster in clusters)
    
    def test_get_spiral_neighbors(self, coordinator, sample_vectors):
        """Test finding spiral neighbors."""
        coordinates = coordinator.transform_batch(sample_vectors)
        target = coordinates[0]
        
        neighbors = coordinator.get_spiral_neighbors(target, coordinates[1:], k=3)
        assert len(neighbors) <= 3
        
        # Test with empty coordinates
        neighbors = coordinator.get_spiral_neighbors(target, [], k=3)
        assert len(neighbors) == 0
    
    def test_compression_potential_estimation(self, coordinator):
        """Test compression potential estimation."""
        # Create vectors with good spiral locality
        angles = np.linspace(0, np.pi/4, 10)
        vectors = []
        for angle in angles:
            vector = np.array([np.cos(angle), np.sin(angle)] + [0] * 126)
            vectors.append(vector)
        
        coordinates = coordinator.transform_batch(np.array(vectors))
        potential = coordinator.estimate_compression_potential(coordinates)
        
        assert 0.0 <= potential <= 1.0
        assert potential > 0.5  # Should have good compression potential
    
    def test_adaptive_reference_update(self):
        """Test adaptive reference vector updates."""
        coord = SpiralCoordinator(dimensions=4, adaptive_reference=True)
        original_ref = coord.reference_vector.copy()
        
        # Transform many vectors to trigger reference update
        for i in range(200):
            vector = np.random.randn(4)
            coord.transform(vector)
        
        # Reference should have been updated
        assert coord.reference_updates > 0
        assert not np.array_equal(coord.reference_vector, original_ref)
    
    def test_statistics(self, coordinator, sample_vectors):
        """Test getting coordinator statistics."""
        # Transform some vectors
        coordinator.transform_batch(sample_vectors)
        
        stats = coordinator.get_statistics()
        
        assert stats["dimensions"] == 128
        assert stats["spiral_constant"] == 1.618
        assert stats["transform_count"] == len(sample_vectors)
        assert "reference_norm" in stats
        assert stats["adaptive_reference"] is True
    
    def test_save_and_load_state(self, coordinator, sample_vectors):
        """Test saving and loading coordinator state."""
        # Transform some vectors to change state
        coordinator.transform_batch(sample_vectors)
        
        # Save state
        state = coordinator.save_state()
        
        # Create new coordinator from state
        loaded_coord = SpiralCoordinator.load_state(state)
        
        assert loaded_coord.dimensions == coordinator.dimensions
        assert loaded_coord.spiral_constant == coordinator.spiral_constant
        assert loaded_coord.transform_count == coordinator.transform_count
        assert np.array_equal(loaded_coord.reference_vector, coordinator.reference_vector)
    
    def test_reference_vector_normalization(self):
        """Test that reference vector remains normalized."""
        coord = SpiralCoordinator(dimensions=10)
        
        # Check initial normalization
        assert np.isclose(np.linalg.norm(coord.reference_vector), 1.0)
        
        # Transform vectors and check normalization is maintained
        for _ in range(200):
            vector = np.random.randn(10)
            coord.transform(vector)
            
            # Reference should remain approximately normalized
            assert np.isclose(np.linalg.norm(coord.reference_vector), 1.0, atol=1e-6)
    
    @pytest.mark.parametrize("spiral_constant", [1.0, 1.618, 2.0])
    def test_different_spiral_constants(self, spiral_constant):
        """Test coordinator with different spiral constants."""
        coord = SpiralCoordinator(dimensions=32, spiral_constant=spiral_constant)
        assert coord.spiral_constant == spiral_constant
        
        # Should still work with different constants
        vector = np.random.randn(32)
        spiral_coord = coord.transform(vector)
        assert isinstance(spiral_coord, SpiralCoordinate)
    
    def test_zero_vector_handling(self, coordinator):
        """Test handling of zero vectors."""
        zero_vector = np.zeros(128)
        spiral_coord = coordinator.transform(zero_vector)
        
        assert spiral_coord.radius == 0.0
        assert isinstance(spiral_coord.theta, float)
    
    def test_deterministic_behavior(self):
        """Test that transformation is deterministic."""
        np.random.seed(42)
        coord1 = SpiralCoordinator(dimensions=64)
        vector = np.random.randn(64)
        result1 = coord1.transform(vector)
        
        np.random.seed(42)
        coord2 = SpiralCoordinator(dimensions=64)
        result2 = coord2.transform(vector)
        
        assert np.isclose(result1.theta, result2.theta)
        assert np.isclose(result1.radius, result2.radius)