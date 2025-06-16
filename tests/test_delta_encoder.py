"""
Unit tests for DeltaEncoder class.
"""

import pytest
import numpy as np
from spiraldelta.delta_encoder import DeltaEncoder, ProductQuantizer
from spiraldelta.types import CompressedSequence


class TestProductQuantizer:
    """Test cases for ProductQuantizer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        return np.random.randn(1000, 64)
    
    def test_initialization(self):
        """Test PQ initialization."""
        pq = ProductQuantizer(dimensions=64, n_subspaces=8, n_bits=8)
        
        assert pq.dimensions == 64
        assert pq.n_subspaces == 8
        assert pq.n_bits == 8
        assert pq.codebook_size == 256
        assert pq.subspace_dim == 8
        assert not pq.is_trained
    
    def test_invalid_dimensions(self):
        """Test invalid dimension configuration."""
        with pytest.raises(ValueError):
            ProductQuantizer(dimensions=63, n_subspaces=8)  # Not divisible
    
    def test_training(self, sample_data):
        """Test PQ training."""
        pq = ProductQuantizer(dimensions=64, n_subspaces=8, n_bits=4)
        pq.train(sample_data)
        
        assert pq.is_trained
        assert len(pq.codebooks) == 8
        assert all(codebook is not None for codebook in pq.codebooks)
        
        # Check codebook shapes
        for codebook in pq.codebooks:
            assert codebook.shape == (16, 8)  # 2^4 codes, 8 dims per subspace
    
    def test_encode_decode(self, sample_data):
        """Test encoding and decoding."""
        pq = ProductQuantizer(dimensions=64, n_subspaces=8, n_bits=4)
        pq.train(sample_data)
        
        # Test single vector
        test_vector = sample_data[0]
        codes = pq.encode(test_vector)
        decoded = pq.decode(codes)
        
        assert codes.shape == (8,)
        assert decoded.shape == (64,)
        
        # Should be reasonably close (quantization error expected)
        reconstruction_error = np.linalg.norm(test_vector - decoded)
        assert reconstruction_error < 10.0  # Reasonable bound for test data
    
    def test_compression_ratio_bounds(self):
        """Test compression ratio calculation bounds."""
        encoder = DeltaEncoder(
            quantization_levels=4,
            compression_target=0.6,
            n_subspaces=8,
            n_bits=8,
            anchor_stride=32
        )
        
        # Create training data with known properties
        np.random.seed(42)
        sequences = []
        for _ in range(5):
            # Create correlated vectors for better compression
            base_vector = np.random.randn(128)
            sequence = [base_vector + np.random.randn(128) * 0.1 for _ in range(64)]
            sequences.append(sequence)
        
        # Train encoder
        encoder.train(sequences)
        
        # Test compression
        test_sequence = sequences[0]
        compressed = encoder.encode_sequence(test_sequence)
        
        # Validate compression ratio bounds
        assert 0.30 <= compressed.compression_ratio <= 0.70
        
        # Test decoding
        decoded = encoder.decode_sequence(compressed)
        assert len(decoded) == len(test_sequence)
        
        # Check reconstruction quality
        total_error = 0
        for orig, recon in zip(test_sequence, decoded):
            error = np.linalg.norm(orig - recon)
            total_error += error
        
        avg_error = total_error / len(test_sequence)
        assert avg_error < 5.0  # Reasonable reconstruction quality
    
    def test_size_estimation_accuracy(self):
        """Test compressed size estimation accuracy."""
        encoder = DeltaEncoder()
        
        # Create test data
        np.random.seed(123)
        anchors = [np.random.randn(64).astype(np.float32) for _ in range(4)]
        delta_codes = []
        
        for level in range(2):
            level_codes = []
            for _ in range(10):
                codes = np.random.randint(0, 255, size=8, dtype=np.uint8)
                level_codes.append(codes)
            delta_codes.append(level_codes)
        
        # Estimate size
        estimated_size = encoder._estimate_compressed_size(anchors, delta_codes)
        
        # Should be reasonable
        anchor_size = sum(a.nbytes for a in anchors)  # 4 * 64 * 4 = 1024 bytes
        delta_size = 2 * 10 * 8  # 160 bytes
        expected_min_size = anchor_size + delta_size
        
        assert estimated_size >= expected_min_size
        assert estimated_size < expected_min_size * 2  # Not too much overhead
        pq = ProductQuantizer(dimensions=64, n_subspaces=8, n_bits=8)
        pq.train(sample_data)
        
        # Test single vector
        vector = sample_data[0]
        codes = pq.encode(vector)
        decoded = pq.decode(codes)
        
        assert codes.shape == (8,)
        assert codes.dtype == np.uint8
        assert decoded.shape == (64,)
        
        # Decoded vector should be similar to original
        similarity = np.dot(vector, decoded) / (np.linalg.norm(vector) * np.linalg.norm(decoded))
        assert similarity > 0.5  # Reasonable reconstruction quality
    
    def test_encode_without_training(self):
        """Test encoding without training."""
        pq = ProductQuantizer(dimensions=64, n_subspaces=8)
        vector = np.random.randn(64)
        
        with pytest.raises(ValueError):
            pq.encode(vector)
    
    def test_decode_without_training(self):
        """Test decoding without training."""
        pq = ProductQuantizer(dimensions=64, n_subspaces=8)
        codes = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
        
        with pytest.raises(ValueError):
            pq.decode(codes)
    
    def test_invalid_vector_dimensions(self, sample_data):
        """Test encoding vector with wrong dimensions."""
        pq = ProductQuantizer(dimensions=64, n_subspaces=8)
        pq.train(sample_data)
        
        wrong_vector = np.random.randn(32)  # Wrong size
        
        with pytest.raises(ValueError):
            pq.encode(wrong_vector)


class TestDeltaEncoder:
    """Test cases for DeltaEncoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create a test encoder."""
        return DeltaEncoder(
            quantization_levels=4,
            compression_target=0.5,
            n_subspaces=8,
            n_bits=8,
            anchor_stride=16
        )
    
    @pytest.fixture
    def sample_sequences(self):
        """Create sample vector sequences."""
        np.random.seed(42)
        sequences = []
        
        for _ in range(5):
            # Create correlated vectors for better compression
            base = np.random.randn(128)
            sequence = []
            
            for i in range(64):
                # Add small random delta to create correlation
                delta = np.random.randn(128) * 0.1
                vector = base + delta * (i / 64.0)
                sequence.append(vector)
            
            sequences.append(sequence)
        
        return sequences
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = DeltaEncoder(quantization_levels=3, compression_target=0.6)
        
        assert encoder.quantization_levels == 3
        assert encoder.compression_target == 0.6
        assert encoder.n_subspaces == 8  # Default
        assert encoder.n_bits == 8      # Default
        assert encoder.anchor_stride == 64  # Default
        assert not encoder.is_trained
        assert encoder.encode_count == 0
    
    def test_anchor_selection(self, encoder, sample_sequences):
        """Test anchor point selection."""
        sequence = sample_sequences[0]
        anchor_indices, anchor_vectors = encoder._select_anchors(sequence)
        
        assert len(anchor_indices) == len(anchor_vectors)
        assert 0 in anchor_indices  # First vector should be anchor
        assert len(sequence) - 1 in anchor_indices  # Last vector should be anchor
        
        # Check anchor stride
        for i in range(1, len(anchor_indices) - 1):
            prev_idx = anchor_indices[i-1]
            curr_idx = anchor_indices[i]
            assert curr_idx - prev_idx <= encoder.anchor_stride
    
    def test_anchor_selection_empty(self, encoder):
        """Test anchor selection with empty sequence."""
        indices, vectors = encoder._select_anchors([])
        assert indices == []
        assert vectors == []
    
    def test_hierarchical_deltas(self, encoder, sample_sequences):
        """Test hierarchical delta computation."""
        sequence = sample_sequences[0]
        anchor_indices, _ = encoder._select_anchors(sequence)
        deltas_by_level = encoder._compute_hierarchical_deltas(sequence, anchor_indices)
        
        assert len(deltas_by_level) == encoder.quantization_levels
        
        # Each level should have deltas for non-anchor vectors
        expected_delta_count = len(sequence) - len(anchor_indices)
        for level_deltas in deltas_by_level:
            assert len(level_deltas) == expected_delta_count
    
    def test_training(self, encoder, sample_sequences):
        """Test encoder training."""
        encoder.train(sample_sequences)
        
        assert encoder.is_trained
        assert len(encoder.pq_encoders) <= encoder.quantization_levels
        
        # Check that PQ encoders are trained
        for level, pq in encoder.pq_encoders.items():
            assert pq.is_trained
    
    def test_training_empty_sequences(self, encoder):
        """Test training with empty sequences."""
        encoder.train([])
        
        # Should complete without error
        assert encoder.is_trained
        assert len(encoder.pq_encoders) == 0
    
    def test_encode_sequence(self, encoder, sample_sequences):
        """Test sequence encoding."""
        encoder.train(sample_sequences)
        
        sequence = sample_sequences[0]
        compressed = encoder.encode_sequence(sequence)
        
        assert isinstance(compressed, CompressedSequence)
        assert len(compressed.anchors) > 0
        assert compressed.compression_ratio > 0
        assert compressed.metadata["sequence_length"] == len(sequence)
        assert compressed.metadata["dimensions"] == sequence[0].shape[0]
    
    def test_encode_without_training(self, encoder, sample_sequences):
        """Test encoding without training."""
        sequence = sample_sequences[0]
        
        with pytest.raises(ValueError):
            encoder.encode_sequence(sequence)
    
    def test_encode_empty_sequence(self, encoder):
        """Test encoding empty sequence."""
        encoder.train([[np.random.randn(64) for _ in range(10)]])
        
        with pytest.raises(ValueError):
            encoder.encode_sequence([])
    
    def test_decode_sequence(self, encoder, sample_sequences):
        """Test sequence decoding."""
        encoder.train(sample_sequences)
        
        sequence = sample_sequences[0]
        compressed = encoder.encode_sequence(sequence)
        decoded = encoder.decode_sequence(compressed)
        
        assert len(decoded) == len(sequence)
        assert all(vec.shape == sequence[0].shape for vec in decoded)
        
        # Check reconstruction quality
        similarities = []
        for orig, recon in zip(sequence, decoded):
            if np.linalg.norm(orig) > 0 and np.linalg.norm(recon) > 0:
                sim = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon))
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        assert avg_similarity > 0.7  # Reasonable reconstruction quality
    
    def test_decode_without_training(self, encoder, sample_sequences):
        """Test decoding without training."""
        # Create a fake compressed sequence
        compressed = CompressedSequence(
            anchors=[np.random.randn(64)],
            delta_codes=[],
            metadata={"anchor_indices": [0], "sequence_length": 1, "dimensions": 64},
            compression_ratio=0.5
        )
        
        with pytest.raises(ValueError):
            encoder.decode_sequence(compressed)
    
    def test_compression_ratio_estimation(self, encoder):
        """Test compression ratio estimation."""
        # Create vectors with good correlation
        base = np.random.randn(64)
        correlated_vectors = [base + np.random.randn(64) * 0.1 for _ in range(10)]
        
        ratio = encoder.estimate_compression_ratio(correlated_vectors)
        assert 0.0 <= ratio <= 0.9
        
        # Should give better compression for correlated vectors
        random_vectors = [np.random.randn(64) for _ in range(10)]
        random_ratio = encoder.estimate_compression_ratio(random_vectors)
        
        assert ratio >= random_ratio
    
    def test_compression_size_estimation(self, encoder, sample_sequences):
        """Test compressed size estimation."""
        encoder.train(sample_sequences)
        
        sequence = sample_sequences[0]
        compressed = encoder.encode_sequence(sequence)
        
        anchors = compressed.anchors
        delta_codes = compressed.delta_codes
        
        estimated_size = encoder._estimate_compressed_size(anchors, delta_codes)
        assert estimated_size > 0
        
        # Should be smaller than original size
        original_size = len(sequence) * sequence[0].shape[0] * 4  # float32
        assert estimated_size < original_size
    
    def test_statistics(self, encoder, sample_sequences):
        """Test getting encoder statistics."""
        # Initial statistics
        stats = encoder.get_statistics()
        assert stats["encode_count"] == 0
        assert stats["average_compression_ratio"] == 0.0
        assert not stats["is_trained"]
        
        # After training and encoding
        encoder.train(sample_sequences)
        compressed = encoder.encode_sequence(sample_sequences[0])
        
        stats = encoder.get_statistics()
        assert stats["encode_count"] == 1
        assert stats["average_compression_ratio"] > 0
        assert stats["is_trained"]
    
    def test_save_and_load_state(self, encoder, sample_sequences):
        """Test saving and loading encoder state."""
        # Train encoder
        encoder.train(sample_sequences)
        encoder.encode_sequence(sample_sequences[0])
        
        # Save state
        state_bytes = encoder.save_state()
        
        # Load state
        loaded_encoder = DeltaEncoder.load_state(state_bytes)
        
        assert loaded_encoder.quantization_levels == encoder.quantization_levels
        assert loaded_encoder.compression_target == encoder.compression_target
        assert loaded_encoder.is_trained == encoder.is_trained
        assert loaded_encoder.encode_count == encoder.encode_count
        
        # Should be able to encode with loaded encoder
        new_compressed = loaded_encoder.encode_sequence(sample_sequences[1])
        assert isinstance(new_compressed, CompressedSequence)
    
    @pytest.mark.parametrize("quantization_levels", [2, 3, 4, 6])
    def test_different_quantization_levels(self, quantization_levels, sample_sequences):
        """Test encoder with different quantization levels."""
        encoder = DeltaEncoder(quantization_levels=quantization_levels)
        encoder.train(sample_sequences)
        
        sequence = sample_sequences[0]
        compressed = encoder.encode_sequence(sequence)
        decoded = encoder.decode_sequence(compressed)
        
        assert len(decoded) == len(sequence)
        assert compressed.metadata["quantization_levels"] == quantization_levels
    
    def test_round_trip_consistency(self, encoder, sample_sequences):
        """Test encode-decode round trip consistency."""
        encoder.train(sample_sequences)
        
        for sequence in sample_sequences[:3]:  # Test first 3 sequences
            compressed = encoder.encode_sequence(sequence)
            decoded = encoder.decode_sequence(compressed)
            
            # Should maintain sequence length
            assert len(decoded) == len(sequence)
            
            # Anchor points should be exactly preserved
            anchor_indices = compressed.metadata["anchor_indices"]
            for anchor_idx in anchor_indices:
                if anchor_idx < len(sequence) and anchor_idx < len(decoded):
                    original = sequence[anchor_idx]
                    reconstructed = decoded[anchor_idx]
                    assert np.allclose(original, reconstructed, atol=1e-6)
    
    def test_compression_improvement_with_correlation(self):
        """Test that correlated vectors achieve better compression."""
        encoder = DeltaEncoder(quantization_levels=3, anchor_stride=8)
        
        # Create highly correlated sequence
        base = np.random.randn(64)
        correlated_sequence = []
        for i in range(32):
            vector = base + np.random.randn(64) * 0.05  # Small noise
            correlated_sequence.append(vector)
        
        # Create random sequence
        random_sequence = [np.random.randn(64) for _ in range(32)]
        
        # Train on both
        encoder.train([correlated_sequence, random_sequence])
        
        # Compare compression ratios
        corr_compressed = encoder.encode_sequence(correlated_sequence)
        rand_compressed = encoder.encode_sequence(random_sequence)
        
        # Correlated sequence should compress better
        assert corr_compressed.compression_ratio >= rand_compressed.compression_ratio