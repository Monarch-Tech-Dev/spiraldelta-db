#!/usr/bin/env python3
"""
Basic test of SpiralDeltaDB core functionality without external dependencies.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that we can import all modules."""
    print("Testing imports...")
    
    try:
        from spiraldelta.types import SpiralCoordinate, CompressedSequence, SearchResult, DatabaseStats
        print("✓ Types imported successfully")
        
        from spiraldelta.spiral_coordinator import SpiralCoordinator
        print("✓ SpiralCoordinator imported successfully")
        
        # Test basic spiral coordinator functionality
        coord = SpiralCoordinator(dimensions=64)
        vector = np.random.randn(64)
        spiral_coord = coord.transform(vector)
        print(f"✓ Spiral transformation: theta={spiral_coord.theta:.3f}, radius={spiral_coord.radius:.3f}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_spiral_math():
    """Test spiral coordinate math."""
    print("\nTesting spiral mathematics...")
    
    try:
        from spiraldelta.spiral_coordinator import SpiralCoordinator
        
        # Test with known vectors
        coord = SpiralCoordinator(dimensions=3)
        
        # Test unit vectors
        v1 = np.array([1., 0., 0.])
        v2 = np.array([0., 1., 0.])
        
        spiral1 = coord.transform(v1)
        spiral2 = coord.transform(v2)
        
        print(f"✓ Vector [1,0,0]: theta={spiral1.theta:.3f}, radius={spiral1.radius:.3f}")
        print(f"✓ Vector [0,1,0]: theta={spiral2.theta:.3f}, radius={spiral2.radius:.3f}")
        
        # Test batch transformation
        vectors = np.array([v1, v2, [0., 0., 1.]])
        batch_coords = coord.transform_batch(vectors)
        print(f"✓ Batch transformation: {len(batch_coords)} vectors transformed")
        
        # Test sorting
        sorted_coords = coord.sort_by_spiral(batch_coords)
        print(f"✓ Spiral sorting: angles = {[c.theta for c in sorted_coords]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Spiral math test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_type_definitions():
    """Test type definitions."""
    print("\nTesting type definitions...")
    
    try:
        from spiraldelta.types import SpiralCoordinate, SearchResult, DatabaseStats
        
        # Test SpiralCoordinate
        vector = np.array([1., 2., 3.])
        coord = SpiralCoordinate(theta=1.5, radius=2.0, vector=vector)
        print(f"✓ SpiralCoordinate: theta={coord.theta}, radius={coord.radius}")
        
        # Test SearchResult
        result = SearchResult(
            id=1,
            similarity=0.95,
            vector=vector,
            metadata={"test": "value"},
            distance=0.05
        )
        print(f"✓ SearchResult: id={result.id}, similarity={result.similarity}")
        
        # Test DatabaseStats
        stats = DatabaseStats(
            vector_count=100,
            storage_size_mb=10.5,
            compression_ratio=0.7,
            avg_query_time_ms=5.2,
            index_size_mb=2.1,
            memory_usage_mb=15.0,
            dimensions=128
        )
        print(f"✓ DatabaseStats: {stats.vector_count} vectors, {stats.compression_ratio:.1%} compression")
        
        return True
        
    except Exception as e:
        print(f"✗ Type definitions test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("SpiralDeltaDB Basic Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_spiral_math, 
        test_type_definitions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())