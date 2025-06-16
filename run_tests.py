#!/usr/bin/env python
"""
Simple test runner for SpiralDeltaDB.
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run the test suite."""
    project_root = Path(__file__).parent
    
    # Add src to Python path
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        # Try to run with pytest if available
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "-x"  # Stop on first failure
        ], cwd=project_root)
        
        return result.returncode
        
    except FileNotFoundError:
        print("pytest not found. Running basic import tests...")
        
        # Basic import test
        try:
            import spiraldelta
            from spiraldelta import SpiralDeltaDB
            import numpy as np
            
            print("✓ Basic imports successful")
            
            # Create a simple test
            print("✓ Running basic functionality test...")
            
            with SpiralDeltaDB(dimensions=64, storage_path="test.db") as db:
                # Insert some test data
                vectors = np.random.randn(10, 64)
                vector_ids = db.insert(vectors)
                print(f"✓ Inserted {len(vector_ids)} vectors")
                
                # Search
                query = vectors[0]
                results = db.search(query, k=3)
                print(f"✓ Search returned {len(results)} results")
                
                # Get stats
                stats = db.get_stats()
                print(f"✓ Database stats: {stats.vector_count} vectors")
            
            print("✓ Basic functionality test passed!")
            return 0
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(run_tests())