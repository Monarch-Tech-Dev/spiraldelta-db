#!/usr/bin/env python3
"""
GloVe-300 dataset-specific optimization for SpiralDeltaDB.

This script implements parameter tuning to achieve target compression ratios
while maintaining search quality on GloVe-300 embeddings.
"""

import numpy as np
import sys
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from spiraldelta import SpiralDeltaDB
from spiraldelta.types import SearchResult
# Import from the same directory
import importlib.util
spec = importlib.util.spec_from_file_location("create_synthetic_glove", 
                                              Path(__file__).parent / "create_synthetic_glove.py")
create_synthetic_glove = importlib.util.module_from_spec(spec)
spec.loader.exec_module(create_synthetic_glove)
SyntheticGloVeGenerator = create_synthetic_glove.SyntheticGloVeGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    parameters: Dict[str, Any]
    compression_ratio: float
    search_quality: float
    insert_time: float
    search_time: float
    memory_usage_mb: float


class GloVeOptimizer:
    """Optimize SpiralDeltaDB parameters for GloVe-300 dataset."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize optimizer.
        
        Args:
            data_dir: Directory containing GloVe data
        """
        self.data_dir = Path(data_dir)
        self.generator = SyntheticGloVeGenerator(data_dir)
        
    def load_glove_data(self, max_vectors: int = 10000) -> Tuple[np.ndarray, List[str]]:
        """
        Load GloVe dataset for optimization.
        
        Args:
            max_vectors: Maximum vectors to use for optimization
            
        Returns:
            Tuple of (vectors, words)
        """
        try:
            vectors, words = self.generator.load_dataset("synthetic_glove_300d")
            if max_vectors and max_vectors < len(vectors):
                indices = np.random.choice(len(vectors), max_vectors, replace=False)
                vectors = vectors[indices]
                words = [words[i] for i in indices]
            return vectors, words
        except FileNotFoundError:
            logger.warning("Synthetic dataset not found, generating new one")
            vectors, words = self.generator.generate_realistic_embeddings(
                n_vectors=max_vectors, dimensions=300
            )
            self.generator.save_dataset(vectors, words, "synthetic_glove_300d")
            return vectors, words
    
    def optimize_parameters(
        self, 
        target_compression: float = 0.65,
        max_vectors: int = 10000
    ) -> OptimizationResult:
        """
        Optimize parameters to achieve target compression ratio.
        
        Args:
            target_compression: Target compression ratio (0.0-1.0)
            max_vectors: Maximum vectors for optimization
            
        Returns:
            OptimizationResult with best parameters
        """
        logger.info(f"Optimizing for target compression: {target_compression:.2f}")
        
        # Load test data
        vectors, words = self.load_glove_data(max_vectors)
        logger.info(f"Using {len(vectors)} vectors for optimization")
        
        # Parameter grid for optimization
        param_combinations = self._generate_parameter_grid(target_compression)
        
        best_result = None
        best_score = -float('inf')
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing parameter set {i+1}/{len(param_combinations)}: {params}")
            
            try:
                result = self._evaluate_parameters(vectors, words, params)
                
                # Score based on compression ratio and search quality
                compression_score = 1.0 - abs(result.compression_ratio - target_compression)
                quality_penalty = max(0, 0.95 - result.search_quality) * 2  # Penalize quality < 95%
                score = compression_score - quality_penalty
                
                logger.info(
                    f"  Compression: {result.compression_ratio:.3f}, "
                    f"Quality: {result.search_quality:.3f}, "
                    f"Score: {score:.3f}"
                )
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    logger.info(f"  New best result!")
                
            except Exception as e:
                logger.warning(f"  Failed: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("No valid parameter combination found")
        
        logger.info(f"Best parameters: {best_result.parameters}")
        logger.info(f"Achieved compression: {best_result.compression_ratio:.3f}")
        logger.info(f"Search quality: {best_result.search_quality:.3f}")
        
        return best_result
    
    def _generate_parameter_grid(self, target_compression: float) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for optimization.
        
        Args:
            target_compression: Target compression ratio
            
        Returns:
            List of parameter dictionaries
        """
        # Base parameters for GloVe-300
        base_params = {
            "dimensions": 300,
            "compression_ratio": target_compression,
            "auto_train_threshold": 1000,
        }
        
        # Parameter variations to try
        variations = [
            # Aggressive compression settings
            {
                **base_params,
                "quantization_levels": 6,
                "n_subspaces": 15,  # 300/15 = 20 dims per subspace
                "n_bits": 6,  # 64 codes per subspace
                "anchor_stride": 32,
                "spiral_constant": 1.618,
            },
            # Balanced compression settings
            {
                **base_params,
                "quantization_levels": 4,
                "n_subspaces": 12,  # 300/12 = 25 dims per subspace
                "n_bits": 7,  # 128 codes per subspace
                "anchor_stride": 48,
                "spiral_constant": 1.618,
            },
            # Conservative compression settings
            {
                **base_params,
                "quantization_levels": 3,
                "n_subspaces": 10,  # 300/10 = 30 dims per subspace
                "n_bits": 8,  # 256 codes per subspace
                "anchor_stride": 64,
                "spiral_constant": 1.618,
            },
            # High-precision settings
            {
                **base_params,
                "quantization_levels": 2,
                "n_subspaces": 8,   # 300/8 = 37.5, will adjust
                "n_bits": 8,
                "anchor_stride": 24,
                "spiral_constant": 1.618,
            },
            # Alternative spiral constants
            {
                **base_params,
                "quantization_levels": 4,
                "n_subspaces": 12,
                "n_bits": 7,
                "anchor_stride": 48,
                "spiral_constant": 2.0,  # Different spiral parameter
            },
            {
                **base_params,
                "quantization_levels": 4,
                "n_subspaces": 12,
                "n_bits": 7,
                "anchor_stride": 48,
                "spiral_constant": 1.414,  # sqrt(2)
            },
        ]
        
        return variations
    
    def _evaluate_parameters(
        self, 
        vectors: np.ndarray, 
        words: List[str], 
        params: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Evaluate a specific parameter combination.
        
        Args:
            vectors: Test vectors
            words: Test words
            params: Parameters to test
            
        Returns:
            OptimizationResult
        """
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create database with test parameters
            db = SpiralDeltaDB(
                storage_path=f"{temp_dir}/test.db",
                **params
            )
            
            # Measure insertion time
            start_time = time.time()
            vector_ids = db.insert(vectors)
            insert_time = time.time() - start_time
            
            # Get compression statistics
            stats = db.get_stats()
            
            # Evaluate search quality
            search_quality = self._evaluate_search_quality(db, vectors, vector_ids)
            
            # Measure search time
            search_time = self._measure_search_time(db, vectors[:100])  # Use subset for speed
            
            return OptimizationResult(
                parameters=params.copy(),
                compression_ratio=stats.compression_ratio,
                search_quality=search_quality,
                insert_time=insert_time,
                search_time=search_time,
                memory_usage_mb=stats.memory_usage_mb,
            )
    
    def _evaluate_search_quality(
        self, 
        db: SpiralDeltaDB, 
        vectors: np.ndarray, 
        vector_ids: List[int]
    ) -> float:
        """
        Evaluate search quality by comparing compressed vs original vectors.
        
        Args:
            db: Database instance
            vectors: Original vectors
            vector_ids: Inserted vector IDs
            
        Returns:
            Average recall at k=10
        """
        n_queries = min(100, len(vectors) // 10)  # Test with subset
        total_recall = 0.0
        
        for i in range(n_queries):
            query_vector = vectors[i]
            
            # Get ground truth (brute force search on original vectors)
            similarities = np.dot(vectors, query_vector)
            true_top_k = np.argsort(similarities)[-10:][::-1]
            
            # Get compressed search results
            try:
                search_results = db.search(query_vector, k=10)
                found_ids = [r.vector_id for r in search_results]
                
                # Calculate recall
                recall = len(set(true_top_k) & set(found_ids)) / 10.0
                total_recall += recall
                
            except Exception as e:
                logger.warning(f"Search failed for query {i}: {e}")
                continue
        
        return total_recall / n_queries if n_queries > 0 else 0.0
    
    def _measure_search_time(self, db: SpiralDeltaDB, query_vectors: np.ndarray) -> float:
        """
        Measure average search time.
        
        Args:
            db: Database instance
            query_vectors: Query vectors for timing
            
        Returns:
            Average search time in milliseconds
        """
        times = []
        
        for query in query_vectors[:20]:  # Test with small subset
            start_time = time.time()
            try:
                db.search(query, k=10)
                search_time = (time.time() - start_time) * 1000  # Convert to ms
                times.append(search_time)
            except Exception:
                continue
        
        return np.mean(times) if times else 0.0
    
    def save_optimization_results(self, result: OptimizationResult, filepath: str) -> None:
        """
        Save optimization results to file.
        
        Args:
            result: Optimization result
            filepath: Output file path
        """
        data = {
            "parameters": result.parameters,
            "performance": {
                "compression_ratio": result.compression_ratio,
                "search_quality": result.search_quality,
                "insert_time": result.insert_time,
                "search_time": result.search_time,
                "memory_usage_mb": result.memory_usage_mb,
            },
            "timestamp": time.time(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize SpiralDeltaDB for GloVe-300")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--target-compression", type=float, default=0.65, 
                       help="Target compression ratio")
    parser.add_argument("--max-vectors", type=int, default=10000,
                       help="Maximum vectors for optimization")
    parser.add_argument("--output", default="glove_optimization_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = GloVeOptimizer(args.data_dir)
    result = optimizer.optimize_parameters(
        target_compression=args.target_compression,
        max_vectors=args.max_vectors
    )
    
    # Save results
    optimizer.save_optimization_results(result, args.output)
    
    # Display summary
    print(f"\n🎯 GloVe-300 Optimization Results:")
    print(f"Target compression: {args.target_compression:.1%}")
    print(f"Achieved compression: {result.compression_ratio:.1%}")
    print(f"Search quality: {result.search_quality:.1%}")
    print(f"Insert time: {result.insert_time:.2f}s")
    print(f"Search time: {result.search_time:.2f}ms")
    print(f"Memory usage: {result.memory_usage_mb:.1f}MB")
    print(f"\nOptimal parameters:")
    for key, value in result.parameters.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()