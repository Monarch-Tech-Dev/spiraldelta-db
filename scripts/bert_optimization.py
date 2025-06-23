#!/usr/bin/env python3
"""
BERT-768 optimization script for SpiralDeltaDB.

This script tunes DeltaEncoder parameters specifically for BERT embeddings
to achieve the target 66.8% compression ratio with minimal quality loss.
"""

import numpy as np
import sys
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from itertools import product
import pickle

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from spiraldelta import SpiralDeltaDB
from spiraldelta.delta_encoder import DeltaEncoder
from spiraldelta.spiral_coordinator import SpiralCoordinator
from spiraldelta.types import CompressedSequence

# Import dataset manager
sys.path.append(str(Path(__file__).parent))
from bert_dataset_manager import BERTDatasetManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTOptimizer:
    """Optimizer for BERT-768 embeddings with SpiralDeltaDB."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize BERT optimizer."""
        self.data_dir = Path(data_dir)
        self.dataset_manager = BERTDatasetManager(data_dir)
        
        # Target compression ratio
        self.target_compression = 0.668  # 66.8%
        
        # Parameter search space for optimization
        self.param_grid = {
            'quantization_levels': [2, 3, 4, 5],
            'n_subspaces': [4, 8, 12, 16],
            'n_bits': [6, 7, 8],
            'anchor_stride': [32, 64, 128],
            'spiral_constant': [1.618, 2.0, 1.5, 1.8]
        }
        
        # Best parameters found
        self.best_params = None
        self.best_compression = 0.0
        self.best_quality = 0.0
        
        # Results cache
        self.optimization_results = []
    
    def load_bert_data(self, max_vectors: int = 50000) -> Tuple[np.ndarray, List[str]]:
        """Load BERT dataset for optimization."""
        logger.info(f"Loading BERT dataset: {max_vectors} vectors")
        
        vectors, metadata, info = self.dataset_manager.get_bert_dataset(max_vectors=max_vectors)
        
        logger.info(f"Loaded {len(vectors)} BERT vectors ({vectors.shape[1]}D)")
        logger.info(f"Dataset source: {info['source']}")
        logger.info(f"Vector norm stats: mean={np.mean(np.linalg.norm(vectors, axis=1)):.3f}")
        
        return vectors, metadata
    
    def evaluate_compression_quality(
        self, 
        original_vectors: List[np.ndarray],
        reconstructed_vectors: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate compression quality metrics.
        
        Args:
            original_vectors: Original vector sequence
            reconstructed_vectors: Reconstructed vector sequence
            
        Returns:
            Dictionary with quality metrics
        """
        if len(original_vectors) != len(reconstructed_vectors):
            raise ValueError("Vector sequences must have same length")
        
        # Convert to numpy arrays for efficient computation
        orig_array = np.array(original_vectors)
        recon_array = np.array(reconstructed_vectors)
        
        # Mean squared error
        mse = np.mean((orig_array - recon_array) ** 2)
        
        # Cosine similarity (most important for embeddings)
        cos_sims = []
        for orig, recon in zip(original_vectors, reconstructed_vectors):
            # Avoid division by zero
            orig_norm = np.linalg.norm(orig)
            recon_norm = np.linalg.norm(recon)
            
            if orig_norm > 1e-8 and recon_norm > 1e-8:
                cos_sim = np.dot(orig, recon) / (orig_norm * recon_norm)
                cos_sims.append(cos_sim)
        
        mean_cosine_similarity = np.mean(cos_sims) if cos_sims else 0.0
        
        # Relative error
        orig_norms = np.linalg.norm(orig_array, axis=1)
        error_norms = np.linalg.norm(orig_array - recon_array, axis=1)
        relative_errors = error_norms / (orig_norms + 1e-8)
        mean_relative_error = np.mean(relative_errors)
        
        return {
            "mse": float(mse),
            "cosine_similarity": float(mean_cosine_similarity),
            "relative_error": float(mean_relative_error),
            "quality_score": float(mean_cosine_similarity * (1.0 - mean_relative_error))
        }
    
    def test_parameter_combination(
        self,
        vectors: np.ndarray,
        params: Dict[str, any],
        test_size: int = 5000
    ) -> Dict[str, any]:
        """
        Test a specific parameter combination.
        
        Args:
            vectors: BERT vectors to test on
            params: Parameter dictionary
            test_size: Number of vectors to test (for speed)
            
        Returns:
            Results dictionary
        """
        logger.info(f"Testing params: {params}")
        
        # Limit test size for efficiency
        test_vectors = vectors[:test_size] if len(vectors) > test_size else vectors
        
        try:
            # Initialize components
            spiral_coordinator = SpiralCoordinator(
                dimensions=768,
                spiral_constant=params['spiral_constant']
            )
            
            delta_encoder = DeltaEncoder(
                quantization_levels=params['quantization_levels'],
                compression_target=self.target_compression,
                n_subspaces=params['n_subspaces'],
                n_bits=params['n_bits'],
                anchor_stride=params['anchor_stride']
            )
            
            # Convert to spiral coordinates and sort
            start_time = time.time()
            
            # Transform and sort vectors by spiral coordinate
            sorted_coords = spiral_coordinator.sort_by_spiral(test_vectors)
            sorted_vectors = [coord.vector for coord in sorted_coords]
            
            spiral_time = time.time() - start_time
            
            # Train delta encoder
            start_time = time.time()
            
            # Create training sequences (split into chunks for realistic usage)
            chunk_size = min(1000, len(sorted_vectors) // 4)
            training_sequences = [
                sorted_vectors[i:i+chunk_size] 
                for i in range(0, len(sorted_vectors), chunk_size)
                if i+chunk_size <= len(sorted_vectors)
            ]
            
            if not training_sequences:
                training_sequences = [sorted_vectors]
            
            delta_encoder.train(training_sequences)
            train_time = time.time() - start_time
            
            # Test compression
            start_time = time.time()
            compressed = delta_encoder.encode_sequence(sorted_vectors)
            encode_time = time.time() - start_time
            
            # Test decompression
            start_time = time.time()
            reconstructed = delta_encoder.decode_sequence(compressed)
            decode_time = time.time() - start_time
            
            # Evaluate quality
            quality_metrics = self.evaluate_compression_quality(sorted_vectors, reconstructed)
            
            # Calculate actual compression ratio
            original_size = len(test_vectors) * 768 * 4  # float32
            compressed_size = delta_encoder._estimate_compressed_size(
                compressed.anchors, compressed.delta_codes
            )
            actual_compression = max(0.0, 1.0 - (compressed_size / original_size)) if original_size > 0 else 0.0
            
            result = {
                "params": params.copy(),
                "compression_ratio": float(compressed.compression_ratio),
                "actual_compression": float(actual_compression),
                "quality_metrics": quality_metrics,
                "timing": {
                    "spiral_transform": spiral_time,
                    "training": train_time,
                    "encoding": encode_time,
                    "decoding": decode_time,
                    "total": spiral_time + train_time + encode_time + decode_time
                },
                "test_vectors": len(test_vectors),
                "success": True,
                "error": None
            }
            
            logger.info(f"Compression: {result['compression_ratio']:.3f}, "
                       f"Quality: {quality_metrics['cosine_similarity']:.3f}, "
                       f"Time: {result['timing']['total']:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Parameter test failed: {e}")
            return {
                "params": params.copy(),
                "compression_ratio": 0.0,
                "actual_compression": 0.0,
                "quality_metrics": {"cosine_similarity": 0.0, "quality_score": 0.0},
                "timing": {"total": 0.0},
                "test_vectors": 0,
                "success": False,
                "error": str(e)
            }
    
    def grid_search_optimization(
        self,
        vectors: np.ndarray,
        max_combinations: int = 20,
        test_size: int = 5000
    ) -> List[Dict[str, any]]:
        """
        Perform grid search to find optimal parameters.
        
        Args:
            vectors: BERT vectors for optimization
            max_combinations: Maximum parameter combinations to test
            test_size: Vectors per test (for speed)
            
        Returns:
            List of results sorted by performance
        """
        logger.info("Starting grid search optimization for BERT-768")
        logger.info(f"Target compression ratio: {self.target_compression:.1%}")
        
        # Generate parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        all_combinations = list(product(*param_values))
        
        # Limit combinations for efficiency
        if len(all_combinations) > max_combinations:
            # Sample combinations strategically
            np.random.seed(42)
            selected_indices = np.random.choice(
                len(all_combinations), 
                max_combinations, 
                replace=False
            )
            combinations = [all_combinations[i] for i in selected_indices]
        else:
            combinations = all_combinations
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        results = []
        for i, param_values in enumerate(combinations):
            params = dict(zip(param_names, param_values))
            
            logger.info(f"Test {i+1}/{len(combinations)}: {params}")
            
            result = self.test_parameter_combination(vectors, params, test_size)
            results.append(result)
            
            # Early exit for clearly good results
            if (result['success'] and 
                result['compression_ratio'] >= self.target_compression * 0.95 and
                result['quality_metrics']['cosine_similarity'] >= 0.95):
                logger.info(f"Found excellent result early: {result['compression_ratio']:.3f} compression")
        
        # Sort by composite score (compression ratio + quality)
        def score_result(result):
            if not result['success']:
                return -1.0
            
            compression_score = result['compression_ratio']
            quality_score = result['quality_metrics']['cosine_similarity']
            
            # Penalize if far from target compression
            target_penalty = abs(result['compression_ratio'] - self.target_compression)
            
            # Composite score favoring results near target with high quality
            return quality_score * compression_score - target_penalty * 0.5
        
        results.sort(key=score_result, reverse=True)
        
        # Store best result
        if results and results[0]['success']:
            self.best_params = results[0]['params']
            self.best_compression = results[0]['compression_ratio']
            self.best_quality = results[0]['quality_metrics']['cosine_similarity']
        
        self.optimization_results = results
        return results
    
    def run_targeted_optimization(
        self,
        vectors: np.ndarray,
        test_size: int = 10000
    ) -> Dict[str, any]:
        """
        Run targeted optimization focused on achieving 66.8% compression.
        
        Args:
            vectors: BERT vectors for optimization
            test_size: Number of vectors to test
            
        Returns:
            Best configuration result
        """
        logger.info("Running targeted optimization for 66.8% compression ratio")
        
        # Focused parameter search based on BERT characteristics
        focused_params = [
            # Configuration 1: High compression, moderate quality
            {
                'quantization_levels': 4,
                'n_subspaces': 12,
                'n_bits': 7,
                'anchor_stride': 64,
                'spiral_constant': 1.618
            },
            # Configuration 2: Balanced compression and quality
            {
                'quantization_levels': 3,
                'n_subspaces': 8,
                'n_bits': 8,
                'anchor_stride': 48,
                'spiral_constant': 1.8
            },
            # Configuration 3: Aggressive compression
            {
                'quantization_levels': 5,
                'n_subspaces': 16,
                'n_bits': 6,
                'anchor_stride': 32,
                'spiral_constant': 2.0
            },
            # Configuration 4: Quality-focused
            {
                'quantization_levels': 2,
                'n_subspaces': 6,
                'n_bits': 8,
                'anchor_stride': 96,
                'spiral_constant': 1.5
            }
        ]
        
        logger.info(f"Testing {len(focused_params)} targeted configurations")
        
        results = []
        for i, params in enumerate(focused_params):
            logger.info(f"Testing configuration {i+1}: {params}")
            result = self.test_parameter_combination(vectors, params, test_size)
            results.append(result)
        
        # Find best result closest to target
        best_result = None
        best_score = -1.0
        
        for result in results:
            if not result['success']:
                continue
            
            compression_diff = abs(result['compression_ratio'] - self.target_compression)
            quality_score = result['quality_metrics']['cosine_similarity']
            
            # Score prioritizes quality but penalizes distance from target
            score = quality_score - compression_diff * 2.0
            
            if score > best_score:
                best_score = score
                best_result = result
        
        if best_result:
            self.best_params = best_result['params']
            self.best_compression = best_result['compression_ratio']
            self.best_quality = best_result['quality_metrics']['cosine_similarity']
            
            logger.info(f"Best configuration found:")
            logger.info(f"  Compression: {self.best_compression:.1%}")
            logger.info(f"  Quality (cosine sim): {self.best_quality:.3f}")
            logger.info(f"  Parameters: {self.best_params}")
        
        return best_result
    
    def save_optimization_results(self, output_path: str = "bert_optimization_results.json"):
        """Save optimization results to file."""
        results_data = {
            "target_compression": self.target_compression,
            "best_params": self.best_params,
            "best_compression": self.best_compression,
            "best_quality": self.best_quality,
            "optimization_results": self.optimization_results,
            "timestamp": time.time()
        }
        
        output_file = self.data_dir / output_path
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved optimization results to {output_file}")
        return output_file


def main():
    """Main function for BERT optimization."""
    parser = argparse.ArgumentParser(description="BERT-768 optimization for SpiralDeltaDB")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--max-vectors", type=int, default=50000, help="Maximum vectors to load")
    parser.add_argument("--test-size", type=int, default=5000, help="Vectors per test")
    parser.add_argument("--grid-search", action="store_true", help="Run grid search optimization")
    parser.add_argument("--targeted", action="store_true", help="Run targeted optimization")
    parser.add_argument("--max-combinations", type=int, default=20, help="Max grid search combinations")
    parser.add_argument("--output", default="bert_optimization_results.json", help="Output file")
    
    args = parser.parse_args()
    
    optimizer = BERTOptimizer(args.data_dir)
    
    # Load BERT data
    vectors, metadata = optimizer.load_bert_data(args.max_vectors)
    
    if args.grid_search:
        print("🔍 Running Grid Search Optimization")
        print("=" * 50)
        results = optimizer.grid_search_optimization(
            vectors, 
            max_combinations=args.max_combinations,
            test_size=args.test_size
        )
        
        print(f"\n📊 Top 5 Results:")
        for i, result in enumerate(results[:5]):
            if result['success']:
                print(f"{i+1}. Compression: {result['compression_ratio']:.1%}, "
                      f"Quality: {result['quality_metrics']['cosine_similarity']:.3f}")
    
    elif args.targeted:
        print("🎯 Running Targeted Optimization")
        print("=" * 50)
        result = optimizer.run_targeted_optimization(vectors, test_size=args.test_size)
        
        if result and result['success']:
            print(f"\n🏆 Best Configuration:")
            print(f"Compression Ratio: {result['compression_ratio']:.1%}")
            print(f"Quality Score: {result['quality_metrics']['cosine_similarity']:.3f}")
            print(f"Parameters: {json.dumps(result['params'], indent=2)}")
        else:
            print("❌ Optimization failed")
    
    else:
        print("Use --grid-search or --targeted to run optimization")
        return
    
    # Save results
    optimizer.save_optimization_results(args.output)
    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()