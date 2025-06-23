#!/usr/bin/env python3
"""
Quality optimization for BERT-768 embeddings.

This script fine-tunes DeltaEncoder parameters to improve reconstruction quality
(target: cosine similarity ≥0.25) while maintaining 66.8%+ compression ratio.
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

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from spiraldelta.delta_encoder import DeltaEncoder
from spiraldelta.spiral_coordinator import SpiralCoordinator

# Import dataset manager
sys.path.append(str(Path(__file__).parent))
from bert_dataset_manager import BERTDatasetManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityOptimizer:
    """Advanced quality optimization for BERT embeddings."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize quality optimizer."""
        self.data_dir = Path(data_dir)
        self.dataset_manager = BERTDatasetManager(data_dir)
        
        # Quality targets
        self.target_quality = 0.25  # Cosine similarity
        self.min_compression = 0.668  # Must maintain 66.8%+ compression
        
        # Current best from previous optimization
        self.baseline_params = {
            'quantization_levels': 4,
            'n_subspaces': 12,
            'n_bits': 7,
            'anchor_stride': 64,
            'spiral_constant': 1.618
        }
        
        # Quality-focused parameter search space
        self.quality_param_grid = {
            # Reduce quantization levels for better quality
            'quantization_levels': [2, 3, 4],
            
            # Reduce subspaces for better reconstruction
            'n_subspaces': [6, 8, 10, 12],
            
            # Increase bits for higher precision
            'n_bits': [7, 8, 9],
            
            # Smaller anchor stride for more anchor points
            'anchor_stride': [32, 48, 64, 80],
            
            # Test different spiral constants
            'spiral_constant': [1.5, 1.618, 1.8]
        }
        
        # Results tracking
        self.optimization_results = []
        self.best_quality_params = None
        self.best_balanced_params = None
        
    def load_bert_data(self, max_vectors: int = 25000) -> Tuple[np.ndarray, List[str]]:
        """Load BERT dataset for quality optimization."""
        logger.info(f"Loading BERT dataset: {max_vectors} vectors")
        
        vectors, metadata, info = self.dataset_manager.get_bert_dataset(max_vectors=max_vectors)
        
        logger.info(f"Loaded {len(vectors)} BERT vectors ({vectors.shape[1]}D)")
        return vectors, metadata
    
    def evaluate_detailed_quality(
        self,
        original_vectors: List[np.ndarray],
        reconstructed_vectors: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Enhanced quality evaluation with multiple metrics.
        
        Args:
            original_vectors: Original vector sequence
            reconstructed_vectors: Reconstructed vector sequence
            
        Returns:
            Comprehensive quality metrics
        """
        orig_array = np.array(original_vectors)
        recon_array = np.array(reconstructed_vectors)
        
        # Individual cosine similarities
        cos_sims = []
        for orig, recon in zip(original_vectors, reconstructed_vectors):
            orig_norm = np.linalg.norm(orig)
            recon_norm = np.linalg.norm(recon)
            
            if orig_norm > 1e-8 and recon_norm > 1e-8:
                cos_sim = np.dot(orig, recon) / (orig_norm * recon_norm)
                cos_sims.append(cos_sim)
        
        cos_sims = np.array(cos_sims)
        
        # Multiple quality metrics
        metrics = {
            # Primary metric: cosine similarity
            "cosine_similarity": float(np.mean(cos_sims)),
            "cosine_similarity_std": float(np.std(cos_sims)),
            "cosine_similarity_min": float(np.min(cos_sims)),
            "cosine_similarity_p10": float(np.percentile(cos_sims, 10)),
            
            # Secondary metrics
            "mse": float(np.mean((orig_array - recon_array) ** 2)),
            "mae": float(np.mean(np.abs(orig_array - recon_array))),
            
            # Relative errors
            "relative_l2_error": float(np.mean(
                np.linalg.norm(orig_array - recon_array, axis=1) /
                (np.linalg.norm(orig_array, axis=1) + 1e-8)
            )),
            
            # Vector magnitude preservation
            "magnitude_preservation": float(np.mean(
                np.abs(np.linalg.norm(orig_array, axis=1) - 
                       np.linalg.norm(recon_array, axis=1)) /
                (np.linalg.norm(orig_array, axis=1) + 1e-8)
            )),
            
            # Quality score (weighted combination)
            "quality_score": float(
                np.mean(cos_sims) * 0.8 +  # Primary: cosine similarity
                (1.0 - np.mean(np.abs(orig_array - recon_array))) * 0.2  # Secondary: MAE
            )
        }
        
        return metrics
    
    def test_quality_focused_params(
        self,
        vectors: np.ndarray,
        params: Dict[str, any],
        test_size: int = 3000
    ) -> Dict[str, any]:
        """
        Test parameters with focus on quality metrics.
        
        Args:
            vectors: BERT vectors to test
            params: Parameter configuration
            test_size: Number of vectors to test
            
        Returns:
            Detailed results with quality focus
        """
        logger.info(f"Testing quality-focused params: {params}")
        
        test_vectors = vectors[:test_size]
        
        try:
            # Initialize components
            spiral_coordinator = SpiralCoordinator(
                dimensions=768,
                spiral_constant=params['spiral_constant']
            )
            
            delta_encoder = DeltaEncoder(
                quantization_levels=params['quantization_levels'],
                compression_target=self.min_compression,
                n_subspaces=params['n_subspaces'],
                n_bits=params['n_bits'],
                anchor_stride=params['anchor_stride']
            )
            
            # Process vectors
            start_time = time.time()
            sorted_coords = spiral_coordinator.sort_by_spiral(test_vectors)
            sorted_vectors = [coord.vector for coord in sorted_coords]
            spiral_time = time.time() - start_time
            
            # Training with smaller chunks for quality
            start_time = time.time()
            chunk_size = min(500, len(sorted_vectors) // 6)  # Smaller chunks
            training_sequences = [
                sorted_vectors[i:i+chunk_size] 
                for i in range(0, len(sorted_vectors), chunk_size)
                if i+chunk_size <= len(sorted_vectors)
            ]
            if not training_sequences:
                training_sequences = [sorted_vectors]
            
            delta_encoder.train(training_sequences)
            train_time = time.time() - start_time
            
            # Compression
            start_time = time.time()
            compressed = delta_encoder.encode_sequence(sorted_vectors)
            encode_time = time.time() - start_time
            
            # Decompression
            start_time = time.time()
            reconstructed = delta_encoder.decode_sequence(compressed)
            decode_time = time.time() - start_time
            
            # Enhanced quality evaluation
            quality_metrics = self.evaluate_detailed_quality(sorted_vectors, reconstructed)
            
            # Calculate compression metrics
            original_size = len(test_vectors) * 768 * 4
            compressed_size = delta_encoder._estimate_compressed_size(
                compressed.anchors, compressed.delta_codes
            )
            actual_compression = 1.0 - (compressed_size / original_size) if original_size > 0 else 0.0
            
            # Success criteria
            quality_target_met = quality_metrics['cosine_similarity'] >= self.target_quality
            compression_target_met = compressed.compression_ratio >= self.min_compression
            
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
                "success_criteria": {
                    "quality_target_met": quality_target_met,
                    "compression_target_met": compression_target_met,
                    "both_targets_met": quality_target_met and compression_target_met
                },
                "test_vectors": len(test_vectors),
                "success": True,
                "error": None
            }
            
            logger.info(f"Quality: {quality_metrics['cosine_similarity']:.3f} "
                       f"({'✅' if quality_target_met else '❌'} ≥{self.target_quality}), "
                       f"Compression: {compressed.compression_ratio:.1%} "
                       f"({'✅' if compression_target_met else '❌'} ≥{self.min_compression:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Quality test failed: {e}")
            return {
                "params": params.copy(),
                "compression_ratio": 0.0,
                "quality_metrics": {"cosine_similarity": 0.0},
                "success_criteria": {
                    "quality_target_met": False,
                    "compression_target_met": False,
                    "both_targets_met": False
                },
                "success": False,
                "error": str(e)
            }
    
    def run_quality_grid_search(
        self,
        vectors: np.ndarray,
        max_combinations: int = 30,
        test_size: int = 3000
    ) -> List[Dict[str, any]]:
        """
        Grid search focused on quality improvement.
        
        Args:
            vectors: BERT vectors for optimization
            max_combinations: Maximum parameter combinations to test
            test_size: Vectors per test
            
        Returns:
            Results sorted by quality score
        """
        logger.info("Starting quality-focused grid search")
        logger.info(f"Target: cosine similarity ≥{self.target_quality}, compression ≥{self.min_compression:.1%}")
        
        # Generate parameter combinations
        param_names = list(self.quality_param_grid.keys())
        param_values = list(self.quality_param_grid.values())
        
        all_combinations = list(product(*param_values))
        
        # Prioritize quality-focused combinations
        if len(all_combinations) > max_combinations:
            # Prefer combinations with higher bits, smaller anchor stride, fewer quantization levels
            def quality_score(combo):
                params = dict(zip(param_names, combo))
                score = 0
                score += (params['n_bits'] - 6) * 10  # Higher bits better
                score += (100 - params['anchor_stride']) / 10  # Smaller stride better
                score += (5 - params['quantization_levels']) * 5  # Fewer levels better
                score += (15 - params['n_subspaces']) / 2  # Fewer subspaces better
                return score
            
            all_combinations.sort(key=quality_score, reverse=True)
            combinations = all_combinations[:max_combinations]
        else:
            combinations = all_combinations
        
        logger.info(f"Testing {len(combinations)} quality-focused parameter combinations")
        
        results = []
        successful_results = []
        
        for i, param_values in enumerate(combinations):
            params = dict(zip(param_names, param_values))
            
            logger.info(f"Test {i+1}/{len(combinations)}: {params}")
            
            result = self.test_quality_focused_params(vectors, params, test_size)
            results.append(result)
            
            if result['success']:
                successful_results.append(result)
                
                # Track best results
                if result['success_criteria']['both_targets_met']:
                    if (self.best_balanced_params is None or 
                        result['quality_metrics']['cosine_similarity'] > 
                        self.best_balanced_params['quality_metrics']['cosine_similarity']):
                        self.best_balanced_params = result
                
                if (self.best_quality_params is None or 
                    result['quality_metrics']['cosine_similarity'] > 
                    self.best_quality_params['quality_metrics']['cosine_similarity']):
                    self.best_quality_params = result
        
        # Sort by quality score
        successful_results.sort(key=lambda x: x['quality_metrics']['cosine_similarity'], reverse=True)
        
        self.optimization_results = results
        return successful_results
    
    def run_targeted_quality_optimization(
        self,
        vectors: np.ndarray,
        test_size: int = 5000
    ) -> Dict[str, any]:
        """
        Run targeted optimization for quality improvement.
        
        Args:
            vectors: BERT vectors for optimization
            test_size: Number of vectors to test
            
        Returns:
            Best quality configuration
        """
        logger.info("Running targeted quality optimization")
        
        # Quality-focused configurations
        quality_configs = [
            # Configuration 1: High precision, fewer quantization levels
            {
                'quantization_levels': 2,
                'n_subspaces': 8,
                'n_bits': 8,
                'anchor_stride': 32,
                'spiral_constant': 1.618
            },
            # Configuration 2: Moderate precision, balanced
            {
                'quantization_levels': 3,
                'n_subspaces': 6,
                'n_bits': 8,
                'anchor_stride': 48,
                'spiral_constant': 1.5
            },
            # Configuration 3: High bits, small stride
            {
                'quantization_levels': 3,
                'n_subspaces': 10,
                'n_bits': 9,
                'anchor_stride': 32,
                'spiral_constant': 1.8
            },
            # Configuration 4: Conservative compression for quality
            {
                'quantization_levels': 2,
                'n_subspaces': 6,
                'n_bits': 9,
                'anchor_stride': 24,
                'spiral_constant': 1.618
            }
        ]
        
        logger.info(f"Testing {len(quality_configs)} targeted quality configurations")
        
        results = []
        for i, params in enumerate(quality_configs):
            logger.info(f"Testing quality config {i+1}: {params}")
            result = self.test_quality_focused_params(vectors, params, test_size)
            results.append(result)
        
        # Find best configuration
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            logger.error("No successful quality optimization results")
            return None
        
        # Prioritize configurations that meet both targets
        both_targets_met = [r for r in successful_results if r['success_criteria']['both_targets_met']]
        
        if both_targets_met:
            best_result = max(both_targets_met, key=lambda x: x['quality_metrics']['cosine_similarity'])
            logger.info("✅ Found configuration meeting both quality and compression targets")
        else:
            # Fall back to best quality
            best_result = max(successful_results, key=lambda x: x['quality_metrics']['cosine_similarity'])
            logger.info("⚠️ Using best quality configuration (may not meet compression target)")
        
        self.best_balanced_params = best_result
        
        logger.info(f"Best quality configuration:")
        logger.info(f"  Quality: {best_result['quality_metrics']['cosine_similarity']:.3f}")
        logger.info(f"  Compression: {best_result['compression_ratio']:.1%}")
        logger.info(f"  Parameters: {best_result['params']}")
        
        return best_result
    
    def save_quality_results(self, output_path: str = "quality_optimization_results.json"):
        """Save quality optimization results."""
        results_data = {
            "quality_optimization": {
                "target_quality": self.target_quality,
                "min_compression": self.min_compression,
                "baseline_params": self.baseline_params,
                "best_quality_params": self.best_quality_params,
                "best_balanced_params": self.best_balanced_params,
                "all_results": self.optimization_results,
                "timestamp": time.time()
            }
        }
        
        output_file = self.data_dir / output_path
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved quality optimization results to {output_file}")
        return output_file


def main():
    """Main function for quality optimization."""
    parser = argparse.ArgumentParser(description="Quality optimization for BERT-768")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--max-vectors", type=int, default=25000, help="Maximum vectors to load")
    parser.add_argument("--test-size", type=int, default=3000, help="Vectors per test")
    parser.add_argument("--grid-search", action="store_true", help="Run quality grid search")
    parser.add_argument("--targeted", action="store_true", help="Run targeted quality optimization")
    parser.add_argument("--max-combinations", type=int, default=30, help="Max grid search combinations")
    parser.add_argument("--output", default="quality_optimization_results.json", help="Output file")
    
    args = parser.parse_args()
    
    optimizer = QualityOptimizer(args.data_dir)
    
    # Load BERT data
    vectors, metadata = optimizer.load_bert_data(args.max_vectors)
    
    if args.grid_search:
        print("🔍 Running Quality-Focused Grid Search")
        print("=" * 60)
        results = optimizer.run_quality_grid_search(
            vectors, 
            max_combinations=args.max_combinations,
            test_size=args.test_size
        )
        
        print(f"\n📊 Top 5 Quality Results:")
        for i, result in enumerate(results[:5]):
            if result['success']:
                quality = result['quality_metrics']['cosine_similarity']
                compression = result['compression_ratio']
                both_met = result['success_criteria']['both_targets_met']
                status = "✅ BOTH" if both_met else "⚠️ PARTIAL"
                print(f"{i+1}. {status} - Quality: {quality:.3f}, Compression: {compression:.1%}")
    
    elif args.targeted:
        print("🎯 Running Targeted Quality Optimization")
        print("=" * 60)
        result = optimizer.run_targeted_quality_optimization(vectors, test_size=args.test_size)
        
        if result and result['success']:
            print(f"\n🏆 Best Quality Configuration:")
            print(f"Quality Score: {result['quality_metrics']['cosine_similarity']:.3f}")
            print(f"Compression Ratio: {result['compression_ratio']:.1%}")
            print(f"Both Targets Met: {'✅' if result['success_criteria']['both_targets_met'] else '❌'}")
            print(f"Parameters: {json.dumps(result['params'], indent=2)}")
        else:
            print("❌ Quality optimization failed")
    
    else:
        print("Use --grid-search or --targeted to run quality optimization")
        return
    
    # Save results
    optimizer.save_quality_results(args.output)
    print(f"\n💾 Quality optimization results saved to {args.output}")


if __name__ == "__main__":
    main()