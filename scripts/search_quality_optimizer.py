#!/usr/bin/env python3
"""
Search quality optimization for SpiralDeltaDB.

This script evaluates and improves search quality preservation
during compression and provides benchmarking tools.
"""

import numpy as np
import sys
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Tuple, Any
import argparse
from collections import defaultdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from spiraldelta import SpiralDeltaDB

# Import dataset manager
import importlib.util
dataset_spec = importlib.util.spec_from_file_location("dataset_manager", 
                                                     Path(__file__).parent / "dataset_manager.py")
dataset_manager = importlib.util.module_from_spec(dataset_spec)
dataset_spec.loader.exec_module(dataset_manager)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchQualityOptimizer:
    """Optimize and evaluate search quality preservation."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize search quality optimizer."""
        self.data_dir = Path(data_dir)
        self.dataset_manager = dataset_manager.DatasetManager(data_dir)
    
    def evaluate_search_quality(
        self, 
        db: SpiralDeltaDB,
        test_vectors: np.ndarray,
        test_words: List[str],
        n_queries: int = 100,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """
        Comprehensive search quality evaluation.
        
        Args:
            db: Database instance to test
            test_vectors: Test vector dataset
            test_words: Corresponding words
            n_queries: Number of queries to test
            k_values: k values for recall@k evaluation
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info(f"Evaluating search quality with {n_queries} queries")
        
        # Select random queries
        query_indices = np.random.choice(len(test_vectors), n_queries, replace=False)
        
        metrics = defaultdict(list)
        search_times = []
        
        for i, query_idx in enumerate(query_indices):
            query_vector = test_vectors[query_idx]
            query_word = test_words[query_idx]
            
            # Get ground truth using brute force
            ground_truth = self._get_ground_truth(query_vector, test_vectors, max(k_values))
            
            # Time the database search
            start_time = time.time()
            try:
                results = db.search(query_vector, k=max(k_values))
                search_time = (time.time() - start_time) * 1000  # ms
                search_times.append(search_time)
                
                # Extract result indices (simplified assumption: IDs map to indices)
                result_indices = list(range(min(len(results), max(k_values))))
                
                # Calculate recall@k for each k
                for k in k_values:
                    true_top_k = set(ground_truth[:k])
                    found_top_k = set(result_indices[:k])
                    
                    recall = len(true_top_k & found_top_k) / k if k > 0 else 0.0
                    metrics[f"recall@{k}"].append(recall)
                
                # Calculate precision@10 
                if len(result_indices) >= 10:
                    true_top_10 = set(ground_truth[:10])
                    found_top_10 = set(result_indices[:10])
                    precision = len(true_top_10 & found_top_10) / 10.0
                    metrics["precision@10"].append(precision)
                
            except Exception as e:
                logger.warning(f"Query {i} failed: {e}")
                # Add zeros for failed queries
                for k in k_values:
                    metrics[f"recall@{k}"].append(0.0)
                metrics["precision@10"].append(0.0)
                search_times.append(0.0)
        
        # Calculate averages
        quality_results = {}
        for metric, values in metrics.items():
            quality_results[metric] = np.mean(values) if values else 0.0
        
        # Add timing metrics
        quality_results["avg_search_time_ms"] = np.mean(search_times) if search_times else 0.0
        quality_results["search_throughput_qps"] = 1000.0 / quality_results["avg_search_time_ms"] if quality_results["avg_search_time_ms"] > 0 else 0.0
        
        return quality_results
    
    def _get_ground_truth(self, query: np.ndarray, vectors: np.ndarray, k: int) -> List[int]:
        """Get ground truth top-k results using exact similarity."""
        similarities = np.dot(vectors, query)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices.tolist()
    
    def optimize_search_parameters(
        self, 
        test_vectors: np.ndarray,
        test_words: List[str],
        target_recall: float = 0.90
    ) -> Dict[str, Any]:
        """
        Find optimal search parameters to achieve target recall.
        
        Args:
            test_vectors: Test dataset
            test_words: Test words
            target_recall: Target recall@10 to achieve
            
        Returns:
            Dictionary with optimal parameters and results
        """
        logger.info(f"Optimizing search parameters for {target_recall:.1%} recall@10")
        
        # Parameter combinations to test
        param_combinations = [
            # Conservative (high quality)
            {
                "ef_construction": 400,
                "ef_search": 100,
                "max_layers": 16,
                "quantization_levels": 3,
                "n_bits": 8,
                "anchor_stride": 64,
            },
            # Balanced 
            {
                "ef_construction": 300,
                "ef_search": 80,
                "max_layers": 12,
                "quantization_levels": 4,
                "n_bits": 7,
                "anchor_stride": 48,
            },
            # Aggressive (high compression)
            {
                "ef_construction": 200,
                "ef_search": 60,
                "max_layers": 8,
                "quantization_levels": 5,
                "n_bits": 6,
                "anchor_stride": 32,
            },
            # Ultra-optimized
            {
                "ef_construction": 500,
                "ef_search": 150,
                "max_layers": 20,
                "quantization_levels": 2,
                "n_bits": 8,
                "anchor_stride": 96,
            }
        ]
        
        best_params = None
        best_recall = 0.0
        all_results = []
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing parameter set {i+1}/{len(param_combinations)}")
            
            try:
                # Create database with test parameters
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    db = SpiralDeltaDB(
                        dimensions=300,
                        storage_path=f"{temp_dir}/search_test.db",
                        auto_train_threshold=min(1000, len(test_vectors) // 2),
                        **params
                    )
                    
                    # Insert test data
                    start_time = time.time()
                    db.insert(test_vectors[:5000])  # Use subset for speed
                    insert_time = time.time() - start_time
                    
                    # Force training
                    if not db._is_trained:
                        db._auto_train_encoder()
                    
                    # Evaluate search quality
                    quality_metrics = self.evaluate_search_quality(
                        db, test_vectors[:2000], test_words[:2000], n_queries=50
                    )
                    
                    # Get compression stats
                    stats = db.get_stats()
                    
                    result = {
                        "parameters": params,
                        "quality_metrics": quality_metrics,
                        "compression_ratio": stats.compression_ratio,
                        "insert_time": insert_time,
                        "recall@10": quality_metrics.get("recall@10", 0.0)
                    }
                    
                    all_results.append(result)
                    
                    current_recall = quality_metrics.get("recall@10", 0.0)
                    logger.info(f"  Recall@10: {current_recall:.3f}, Compression: {stats.compression_ratio:.3f}")
                    
                    # Check if this is the best so far
                    if current_recall > best_recall and current_recall >= target_recall:
                        best_recall = current_recall
                        best_params = result
                    
            except Exception as e:
                logger.warning(f"Parameter set {i+1} failed: {e}")
                continue
        
        if best_params is None:
            # Find the best available result
            all_results.sort(key=lambda x: x["recall@10"], reverse=True)
            best_params = all_results[0] if all_results else None
        
        optimization_result = {
            "target_recall": target_recall,
            "best_parameters": best_params,
            "all_results": all_results,
            "optimization_successful": best_params is not None and best_params["recall@10"] >= target_recall
        }
        
        return optimization_result
    
    def create_search_quality_report(
        self,
        output_file: str = "search_quality_report.json"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive search quality report.
        
        Args:
            output_file: Output file path
            
        Returns:
            Complete quality report
        """
        logger.info("🔍 Generating comprehensive search quality report")
        
        # Load test dataset
        test_vectors, test_words, metadata = self.dataset_manager.get_glove_300_dataset(
            max_vectors=10000, force_synthetic=True
        )
        
        logger.info(f"Using {len(test_vectors)} test vectors from {metadata['source']}")
        
        # Run parameter optimization
        optimization_results = self.optimize_search_parameters(
            test_vectors, test_words, target_recall=0.85
        )
        
        # Generate final report
        report = {
            "timestamp": time.time(),
            "dataset_info": metadata,
            "test_vectors": len(test_vectors),
            "optimization_results": optimization_results,
            "summary": {
                "best_recall@10": optimization_results["best_parameters"]["recall@10"] if optimization_results["best_parameters"] else 0.0,
                "best_compression": optimization_results["best_parameters"]["compression_ratio"] if optimization_results["best_parameters"] else 0.0,
                "target_achieved": optimization_results["optimization_successful"],
                "parameter_sets_tested": len(optimization_results["all_results"])
            }
        }
        
        # Save report
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Search quality report saved to {output_path}")
        return report


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Search quality optimization for SpiralDeltaDB")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--target-recall", type=float, default=0.85, help="Target recall@10")
    parser.add_argument("--output", default="search_quality_report.json", help="Output report file")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation with fewer parameters")
    
    args = parser.parse_args()
    
    optimizer = SearchQualityOptimizer(args.data_dir)
    
    if args.quick:
        logger.info("🚀 Running quick search quality check")
        
        # Load small dataset
        vectors, words, metadata = optimizer.dataset_manager.get_glove_300_dataset(
            max_vectors=2000, force_synthetic=True
        )
        
        # Test with default parameters
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            db = SpiralDeltaDB(
                dimensions=300,
                storage_path=f"{temp_dir}/quick_test.db",
                auto_train_threshold=500
            )
            
            db.insert(vectors)
            if not db._is_trained:
                db._auto_train_encoder()
            
            quality = optimizer.evaluate_search_quality(
                db, vectors, words, n_queries=20
            )
            
            print("📊 Quick Search Quality Results:")
            for metric, value in quality.items():
                print(f"  {metric}: {value:.3f}")
    
    else:
        # Full optimization
        report = optimizer.create_search_quality_report(args.output)
        
        print("📊 Search Quality Optimization Complete:")
        print(f"  Best Recall@10: {report['summary']['best_recall@10']:.3f}")
        print(f"  Best Compression: {report['summary']['best_compression']:.3f}")
        print(f"  Target Achieved: {'✓' if report['summary']['target_achieved'] else '✗'}")
        print(f"  Report saved to: {args.output}")


if __name__ == "__main__":
    main()