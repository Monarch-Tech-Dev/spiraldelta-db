#!/usr/bin/env python3
"""
Comprehensive BERT-768 benchmarking for SpiralDeltaDB.

This script performs detailed performance benchmarking of the optimized
DeltaEncoder parameters for BERT embeddings with the target 66.8% compression ratio.
"""

import numpy as np
import sys
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Tuple, Optional
import argparse
# import matplotlib.pyplot as plt  # Optional - for plotting
# import seaborn as sns  # Optional - for plotting
# from tabulate import tabulate  # Optional - for pretty tables

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from spiraldelta import SpiralDeltaDB
from spiraldelta.delta_encoder import DeltaEncoder
from spiraldelta.spiral_coordinator import SpiralCoordinator

# Import dataset manager
sys.path.append(str(Path(__file__).parent))
from bert_dataset_manager import BERTDatasetManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTBenchmark:
    """Comprehensive benchmark suite for BERT-768 embeddings."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize BERT benchmark."""
        self.data_dir = Path(data_dir)
        self.dataset_manager = BERTDatasetManager(data_dir)
        
        # Optimized parameters from optimization
        self.optimized_params = {
            'quantization_levels': 4,
            'n_subspaces': 12,
            'n_bits': 7,
            'anchor_stride': 64,
            'spiral_constant': 1.618
        }
        
        # Benchmark results
        self.results = {}
        
    def load_bert_data(self, max_vectors: int = 100000) -> Tuple[np.ndarray, List[str]]:
        """Load BERT dataset for benchmarking."""
        logger.info(f"Loading BERT dataset: {max_vectors} vectors")
        
        vectors, metadata, info = self.dataset_manager.get_bert_dataset(max_vectors=max_vectors)
        
        logger.info(f"Loaded {len(vectors)} BERT vectors ({vectors.shape[1]}D)")
        logger.info(f"Vector norm stats: mean={np.mean(np.linalg.norm(vectors, axis=1)):.3f}")
        
        return vectors, metadata
    
    def benchmark_compression_scaling(
        self,
        vectors: np.ndarray,
        test_sizes: List[int] = [1000, 2500, 5000, 10000, 25000]
    ) -> Dict[str, any]:
        """
        Benchmark compression performance across different dataset sizes.
        
        Args:
            vectors: BERT vectors for testing
            test_sizes: List of dataset sizes to test
            
        Returns:
            Scaling benchmark results
        """
        logger.info("Running compression scaling benchmark")
        
        results = []
        
        for size in test_sizes:
            if size > len(vectors):
                logger.warning(f"Skipping size {size} (exceeds available vectors)")
                continue
            
            logger.info(f"Testing compression with {size} vectors...")
            
            # Use subset of vectors
            test_vectors = vectors[:size]
            
            try:
                # Initialize components with optimized parameters
                spiral_coordinator = SpiralCoordinator(
                    dimensions=768,
                    spiral_constant=self.optimized_params['spiral_constant']
                )
                
                delta_encoder = DeltaEncoder(
                    quantization_levels=self.optimized_params['quantization_levels'],
                    compression_target=0.668,
                    n_subspaces=self.optimized_params['n_subspaces'],
                    n_bits=self.optimized_params['n_bits'],
                    anchor_stride=self.optimized_params['anchor_stride']
                )
                
                # Spiral transformation timing
                start_time = time.time()
                sorted_coords = spiral_coordinator.sort_by_spiral(test_vectors)
                sorted_vectors = [coord.vector for coord in sorted_coords]
                spiral_time = time.time() - start_time
                
                # Training timing
                start_time = time.time()
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
                
                # Compression timing
                start_time = time.time()
                compressed = delta_encoder.encode_sequence(sorted_vectors)
                encode_time = time.time() - start_time
                
                # Decompression timing
                start_time = time.time()
                reconstructed = delta_encoder.decode_sequence(compressed)
                decode_time = time.time() - start_time
                
                # Quality evaluation
                quality_metrics = self._evaluate_quality(sorted_vectors, reconstructed)
                
                # Memory usage
                original_size = len(test_vectors) * 768 * 4  # float32
                compressed_size = delta_encoder._estimate_compressed_size(
                    compressed.anchors, compressed.delta_codes
                )
                
                result = {
                    "dataset_size": size,
                    "compression_ratio": float(compressed.compression_ratio),
                    "actual_compression": float(1.0 - (compressed_size / original_size)) if original_size > 0 else 0.0,
                    "quality_metrics": quality_metrics,
                    "timing": {
                        "spiral_transform": spiral_time,
                        "training": train_time,
                        "encoding": encode_time,
                        "decoding": decode_time,
                        "total": spiral_time + train_time + encode_time + decode_time
                    },
                    "throughput": {
                        "encoding_vectors_per_sec": size / encode_time if encode_time > 0 else 0,
                        "decoding_vectors_per_sec": size / decode_time if decode_time > 0 else 0,
                        "total_vectors_per_sec": size / (spiral_time + train_time + encode_time + decode_time) if (spiral_time + train_time + encode_time + decode_time) > 0 else 0
                    },
                    "memory": {
                        "original_size_mb": original_size / (1024**2),
                        "compressed_size_mb": compressed_size / (1024**2),
                        "memory_savings_mb": (original_size - compressed_size) / (1024**2)
                    }
                }
                
                results.append(result)
                
                logger.info(f"Size {size}: {result['compression_ratio']:.1%} compression, "
                           f"{quality_metrics['cosine_similarity']:.3f} quality, "
                           f"{result['timing']['total']:.2f}s total time")
                
            except Exception as e:
                logger.error(f"Failed to benchmark size {size}: {e}")
                results.append({
                    "dataset_size": size,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "scaling_results": results,
            "optimized_params": self.optimized_params
        }
    
    def benchmark_parameter_sensitivity(
        self,
        vectors: np.ndarray,
        base_params: Dict[str, any] = None,
        test_size: int = 5000
    ) -> Dict[str, any]:
        """
        Benchmark sensitivity to parameter changes.
        
        Args:
            vectors: BERT vectors for testing
            base_params: Base parameters to vary
            test_size: Number of vectors to test
            
        Returns:
            Parameter sensitivity results
        """
        if base_params is None:
            base_params = self.optimized_params.copy()
        
        logger.info(f"Running parameter sensitivity analysis with {test_size} vectors")
        
        test_vectors = vectors[:test_size]
        
        # Parameters to vary
        param_variations = {
            'quantization_levels': [2, 3, 4, 5, 6],
            'n_subspaces': [4, 6, 8, 12, 16],
            'n_bits': [5, 6, 7, 8],
            'anchor_stride': [32, 48, 64, 96, 128]
        }
        
        sensitivity_results = {}
        
        for param_name, param_values in param_variations.items():
            logger.info(f"Testing sensitivity for {param_name}")
            
            param_results = []
            
            for param_value in param_values:
                # Create modified parameters
                test_params = base_params.copy()
                test_params[param_name] = param_value
                
                try:
                    result = self._single_parameter_test(test_vectors, test_params)
                    result["param_value"] = param_value
                    param_results.append(result)
                    
                    logger.info(f"  {param_name}={param_value}: "
                               f"{result['compression_ratio']:.1%} compression, "
                               f"{result['quality_metrics']['cosine_similarity']:.3f} quality")
                    
                except Exception as e:
                    logger.error(f"Failed test for {param_name}={param_value}: {e}")
                    param_results.append({
                        "param_value": param_value,
                        "error": str(e),
                        "success": False
                    })
            
            sensitivity_results[param_name] = param_results
        
        return {
            "sensitivity_results": sensitivity_results,
            "base_params": base_params,
            "test_size": test_size
        }
    
    def _single_parameter_test(
        self,
        vectors: np.ndarray,
        params: Dict[str, any]
    ) -> Dict[str, any]:
        """Run a single parameter configuration test."""
        # Initialize components
        spiral_coordinator = SpiralCoordinator(
            dimensions=768,
            spiral_constant=params['spiral_constant']
        )
        
        delta_encoder = DeltaEncoder(
            quantization_levels=params['quantization_levels'],
            compression_target=0.668,
            n_subspaces=params['n_subspaces'],
            n_bits=params['n_bits'],
            anchor_stride=params['anchor_stride']
        )
        
        # Process vectors
        sorted_coords = spiral_coordinator.sort_by_spiral(vectors)
        sorted_vectors = [coord.vector for coord in sorted_coords]
        
        # Train and encode
        chunk_size = min(500, len(sorted_vectors) // 4)
        training_sequences = [
            sorted_vectors[i:i+chunk_size] 
            for i in range(0, len(sorted_vectors), chunk_size)
            if i+chunk_size <= len(sorted_vectors)
        ]
        if not training_sequences:
            training_sequences = [sorted_vectors]
        
        delta_encoder.train(training_sequences)
        compressed = delta_encoder.encode_sequence(sorted_vectors)
        reconstructed = delta_encoder.decode_sequence(compressed)
        
        # Evaluate
        quality_metrics = self._evaluate_quality(sorted_vectors, reconstructed)
        
        return {
            "compression_ratio": float(compressed.compression_ratio),
            "quality_metrics": quality_metrics,
            "success": True
        }
    
    def _evaluate_quality(
        self,
        original_vectors: List[np.ndarray],
        reconstructed_vectors: List[np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate compression quality metrics."""
        orig_array = np.array(original_vectors)
        recon_array = np.array(reconstructed_vectors)
        
        # Mean squared error
        mse = np.mean((orig_array - recon_array) ** 2)
        
        # Cosine similarity
        cos_sims = []
        for orig, recon in zip(original_vectors, reconstructed_vectors):
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
    
    def benchmark_real_world_performance(
        self,
        vectors: np.ndarray,
        query_vectors: np.ndarray = None,
        test_size: int = 25000
    ) -> Dict[str, any]:
        """
        Benchmark real-world usage scenarios.
        
        Args:
            vectors: BERT vectors for database
            query_vectors: Query vectors for search (optional)
            test_size: Size of test database
            
        Returns:
            Real-world performance results
        """
        logger.info(f"Running real-world performance benchmark with {test_size} vectors")
        
        test_vectors = vectors[:test_size]
        
        if query_vectors is None:
            # Use subset of test vectors as queries
            query_vectors = test_vectors[-100:]  # Last 100 vectors as queries
        
        try:
            # Initialize SpiralDeltaDB with optimized parameters
            db = SpiralDeltaDB(
                dimensions=768,
                compression_ratio=1.0 - 0.668,  # Convert to storage ratio
                spiral_constant=self.optimized_params['spiral_constant'],
                quantization_levels=self.optimized_params['quantization_levels'],
                n_subspaces=self.optimized_params['n_subspaces'],
                n_bits=self.optimized_params['n_bits'],
                anchor_stride=self.optimized_params['anchor_stride']
            )
            
            # Bulk insert timing
            metadata = [{"doc_id": i, "text": f"document_{i}"} for i in range(len(test_vectors))]
            
            start_time = time.time()
            vector_ids = db.insert(test_vectors, metadata)
            insert_time = time.time() - start_time
            
            # Search timing
            search_times = []
            search_results = []
            
            for i, query in enumerate(query_vectors):
                start_time = time.time()
                results = db.search(query, k=10)
                search_time = time.time() - start_time
                search_times.append(search_time)
                search_results.append(results)
                
                if i % 10 == 0:
                    logger.info(f"Search {i+1}/{len(query_vectors)}: {search_time*1000:.2f}ms")
            
            # Database statistics
            stats = db.get_stats()
            
            return {
                "database_size": len(test_vectors),
                "query_count": len(query_vectors),
                "insertion": {
                    "total_time_seconds": insert_time,
                    "vectors_per_second": len(test_vectors) / insert_time if insert_time > 0 else 0,
                    "time_per_vector_ms": (insert_time * 1000) / len(test_vectors) if len(test_vectors) > 0 else 0
                },
                "search": {
                    "mean_time_ms": np.mean(search_times) * 1000,
                    "median_time_ms": np.median(search_times) * 1000,
                    "p95_time_ms": np.percentile(search_times, 95) * 1000,
                    "queries_per_second": len(query_vectors) / sum(search_times) if sum(search_times) > 0 else 0
                },
                "database_stats": {
                    "compression_ratio": stats.compression_ratio,
                    "storage_size_mb": stats.storage_size_mb,
                    "memory_usage_mb": stats.memory_usage_mb
                },
                "optimized_params": self.optimized_params
            }
            
        except Exception as e:
            logger.error(f"Real-world benchmark failed: {e}")
            return {"error": str(e), "success": False}
    
    def generate_benchmark_report(self, results: Dict[str, any], output_path: str = "bert_benchmark_report.json"):
        """Generate comprehensive benchmark report."""
        report = {
            "benchmark_summary": {
                "timestamp": time.time(),
                "optimized_parameters": self.optimized_params,
                "target_compression": 0.668,
                "achieved_compression": results.get("scaling_results", {}).get("scaling_results", [{}])[-1].get("compression_ratio", 0.0) if results.get("scaling_results") else 0.0
            },
            "detailed_results": results
        }
        
        output_file = self.data_dir / output_path
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved benchmark report to {output_file}")
        
        # Generate summary table
        self._print_summary_table(results)
        
        return output_file
    
    def _print_summary_table(self, results: Dict[str, any]):
        """Print formatted summary table."""
        print("\n📊 BERT-768 Optimization Benchmark Summary")
        print("=" * 80)
        
        # Scaling results table
        if "scaling_results" in results and results["scaling_results"]["scaling_results"]:
            scaling_data = []
            for result in results["scaling_results"]["scaling_results"]:
                if result.get("success", True):
                    scaling_data.append([
                        result["dataset_size"],
                        f"{result['compression_ratio']:.1%}",
                        f"{result['quality_metrics']['cosine_similarity']:.3f}",
                        f"{result['timing']['total']:.2f}s",
                        f"{result['throughput']['encoding_vectors_per_sec']:.0f}/s",
                        f"{result['memory']['memory_savings_mb']:.1f} MB"
                    ])
            
            if scaling_data:
                print("\n🔢 Scaling Performance:")
                print(f"{'Dataset Size':<12} {'Compression':<12} {'Quality':<10} {'Total Time':<12} {'Encoding Rate':<15} {'Memory Saved'}")
                print("-" * 80)
                for row in scaling_data:
                    print(f"{row[0]:<12} {row[1]:<12} {row[2]:<10} {row[3]:<12} {row[4]:<15} {row[5]}")
        
        # Parameter sensitivity summary
        if "sensitivity_results" in results:
            print(f"\n🎛️  Parameter Sensitivity Analysis:")
            for param_name, param_results in results["sensitivity_results"]["sensitivity_results"].items():
                successful_results = [r for r in param_results if r.get("success", True)]
                if successful_results:
                    best_result = max(successful_results, key=lambda x: x["quality_metrics"]["cosine_similarity"])
                    print(f"  {param_name}: Best value = {best_result['param_value']} "
                          f"(Quality: {best_result['quality_metrics']['cosine_similarity']:.3f})")
        
        # Real-world performance
        if "real_world" in results and results["real_world"].get("success", True):
            rw = results["real_world"]
            print(f"\n🚀 Real-world Performance:")
            print(f"  Database Size: {rw['database_size']} vectors")
            print(f"  Insertion Rate: {rw['insertion']['vectors_per_second']:.0f} vectors/sec")
            print(f"  Search Latency: {rw['search']['mean_time_ms']:.2f}ms (mean)")
            print(f"  Query Throughput: {rw['search']['queries_per_second']:.0f} QPS")
            print(f"  Storage Savings: {rw['database_stats']['compression_ratio']:.1%}")


def main():
    """Main function for BERT benchmarking."""
    parser = argparse.ArgumentParser(description="BERT-768 benchmarking for SpiralDeltaDB")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--max-vectors", type=int, default=50000, help="Maximum vectors to load")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    parser.add_argument("--sensitivity", action="store_true", help="Run parameter sensitivity analysis")
    parser.add_argument("--real-world", action="store_true", help="Run real-world performance benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--output", default="bert_benchmark_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    benchmark = BERTBenchmark(args.data_dir)
    
    # Load BERT data
    vectors, metadata = benchmark.load_bert_data(args.max_vectors)
    
    results = {}
    
    if args.scaling or args.all:
        print("🔢 Running Scaling Benchmark...")
        results["scaling_results"] = benchmark.benchmark_compression_scaling(vectors)
    
    if args.sensitivity or args.all:
        print("🎛️  Running Parameter Sensitivity Analysis...")
        results["sensitivity_results"] = benchmark.benchmark_parameter_sensitivity(vectors)
    
    if args.real_world or args.all:
        print("🚀 Running Real-world Performance Benchmark...")
        results["real_world"] = benchmark.benchmark_real_world_performance(vectors)
    
    if not results:
        print("No benchmarks selected. Use --all or specific benchmark flags.")
        return
    
    # Generate report
    benchmark.generate_benchmark_report(results, args.output)


if __name__ == "__main__":
    main()