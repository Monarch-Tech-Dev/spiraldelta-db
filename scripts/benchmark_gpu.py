#!/usr/bin/env python3
"""
GPU Performance Benchmark for SpiralDelta

Comprehensive benchmarking of GPU vs CPU performance across different scenarios.
"""

import numpy as np
import time
import json
import argparse
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from spiraldelta.gpu_acceleration import GpuAccelerationEngine, GpuConfig, check_gpu_availability
    GPU_MODULE_AVAILABLE = True
except ImportError:
    GPU_MODULE_AVAILABLE = False
    print("Warning: GPU acceleration module not available")

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    scenario: str
    n_vectors: int
    dimension: int
    k: int
    gpu_time: float
    cpu_time: float
    speedup: float
    gpu_memory_gb: float
    accuracy_match: bool

class PerformanceBenchmark:
    """Comprehensive GPU performance benchmarking"""
    
    def __init__(self, config: GpuConfig = None):
        self.config = config or GpuConfig()
        self.gpu_engine = None
        self.results: List[BenchmarkResult] = []
        
        if GPU_MODULE_AVAILABLE:
            self.gpu_engine = GpuAccelerationEngine(self.config)
            print(f"GPU Available: {self.gpu_engine.is_gpu_available()}")
            if self.gpu_engine.is_gpu_available():
                device_info = self.gpu_engine.get_device_info()
                print(f"GPU Device: {device_info}")
        else:
            print("GPU acceleration not available - CPU-only benchmarks")
    
    def run_similarity_search_benchmark(
        self,
        vector_sizes: List[int] = [1000, 5000, 10000, 50000],
        dimensions: List[int] = [128, 256, 512, 768, 1024],
        k_values: List[int] = [1, 10, 100],
        num_queries: int = 100
    ):
        """Benchmark similarity search performance"""
        print("🔍 Running similarity search benchmarks...")
        
        for n_vectors in vector_sizes:
            for dim in dimensions:
                for k in k_values:
                    print(f"  Testing: {n_vectors} vectors, {dim}D, k={k}")
                    
                    # Generate test data
                    np.random.seed(42)  # For reproducible results
                    queries = np.random.randn(num_queries, dim).astype(np.float32)
                    index = np.random.randn(n_vectors, dim).astype(np.float32)
                    
                    # Normalize vectors for fair comparison
                    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
                    index = index / np.linalg.norm(index, axis=1, keepdims=True)
                    
                    result = self._benchmark_single_scenario(
                        "similarity_search", queries, index, k
                    )
                    
                    if result:
                        self.results.append(result)
    
    def run_index_construction_benchmark(
        self,
        vector_sizes: List[int] = [1000, 5000, 10000],
        dimensions: List[int] = [128, 256, 512],
        index_types: List[str] = ["spiral", "hnsw"]
    ):
        """Benchmark index construction performance"""
        print("🏗️ Running index construction benchmarks...")
        
        for n_vectors in vector_sizes:
            for dim in dimensions:
                for index_type in index_types:
                    print(f"  Testing: {n_vectors} vectors, {dim}D, {index_type}")
                    
                    # Generate test data
                    np.random.seed(42)
                    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
                    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
                    
                    result = self._benchmark_index_construction(
                        vectors, index_type
                    )
                    
                    if result:
                        self.results.append(result)
    
    def run_batch_operations_benchmark(
        self,
        vector_sizes: List[int] = [1000, 10000, 100000],
        dimensions: List[int] = [128, 512, 1024],
        operations: List[str] = ["normalize", "pca_reduce", "quantize"]
    ):
        """Benchmark batch operations performance"""
        print("⚡ Running batch operations benchmarks...")
        
        for n_vectors in vector_sizes:
            for dim in dimensions:
                for operation in operations:
                    print(f"  Testing: {n_vectors} vectors, {dim}D, {operation}")
                    
                    # Generate test data
                    np.random.seed(42)
                    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
                    
                    result = self._benchmark_batch_operation(
                        vectors, operation
                    )
                    
                    if result:
                        self.results.append(result)
    
    def _benchmark_single_scenario(
        self,
        scenario: str,
        queries: np.ndarray,
        index: np.ndarray,
        k: int
    ) -> BenchmarkResult:
        """Benchmark a single scenario"""
        gpu_time = None
        cpu_time = None
        gpu_memory = 0.0
        accuracy_match = True
        
        # GPU benchmark
        if self.gpu_engine and self.gpu_engine.is_gpu_available():
            try:
                # Warmup
                _ = self.gpu_engine.similarity_search(queries[:10], index[:100], k)
                
                # Actual benchmark
                start_time = time.time()
                gpu_results = self.gpu_engine.similarity_search(queries, index, k)
                gpu_time = time.time() - start_time
                
                # Get GPU memory usage
                device_info = self.gpu_engine.get_device_info()
                gpu_memory = device_info.get('total_memory_gb', 0)
                
            except Exception as e:
                print(f"    GPU benchmark failed: {e}")
                gpu_results = None
        
        # CPU benchmark
        try:
            # Warmup
            if self.gpu_engine:
                _ = self.gpu_engine._cpu_similarity_search(queries[:10], index[:100], k, "cosine")
            
            # Actual benchmark
            start_time = time.time()
            if self.gpu_engine:
                cpu_results = self.gpu_engine._cpu_similarity_search(queries, index, k, "cosine")
            else:
                cpu_results = self._fallback_cpu_search(queries, index, k)
            cpu_time = time.time() - start_time
            
        except Exception as e:
            print(f"    CPU benchmark failed: {e}")
            return None
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if (gpu_time and cpu_time) else 0.0
        
        # Check accuracy (simplified)
        if gpu_time and cpu_time:
            try:
                # Compare first few results for accuracy
                gpu_indices = [result[0][0] for result in gpu_results[:5]]
                cpu_indices = [result[0][0] for result in cpu_results[:5]]
                accuracy_match = len(set(gpu_indices) & set(cpu_indices)) >= len(gpu_indices) * 0.8
            except:
                accuracy_match = False
        
        return BenchmarkResult(
            scenario=scenario,
            n_vectors=index.shape[0],
            dimension=index.shape[1],
            k=k,
            gpu_time=gpu_time or 0.0,
            cpu_time=cpu_time or 0.0,
            speedup=speedup,
            gpu_memory_gb=gpu_memory,
            accuracy_match=accuracy_match
        )
    
    def _benchmark_index_construction(
        self,
        vectors: np.ndarray,
        index_type: str
    ) -> BenchmarkResult:
        """Benchmark index construction"""
        gpu_time = None
        cpu_time = None
        gpu_memory = 0.0
        
        # GPU benchmark
        if self.gpu_engine and self.gpu_engine.is_gpu_available():
            try:
                start_time = time.time()
                gpu_index = self.gpu_engine.build_index(vectors, index_type)
                gpu_time = time.time() - start_time
                
                device_info = self.gpu_engine.get_device_info()
                gpu_memory = device_info.get('total_memory_gb', 0)
                
            except Exception as e:
                print(f"    GPU index construction failed: {e}")
        
        # CPU benchmark
        try:
            start_time = time.time()
            if self.gpu_engine:
                cpu_index = self.gpu_engine._cpu_build_index(vectors, index_type)
            else:
                cpu_index = self._fallback_cpu_index(vectors, index_type)
            cpu_time = time.time() - start_time
            
        except Exception as e:
            print(f"    CPU index construction failed: {e}")
            return None
        
        speedup = cpu_time / gpu_time if (gpu_time and cpu_time) else 0.0
        
        return BenchmarkResult(
            scenario=f"index_{index_type}",
            n_vectors=vectors.shape[0],
            dimension=vectors.shape[1],
            k=0,
            gpu_time=gpu_time or 0.0,
            cpu_time=cpu_time or 0.0,
            speedup=speedup,
            gpu_memory_gb=gpu_memory,
            accuracy_match=True
        )
    
    def _benchmark_batch_operation(
        self,
        vectors: np.ndarray,
        operation: str
    ) -> BenchmarkResult:
        """Benchmark batch operations"""
        gpu_time = None
        cpu_time = None
        gpu_memory = 0.0
        
        # GPU benchmark
        if self.gpu_engine and self.gpu_engine.is_gpu_available():
            try:
                start_time = time.time()
                gpu_result = self.gpu_engine.batch_operations(vectors, operation)
                gpu_time = time.time() - start_time
                
                device_info = self.gpu_engine.get_device_info()
                gpu_memory = device_info.get('total_memory_gb', 0)
                
            except Exception as e:
                print(f"    GPU batch operation failed: {e}")
        
        # CPU benchmark
        try:
            start_time = time.time()
            if self.gpu_engine:
                cpu_result = self.gpu_engine._cpu_batch_operations(vectors, operation)
            else:
                cpu_result = self._fallback_cpu_batch_op(vectors, operation)
            cpu_time = time.time() - start_time
            
        except Exception as e:
            print(f"    CPU batch operation failed: {e}")
            return None
        
        speedup = cpu_time / gpu_time if (gpu_time and cpu_time) else 0.0
        
        return BenchmarkResult(
            scenario=f"batch_{operation}",
            n_vectors=vectors.shape[0],
            dimension=vectors.shape[1],
            k=0,
            gpu_time=gpu_time or 0.0,
            cpu_time=cpu_time or 0.0,
            speedup=speedup,
            gpu_memory_gb=gpu_memory,
            accuracy_match=True
        )
    
    def _fallback_cpu_search(self, queries, index, k):
        """Fallback CPU search when GPU engine not available"""
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(queries, index)
        top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        
        results = []
        for i, indices in enumerate(top_k_indices):
            scores = similarities[i, indices]
            query_results = [(int(idx), float(score)) for idx, score in zip(indices, scores)]
            results.append(query_results)
        return results
    
    def _fallback_cpu_index(self, vectors, index_type):
        """Fallback CPU index construction"""
        import pickle
        if index_type == "spiral":
            norms = np.linalg.norm(vectors, axis=1)
            order = np.argsort(norms)
            return pickle.dumps({"type": "spiral", "order": order})
        else:
            return pickle.dumps({"type": index_type, "vectors": vectors})
    
    def _fallback_cpu_batch_op(self, vectors, operation):
        """Fallback CPU batch operations"""
        if operation == "normalize":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / (norms + 1e-8)
        elif operation == "pca_reduce":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(128, vectors.shape[1]))
            return pca.fit_transform(vectors).astype(np.float32)
        else:
            return vectors
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Aggregate results by scenario
        scenarios = {}
        for result in self.results:
            scenario = result.scenario
            if scenario not in scenarios:
                scenarios[scenario] = []
            scenarios[scenario].append(result)
        
        # Calculate statistics
        report = {
            "total_benchmarks": len(self.results),
            "gpu_available": self.gpu_engine.is_gpu_available() if self.gpu_engine else False,
            "scenarios": {},
            "summary": {}
        }
        
        all_speedups = []
        
        for scenario, results in scenarios.items():
            speedups = [r.speedup for r in results if r.speedup > 0]
            gpu_times = [r.gpu_time for r in results if r.gpu_time > 0]
            cpu_times = [r.cpu_time for r in results if r.cpu_time > 0]
            
            report["scenarios"][scenario] = {
                "count": len(results),
                "avg_speedup": np.mean(speedups) if speedups else 0,
                "max_speedup": np.max(speedups) if speedups else 0,
                "avg_gpu_time": np.mean(gpu_times) if gpu_times else 0,
                "avg_cpu_time": np.mean(cpu_times) if cpu_times else 0,
                "accuracy_rate": np.mean([r.accuracy_match for r in results])
            }
            
            all_speedups.extend(speedups)
        
        # Overall summary
        if all_speedups:
            report["summary"] = {
                "overall_avg_speedup": np.mean(all_speedups),
                "overall_max_speedup": np.max(all_speedups),
                "scenarios_with_speedup": len([s for s in all_speedups if s > 1.0]),
                "total_scenarios": len(all_speedups)
            }
        
        return report
    
    def save_results(self, filename: str):
        """Save results to JSON file"""
        data = {
            "config": asdict(self.config),
            "results": [asdict(result) for result in self.results],
            "report": self.generate_report()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Results saved to {filename}")
    
    def create_visualizations(self, output_dir: str = "benchmark_plots"):
        """Create performance visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results:
            print("No results to visualize")
            return
        
        # Speedup by vector count
        plt.figure(figsize=(12, 8))
        
        # Group by scenario
        scenarios = {}
        for result in self.results:
            if result.speedup > 0:
                scenario = result.scenario
                if scenario not in scenarios:
                    scenarios[scenario] = {"n_vectors": [], "speedups": []}
                scenarios[scenario]["n_vectors"].append(result.n_vectors)
                scenarios[scenario]["speedups"].append(result.speedup)
        
        # Plot speedup vs vector count
        plt.subplot(2, 2, 1)
        for scenario, data in scenarios.items():
            plt.scatter(data["n_vectors"], data["speedups"], label=scenario, alpha=0.7)
        plt.xlabel("Number of Vectors")
        plt.ylabel("GPU Speedup (x)")
        plt.title("GPU Speedup vs Vector Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Speedup by dimension
        plt.subplot(2, 2, 2)
        for scenario, data in scenarios.items():
            dimensions = [r.dimension for r in self.results if r.scenario == scenario and r.speedup > 0]
            speedups = [r.speedup for r in self.results if r.scenario == scenario and r.speedup > 0]
            plt.scatter(dimensions, speedups, label=scenario, alpha=0.7)
        plt.xlabel("Vector Dimension")
        plt.ylabel("GPU Speedup (x)")
        plt.title("GPU Speedup vs Dimension")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Time comparison
        plt.subplot(2, 2, 3)
        gpu_times = [r.gpu_time for r in self.results if r.gpu_time > 0]
        cpu_times = [r.cpu_time for r in self.results if r.cpu_time > 0]
        plt.scatter(cpu_times, gpu_times, alpha=0.6)
        plt.plot([0, max(cpu_times)], [0, max(cpu_times)], 'r--', label='Equal time')
        plt.xlabel("CPU Time (s)")
        plt.ylabel("GPU Time (s)")
        plt.title("GPU vs CPU Execution Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Speedup distribution
        plt.subplot(2, 2, 4)
        all_speedups = [r.speedup for r in self.results if r.speedup > 0]
        plt.hist(all_speedups, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel("Speedup Factor")
        plt.ylabel("Frequency")
        plt.title("Distribution of GPU Speedups")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gpu_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualizations saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="GPU Performance Benchmark for SpiralDelta")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--full", action="store_true", help="Run comprehensive benchmark")
    parser.add_argument("--similarity-only", action="store_true", help="Only test similarity search")
    parser.add_argument("--output", default="gpu_benchmark_results.json", help="Output file")
    parser.add_argument("--plots", action="store_true", help="Generate visualization plots")
    
    args = parser.parse_args()
    
    print("🚀 SpiralDelta GPU Performance Benchmark")
    print("=" * 50)
    
    # Initialize benchmark
    config = GpuConfig(
        memory_limit_gb=6.0,
        max_batch_size=5000 if args.quick else 10000
    )
    
    benchmark = PerformanceBenchmark(config)
    
    # Run benchmarks based on arguments
    if args.quick:
        print("🏃 Running quick benchmark...")
        benchmark.run_similarity_search_benchmark(
            vector_sizes=[1000, 5000],
            dimensions=[128, 256],
            k_values=[10],
            num_queries=50
        )
    elif args.similarity_only:
        print("🔍 Running similarity search benchmark only...")
        benchmark.run_similarity_search_benchmark()
    elif args.full:
        print("🔥 Running comprehensive benchmark...")
        benchmark.run_similarity_search_benchmark()
        benchmark.run_index_construction_benchmark()
        benchmark.run_batch_operations_benchmark()
    else:
        print("🎯 Running standard benchmark...")
        benchmark.run_similarity_search_benchmark(
            vector_sizes=[1000, 5000, 10000],
            dimensions=[128, 256, 512],
            k_values=[1, 10, 100]
        )
    
    # Generate and display report
    report = benchmark.generate_report()
    print("\n📊 Benchmark Report")
    print("=" * 30)
    
    if "summary" in report:
        summary = report["summary"]
        print(f"Overall Average Speedup: {summary.get('overall_avg_speedup', 0):.1f}x")
        print(f"Maximum Speedup Achieved: {summary.get('overall_max_speedup', 0):.1f}x")
        print(f"Scenarios with Speedup: {summary.get('scenarios_with_speedup', 0)}/{summary.get('total_scenarios', 0)}")
    
    for scenario, stats in report.get("scenarios", {}).items():
        print(f"\n{scenario}:")
        print(f"  Average Speedup: {stats['avg_speedup']:.1f}x")
        print(f"  Average GPU Time: {stats['avg_gpu_time']:.3f}s")
        print(f"  Average CPU Time: {stats['avg_cpu_time']:.3f}s")
        print(f"  Accuracy Rate: {stats['accuracy_rate']:.1%}")
    
    # Save results
    benchmark.save_results(args.output)
    
    # Generate plots if requested
    if args.plots:
        benchmark.create_visualizations()
    
    print(f"\n✅ Benchmark completed! Results saved to {args.output}")

if __name__ == "__main__":
    main()