"""
GPU Acceleration Interface for SpiralDeltaDB

This module provides Python bindings for GPU-accelerated operations,
including vector similarity search, index construction, and spiral ordering.
"""

import os
import sys
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Try to import GPU acceleration module
try:
    import spiraldelta_gpu
    GPU_AVAILABLE = True
    logger.info("GPU acceleration module loaded successfully")
except ImportError as e:
    GPU_AVAILABLE = False
    logger.warning(f"GPU acceleration not available: {e}")
    # Create mock module for fallback
    class MockGpuModule:
        class GpuAccelerator:
            @staticmethod
            def is_available():
                return False
            
            def __init__(self, *args, **kwargs):
                raise RuntimeError("GPU acceleration not available")
        
        @staticmethod
        def check_cuda_availability():
            return False
            
        @staticmethod
        def get_gpu_memory_info():
            return {"total_gb": 0, "available_gb": 0}
    
    spiraldelta_gpu = MockGpuModule()

@dataclass
class GpuConfig:
    """Configuration for GPU acceleration"""
    device_id: int = 0
    max_batch_size: int = 10000
    memory_limit_gb: float = 8.0
    enable_mixed_precision: bool = True
    enable_tensor_cores: bool = True
    stream_count: int = 4

class GpuAccelerationEngine:
    """
    High-level interface for GPU-accelerated SpiralDelta operations.
    
    This class provides a unified interface for GPU acceleration with automatic
    fallback to CPU implementations when GPU is not available.
    """
    
    def __init__(self, config: Optional[GpuConfig] = None, fallback_enabled: bool = True):
        """
        Initialize GPU acceleration engine.
        
        Args:
            config: GPU configuration parameters
            fallback_enabled: Whether to fallback to CPU when GPU unavailable
        """
        self.config = config or GpuConfig()
        self.fallback_enabled = fallback_enabled
        self.gpu_accelerator = None
        self.gpu_available = False
        
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU acceleration if available"""
        if not GPU_AVAILABLE:
            if not self.fallback_enabled:
                raise RuntimeError("GPU acceleration required but not available")
            logger.info("GPU acceleration not available, will use CPU fallback")
            return
        
        try:
            if spiraldelta_gpu.check_cuda_availability():
                self.gpu_accelerator = spiraldelta_gpu.GpuAccelerator(self.config.__dict__)
                self.gpu_available = True
                logger.info("GPU acceleration initialized successfully")
                
                # Log GPU information
                gpu_info = self.gpu_accelerator.get_device_info()
                logger.info(f"GPU Device: {gpu_info.get('device_id')}")
                logger.info(f"GPU Memory: {gpu_info.get('total_memory_gb', 0):.2f} GB")
            else:
                if not self.fallback_enabled:
                    raise RuntimeError("CUDA not available")
                logger.warning("CUDA not available, using CPU fallback")
        except Exception as e:
            if not self.fallback_enabled:
                raise RuntimeError(f"Failed to initialize GPU acceleration: {e}")
            logger.warning(f"Failed to initialize GPU acceleration: {e}, using CPU fallback")
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available"""
        return self.gpu_available
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information"""
        if self.gpu_available:
            return self.gpu_accelerator.get_device_info()
        else:
            return {
                "device_type": "CPU",
                "gpu_available": False,
                "fallback_mode": True
            }
    
    def similarity_search(
        self,
        query_vectors: np.ndarray,
        index_vectors: np.ndarray,
        k: int = 10,
        metric: str = "cosine",
        batch_size: Optional[int] = None
    ) -> List[List[Tuple[int, float]]]:
        """
        Perform similarity search with GPU acceleration.
        
        Args:
            query_vectors: Query vectors as numpy array (N x D)
            index_vectors: Index vectors as numpy array (M x D)
            k: Number of nearest neighbors to return
            metric: Similarity metric ("cosine", "euclidean", "dot", "manhattan")
            batch_size: Batch size for processing (None for auto)
            
        Returns:
            List of [(index, score)] for each query vector
        """
        start_time = time.time()
        
        # Validate inputs
        if query_vectors.ndim != 2 or index_vectors.ndim != 2:
            raise ValueError("Query and index vectors must be 2D arrays")
        
        if query_vectors.shape[1] != index_vectors.shape[1]:
            raise ValueError("Query and index vectors must have same dimension")
        
        # Convert to appropriate format
        query_list = query_vectors.tolist()
        index_list = index_vectors.tolist()
        
        if self.gpu_available:
            try:
                # Use GPU acceleration
                results = self.gpu_accelerator.similarity_search(
                    query_list, index_list, k, metric
                )
                
                elapsed = time.time() - start_time
                logger.debug(f"GPU similarity search completed in {elapsed:.3f}s")
                return results
                
            except Exception as e:
                if not self.fallback_enabled:
                    raise RuntimeError(f"GPU similarity search failed: {e}")
                logger.warning(f"GPU search failed: {e}, falling back to CPU")
        
        # CPU fallback
        return self._cpu_similarity_search(query_vectors, index_vectors, k, metric)
    
    def build_index(
        self,
        vectors: np.ndarray,
        index_type: str = "spiral",
        **kwargs
    ) -> bytes:
        """
        Build optimized index with GPU acceleration.
        
        Args:
            vectors: Input vectors as numpy array (N x D)
            index_type: Type of index ("spiral", "hnsw", "ivf", "lsh")
            **kwargs: Additional parameters for index construction
            
        Returns:
            Serialized index data
        """
        start_time = time.time()
        
        if vectors.ndim != 2:
            raise ValueError("Vectors must be 2D array")
        
        vector_list = vectors.tolist()
        
        if self.gpu_available:
            try:
                # Use GPU acceleration
                index_data = self.gpu_accelerator.build_index(vector_list, index_type)
                
                elapsed = time.time() - start_time
                logger.info(f"GPU index construction completed in {elapsed:.3f}s")
                return index_data
                
            except Exception as e:
                if not self.fallback_enabled:
                    raise RuntimeError(f"GPU index construction failed: {e}")
                logger.warning(f"GPU index construction failed: {e}, falling back to CPU")
        
        # CPU fallback
        return self._cpu_build_index(vectors, index_type, **kwargs)
    
    def batch_operations(
        self,
        vectors: np.ndarray,
        operation: str,
        **params
    ) -> np.ndarray:
        """
        Perform batch vector operations with GPU acceleration.
        
        Args:
            vectors: Input vectors as numpy array (N x D)
            operation: Operation type ("normalize", "pca_reduce", "quantize")
            **params: Operation-specific parameters
            
        Returns:
            Processed vectors as numpy array
        """
        start_time = time.time()
        
        if vectors.ndim != 2:
            raise ValueError("Vectors must be 2D array")
        
        vector_list = vectors.tolist()
        
        if self.gpu_available:
            try:
                # Use GPU acceleration
                results = self.gpu_accelerator.batch_operations(
                    vector_list, operation, params
                )
                
                elapsed = time.time() - start_time
                logger.debug(f"GPU batch operations completed in {elapsed:.3f}s")
                return np.array(results, dtype=np.float32)
                
            except Exception as e:
                if not self.fallback_enabled:
                    raise RuntimeError(f"GPU batch operations failed: {e}")
                logger.warning(f"GPU batch operations failed: {e}, falling back to CPU")
        
        # CPU fallback
        return self._cpu_batch_operations(vectors, operation, **params)
    
    def optimize_memory(self):
        """Optimize GPU memory usage"""
        if self.gpu_available:
            try:
                self.gpu_accelerator.optimize_memory()
                logger.info("GPU memory optimized")
            except Exception as e:
                logger.warning(f"GPU memory optimization failed: {e}")
    
    def benchmark_performance(
        self,
        vector_sizes: List[int] = [1000, 5000, 10000],
        dimensions: List[int] = [128, 256, 512, 768],
        k_values: List[int] = [1, 10, 100],
    ) -> Dict[str, Any]:
        """
        Benchmark GPU vs CPU performance across different scenarios.
        
        Args:
            vector_sizes: List of vector counts to test
            dimensions: List of dimensions to test
            k_values: List of k values to test
            
        Returns:
            Performance benchmark results
        """
        results = {
            "gpu_available": self.gpu_available,
            "benchmarks": []
        }
        
        for n_vectors in vector_sizes:
            for dim in dimensions:
                for k in k_values:
                    # Generate test data
                    query_vectors = np.random.randn(100, dim).astype(np.float32)
                    index_vectors = np.random.randn(n_vectors, dim).astype(np.float32)
                    
                    benchmark = {
                        "n_vectors": n_vectors,
                        "dimension": dim,
                        "k": k,
                        "gpu_time": None,
                        "cpu_time": None,
                        "speedup": None
                    }
                    
                    # GPU benchmark
                    if self.gpu_available:
                        try:
                            start_time = time.time()
                            _ = self.similarity_search(query_vectors, index_vectors, k)
                            gpu_time = time.time() - start_time
                            benchmark["gpu_time"] = gpu_time
                        except Exception as e:
                            logger.warning(f"GPU benchmark failed: {e}")
                    
                    # CPU benchmark
                    try:
                        start_time = time.time()
                        _ = self._cpu_similarity_search(query_vectors, index_vectors, k, "cosine")
                        cpu_time = time.time() - start_time
                        benchmark["cpu_time"] = cpu_time
                    except Exception as e:
                        logger.warning(f"CPU benchmark failed: {e}")
                    
                    # Calculate speedup
                    if benchmark["gpu_time"] and benchmark["cpu_time"]:
                        benchmark["speedup"] = benchmark["cpu_time"] / benchmark["gpu_time"]
                    
                    results["benchmarks"].append(benchmark)
                    
                    logger.info(f"Benchmark: {n_vectors} vectors, {dim}D, k={k} - "
                              f"GPU: {benchmark['gpu_time']:.3f}s, CPU: {benchmark['cpu_time']:.3f}s, "
                              f"Speedup: {benchmark.get('speedup', 'N/A'):.1f}x")
        
        return results
    
    # CPU fallback implementations
    
    def _cpu_similarity_search(
        self,
        query_vectors: np.ndarray,
        index_vectors: np.ndarray,
        k: int,
        metric: str
    ) -> List[List[Tuple[int, float]]]:
        """CPU fallback for similarity search"""
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        if metric == "cosine":
            similarities = cosine_similarity(query_vectors, index_vectors)
            # Higher is better for cosine similarity
            top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        elif metric == "euclidean":
            distances = euclidean_distances(query_vectors, index_vectors)
            # Lower is better for euclidean distance
            top_k_indices = np.argsort(distances, axis=1)[:, :k]
            similarities = -distances  # Convert to similarity-like scores
        elif metric == "dot":
            similarities = np.dot(query_vectors, index_vectors.T)
            top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        results = []
        for i, indices in enumerate(top_k_indices):
            if metric == "euclidean":
                scores = similarities[i, indices]
            else:
                scores = similarities[i, indices]
            
            query_results = [(int(idx), float(score)) for idx, score in zip(indices, scores)]
            results.append(query_results)
        
        return results
    
    def _cpu_build_index(self, vectors: np.ndarray, index_type: str, **kwargs) -> bytes:
        """CPU fallback for index construction"""
        import pickle
        
        if index_type == "spiral":
            # Simple spiral ordering based on vector norms
            norms = np.linalg.norm(vectors, axis=1)
            order = np.argsort(norms)
            index_data = {"type": "spiral", "order": order, "vectors": vectors}
        elif index_type == "hnsw":
            # Placeholder HNSW implementation
            index_data = {"type": "hnsw", "vectors": vectors}
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        return pickle.dumps(index_data)
    
    def _cpu_batch_operations(self, vectors: np.ndarray, operation: str, **params) -> np.ndarray:
        """CPU fallback for batch operations"""
        if operation == "normalize":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / (norms + 1e-8)
        elif operation == "pca_reduce":
            from sklearn.decomposition import PCA
            target_dim = params.get("target_dim", 128)
            pca = PCA(n_components=target_dim)
            return pca.fit_transform(vectors).astype(np.float32)
        elif operation == "quantize":
            bits = params.get("bits", 8)
            scale = (2 ** bits - 1)
            normalized = (vectors - vectors.min()) / (vectors.max() - vectors.min())
            quantized = np.round(normalized * scale) / scale
            return quantized.astype(np.float32)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

# Global GPU acceleration instance
_gpu_engine = None

def get_gpu_engine(config: Optional[GpuConfig] = None, fallback_enabled: bool = True) -> GpuAccelerationEngine:
    """Get global GPU acceleration engine instance"""
    global _gpu_engine
    if _gpu_engine is None:
        _gpu_engine = GpuAccelerationEngine(config, fallback_enabled)
    return _gpu_engine

def check_gpu_availability() -> bool:
    """Check if GPU acceleration is available"""
    return GPU_AVAILABLE and spiraldelta_gpu.check_cuda_availability()

def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information"""
    if GPU_AVAILABLE:
        return spiraldelta_gpu.get_gpu_memory_info()
    else:
        return {"total_gb": 0, "available_gb": 0}

# Convenience functions for common operations

def gpu_similarity_search(
    query_vectors: np.ndarray,
    index_vectors: np.ndarray,
    k: int = 10,
    metric: str = "cosine"
) -> List[List[Tuple[int, float]]]:
    """Convenience function for GPU-accelerated similarity search"""
    engine = get_gpu_engine()
    return engine.similarity_search(query_vectors, index_vectors, k, metric)

def gpu_build_index(vectors: np.ndarray, index_type: str = "spiral", **kwargs) -> bytes:
    """Convenience function for GPU-accelerated index construction"""
    engine = get_gpu_engine()
    return engine.build_index(vectors, index_type, **kwargs)

def gpu_normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Convenience function for GPU-accelerated vector normalization"""
    engine = get_gpu_engine()
    return engine.batch_operations(vectors, "normalize")