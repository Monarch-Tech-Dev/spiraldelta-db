#!/usr/bin/env python3
"""
Unified dataset manager for SpiralDeltaDB benchmarking.

This script provides a unified interface for loading real GloVe datasets
or generating synthetic alternatives when real data is unavailable.
"""

import numpy as np
import sys
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Tuple, Optional
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our existing modules
import importlib.util
glove_spec = importlib.util.spec_from_file_location("download_glove", 
                                                   Path(__file__).parent / "download_glove.py")
download_glove = importlib.util.module_from_spec(glove_spec)
glove_spec.loader.exec_module(download_glove)

synthetic_spec = importlib.util.spec_from_file_location("create_synthetic_glove", 
                                                       Path(__file__).parent / "create_synthetic_glove.py")
create_synthetic_glove = importlib.util.module_from_spec(synthetic_spec)
synthetic_spec.loader.exec_module(create_synthetic_glove)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetManager:
    """Unified manager for real and synthetic GloVe datasets."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory for dataset storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.glove_downloader = download_glove.GloVeDownloader(data_dir)
        self.synthetic_generator = create_synthetic_glove.SyntheticGloVeGenerator(data_dir)
    
    def get_glove_300_dataset(
        self, 
        max_vectors: Optional[int] = None,
        prefer_real: bool = True,
        force_synthetic: bool = False
    ) -> Tuple[np.ndarray, List[str], Dict[str, any]]:
        """
        Get GloVe-300 dataset, trying real first then synthetic fallback.
        
        Args:
            max_vectors: Maximum vectors to return (None for all)
            prefer_real: Try to download real GloVe first
            force_synthetic: Force use of synthetic data
            
        Returns:
            Tuple of (vectors, words, metadata)
        """
        metadata = {"source": "unknown", "total_vectors": 0, "dimensions": 300}
        
        if force_synthetic:
            logger.info("🔬 Using synthetic GloVe dataset (forced)")
            return self._get_synthetic_dataset(max_vectors, metadata)
        
        if prefer_real:
            try:
                logger.info("🌐 Attempting to load real GloVe dataset...")
                return self._get_real_dataset(max_vectors, metadata)
            except Exception as e:
                logger.warning(f"Real GloVe dataset unavailable: {e}")
                logger.info("🔬 Falling back to synthetic dataset")
                return self._get_synthetic_dataset(max_vectors, metadata)
        else:
            logger.info("🔬 Using synthetic GloVe dataset")
            return self._get_synthetic_dataset(max_vectors, metadata)
    
    def _get_real_dataset(
        self, 
        max_vectors: Optional[int], 
        metadata: Dict[str, any]
    ) -> Tuple[np.ndarray, List[str], Dict[str, any]]:
        """Load real GloVe dataset."""
        
        # Try to download if needed
        try:
            self.glove_downloader.download_if_needed()
        except Exception as e:
            raise RuntimeError(f"Failed to download GloVe dataset: {e}")
        
        # Load the processed data
        try:
            vectors, words = self.glove_downloader.load_data(max_vectors)
            
            metadata.update({
                "source": "real_glove",
                "total_vectors": len(vectors),
                "url": self.glove_downloader.GLOVE_URL,
                "processing_time": time.time()
            })
            
            logger.info(f"✅ Loaded real GloVe dataset: {len(vectors)} vectors")
            return vectors, words, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load GloVe data: {e}")
    
    def _get_synthetic_dataset(
        self, 
        max_vectors: Optional[int], 
        metadata: Dict[str, any]
    ) -> Tuple[np.ndarray, List[str], Dict[str, any]]:
        """Generate or load synthetic dataset."""
        
        n_vectors = max_vectors if max_vectors else 50000
        
        try:
            # Try to load existing synthetic dataset
            vectors, words = self.synthetic_generator.load_dataset("synthetic_glove_300d")
            
            if max_vectors and max_vectors < len(vectors):
                indices = np.random.choice(len(vectors), max_vectors, replace=False)
                vectors = vectors[indices]
                words = [words[i] for i in indices]
            
            logger.info(f"✅ Loaded existing synthetic dataset: {len(vectors)} vectors")
            
        except FileNotFoundError:
            # Generate new synthetic dataset
            logger.info(f"🔬 Generating new synthetic GloVe dataset: {n_vectors} vectors")
            vectors, words = self.synthetic_generator.generate_realistic_embeddings(
                n_vectors=n_vectors, dimensions=300
            )
            
            # Save for future use
            self.synthetic_generator.save_dataset(vectors, words, "synthetic_glove_300d")
            logger.info("💾 Saved synthetic dataset for future use")
        
        metadata.update({
            "source": "synthetic",
            "total_vectors": len(vectors),
            "generation_method": "semantic_clustering",
            "processing_time": time.time()
        })
        
        return vectors, words, metadata
    
    def get_dataset_info(self) -> Dict[str, any]:
        """Get information about available datasets."""
        info = {
            "real_glove_available": False,
            "synthetic_glove_available": False,
            "data_directory": str(self.data_dir),
            "datasets": []
        }
        
        # Check for real GloVe
        if self.glove_downloader.vectors_path.exists():
            info["real_glove_available"] = True
            try:
                vectors, words = self.glove_downloader.load_data(max_vectors=1)
                info["datasets"].append({
                    "name": "real_glove_300d",
                    "source": "stanford_glove",
                    "vectors": len(words),  # Approximate
                    "dimensions": 300,
                    "file_size_mb": self.glove_downloader.vectors_path.stat().st_size / (1024**2)
                })
            except Exception:
                pass
        
        # Check for synthetic GloVe
        synthetic_vectors_path = self.data_dir / "synthetic_glove_300d_vectors.npy"
        if synthetic_vectors_path.exists():
            info["synthetic_glove_available"] = True
            try:
                vectors, words = self.synthetic_generator.load_dataset("synthetic_glove_300d")
                info["datasets"].append({
                    "name": "synthetic_glove_300d", 
                    "source": "generated",
                    "vectors": len(vectors),
                    "dimensions": vectors.shape[1],
                    "file_size_mb": synthetic_vectors_path.stat().st_size / (1024**2)
                })
            except Exception:
                pass
        
        return info
    
    def benchmark_dataset_loading(self) -> Dict[str, any]:
        """Benchmark dataset loading performance."""
        results = {"real": None, "synthetic": None}
        
        # Test synthetic (faster)
        try:
            start_time = time.time()
            vectors, words, metadata = self.get_glove_300_dataset(
                max_vectors=10000, 
                force_synthetic=True
            )
            load_time = time.time() - start_time
            
            results["synthetic"] = {
                "load_time_seconds": load_time,
                "vectors_loaded": len(vectors),
                "vectors_per_second": len(vectors) / load_time,
                "memory_mb": vectors.nbytes / (1024**2)
            }
        except Exception as e:
            results["synthetic"] = {"error": str(e)}
        
        # Test real (if available)
        try:
            start_time = time.time()
            vectors, words, metadata = self.get_glove_300_dataset(
                max_vectors=10000, 
                prefer_real=True,
                force_synthetic=False
            )
            load_time = time.time() - start_time
            
            if metadata["source"] == "real_glove":
                results["real"] = {
                    "load_time_seconds": load_time,
                    "vectors_loaded": len(vectors),
                    "vectors_per_second": len(vectors) / load_time,
                    "memory_mb": vectors.nbytes / (1024**2)
                }
        except Exception as e:
            results["real"] = {"error": str(e)}
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Unified dataset manager for SpiralDeltaDB")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--info", action="store_true", help="Show dataset information")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark loading performance")
    parser.add_argument("--max-vectors", type=int, help="Maximum vectors to load")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset")
    parser.add_argument("--test-load", action="store_true", help="Test loading dataset")
    
    args = parser.parse_args()
    
    manager = DatasetManager(args.data_dir)
    
    if args.info:
        print("📊 Dataset Information:")
        print("=" * 50)
        info = manager.get_dataset_info()
        print(json.dumps(info, indent=2))
    
    elif args.benchmark:
        print("⚡ Dataset Loading Benchmark:")
        print("=" * 50)
        results = manager.benchmark_dataset_loading()
        print(json.dumps(results, indent=2))
    
    elif args.test_load:
        print("🧪 Testing Dataset Loading:")
        print("=" * 50)
        try:
            vectors, words, metadata = manager.get_glove_300_dataset(
                max_vectors=args.max_vectors,
                force_synthetic=args.synthetic
            )
            
            print(f"✅ Successfully loaded dataset:")
            print(f"  Source: {metadata['source']}")
            print(f"  Vectors: {len(vectors)}")
            print(f"  Dimensions: {vectors.shape[1]}")
            print(f"  Words sample: {words[:5]}")
            print(f"  Memory usage: {vectors.nbytes / (1024**2):.1f} MB")
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
    
    else:
        print("Use --help for usage information")


if __name__ == "__main__":
    main()