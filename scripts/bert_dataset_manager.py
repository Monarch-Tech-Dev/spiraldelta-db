#!/usr/bin/env python3
"""
BERT embeddings dataset manager for SpiralDeltaDB optimization.

This script downloads and manages BERT-768 embedding datasets for
compression benchmarking with target 66.8% compression ratio.
"""

import numpy as np
import sys
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Tuple, Optional
import argparse
import urllib.request
import gzip
# import h5py  # Optional - only needed for real datasets
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTDatasetManager:
    """Manager for BERT-768 embedding datasets."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize BERT dataset manager.
        
        Args:
            data_dir: Directory for dataset storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and paths
        self.bert_urls = {
            "wiki_bert_768": "https://dl.fbaipublicfiles.com/mcontribs/BERT/wiki_bert_768_1m.h5",
            "news_bert_768": "https://dl.fbaipublicfiles.com/mcontribs/BERT/news_bert_768_1m.h5"
        }
        
        self.dataset_paths = {
            name: self.data_dir / f"{name}.h5" 
            for name in self.bert_urls.keys()
        }
        
        # Synthetic dataset path
        self.synthetic_path = self.data_dir / "synthetic_bert_768_1m.npz"
        
    def generate_synthetic_bert_dataset(
        self, 
        n_vectors: int = 1000000,
        dimensions: int = 768,
        save: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate synthetic BERT-like embeddings with realistic properties.
        
        Args:
            n_vectors: Number of vectors to generate
            dimensions: Vector dimensionality (768 for BERT)
            save: Whether to save the dataset
            
        Returns:
            Tuple of (vectors, metadata)
        """
        logger.info(f"Generating synthetic BERT dataset: {n_vectors} vectors, {dimensions}D")
        
        # BERT embeddings have specific statistical properties
        # They tend to be:
        # 1. Normalized (roughly unit length)
        # 2. Have semantic clustering
        # 3. Show layered structure from transformer
        
        # Generate clustered embeddings to simulate semantic structure
        n_clusters = min(10000, n_vectors // 10)  # ~10% cluster centers
        cluster_centers = np.random.randn(n_clusters, dimensions)
        
        # Normalize cluster centers
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        
        # Generate vectors around clusters with BERT-like variance
        vectors = []
        metadata = []
        
        batch_size = 10000
        for i in tqdm(range(0, n_vectors, batch_size), desc="Generating vectors"):
            batch_end = min(i + batch_size, n_vectors)
            batch_size_actual = batch_end - i
            
            # Assign each vector to a cluster
            cluster_assignments = np.random.randint(0, n_clusters, batch_size_actual)
            
            # Generate vectors around assigned clusters
            batch_vectors = []
            for j, cluster_id in enumerate(cluster_assignments):
                # Start from cluster center
                base_vector = cluster_centers[cluster_id].copy()
                
                # Add semantic noise (smaller variance, clustered structure)
                semantic_noise = np.random.randn(dimensions) * 0.15
                
                # Add small random component (transformer layer effects)  
                layer_noise = np.random.randn(dimensions) * 0.05
                
                # Combine and normalize (BERT outputs are roughly normalized)
                vector = base_vector + semantic_noise + layer_noise
                vector = vector / (np.linalg.norm(vector) + 1e-8)  # Avoid division by zero
                
                batch_vectors.append(vector)
                metadata.append(f"doc_{i + j}")
            
            vectors.extend(batch_vectors)
        
        vectors = np.array(vectors, dtype=np.float32)
        
        logger.info(f"Generated {len(vectors)} synthetic BERT vectors")
        logger.info(f"Vector stats: mean_norm={np.mean(np.linalg.norm(vectors, axis=1)):.3f}")
        
        if save:
            self._save_synthetic_dataset(vectors, metadata)
        
        return vectors, metadata
    
    def _save_synthetic_dataset(self, vectors: np.ndarray, metadata: List[str]) -> None:
        """Save synthetic dataset to disk."""
        logger.info(f"Saving synthetic BERT dataset to {self.synthetic_path}")
        
        np.savez_compressed(
            self.synthetic_path,
            vectors=vectors,
            metadata=metadata,
            dimensions=vectors.shape[1],
            n_vectors=len(vectors),
            dataset_type="synthetic_bert_768"
        )
        
        logger.info(f"Saved dataset: {vectors.nbytes / (1024**2):.1f} MB")
    
    def load_synthetic_dataset(self) -> Tuple[np.ndarray, List[str], Dict]:
        """Load synthetic BERT dataset."""
        if not self.synthetic_path.exists():
            raise FileNotFoundError(f"Synthetic dataset not found: {self.synthetic_path}")
        
        logger.info(f"Loading synthetic BERT dataset from {self.synthetic_path}")
        
        data = np.load(self.synthetic_path, allow_pickle=True)
        vectors = data['vectors']
        metadata = data['metadata'].tolist()
        
        dataset_info = {
            "source": "synthetic",
            "dimensions": int(data['dimensions']),
            "n_vectors": int(data['n_vectors']),
            "dataset_type": str(data['dataset_type']),
            "file_size_mb": self.synthetic_path.stat().st_size / (1024**2)
        }
        
        logger.info(f"Loaded {len(vectors)} vectors ({vectors.shape[1]}D)")
        return vectors, metadata, dataset_info
    
    def download_real_bert_dataset(self, dataset_name: str = "wiki_bert_768") -> bool:
        """
        Download real BERT dataset (if available).
        
        Args:
            dataset_name: Name of dataset to download
            
        Returns:
            Success status
        """
        if dataset_name not in self.bert_urls:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        url = self.bert_urls[dataset_name]
        output_path = self.dataset_paths[dataset_name]
        
        if output_path.exists():
            logger.info(f"Dataset already exists: {output_path}")
            return True
        
        logger.info(f"Downloading BERT dataset: {dataset_name}")
        logger.info(f"URL: {url}")
        
        try:
            # Note: These URLs are placeholders - real BERT datasets would need actual URLs
            logger.warning("Real BERT dataset URLs are placeholders - using synthetic fallback")
            return False
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False
    
    def get_bert_dataset(
        self,
        max_vectors: Optional[int] = None,
        prefer_real: bool = False
    ) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Get BERT-768 dataset, generating synthetic if real unavailable.
        
        Args:
            max_vectors: Maximum vectors to return
            prefer_real: Try real dataset first
            
        Returns:
            Tuple of (vectors, metadata, info)
        """
        if prefer_real:
            try:
                logger.info("Attempting to load real BERT dataset...")
                success = self.download_real_bert_dataset()
                if success:
                    return self._load_real_dataset(max_vectors)
            except Exception as e:
                logger.warning(f"Real BERT dataset unavailable: {e}")
        
        # Use synthetic dataset
        try:
            vectors, metadata, info = self.load_synthetic_dataset()
            
            if max_vectors and max_vectors < len(vectors):
                indices = np.random.choice(len(vectors), max_vectors, replace=False)
                vectors = vectors[indices]
                metadata = [metadata[i] for i in indices]
                info["n_vectors"] = len(vectors)
            
            return vectors, metadata, info
            
        except FileNotFoundError:
            # Generate new synthetic dataset
            n_vectors = max_vectors if max_vectors else 1000000
            logger.info(f"Generating new synthetic BERT dataset: {n_vectors} vectors")
            
            vectors, metadata = self.generate_synthetic_bert_dataset(n_vectors)
            
            info = {
                "source": "synthetic",
                "dimensions": 768,
                "n_vectors": len(vectors),
                "dataset_type": "synthetic_bert_768",
                "generated_at": time.time()
            }
            
            return vectors, metadata, info
    
    def benchmark_dataset_loading(self) -> Dict:
        """Benchmark dataset loading performance."""
        results = {}
        
        try:
            start_time = time.time()
            vectors, metadata, info = self.get_bert_dataset(max_vectors=100000)
            load_time = time.time() - start_time
            
            results["bert_768"] = {
                "load_time_seconds": load_time,
                "vectors_loaded": len(vectors),
                "vectors_per_second": len(vectors) / load_time,
                "memory_mb": vectors.nbytes / (1024**2),
                "source": info["source"]
            }
            
        except Exception as e:
            results["bert_768"] = {"error": str(e)}
        
        return results
    
    def get_dataset_info(self) -> Dict:
        """Get information about available BERT datasets."""
        info = {
            "real_bert_available": False,
            "synthetic_bert_available": self.synthetic_path.exists(),
            "data_directory": str(self.data_dir),
            "datasets": []
        }
        
        # Check synthetic dataset
        if self.synthetic_path.exists():
            try:
                data = np.load(self.synthetic_path, allow_pickle=True)
                info["datasets"].append({
                    "name": "synthetic_bert_768",
                    "source": "generated",
                    "vectors": int(data['n_vectors']),
                    "dimensions": int(data['dimensions']),
                    "file_size_mb": self.synthetic_path.stat().st_size / (1024**2)
                })
            except Exception:
                pass
        
        # Check for real datasets (h5py functionality disabled for now)
        # for name, path in self.dataset_paths.items():
        #     if path.exists():
        #         info["real_bert_available"] = True
        #         try:
        #             with h5py.File(path, 'r') as f:
        #                 vectors = f.get('vectors', f.get('embeddings'))
        #                 if vectors is not None:
        #                     info["datasets"].append({
        #                         "name": name,
        #                         "source": "real",
        #                         "vectors": vectors.shape[0],
        #                         "dimensions": vectors.shape[1],
        #                         "file_size_mb": path.stat().st_size / (1024**2)
        #                     })
        #         except Exception:
        #             pass
        
        return info


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="BERT dataset manager for SpiralDeltaDB")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--info", action="store_true", help="Show dataset information")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark loading performance")
    parser.add_argument("--generate", type=int, help="Generate synthetic dataset with N vectors")
    parser.add_argument("--max-vectors", type=int, help="Maximum vectors to load")
    parser.add_argument("--test-load", action="store_true", help="Test loading dataset")
    
    args = parser.parse_args()
    
    manager = BERTDatasetManager(args.data_dir)
    
    if args.info:
        print("📊 BERT Dataset Information:")
        print("=" * 50)
        info = manager.get_dataset_info()
        print(json.dumps(info, indent=2))
    
    elif args.benchmark:
        print("⚡ BERT Dataset Loading Benchmark:")
        print("=" * 50)
        results = manager.benchmark_dataset_loading()
        print(json.dumps(results, indent=2))
    
    elif args.generate:
        print(f"🔬 Generating synthetic BERT dataset: {args.generate} vectors")
        print("=" * 50)
        start_time = time.time()
        vectors, metadata = manager.generate_synthetic_bert_dataset(args.generate)
        generation_time = time.time() - start_time
        
        print(f"✅ Generated {len(vectors)} vectors in {generation_time:.1f}s")
        print(f"Generation rate: {len(vectors) / generation_time:.0f} vectors/sec")
        print(f"Memory usage: {vectors.nbytes / (1024**2):.1f} MB")
    
    elif args.test_load:
        print("🧪 Testing BERT Dataset Loading:")
        print("=" * 50)
        try:
            vectors, metadata, info = manager.get_bert_dataset(max_vectors=args.max_vectors)
            
            print(f"✅ Successfully loaded BERT dataset:")
            print(f"  Source: {info['source']}")
            print(f"  Vectors: {len(vectors)}")
            print(f"  Dimensions: {vectors.shape[1]}")
            print(f"  Memory usage: {vectors.nbytes / (1024**2):.1f} MB")
            print(f"  Vector norm stats: mean={np.mean(np.linalg.norm(vectors, axis=1)):.3f}")
            
        except Exception as e:
            print(f"❌ Failed to load BERT dataset: {e}")
    
    else:
        print("Use --help for usage information")


if __name__ == "__main__":
    main()