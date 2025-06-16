#!/usr/bin/env python3
"""
Create synthetic GloVe-like dataset for SpiralDeltaDB benchmarking.

This script generates realistic word embedding vectors that mimic the
properties of GloVe embeddings for testing compression and search quality.
"""

import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticGloVeGenerator:
    """Generate synthetic GloVe-like embeddings."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize generator.
        
        Args:
            data_dir: Directory to store generated data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_realistic_embeddings(
        self, 
        n_vectors: int = 50000,
        dimensions: int = 300,
        seed: int = 42
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate realistic word embeddings with GloVe-like properties.
        
        Args:
            n_vectors: Number of vectors to generate
            dimensions: Vector dimensionality
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (vectors, words)
        """
        np.random.seed(seed)
        logger.info(f"Generating {n_vectors} synthetic embeddings with {dimensions} dimensions")
        
        # Create word vocabulary
        words = self._generate_vocabulary(n_vectors)
        
        # Generate base semantic clusters
        n_clusters = min(100, n_vectors // 10)  # Reasonable number of semantic clusters
        cluster_centers = np.random.randn(n_clusters, dimensions).astype(np.float32)
        
        # Normalize cluster centers (GloVe vectors are typically normalized)
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        
        vectors = []
        
        for i in range(n_vectors):
            # Assign to cluster with some probability distribution
            # More frequent words get assigned to larger clusters
            cluster_weights = np.exp(-0.1 * np.arange(n_clusters))  # Zipfian-like distribution
            cluster_weights /= cluster_weights.sum()
            cluster_id = np.random.choice(n_clusters, p=cluster_weights)
            
            # Generate vector near cluster center
            cluster_center = cluster_centers[cluster_id]
            
            # Add noise (smaller for more frequent words)
            frequency_rank = i + 1  # Higher rank = less frequent
            noise_scale = 0.1 + 0.3 * (frequency_rank / n_vectors)  # More noise for rare words
            
            vector = cluster_center + np.random.randn(dimensions) * noise_scale
            
            # Normalize (GloVe property)
            vector = vector / max(np.linalg.norm(vector), 1e-8)
            
            # Add some correlation structure (nearby words should be similar)
            if i > 0 and np.random.random() < 0.3:  # 30% chance of correlation
                prev_vector = vectors[max(0, i - np.random.randint(1, min(10, i + 1)))]
                correlation_strength = np.random.uniform(0.1, 0.5)
                vector = (1 - correlation_strength) * vector + correlation_strength * prev_vector
                vector = vector / max(np.linalg.norm(vector), 1e-8)
            
            vectors.append(vector.astype(np.float32))
        
        vectors_array = np.array(vectors, dtype=np.float32)
        
        logger.info(f"Generated embeddings with shape {vectors_array.shape}")
        logger.info(f"Vector norm stats: mean={np.mean(np.linalg.norm(vectors_array, axis=1)):.3f}, "
                   f"std={np.std(np.linalg.norm(vectors_array, axis=1)):.3f}")
        
        return vectors_array, words
    
    def _generate_vocabulary(self, n_words: int) -> List[str]:
        """
        Generate realistic vocabulary following Zipfian distribution.
        
        Args:
            n_words: Number of words to generate
            
        Returns:
            List of word strings
        """
        words = []
        
        # Common English words (most frequent)
        common_words = [
            "the", "of", "and", "a", "to", "in", "is", "you", "that", "it",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
            "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
            "but", "not", "what", "all", "were", "we", "when", "your", "can", "said"
        ]
        
        # Add common words first
        words.extend(common_words[:min(len(common_words), n_words)])
        
        # Generate synthetic words for the rest
        prefixes = ["pre", "un", "re", "in", "dis", "en", "non", "over", "under", "out"]
        roots = ["act", "form", "struct", "spect", "port", "dict", "fac", "mit", "ject", "duc"]
        suffixes = ["ing", "ed", "er", "est", "ly", "tion", "ness", "ment", "able", "ful"]
        
        while len(words) < n_words:
            # Generate compound words
            if np.random.random() < 0.3:
                word = np.random.choice(prefixes) + np.random.choice(roots) + np.random.choice(suffixes)
            else:
                # Generate simple words
                word = np.random.choice(roots) + np.random.choice(suffixes)
            
            # Add unique identifier to avoid duplicates
            word = f"{word}_{len(words)}"
            words.append(word)
        
        return words[:n_words]
    
    def save_dataset(
        self, 
        vectors: np.ndarray, 
        words: List[str], 
        name: str = "synthetic_glove_300d"
    ) -> None:
        """
        Save dataset to files.
        
        Args:
            vectors: Vector array
            words: Word list
            name: Dataset name prefix
        """
        vectors_path = self.data_dir / f"{name}_vectors.npy"
        words_path = self.data_dir / f"{name}_words.txt"
        info_path = self.data_dir / f"{name}_info.json"
        
        # Save vectors
        np.save(vectors_path, vectors)
        
        # Save words
        with open(words_path, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(f"{word}\n")
        
        # Save dataset info
        info = {
            "n_vectors": len(vectors),
            "dimensions": vectors.shape[1],
            "dtype": str(vectors.dtype),
            "memory_mb": vectors.nbytes / (1024**2),
            "vector_norm_stats": {
                "mean": float(np.mean(np.linalg.norm(vectors, axis=1))),
                "std": float(np.std(np.linalg.norm(vectors, axis=1))),
                "min": float(np.min(np.linalg.norm(vectors, axis=1))),
                "max": float(np.max(np.linalg.norm(vectors, axis=1)))
            }
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Saved dataset:")
        logger.info(f"  Vectors: {vectors_path}")
        logger.info(f"  Words: {words_path}")
        logger.info(f"  Info: {info_path}")
    
    def load_dataset(self, name: str = "synthetic_glove_300d") -> Tuple[np.ndarray, List[str]]:
        """
        Load saved dataset.
        
        Args:
            name: Dataset name prefix
            
        Returns:
            Tuple of (vectors, words)
        """
        vectors_path = self.data_dir / f"{name}_vectors.npy"
        words_path = self.data_dir / f"{name}_words.txt"
        
        if not vectors_path.exists() or not words_path.exists():
            raise FileNotFoundError(f"Dataset {name} not found in {self.data_dir}")
        
        # Load vectors
        vectors = np.load(vectors_path)
        
        # Load words
        with open(words_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        
        logger.info(f"Loaded dataset {name}: {vectors.shape} vectors")
        return vectors, words


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic GloVe-like dataset")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--n-vectors", type=int, default=50000, help="Number of vectors")
    parser.add_argument("--dimensions", type=int, default=300, help="Vector dimensions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--name", default="synthetic_glove_300d", help="Dataset name")
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = SyntheticGloVeGenerator(args.data_dir)
    vectors, words = generator.generate_realistic_embeddings(
        n_vectors=args.n_vectors,
        dimensions=args.dimensions,
        seed=args.seed
    )
    
    # Save dataset
    generator.save_dataset(vectors, words, args.name)
    
    print(f"\nSynthetic GloVe Dataset Summary:")
    print(f"Vectors: {vectors.shape}")
    print(f"Dimensions: {vectors.shape[1]}")
    print(f"Data type: {vectors.dtype}")
    print(f"Memory usage: {vectors.nbytes / (1024**2):.1f} MB")
    print(f"Sample words: {words[:10]}")


if __name__ == "__main__":
    main()