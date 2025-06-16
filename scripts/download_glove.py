#!/usr/bin/env python3
"""
Download and prepare GloVe-300 dataset for SpiralDeltaDB benchmarking.

This script downloads the GloVe 6B tokens dataset and extracts the 300-dimensional
vectors for testing compression and search quality.
"""

import os
import urllib.request
import zipfile
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import List, Tuple
import tqdm
import ssl
import requests
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GloVeDownloader:
    """Download and process GloVe embeddings."""
    
    GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
    GLOVE_300D_FILE = "glove.6B.300d.txt"
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize GloVe downloader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.zip_path = self.data_dir / "glove.6B.zip"
        self.txt_path = self.data_dir / self.GLOVE_300D_FILE
        self.vectors_path = self.data_dir / "glove_300d_vectors.npy"
        self.words_path = self.data_dir / "glove_300d_words.txt"
    
    def download_if_needed(self) -> None:
        """Download GloVe dataset if not already present."""
        if self.vectors_path.exists() and self.words_path.exists():
            logger.info("GloVe-300 dataset already processed")
            return
        
        if not self.zip_path.exists():
            logger.info(f"Downloading GloVe dataset from {self.GLOVE_URL}")
            self._download_with_progress(self.GLOVE_URL, self.zip_path)
        
        if not self.txt_path.exists():
            logger.info("Extracting GloVe-300 file from archive")
            self._extract_300d_file()
        
        logger.info("Processing GloVe text file to numpy arrays")
        self._process_text_file()
    
    def _download_with_progress(self, url: str, filepath: Path) -> None:
        """Download file with progress bar using multiple methods."""
        
        # Try method 1: requests with robust SSL handling
        try:
            self._download_with_requests(url, filepath)
            return
        except Exception as e:
            logger.warning(f"Requests download failed: {e}")
        
        # Try method 2: urllib with SSL bypass
        try:
            self._download_with_urllib(url, filepath)
            return
        except Exception as e:
            logger.warning(f"Urllib download failed: {e}")
        
        # Try method 3: Alternative URLs
        alternative_urls = [
            "https://github.com/stanfordnlp/GloVe/releases/download/v1.2/glove.6B.zip",
            "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        ]
        
        for alt_url in alternative_urls:
            try:
                logger.info(f"Trying alternative URL: {alt_url}")
                self._download_with_requests(alt_url, filepath)
                return
            except Exception as e:
                logger.warning(f"Alternative URL failed: {e}")
                continue
        
        raise RuntimeError("All download methods failed")
    
    def _download_with_requests(self, url: str, filepath: Path) -> None:
        """Download using requests library."""
        response = requests.get(url, stream=True, timeout=30, verify=False)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded {filepath}")
    
    def _download_with_urllib(self, url: str, filepath: Path) -> None:
        """Download using urllib with SSL bypass."""
        # Create unverified SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        def progress_hook(block_num, block_size, total_size):
            if hasattr(progress_hook, 'pbar'):
                progress_hook.pbar.update(block_size)
            else:
                progress_hook.pbar = tqdm.tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True, 
                    desc="Downloading"
                )
        
        # Install SSL context
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        if hasattr(progress_hook, 'pbar'):
            progress_hook.pbar.close()
        logger.info(f"Downloaded {filepath}")
    
    def _extract_300d_file(self) -> None:
        """Extract only the 300d file from the zip archive."""
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            # Extract only the 300d file
            zip_ref.extract(self.GLOVE_300D_FILE, self.data_dir)
        logger.info(f"Extracted {self.GLOVE_300D_FILE}")
    
    def _process_text_file(self) -> None:
        """Process GloVe text file into numpy arrays."""
        logger.info("Reading GloVe text file...")
        
        words = []
        vectors = []
        
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm.tqdm(f, desc="Processing lines")):
                parts = line.strip().split()
                if len(parts) != 301:  # word + 300 dimensions
                    logger.warning(f"Skipping malformed line {line_num + 1}")
                    continue
                
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                words.append(word)
                vectors.append(vector)
        
        # Convert to numpy array
        vectors_array = np.array(vectors, dtype=np.float32)
        
        logger.info(f"Processed {len(words)} word vectors with shape {vectors_array.shape}")
        
        # Save processed data
        np.save(self.vectors_path, vectors_array)
        
        with open(self.words_path, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(f"{word}\n")
        
        logger.info(f"Saved vectors to {self.vectors_path}")
        logger.info(f"Saved words to {self.words_path}")
    
    def load_data(self, max_vectors: int = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load processed GloVe data.
        
        Args:
            max_vectors: Maximum number of vectors to load (None for all)
            
        Returns:
            Tuple of (vectors, words)
        """
        if not self.vectors_path.exists():
            raise FileNotFoundError("GloVe data not found. Run download_if_needed() first.")
        
        # Load vectors
        vectors = np.load(self.vectors_path)
        
        # Load words
        with open(self.words_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        
        # Limit if requested
        if max_vectors is not None and max_vectors < len(vectors):
            vectors = vectors[:max_vectors]
            words = words[:max_vectors]
        
        logger.info(f"Loaded {len(vectors)} vectors with shape {vectors.shape}")
        return vectors, words
    
    def get_sample_subset(self, n_samples: int = 1000) -> Tuple[np.ndarray, List[str]]:
        """
        Get a random sample subset for quick testing.
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            Tuple of (vectors, words)
        """
        vectors, words = self.load_data()
        
        # Random sampling
        indices = np.random.choice(len(vectors), size=min(n_samples, len(vectors)), replace=False)
        sampled_vectors = vectors[indices]
        sampled_words = [words[i] for i in indices]
        
        return sampled_vectors, sampled_words


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Download and prepare GloVe-300 dataset")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--sample-size", type=int, help="Create a sample subset of this size")
    parser.add_argument("--max-vectors", type=int, help="Maximum vectors to process")
    
    args = parser.parse_args()
    
    # Download and process
    downloader = GloVeDownloader(args.data_dir)
    downloader.download_if_needed()
    
    # Load and display info
    vectors, words = downloader.load_data(args.max_vectors)
    
    print(f"\nGloVe-300 Dataset Summary:")
    print(f"Vectors: {vectors.shape}")
    print(f"Dimensions: {vectors.shape[1]}")
    print(f"Data type: {vectors.dtype}")
    print(f"Memory usage: {vectors.nbytes / (1024**2):.1f} MB")
    print(f"Sample words: {words[:10]}")
    
    # Create sample if requested
    if args.sample_size:
        sample_vectors, sample_words = downloader.get_sample_subset(args.sample_size)
        sample_path = Path(args.data_dir) / f"glove_sample_{args.sample_size}.npy"
        np.save(sample_path, sample_vectors)
        print(f"\nSaved {args.sample_size} sample vectors to {sample_path}")


if __name__ == "__main__":
    main()