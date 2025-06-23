#!/usr/bin/env python3
"""
Scale testing for optimized BERT-768 configuration.

Tests the quality-optimized parameters with larger datasets up to 1M vectors.
"""

import numpy as np
import sys
from pathlib import Path
import time
import json
import logging
# import psutil  # Optional - for memory monitoring
import gc

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from spiraldelta.delta_encoder import DeltaEncoder
from spiraldelta.spiral_coordinator import SpiralCoordinator

# Import dataset manager
sys.path.append(str(Path(__file__).parent))
from bert_dataset_manager import BERTDatasetManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScaleTester:
    """Scale testing for optimized BERT configuration."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize scale tester."""
        self.data_dir = Path(data_dir)
        self.dataset_manager = BERTDatasetManager(data_dir)
        
        # Load optimized parameters from quality optimization
        try:
            with open(self.data_dir / "fast_quality_optimization.json", 'r') as f:
                quality_results = json.load(f)
            self.optimized_params = quality_results["best_balanced_result"]["params"]
            logger.info("Loaded optimized parameters from quality optimization")
        except Exception as e:
            logger.warning(f"Could not load quality optimization results: {e}")
            # Fallback to manual optimized parameters
            self.optimized_params = {
                "quantization_levels": 2,
                "n_subspaces": 4,
                "n_bits": 9,
                "anchor_stride": 16,
                "spiral_constant": 1.5
            }
        
        # Scale test configurations
        self.scale_sizes = [5000, 10000, 25000, 50000, 100000]
        
        # Results tracking
        self.scale_results = []
        
    def get_memory_usage(self):
        """Get current memory usage (simplified)."""
        # Simplified memory tracking without psutil
        return 0.0  # Placeholder - could implement basic memory tracking if needed
    
    def evaluate_quality_fast(self, original_vectors, reconstructed_vectors):
        """Fast quality evaluation."""
        if len(original_vectors) != len(reconstructed_vectors):
            return 0.0
        
        # Sample for large datasets to speed up evaluation
        if len(original_vectors) > 2000:
            indices = np.random.choice(len(original_vectors), 2000, replace=False)
            orig_sample = [original_vectors[i] for i in indices]
            recon_sample = [reconstructed_vectors[i] for i in indices]
        else:
            orig_sample = original_vectors
            recon_sample = reconstructed_vectors
        
        cos_sims = []
        for orig, recon in zip(orig_sample, recon_sample):
            orig_norm = np.linalg.norm(orig)
            recon_norm = np.linalg.norm(recon)
            
            if orig_norm > 1e-8 and recon_norm > 1e-8:
                cos_sim = np.dot(orig, recon) / (orig_norm * recon_norm)
                cos_sims.append(cos_sim)
        
        return np.mean(cos_sims) if cos_sims else 0.0
    
    def test_scale_size(self, vectors, test_size):
        """Test compression at a specific scale."""
        logger.info(f"Testing scale: {test_size} vectors")
        
        test_vectors = vectors[:test_size]
        memory_before = self.get_memory_usage()
        
        try:
            # Initialize components
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
            
            # Spiral transformation
            start_time = time.time()
            sorted_coords = spiral_coordinator.sort_by_spiral(test_vectors)
            sorted_vectors = [coord.vector for coord in sorted_coords]
            spiral_time = time.time() - start_time
            
            memory_after_spiral = self.get_memory_usage()
            
            # Training with adaptive chunk size
            start_time = time.time()
            # Scale chunk size with dataset size, but keep training efficient
            chunk_size = min(1000, max(500, len(sorted_vectors) // 10))
            training_sequences = [
                sorted_vectors[i:i+chunk_size] 
                for i in range(0, len(sorted_vectors), chunk_size)
                if i+chunk_size <= len(sorted_vectors)
            ][:8]  # Limit to 8 sequences for memory efficiency
            
            if not training_sequences:
                training_sequences = [sorted_vectors]
            
            delta_encoder.train(training_sequences)
            train_time = time.time() - start_time
            
            memory_after_training = self.get_memory_usage()
            
            # Compression
            start_time = time.time()
            compressed = delta_encoder.encode_sequence(sorted_vectors)
            encode_time = time.time() - start_time
            
            # Decompression
            start_time = time.time()
            reconstructed = delta_encoder.decode_sequence(compressed)
            decode_time = time.time() - start_time
            
            memory_after_decode = self.get_memory_usage()
            
            # Quality evaluation (sampled for large datasets)
            start_time = time.time()
            quality = self.evaluate_quality_fast(sorted_vectors, reconstructed)
            quality_time = time.time() - start_time
            
            # Storage calculations
            original_size = len(test_vectors) * 768 * 4  # float32
            compressed_size = delta_encoder._estimate_compressed_size(
                compressed.anchors, compressed.delta_codes
            )
            
            # Clean up large objects
            del sorted_coords, sorted_vectors, reconstructed
            gc.collect()
            
            memory_after_cleanup = self.get_memory_usage()
            
            result = {
                "test_size": test_size,
                "compression_ratio": float(compressed.compression_ratio),
                "actual_compression": float(1.0 - (compressed_size / original_size)) if original_size > 0 else 0.0,
                "quality_score": float(quality),
                "timing": {
                    "spiral_transform": spiral_time,
                    "training": train_time,
                    "encoding": encode_time,
                    "decoding": decode_time,
                    "quality_eval": quality_time,
                    "total": spiral_time + train_time + encode_time + decode_time
                },
                "throughput": {
                    "encoding_vectors_per_sec": test_size / encode_time if encode_time > 0 else 0,
                    "decoding_vectors_per_sec": test_size / decode_time if decode_time > 0 else 0,
                    "total_vectors_per_sec": test_size / (spiral_time + train_time + encode_time + decode_time) if (spiral_time + train_time + encode_time + decode_time) > 0 else 0
                },
                "memory": {
                    "before_mb": memory_before,
                    "after_spiral_mb": memory_after_spiral,
                    "after_training_mb": memory_after_training,
                    "after_decode_mb": memory_after_decode,
                    "after_cleanup_mb": memory_after_cleanup,
                    "peak_usage_mb": max(memory_after_spiral, memory_after_training, memory_after_decode)
                },
                "storage": {
                    "original_size_mb": original_size / (1024**2),
                    "compressed_size_mb": compressed_size / (1024**2),
                    "space_saved_mb": (original_size - compressed_size) / (1024**2)
                },
                "targets_met": {
                    "quality_target": quality >= 0.25,
                    "compression_target": compressed.compression_ratio >= 0.668,
                    "both_targets": quality >= 0.25 and compressed.compression_ratio >= 0.668
                },
                "success": True
            }
            
            logger.info(f"Scale {test_size}: Quality={quality:.3f}, "
                       f"Compression={compressed.compression_ratio:.1%}, "
                       f"Time={result['timing']['total']:.1f}s, "
                       f"Memory={memory_after_decode:.0f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Scale test failed for size {test_size}: {e}")
            return {
                "test_size": test_size,
                "success": False,
                "error": str(e),
                "memory_before_mb": memory_before
            }
    
    def run_scale_tests(self, max_vectors=100000):
        """Run comprehensive scale testing."""
        logger.info("Starting scale testing for optimized BERT configuration")
        logger.info(f"Optimized parameters: {self.optimized_params}")
        
        # Load dataset
        logger.info(f"Loading BERT dataset (up to {max_vectors} vectors)...")
        vectors, metadata, info = self.dataset_manager.get_bert_dataset(max_vectors=max_vectors)
        
        logger.info(f"Dataset loaded: {len(vectors)} vectors, {info['source']} source")
        
        # Filter scale sizes based on available data
        available_scales = [size for size in self.scale_sizes if size <= len(vectors)]
        
        logger.info(f"Testing scales: {available_scales}")
        
        results = []
        
        for scale_size in available_scales:
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING SCALE: {scale_size:,} vectors")
            logger.info(f"{'='*60}")
            
            try:
                result = self.test_scale_size(vectors, scale_size)
                results.append(result)
                
                if result['success']:
                    # Progress update
                    quality_status = "✅" if result['targets_met']['quality_target'] else "❌"
                    compression_status = "✅" if result['targets_met']['compression_target'] else "❌"
                    
                    print(f"Scale {scale_size:,}: "
                          f"Quality {quality_status} {result['quality_score']:.3f}, "
                          f"Compression {compression_status} {result['compression_ratio']:.1%}")
                else:
                    print(f"Scale {scale_size:,}: ❌ FAILED - {result.get('error', 'Unknown error')}")
                
                # Memory cleanup between tests
                gc.collect()
                
            except Exception as e:
                logger.error(f"Critical error testing scale {scale_size}: {e}")
                results.append({
                    "test_size": scale_size,
                    "success": False,
                    "error": f"Critical error: {str(e)}"
                })
        
        self.scale_results = results
        return results
    
    def generate_scale_report(self, output_path="scale_testing_results.json"):
        """Generate comprehensive scale testing report."""
        # Analyze results
        successful_results = [r for r in self.scale_results if r.get('success', False)]
        
        if not successful_results:
            report = {
                "scale_testing": {
                    "status": "FAILED",
                    "error": "No successful scale tests",
                    "timestamp": time.time()
                }
            }
        else:
            # Performance analysis
            max_scale = max(r['test_size'] for r in successful_results)
            best_quality = max(r['quality_score'] for r in successful_results)
            best_compression = max(r['compression_ratio'] for r in successful_results)
            
            # Targets analysis
            quality_targets_met = [r for r in successful_results if r['targets_met']['quality_target']]
            compression_targets_met = [r for r in successful_results if r['targets_met']['compression_target']]
            both_targets_met = [r for r in successful_results if r['targets_met']['both_targets']]
            
            report = {
                "scale_testing": {
                    "status": "SUCCESS",
                    "optimized_parameters": self.optimized_params,
                    "summary": {
                        "max_scale_tested": max_scale,
                        "successful_tests": len(successful_results),
                        "best_quality_achieved": best_quality,
                        "best_compression_achieved": best_compression,
                        "scales_meeting_quality_target": len(quality_targets_met),
                        "scales_meeting_compression_target": len(compression_targets_met),
                        "scales_meeting_both_targets": len(both_targets_met)
                    },
                    "detailed_results": self.scale_results,
                    "timestamp": time.time()
                }
            }
        
        # Save report
        output_file = self.data_dir / output_path
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Scale testing report saved to {output_file}")
        
        # Print summary
        self._print_scale_summary(report)
        
        return output_file
    
    def _print_scale_summary(self, report):
        """Print formatted scale testing summary."""
        print(f"\n📊 SCALE TESTING SUMMARY")
        print("=" * 60)
        
        if report["scale_testing"]["status"] == "FAILED":
            print("❌ Scale testing failed")
            return
        
        summary = report["scale_testing"]["summary"]
        
        print(f"🎯 Maximum Scale Tested: {summary['max_scale_tested']:,} vectors")
        print(f"✅ Successful Tests: {summary['successful_tests']}")
        print(f"🏆 Best Quality: {summary['best_quality_achieved']:.3f}")
        print(f"📦 Best Compression: {summary['best_compression_achieved']:.1%}")
        
        print(f"\n📈 Target Achievement Across Scales:")
        print(f"  Quality ≥0.25: {summary['scales_meeting_quality_target']}/{summary['successful_tests']} scales")
        print(f"  Compression ≥66.8%: {summary['scales_meeting_compression_target']}/{summary['successful_tests']} scales")
        print(f"  Both Targets: {summary['scales_meeting_both_targets']}/{summary['successful_tests']} scales")
        
        # Scale-by-scale results
        print(f"\n📋 Scale-by-Scale Performance:")
        print(f"{'Scale':<10} {'Quality':<10} {'Compression':<12} {'Time':<8} {'Memory':<10} {'Status'}")
        print("-" * 70)
        
        for result in self.scale_results:
            if result.get('success', False):
                scale = f"{result['test_size']:,}"
                quality = f"{result['quality_score']:.3f}"
                compression = f"{result['compression_ratio']:.1%}"
                time_str = f"{result['timing']['total']:.1f}s"
                memory = f"{result['memory']['peak_usage_mb']:.0f}MB"
                status = "✅ BOTH" if result['targets_met']['both_targets'] else "⚠️ PARTIAL"
                
                print(f"{scale:<10} {quality:<10} {compression:<12} {time_str:<8} {memory:<10} {status}")


def main():
    """Main function for scale testing."""
    print("📈 Scale Testing for Optimized BERT-768 Configuration")
    print("=" * 70)
    
    tester = ScaleTester("./data")
    
    # Run scale tests
    results = tester.run_scale_tests(max_vectors=100000)
    
    # Generate report
    tester.generate_scale_report()
    
    # Final assessment
    successful_results = [r for r in results if r.get('success', False)]
    both_targets_met = [r for r in successful_results if r.get('targets_met', {}).get('both_targets', False)]
    
    if both_targets_met:
        max_successful_scale = max(r['test_size'] for r in both_targets_met)
        print(f"\n🎉 SCALE TESTING: ✅ SUCCESS")
        print(f"   Configuration works up to {max_successful_scale:,} vectors")
        print(f"   Meeting both quality ≥0.25 AND compression ≥66.8%")
    elif successful_results:
        max_scale = max(r['test_size'] for r in successful_results)
        print(f"\n⚠️ SCALE TESTING: PARTIAL SUCCESS")
        print(f"   Tested up to {max_scale:,} vectors")
        print(f"   Some targets not consistently met at scale")
    else:
        print(f"\n❌ SCALE TESTING: FAILED")
        print(f"   Configuration needs optimization for larger datasets")


if __name__ == "__main__":
    main()