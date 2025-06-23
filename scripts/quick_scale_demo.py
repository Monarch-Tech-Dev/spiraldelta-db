#!/usr/bin/env python3
"""
Quick scale demonstration for optimized BERT-768 configuration.

Demonstrates scaling behavior with the quality-optimized parameters.
"""

import numpy as np
import sys
from pathlib import Path
import time
import json
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from spiraldelta.delta_encoder import DeltaEncoder
from spiraldelta.spiral_coordinator import SpiralCoordinator

# Import dataset manager
sys.path.append(str(Path(__file__).parent))
from bert_dataset_manager import BERTDatasetManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_test_scale(vectors, params, test_size, max_train_vectors=2000):
    """Quick test at a specific scale with limited training."""
    test_vectors = vectors[:test_size]
    
    try:
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
        
        # Process
        start_time = time.time()
        sorted_coords = spiral_coordinator.sort_by_spiral(test_vectors)
        sorted_vectors = [coord.vector for coord in sorted_coords]
        spiral_time = time.time() - start_time
        
        # Limited training for speed (use subset for training, apply to all)
        start_time = time.time()
        
        # Use limited vectors for training to speed up
        train_vectors = min(max_train_vectors, len(sorted_vectors))
        training_sample = sorted_vectors[:train_vectors]
        
        chunk_size = min(500, train_vectors // 3)
        training_sequences = [
            training_sample[i:i+chunk_size] 
            for i in range(0, len(training_sample), chunk_size)
            if i+chunk_size <= len(training_sample)
        ][:4]  # Limit to 4 sequences
        
        if not training_sequences:
            training_sequences = [training_sample]
        
        delta_encoder.train(training_sequences)
        train_time = time.time() - start_time
        
        # Encode full dataset using trained encoder
        start_time = time.time()
        compressed = delta_encoder.encode_sequence(sorted_vectors)
        encode_time = time.time() - start_time
        
        # Decode
        start_time = time.time()
        reconstructed = delta_encoder.decode_sequence(compressed)
        decode_time = time.time() - start_time
        
        # Quick quality evaluation (sample)
        sample_size = min(1000, len(sorted_vectors))
        indices = np.random.choice(len(sorted_vectors), sample_size, replace=False)
        
        cos_sims = []
        for i in indices:
            orig = sorted_vectors[i]
            recon = reconstructed[i]
            
            orig_norm = np.linalg.norm(orig)
            recon_norm = np.linalg.norm(recon)
            
            if orig_norm > 1e-8 and recon_norm > 1e-8:
                cos_sim = np.dot(orig, recon) / (orig_norm * recon_norm)
                cos_sims.append(cos_sim)
        
        quality = np.mean(cos_sims) if cos_sims else 0.0
        
        return {
            "test_size": test_size,
            "quality": float(quality),
            "compression": float(compressed.compression_ratio),
            "timing": {
                "spiral": spiral_time,
                "training": train_time,
                "encoding": encode_time,
                "decoding": decode_time,
                "total": spiral_time + train_time + encode_time + decode_time
            },
            "training_vectors_used": train_vectors,
            "success": True
        }
        
    except Exception as e:
        return {
            "test_size": test_size,
            "error": str(e),
            "success": False
        }


def main():
    """Quick scale demonstration."""
    print("📈 Quick Scale Demonstration - Optimized BERT-768")
    print("=" * 60)
    
    # Load optimized parameters
    try:
        with open("./data/fast_quality_optimization.json", 'r') as f:
            quality_results = json.load(f)
        optimized_params = quality_results["best_balanced_result"]["params"]
        print(f"✅ Loaded optimized parameters from quality optimization")
    except Exception as e:
        print(f"⚠️ Could not load optimization results: {e}")
        optimized_params = {
            "quantization_levels": 2,
            "n_subspaces": 4,
            "n_bits": 9,
            "anchor_stride": 16,
            "spiral_constant": 1.5
        }
        print(f"Using fallback parameters")
    
    print(f"📋 Parameters: {optimized_params}")
    
    # Load dataset
    print(f"\n📊 Loading BERT dataset...")
    dataset_manager = BERTDatasetManager("./data")
    vectors, _, info = dataset_manager.get_bert_dataset(max_vectors=50000)
    print(f"Dataset: {len(vectors)} vectors, {info['source']} source")
    
    # Scale test sizes
    test_scales = [1000, 2500, 5000, 10000, 25000]
    available_scales = [size for size in test_scales if size <= len(vectors)]
    
    print(f"\n🧪 Testing scales: {available_scales}")
    
    results = []
    
    for scale in available_scales:
        print(f"\n{'='*20} SCALE: {scale:,} vectors {'='*20}")
        
        start_time = time.time()
        result = quick_test_scale(vectors, optimized_params, scale)
        total_time = time.time() - start_time
        
        if result['success']:
            quality = result['quality']
            compression = result['compression']
            
            # Check targets
            quality_met = quality >= 0.25
            compression_met = compression >= 0.668
            both_met = quality_met and compression_met
            
            print(f"Quality: {quality:.3f} {'✅' if quality_met else '❌'} (target ≥0.25)")
            print(f"Compression: {compression:.1%} {'✅' if compression_met else '❌'} (target ≥66.8%)")
            print(f"Both Targets: {'✅' if both_met else '❌'}")
            print(f"Processing Time: {result['timing']['total']:.1f}s")
            print(f"Training Vectors: {result['training_vectors_used']:,}")
            print(f"Throughput: {scale / result['timing']['total']:.0f} vectors/sec")
            
            result['targets_met'] = {
                'quality': quality_met,
                'compression': compression_met,
                'both': both_met
            }
            
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        
        results.append(result)
    
    # Summary
    print(f"\n🏆 SCALE DEMONSTRATION SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        max_scale = max(r['test_size'] for r in successful_results)
        both_targets_results = [r for r in successful_results if r.get('targets_met', {}).get('both', False)]
        
        print(f"📊 Results Overview:")
        print(f"  Maximum scale tested: {max_scale:,} vectors")
        print(f"  Successful tests: {len(successful_results)}/{len(results)}")
        
        if both_targets_results:
            max_both_scale = max(r['test_size'] for r in both_targets_results)
            print(f"  Scales meeting both targets: {len(both_targets_results)}")
            print(f"  Maximum scale with both targets: {max_both_scale:,} vectors")
        
        print(f"\n📋 Scale-by-Scale Results:")
        print(f"{'Scale':<8} {'Quality':<8} {'Comp.':<8} {'Time':<8} {'Rate':<12} {'Status'}")
        print("-" * 60)
        
        for result in successful_results:
            scale = f"{result['test_size']:,}"
            quality = f"{result['quality']:.3f}"
            compression = f"{result['compression']:.0%}"
            time_str = f"{result['timing']['total']:.1f}s"
            rate = f"{result['test_size'] / result['timing']['total']:.0f}/s"
            
            if result.get('targets_met', {}).get('both', False):
                status = "✅ BOTH"
            elif result.get('targets_met', {}).get('quality', False) or result.get('targets_met', {}).get('compression', False):
                status = "⚠️ PARTIAL"
            else:
                status = "❌ NEITHER"
            
            print(f"{scale:<8} {quality:<8} {compression:<8} {time_str:<8} {rate:<12} {status}")
        
        # Save results
        scale_demo_results = {
            "scale_demonstration": {
                "optimized_parameters": optimized_params,
                "max_scale_tested": max_scale,
                "successful_tests": len(successful_results),
                "scales_meeting_both_targets": len(both_targets_results) if both_targets_results else 0,
                "detailed_results": results,
                "timestamp": time.time()
            }
        }
        
        output_path = Path("./data/scale_demonstration_results.json")
        with open(output_path, 'w') as f:
            json.dump(scale_demo_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        
        # Final assessment
        if both_targets_results:
            print(f"\n🎉 SCALE DEMONSTRATION: ✅ SUCCESS")
            print(f"   Configuration scales effectively up to {max_both_scale:,} vectors")
            print(f"   Quality and compression targets maintained at scale")
        elif successful_results:
            print(f"\n⚠️ SCALE DEMONSTRATION: PARTIAL SUCCESS")
            print(f"   Configuration works up to {max_scale:,} vectors")
            print(f"   Some target degradation at larger scales")
        else:
            print(f"\n❌ SCALE DEMONSTRATION: FAILED")
            print(f"   Configuration needs optimization for scaling")
    
    else:
        print("❌ No successful scale tests")


if __name__ == "__main__":
    main()