#!/usr/bin/env python3
"""
Fast quality tuning for BERT-768 embeddings.

Quick optimization to reach quality target ≥0.25 while maintaining compression ≥66.8%.
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


def evaluate_quality_fast(original_vectors, reconstructed_vectors):
    """Fast quality evaluation."""
    cos_sims = []
    for orig, recon in zip(original_vectors, reconstructed_vectors):
        orig_norm = np.linalg.norm(orig)
        recon_norm = np.linalg.norm(recon)
        
        if orig_norm > 1e-8 and recon_norm > 1e-8:
            cos_sim = np.dot(orig, recon) / (orig_norm * recon_norm)
            cos_sims.append(cos_sim)
    
    return np.mean(cos_sims) if cos_sims else 0.0


def test_single_config(vectors, params, test_size=1500):
    """Test a single parameter configuration quickly."""
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
        sorted_coords = spiral_coordinator.sort_by_spiral(test_vectors)
        sorted_vectors = [coord.vector for coord in sorted_coords]
        
        # Small training sequences for speed
        chunk_size = min(400, len(sorted_vectors) // 3)
        training_sequences = [
            sorted_vectors[i:i+chunk_size] 
            for i in range(0, len(sorted_vectors), chunk_size)
            if i+chunk_size <= len(sorted_vectors)
        ][:3]  # Limit to 3 sequences
        
        if not training_sequences:
            training_sequences = [sorted_vectors]
        
        delta_encoder.train(training_sequences)
        compressed = delta_encoder.encode_sequence(sorted_vectors)
        reconstructed = delta_encoder.decode_sequence(compressed)
        
        # Evaluate
        quality = evaluate_quality_fast(sorted_vectors, reconstructed)
        compression = compressed.compression_ratio
        
        return {
            "params": params,
            "quality": quality,
            "compression": compression,
            "success": True
        }
        
    except Exception as e:
        return {
            "params": params,
            "quality": 0.0,
            "compression": 0.0,
            "success": False,
            "error": str(e)
        }


def main():
    """Fast quality optimization."""
    print("⚡ Fast Quality Optimization for BERT-768")
    print("=" * 50)
    
    # Load data
    dataset_manager = BERTDatasetManager("./data")
    vectors, _, _ = dataset_manager.get_bert_dataset(max_vectors=5000)
    
    print(f"📊 Loaded {len(vectors)} vectors for fast optimization")
    
    # Progressive quality-focused configurations
    # Start conservative and gradually increase quality focus
    quality_configs = [
        # Config 1: Fewer quantization levels, smaller anchor stride
        {
            'quantization_levels': 2,
            'n_subspaces': 6,
            'n_bits': 8,
            'anchor_stride': 24,
            'spiral_constant': 1.618
        },
        # Config 2: Even smaller anchor stride, higher precision
        {
            'quantization_levels': 2,
            'n_subspaces': 4,
            'n_bits': 9,
            'anchor_stride': 16,
            'spiral_constant': 1.5
        },
        # Config 3: Minimal quantization, maximum precision
        {
            'quantization_levels': 1,
            'n_subspaces': 4,
            'n_bits': 8,
            'anchor_stride': 20,
            'spiral_constant': 1.618
        },
        # Config 4: Aggressive quality focus
        {
            'quantization_levels': 1,
            'n_subspaces': 3,
            'n_bits': 9,
            'anchor_stride': 12,
            'spiral_constant': 1.8
        },
        # Config 5: Extreme quality (may sacrifice compression)
        {
            'quantization_levels': 1,
            'n_subspaces': 2,
            'n_bits': 8,
            'anchor_stride': 8,
            'spiral_constant': 1.618
        }
    ]
    
    print(f"🧪 Testing {len(quality_configs)} quality-focused configurations...")
    
    results = []
    best_balanced = None
    best_quality = None
    
    for i, params in enumerate(quality_configs):
        print(f"\n🔧 Config {i+1}: {params}")
        
        start_time = time.time()
        result = test_single_config(vectors, params, test_size=1500)
        test_time = time.time() - start_time
        
        if result['success']:
            quality = result['quality']
            compression = result['compression']
            
            # Check targets
            quality_met = quality >= 0.25
            compression_met = compression >= 0.668
            both_met = quality_met and compression_met
            
            print(f"  Quality: {quality:.3f} {'✅' if quality_met else '❌'} (≥0.25)")
            print(f"  Compression: {compression:.1%} {'✅' if compression_met else '❌'} (≥66.8%)")
            print(f"  Both targets: {'✅' if both_met else '❌'}")
            print(f"  Test time: {test_time:.1f}s")
            
            result['quality_met'] = bool(quality_met)
            result['compression_met'] = bool(compression_met)
            result['both_met'] = bool(both_met)
            result['test_time'] = float(test_time)
            result['quality'] = float(result['quality'])
            result['compression'] = float(result['compression'])
            
            # Track best results
            if both_met and (best_balanced is None or quality > best_balanced['quality']):
                best_balanced = result
            
            if best_quality is None or quality > best_quality['quality']:
                best_quality = result
            
            results.append(result)
            
            # Early exit if we achieve both targets
            if both_met:
                print(f"🎉 Found configuration meeting both targets!")
                break
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Results summary
    print(f"\n🏆 QUALITY OPTIMIZATION RESULTS")
    print("=" * 50)
    
    if best_balanced:
        print(f"✅ BEST BALANCED CONFIGURATION:")
        print(f"  Quality: {best_balanced['quality']:.3f}")
        print(f"  Compression: {best_balanced['compression']:.1%}")
        print(f"  Parameters: {best_balanced['params']}")
        
        # Save best configuration
        final_config = {
            "optimization_type": "quality_improvement",
            "target_quality": 0.25,
            "min_compression": 0.668,
            "best_balanced_result": best_balanced,
            "all_results": results,
            "timestamp": time.time(),
            "status": "SUCCESS - Both targets achieved"
        }
        
    elif best_quality:
        print(f"⚠️ BEST QUALITY CONFIGURATION (compression may be lower):")
        print(f"  Quality: {best_quality['quality']:.3f}")
        print(f"  Compression: {best_quality['compression']:.1%}")
        print(f"  Parameters: {best_quality['params']}")
        
        final_config = {
            "optimization_type": "quality_improvement",
            "target_quality": 0.25,
            "min_compression": 0.668,
            "best_quality_result": best_quality,
            "all_results": results,
            "timestamp": time.time(),
            "status": "PARTIAL - Quality improved but targets not both met"
        }
    else:
        print("❌ No successful configurations found")
        final_config = {
            "optimization_type": "quality_improvement",
            "status": "FAILED - No successful configurations",
            "timestamp": time.time()
        }
    
    # Save results
    output_path = Path("./data/fast_quality_optimization.json")
    with open(output_path, 'w') as f:
        json.dump(final_config, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Final assessment
    if best_balanced:
        print(f"\n🎯 QUALITY OPTIMIZATION: ✅ SUCCESS")
        print(f"   Achieved both quality ≥0.25 AND compression ≥66.8%")
        print(f"   Ready for production deployment!")
    elif best_quality and best_quality['quality'] >= 0.20:
        print(f"\n🎯 QUALITY OPTIMIZATION: ⚠️ PARTIAL SUCCESS")  
        print(f"   Improved quality to {best_quality['quality']:.3f}")
        print(f"   May need compression adjustment for production")
    else:
        print(f"\n🎯 QUALITY OPTIMIZATION: ❌ NEEDS MORE WORK")
        print(f"   Consider alternative approaches or parameter ranges")


if __name__ == "__main__":
    main()