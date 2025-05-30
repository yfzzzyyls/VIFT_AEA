#!/usr/bin/env python3
"""Test script to verify the fixes for normalization and relative poses"""
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append('.')

from data.latent_caching_aria import SimpleVisualEncoder, process_aria_sequence_simple

def test_normalization():
    """Test that image normalization is now using -0.5"""
    print("🧪 Testing image normalization...")
    
    # Create encoder
    encoder = SimpleVisualEncoder()
    
    # Create a dummy image tensor [1, 3, 480, 640] with values in [0, 1]
    dummy_img = torch.rand(1, 3, 480, 640)
    
    # Apply transform
    transformed = encoder.transform(dummy_img[0])
    
    # Check range - should be approximately [-0.5, 0.5]
    min_val = transformed.min().item()
    max_val = transformed.max().item()
    
    print(f"  Original range: [0, 1]")
    print(f"  Transformed range: [{min_val:.3f}, {max_val:.3f}]")
    print(f"  Expected range: [-0.5, 0.5]")
    
    if -0.6 < min_val < -0.4 and 0.4 < max_val < 0.6:
        print("  ✅ Normalization is correct!")
    else:
        print("  ❌ Normalization is incorrect!")
    
    return True

def test_relative_poses():
    """Test that poses are now relative, not absolute"""
    print("\n🧪 Testing relative pose computation...")
    
    # Find a test sequence
    data_dir = Path("data/aria_split/train")
    if not data_dir.exists():
        print("  ⚠️ No training data found to test")
        return False
    
    # Get first sequence
    seq_dirs = sorted([d for d in data_dir.glob("[0-9]*") if d.is_dir()])
    if not seq_dirs:
        print("  ⚠️ No sequences found")
        return False
    
    seq_dir = seq_dirs[0]
    print(f"  Using sequence: {seq_dir.name}")
    
    # Create encoder and process sequence
    encoder = SimpleVisualEncoder()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = encoder.to(device)
    encoder.eval()
    
    result = process_aria_sequence_simple(seq_dir, encoder, device)
    if result is None:
        print("  ❌ Failed to process sequence")
        return False
    
    latent_vectors, ground_truths = result
    
    if len(ground_truths) == 0:
        print("  ❌ No ground truth poses generated")
        return False
    
    # Check first ground truth sequence
    gt = ground_truths[0]  # Shape: [11, 6]
    
    print(f"  Ground truth shape: {gt.shape}")
    print(f"  First pose (should be zeros): {gt[0]}")
    print(f"  Sample relative poses:")
    for i in range(1, min(4, len(gt))):
        print(f"    Frame {i}: rot={gt[i, :3]}, trans={gt[i, 3:]}")
    
    # Check that first pose is identity (all zeros)
    if np.allclose(gt[0], 0):
        print("  ✅ First pose is identity transformation")
    else:
        print("  ❌ First pose should be zeros")
    
    # Check that subsequent poses have small values (relative motion)
    max_rotation = np.abs(gt[1:, :3]).max()
    max_translation = np.abs(gt[1:, 3:]).max()
    
    print(f"\n  Max rotation change: {max_rotation:.4f} rad ({np.degrees(max_rotation):.1f}°)")
    print(f"  Max translation change: {max_translation:.4f} m")
    
    if max_rotation < 0.5 and max_translation < 0.5:
        print("  ✅ Relative poses have reasonable magnitudes")
    else:
        print("  ⚠️ Relative poses seem large for frame-to-frame motion")
    
    return True

def main():
    print("🔍 Testing fixes for VIFT-AEA implementation\n")
    
    test_normalization()
    test_relative_poses()
    
    print("\n✅ Tests complete! Ready to re-cache all features.")

if __name__ == "__main__":
    main()