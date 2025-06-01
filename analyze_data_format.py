#!/usr/bin/env python3
import numpy as np
import sys

def analyze_poses(file_path, label):
    poses = np.load(file_path)
    print(f"\n{label}:")
    print(f"Shape: {poses.shape}")
    
    # Check first pose
    print(f"\nFirst pose (should be at origin for relative):")
    print(f"  Translation: {poses[0, :3]}")
    print(f"  Quaternion: {poses[0, 3:]}")
    print(f"  Is at origin? Trans={np.allclose(poses[0, :3], 0, atol=1e-6)}, Quat={np.allclose(poses[0, 3:], [0,0,0,1], atol=1e-6)}")
    
    # Check a few more poses
    print(f"\nNext few poses:")
    for i in range(1, min(5, len(poses))):
        print(f"  Pose {i}: T={poses[i, :3]}, Q={poses[i, 3:]}")
        trans_norm = np.linalg.norm(poses[i, :3])
        print(f"    Translation norm: {trans_norm:.4f}")
        
    # Check if translations look like meters or centimeters
    max_trans = np.max(np.abs(poses[:, :3]))
    print(f"\nMax absolute translation value: {max_trans:.6f}")
    if max_trans < 1.0:
        print("  -> Looks like METERS (needs scaling)")
    else:
        print("  -> Looks like CENTIMETERS (already scaled)")

# Analyze different data versions
print("Analyzing pose data formats...")

# Original pretrained data
analyze_poses("aria_latent_data_pretrained/train/0_gt.npy", "Original pretrained data")

# Fixed data (if exists)
try:
    analyze_poses("aria_latent_data_fixed/train/0_gt.npy", "Fixed preprocessed data")
except:
    print("\nFixed data not found")