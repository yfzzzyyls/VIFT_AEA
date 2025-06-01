#!/usr/bin/env python3
"""
Debug script to investigate why stride=1 is yielding poor results.
"""

import numpy as np
import json
import os
from pathlib import Path
import torch
from scipy.spatial.transform import Rotation
import sys

# Import the quaternion functions
sys.path.append('.')
from train_pretrained_relative import quaternion_multiply, quaternion_inverse, convert_absolute_to_relative


def analyze_pose_data():
    """Analyze the pose data from multiple perspectives."""
    
    print("="*80)
    print("STRIDE=1 DEBUGGING ANALYSIS")
    print("="*80)
    
    # 1. Check the original processed data
    print("\n1. ANALYZING ORIGINAL PROCESSED DATA")
    print("-"*40)
    
    # Load a sample sequence
    seq_path = Path("data/aria_processed/00")
    if seq_path.exists():
        with open(seq_path / "poses.json", 'r') as f:
            poses_data = json.load(f)
        
        print(f"Number of poses in sequence: {len(poses_data)}")
        print(f"First 3 poses:")
        for i in range(min(3, len(poses_data))):
            pose = poses_data[i]
            print(f"  Pose {i}: trans={pose['translation']}, euler={pose['rotation_euler']}")
        
        # Check pose format
        print(f"\nPose format analysis:")
        print(f"  Translation units: meters (need to scale by 100 for cm)")
        print(f"  Rotation format: Euler angles [roll, pitch, yaw]")
        
        # Analyze frame-to-frame motion
        print(f"\nFrame-to-frame motion (first 10 frames):")
        for i in range(1, min(10, len(poses_data))):
            prev_t = np.array(poses_data[i-1]['translation'])
            curr_t = np.array(poses_data[i]['translation'])
            trans_diff = np.linalg.norm(curr_t - prev_t) * 100  # Convert to cm
            
            prev_euler = poses_data[i-1]['rotation_euler']
            curr_euler = poses_data[i]['rotation_euler']
            euler_diff = np.array(curr_euler) - np.array(prev_euler)
            rot_diff = np.linalg.norm(euler_diff) * 180 / np.pi
            
            print(f"  Frame {i-1} -> {i}: trans={trans_diff:.4f}cm, rot={rot_diff:.4f}°")
    
    # 2. Check the generated latent data
    print("\n\n2. ANALYZING GENERATED LATENT DATA (aria_latent_data_pretrained)")
    print("-"*40)
    
    pretrained_dir = Path("aria_latent_data_pretrained/train")
    if pretrained_dir.exists():
        # Load first sample
        if (pretrained_dir / "0_gt.npy").exists():
            poses = np.load(pretrained_dir / "0_gt.npy")
            print(f"Sample 0 shape: {poses.shape}")
            print(f"First pose (should be absolute): {poses[0]}")
            print(f"Second pose: {poses[1]}")
            
            # Check if it's absolute or relative
            if np.linalg.norm(poses[0, :3]) > 1e-6:
                print("  -> Poses appear to be ABSOLUTE (not at origin)")
            else:
                print("  -> Poses appear to be RELATIVE (first at origin)")
    
    # 3. Check the fixed data
    print("\n\n3. ANALYZING FIXED DATA (aria_latent_data_fixed)")
    print("-"*40)
    
    fixed_dir = Path("aria_latent_data_fixed/train")
    if fixed_dir.exists():
        # Load first sample
        if (fixed_dir / "0_gt.npy").exists():
            poses = np.load(fixed_dir / "0_gt.npy")
            print(f"Sample 0 shape: {poses.shape}")
            print(f"First pose (should be at origin): {poses[0]}")
            print(f"Identity check: translation={np.linalg.norm(poses[0, :3]):.6f}, quat_diff={np.linalg.norm(poses[0, 3:] - [0,0,0,1]):.6f}")
            
            print(f"\nFrame-to-frame relative poses:")
            for i in range(1, min(5, len(poses))):
                trans_norm = np.linalg.norm(poses[i, :3])
                quat = poses[i, 3:]
                angle = 2 * np.arccos(np.clip(quat[3], -1, 1)) * 180 / np.pi
                print(f"  Frame {i}: trans={trans_norm:.4f}cm, rot={angle:.4f}°")
            
            # Verify quaternion normalization
            print(f"\nQuaternion normalization check:")
            for i in range(min(5, len(poses))):
                quat_norm = np.linalg.norm(poses[i, 3:])
                print(f"  Frame {i}: |q| = {quat_norm:.6f} (should be 1.0)")
    
    # 4. Test the conversion functions
    print("\n\n4. TESTING CONVERSION FUNCTIONS")
    print("-"*40)
    
    # Create test data
    test_poses = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Origin
        [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 1cm forward
        [0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0],  # Another 1cm forward, 1cm right
    ])
    
    print("Test absolute poses (meters):")
    for i, pose in enumerate(test_poses):
        print(f"  Pose {i}: {pose}")
    
    # Convert to relative
    relative_poses = convert_absolute_to_relative(test_poses.copy())
    relative_poses[:, :3] *= 100  # Scale to cm
    
    print("\nConverted relative poses (cm):")
    for i, pose in enumerate(relative_poses):
        print(f"  Pose {i}: {pose}")
    
    # Verify the conversion
    print("\nVerification:")
    print(f"  First pose at origin: {np.allclose(relative_poses[0], [0,0,0,0,0,0,1])}")
    print(f"  Second pose translation: {relative_poses[1, :3]} (expected [1.0, 0.0, 0.0])")
    print(f"  Third pose translation: {relative_poses[2, :3]} (expected [1.0, 1.0, 0.0])")
    
    # 5. Check for potential issues
    print("\n\n5. POTENTIAL ISSUES ANALYSIS")
    print("-"*40)
    
    print("Checking for common issues:")
    
    # Issue 1: Double conversion
    print("\n[Issue 1] Double conversion check:")
    if fixed_dir.exists() and (fixed_dir / "0_gt.npy").exists():
        poses = np.load(fixed_dir / "0_gt.npy")
        # Try converting again
        double_converted = convert_absolute_to_relative(poses)
        if not np.allclose(double_converted[0], [0,0,0,0,0,0,1]):
            print("  ⚠️  WARNING: Double conversion detected! Data might be converted twice.")
        else:
            print("  ✓ No double conversion detected")
    
    # Issue 2: Quaternion format
    print("\n[Issue 2] Quaternion format check:")
    print("  Expected format: XYZW")
    print("  Identity quaternion: [0, 0, 0, 1]")
    if fixed_dir.exists() and (fixed_dir / "0_gt.npy").exists():
        poses = np.load(fixed_dir / "0_gt.npy")
        identity_quat = poses[0, 3:]
        if np.allclose(identity_quat, [1, 0, 0, 0]):
            print("  ⚠️  WARNING: Might be using WXYZ format instead of XYZW!")
        elif np.allclose(identity_quat, [0, 0, 0, 1]):
            print("  ✓ Correct XYZW format")
        else:
            print(f"  ⚠️  WARNING: Unexpected identity quaternion: {identity_quat}")
    
    # Issue 3: Stride effect
    print("\n[Issue 3] Stride effect analysis:")
    print("  With stride=1: Every frame becomes a training sample")
    print("  With stride=10: Every 10th frame becomes a training sample")
    print("  Potential issue: Overlapping windows might cause data leakage")
    
    # Count samples with different strides
    if seq_path.exists():
        with open(seq_path / "poses.json", 'r') as f:
            poses_data = json.load(f)
        num_frames = len(poses_data)
        window_size = 11
        
        samples_stride_1 = (num_frames - window_size + 1) // 1
        samples_stride_10 = (num_frames - window_size + 1) // 10
        
        print(f"  Sequence length: {num_frames} frames")
        print(f"  Samples with stride=1: {samples_stride_1}")
        print(f"  Samples with stride=10: {samples_stride_10}")
        print(f"  Ratio: {samples_stride_1/samples_stride_10:.1f}x more data")
    
    # Issue 4: Scale consistency
    print("\n[Issue 4] Scale consistency check:")
    if pretrained_dir.exists() and fixed_dir.exists():
        if (pretrained_dir / "0_gt.npy").exists() and (fixed_dir / "0_gt.npy").exists():
            pretrained_poses = np.load(pretrained_dir / "0_gt.npy")
            fixed_poses = np.load(fixed_dir / "0_gt.npy")
            
            # Check translation scales
            if np.linalg.norm(pretrained_poses[0, :3]) > 1e-6:  # If absolute
                print(f"  Pretrained data scale: meters (typical range 0-10m)")
            
            print(f"  Fixed data scale: cm (typical range 0-10cm per frame)")
            print(f"  Fixed data avg translation: {np.mean([np.linalg.norm(p[:3]) for p in fixed_poses[1:]]):.4f}cm")
    
    # 6. Recommendations
    print("\n\n6. RECOMMENDATIONS")
    print("-"*40)
    print("Based on the analysis:")
    print("1. Verify that generate_all_pretrained_latents.py with stride=1 produces relative poses")
    print("2. Check if the training script is applying conversion twice")
    print("3. Ensure quaternion format consistency (XYZW) throughout")
    print("4. Consider if overlapping windows with stride=1 cause issues")
    print("5. Verify that the model architecture handles the increased data volume")


if __name__ == "__main__":
    analyze_pose_data()