#!/usr/bin/env python3
"""
Verify that the quaternion fix is correct by testing the conversion functions.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import sys
sys.path.append('.')

# Import the fixed functions
from train_pretrained_relative import quaternion_multiply, quaternion_inverse, convert_absolute_to_relative


def test_quaternion_operations():
    """Test quaternion operations with known values."""
    print("Testing Quaternion Operations (XYZW format)")
    print("=" * 50)
    
    # Test 1: Identity quaternion
    q_identity = np.array([0, 0, 0, 1])  # XYZW format
    q_inv = quaternion_inverse(q_identity)
    print(f"\nIdentity quaternion: {q_identity}")
    print(f"Inverse of identity: {q_inv}")
    print(f"Should be identity: {np.allclose(q_inv, q_identity)}")
    
    # Test 2: Multiply quaternion by its inverse
    q1 = np.array([0.5, 0.5, 0.5, 0.5])  # XYZW format, normalized
    q1_inv = quaternion_inverse(q1)
    result = quaternion_multiply(q1, q1_inv)
    print(f"\nQuaternion: {q1}")
    print(f"Its inverse: {q1_inv}")
    print(f"q * q^-1 = {result}")
    print(f"Should be identity: {np.allclose(result, [0, 0, 0, 1])}")
    
    # Test 3: Compare with scipy
    print("\n\nComparing with scipy.spatial.transform.Rotation")
    print("=" * 50)
    
    # Create two rotations
    rot1 = Rotation.from_euler('xyz', [30, 45, 60], degrees=True)
    rot2 = Rotation.from_euler('xyz', [10, 20, 30], degrees=True)
    
    # Get quaternions in XYZW format
    q1_scipy = rot1.as_quat()  # scipy returns XYZW
    q2_scipy = rot2.as_quat()  # scipy returns XYZW
    
    # Compute relative rotation using scipy
    rot_relative_scipy = rot1.inv() * rot2
    q_relative_scipy = rot_relative_scipy.as_quat()
    
    # Compute using our functions
    q1_inv_ours = quaternion_inverse(q1_scipy)
    q_relative_ours = quaternion_multiply(q1_inv_ours, q2_scipy)
    
    print(f"\nq1 (scipy): {q1_scipy}")
    print(f"q2 (scipy): {q2_scipy}")
    print(f"\nRelative quaternion (scipy): {q_relative_scipy}")
    print(f"Relative quaternion (ours):  {q_relative_ours}")
    print(f"Match: {np.allclose(q_relative_scipy, q_relative_ours, atol=1e-6)}")
    
    # Convert to angle
    angle_scipy = 2 * np.arccos(np.clip(q_relative_scipy[3], -1, 1)) * 180 / np.pi
    angle_ours = 2 * np.arccos(np.clip(q_relative_ours[3], -1, 1)) * 180 / np.pi
    print(f"\nRelative angle (scipy): {angle_scipy:.2f}°")
    print(f"Relative angle (ours):  {angle_ours:.2f}°")


def test_relative_pose_conversion():
    """Test the full relative pose conversion."""
    print("\n\nTesting Relative Pose Conversion")
    print("=" * 50)
    
    # Create sample absolute poses
    poses = np.zeros((3, 7))
    
    # Pose 0: At origin
    poses[0, :3] = [0, 0, 0]
    poses[0, 3:] = [0, 0, 0, 1]  # Identity quaternion in XYZW
    
    # Pose 1: Small movement
    poses[1, :3] = [0.1, 0.05, 0.02]  # 10cm, 5cm, 2cm
    rot1 = Rotation.from_euler('xyz', [2, 1, 3], degrees=True)
    poses[1, 3:] = rot1.as_quat()  # XYZW format
    
    # Pose 2: Another small movement
    poses[2, :3] = [0.15, 0.08, 0.04]
    rot2 = Rotation.from_euler('xyz', [3, 2, 5], degrees=True)
    poses[2, 3:] = rot2.as_quat()  # XYZW format
    
    # Convert to relative poses
    relative_poses = convert_absolute_to_relative(poses)
    
    print("\nAbsolute poses:")
    for i, pose in enumerate(poses):
        print(f"  Pose {i}: trans={pose[:3]}, quat={pose[3:]}")
    
    print("\nRelative poses:")
    for i, pose in enumerate(relative_poses):
        angle = 2 * np.arccos(np.clip(pose[6], -1, 1)) * 180 / np.pi
        print(f"  Pose {i}: trans={pose[:3]}, quat={pose[3:]}, angle={angle:.2f}°")
    
    # Verify first pose is at origin
    print(f"\nFirst relative pose at origin: {np.allclose(relative_poses[0, :3], [0, 0, 0])}")
    print(f"First relative pose is identity: {np.allclose(relative_poses[0, 3:], [0, 0, 0, 1])}")
    
    # Check relative translations
    expected_trans_1 = poses[1, :3] - poses[0, :3]
    expected_trans_2 = poses[2, :3] - poses[1, :3]
    
    print(f"\nRelative translation 0->1 correct: {np.allclose(relative_poses[1, :3], expected_trans_1)}")
    print(f"Relative translation 1->2 correct: {np.allclose(relative_poses[2, :3], expected_trans_2)}")


def test_with_real_data():
    """Test with actual data from the dataset."""
    print("\n\nTesting with Real Data Sample")
    print("=" * 50)
    
    # Load a sample from the dataset
    import os
    data_dir = "aria_latent_data_pretrained/train"
    
    if os.path.exists(os.path.join(data_dir, "0_gt.npy")):
        poses = np.load(os.path.join(data_dir, "0_gt.npy"))
        print(f"\nLoaded poses shape: {poses.shape}")
        
        # Show first few poses
        print("\nFirst 3 absolute poses:")
        for i in range(min(3, len(poses))):
            print(f"  Pose {i}: trans={poses[i, :3]}, quat={poses[i, 3:]}")
        
        # Convert to relative
        relative_poses = convert_absolute_to_relative(poses)
        
        print("\nFirst 3 relative poses:")
        for i in range(min(3, len(relative_poses))):
            angle = 2 * np.arccos(np.clip(relative_poses[i, 6], -1, 1)) * 180 / np.pi
            print(f"  Pose {i}: trans={relative_poses[i, :3]}, quat={relative_poses[i, 3:]}, angle={angle:.2f}°")
        
        # Check that relative rotations are small (typical for sequential frames)
        angles = []
        for i in range(1, len(relative_poses)):
            angle = 2 * np.arccos(np.clip(relative_poses[i, 6], -1, 1)) * 180 / np.pi
            angles.append(angle)
        
        angles = np.array(angles)
        print(f"\nRelative rotation statistics:")
        print(f"  Mean angle: {np.mean(angles):.2f}°")
        print(f"  Max angle: {np.max(angles):.2f}°")
        print(f"  Min angle: {np.min(angles):.2f}°")
        print(f"  Std angle: {np.std(angles):.2f}°")
        
        # These should be small for consecutive frames
        if np.mean(angles) < 5.0:  # Less than 5 degrees on average
            print("\n✓ Relative rotations are reasonably small - conversion looks correct!")
        else:
            print("\n✗ WARNING: Relative rotations seem too large - there might be an issue!")
    else:
        print("Could not find sample data file.")


if __name__ == "__main__":
    test_quaternion_operations()
    test_relative_pose_conversion()
    test_with_real_data()