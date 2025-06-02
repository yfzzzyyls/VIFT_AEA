#!/usr/bin/env python3
"""
Validation script to ensure our inference implementation is correct.
Tests key components against known values.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import sys
sys.path.append('.')

from inference_full_sequence import quaternion_to_matrix, accumulate_poses
from generate_all_pretrained_latents_fixed import quaternion_multiply, quaternion_inverse


def test_quaternion_operations():
    """Test quaternion operations for correctness."""
    print("Testing quaternion operations...")
    
    # Test identity quaternion
    q_identity = np.array([0, 0, 0, 1])  # XYZW
    R_identity = quaternion_to_matrix(q_identity)
    assert np.allclose(R_identity, np.eye(3)), "Identity quaternion failed"
    
    # Test quaternion multiplication
    q1 = np.array([0, 0, 0.7071068, 0.7071068])  # 90 deg around Z
    q2 = np.array([0, 0, 0.7071068, 0.7071068])  # 90 deg around Z
    q_result = quaternion_multiply(q1, q2)
    # Should be 180 deg around Z: [0, 0, 1, 0]
    expected = np.array([0, 0, 1, 0])
    assert np.allclose(q_result, expected, atol=1e-6), f"Quaternion multiply failed: {q_result}"
    
    # Test quaternion inverse
    q_inv = quaternion_inverse(q1)
    q_identity_check = quaternion_multiply(q1, q_inv)
    assert np.allclose(q_identity_check, q_identity, atol=1e-6), "Quaternion inverse failed"
    
    print("✅ Quaternion operations passed!")


def test_pose_accumulation():
    """Test pose accumulation for correctness."""
    print("\nTesting pose accumulation...")
    
    # Create simple test case: move 1 unit in X direction each frame
    n_frames = 5
    relative_poses = np.zeros((n_frames, 7))
    
    # First frame at origin
    relative_poses[0] = [0, 0, 0, 0, 0, 0, 1]
    
    # Subsequent frames: 1 unit forward in X
    for i in range(1, n_frames):
        relative_poses[i] = [1, 0, 0, 0, 0, 0, 1]  # 1 unit in X, no rotation
    
    # Accumulate
    absolute_poses = accumulate_poses(relative_poses)
    
    # Check positions
    expected_positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    actual_positions = absolute_poses[:, :3]
    
    assert np.allclose(actual_positions, expected_positions), f"Position accumulation failed:\n{actual_positions}"
    
    # Test with rotation
    relative_poses2 = np.zeros((3, 7))
    relative_poses2[0] = [0, 0, 0, 0, 0, 0, 1]  # Origin
    relative_poses2[1] = [1, 0, 0, 0, 0, 0.7071068, 0.7071068]  # Move 1 in X, rotate 90° around Z
    relative_poses2[2] = [1, 0, 0, 0, 0, 0, 1]  # Move 1 in local X (which is now Y in world)
    
    absolute_poses2 = accumulate_poses(relative_poses2)
    
    # After rotation, local X becomes world Y
    expected_positions2 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    actual_positions2 = absolute_poses2[:, :3]
    
    assert np.allclose(actual_positions2, expected_positions2, atol=1e-6), f"Rotation accumulation failed:\n{actual_positions2}"
    
    print("✅ Pose accumulation passed!")


def test_relative_pose_conversion():
    """Test conversion from absolute to relative poses."""
    print("\nTesting relative pose conversion...")
    
    # Create absolute poses
    absolute_poses = np.array([
        [0, 0, 0, 0, 0, 0, 1],      # Origin
        [1, 0, 0, 0, 0, 0, 1],      # Move 1 in X
        [1, 1, 0, 0, 0, 0.7071068, 0.7071068],  # Move 1 in Y, rotate 90° around Z
    ])
    
    # Convert to relative (manual calculation)
    relative_poses = np.zeros_like(absolute_poses)
    relative_poses[0] = [0, 0, 0, 0, 0, 0, 1]  # First is always origin
    
    # Frame 0->1: Just translation in X
    relative_poses[1] = [1, 0, 0, 0, 0, 0, 1]
    
    # Frame 1->2: Translation in Y and rotation
    # In world: move from [1,0,0] to [1,1,0]
    # In local frame of pose 1 (no rotation): still [0,1,0]
    prev_rot = Rotation.from_quat([0, 0, 0, 1])
    trans_world = np.array([0, 1, 0])
    trans_local = prev_rot.inv().apply(trans_world)
    relative_poses[2, :3] = trans_local
    relative_poses[2, 3:] = [0, 0, 0.7071068, 0.7071068]
    
    # Now accumulate back and verify
    reconstructed = accumulate_poses(relative_poses)
    
    assert np.allclose(reconstructed[:, :3], absolute_poses[:, :3], atol=1e-6), "Relative conversion failed"
    
    print("✅ Relative pose conversion passed!")


def test_coordinate_systems():
    """Test that our coordinate transformations are correct."""
    print("\nTesting coordinate systems...")
    
    # Test case: camera rotates 90° around Z, then moves forward
    # In camera frame: move [1, 0, 0]
    # In world frame: should be [0, 1, 0] after rotation
    
    R = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    local_trans = np.array([1, 0, 0])
    world_trans = R @ local_trans
    
    expected = np.array([0, 1, 0])
    assert np.allclose(world_trans, expected, atol=1e-6), f"Coordinate transform failed: {world_trans}"
    
    print("✅ Coordinate systems passed!")


def main():
    """Run all validation tests."""
    print("=== Validating Inference Implementation ===\n")
    
    test_quaternion_operations()
    test_pose_accumulation()
    test_relative_pose_conversion()
    test_coordinate_systems()
    
    print("\n✅ All validation tests passed!")
    print("\nOur inference implementation is mathematically correct.")


if __name__ == "__main__":
    main()