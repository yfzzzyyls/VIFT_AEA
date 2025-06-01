#!/usr/bin/env python3
"""
Test and compare different relative pose conversion methods.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_multiply(q1, q2):
    """Multiply two quaternions in XYZW format."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([x, y, z, w])


def quaternion_inverse(q):
    """Compute quaternion inverse for XYZW format."""
    x, y, z, w = q
    norm_sq = w*w + x*x + y*y + z*z
    return np.array([-x, -y, -z, w]) / norm_sq


def quaternion_to_matrix(q):
    """Convert quaternion to rotation matrix (XYZW convention)."""
    q = q / (np.linalg.norm(q) + 1e-8)
    x, y, z, w = q
    
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])


def convert_absolute_to_relative_wrong(poses):
    """WRONG: Current implementation that doesn't transform translation."""
    seq_len = poses.shape[0]
    relative_poses = np.zeros_like(poses)
    
    relative_poses[0, :3] = [0, 0, 0]
    relative_poses[0, 3:] = [0, 0, 0, 1]
    
    for i in range(1, seq_len):
        prev_trans = poses[i-1, :3]
        prev_rot = poses[i-1, 3:]
        curr_trans = poses[i, :3]
        curr_rot = poses[i, 3:]
        
        # WRONG: This is in world coordinates, not previous frame
        trans_diff = curr_trans - prev_trans
        
        prev_rot_inv = quaternion_inverse(prev_rot)
        rel_rot = quaternion_multiply(prev_rot_inv, curr_rot)
        
        relative_poses[i, :3] = trans_diff
        relative_poses[i, 3:] = rel_rot / np.linalg.norm(rel_rot)
    
    return relative_poses


def convert_absolute_to_relative_correct(poses):
    """CORRECT: Transform translation into previous frame's coordinate system."""
    seq_len = poses.shape[0]
    relative_poses = np.zeros_like(poses)
    
    relative_poses[0, :3] = [0, 0, 0]
    relative_poses[0, 3:] = [0, 0, 0, 1]
    
    for i in range(1, seq_len):
        prev_trans = poses[i-1, :3]
        prev_rot = poses[i-1, 3:]
        curr_trans = poses[i, :3]
        curr_rot = poses[i, 3:]
        
        # Compute translation difference in world coordinates
        trans_diff_world = curr_trans - prev_trans
        
        # Transform to previous frame's coordinate system
        prev_rot_matrix = quaternion_to_matrix(prev_rot)
        trans_diff_local = prev_rot_matrix.T @ trans_diff_world
        
        # Relative rotation
        prev_rot_inv = quaternion_inverse(prev_rot)
        rel_rot = quaternion_multiply(prev_rot_inv, curr_rot)
        
        relative_poses[i, :3] = trans_diff_local
        relative_poses[i, 3:] = rel_rot / np.linalg.norm(rel_rot)
    
    return relative_poses


def test_conversion():
    """Test with a simple example."""
    print("Testing relative pose conversion methods\n")
    
    # Create test scenario: moving forward then turning right
    poses = np.array([
        # Position (x,y,z), Quaternion (x,y,z,w)
        [0, 0, 0, 0, 0, 0, 1],  # Origin, no rotation
        [1, 0, 0, 0, 0, 0, 1],  # Move 1m forward (along x)
        [1, 0, 0, 0, 0, 0.7071, 0.7071],  # Turn 90 degrees around z
        [1, 1, 0, 0, 0, 0.7071, 0.7071],  # Move 1m right (along y in world)
    ])
    
    print("Absolute poses:")
    for i, pose in enumerate(poses):
        print(f"  Frame {i}: pos={pose[:3]}, quat={pose[3:]}")
    
    # Test wrong conversion
    print("\n\nWRONG METHOD (current implementation):")
    relative_wrong = convert_absolute_to_relative_wrong(poses.copy())
    relative_wrong[:, :3] *= 100  # Scale to cm
    
    for i, pose in enumerate(relative_wrong):
        print(f"  Frame {i}: trans={pose[:3]}cm, quat={pose[3:]}")
    
    print("\nAnalysis of wrong method:")
    print(f"  Frame 3 translation: {relative_wrong[3, :3]}cm")
    print("  -> This shows [0, 100, 0] which is in WORLD coordinates")
    print("  -> But from frame 2's perspective (rotated 90°), this should be [100, 0, 0]")
    
    # Test correct conversion
    print("\n\nCORRECT METHOD (with coordinate transformation):")
    relative_correct = convert_absolute_to_relative_correct(poses.copy())
    relative_correct[:, :3] *= 100  # Scale to cm
    
    for i, pose in enumerate(relative_correct):
        print(f"  Frame {i}: trans={pose[:3]}cm, quat={pose[3:]}")
    
    print("\nAnalysis of correct method:")
    print(f"  Frame 3 translation: {relative_correct[3, :3]}cm")
    print("  -> This shows [100, 0, 0] which is correct in frame 2's LOCAL coordinates")
    print("  -> The robot moved forward in its own frame after turning")
    
    # Impact analysis
    print("\n\nIMPACT ON TRAINING:")
    print("The wrong method causes:")
    print("1. Translations are in world coordinates, not local frame")
    print("2. Model learns incorrect motion patterns")
    print("3. Especially problematic when robot rotates")
    print("4. This explains why rotation error is high (0.1743° vs 0.0739°)")


if __name__ == "__main__":
    test_conversion()