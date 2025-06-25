"""
Utility functions for pose manipulation and trajectory computation.
"""

import torch
import numpy as np
from typing import Union, Tuple


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z] or [x, y, z, w]
        
    Returns:
        R: 3x3 rotation matrix
    """
    # Normalize quaternion
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-8:
        # Return identity rotation for zero quaternion
        return np.eye(3)
    q = q / q_norm
    
    # Assume quaternion format [x, y, z, w] (following PyTorch3D convention)
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def integrate_poses(relative_poses: np.ndarray) -> np.ndarray:
    """
    Integrate relative poses to get absolute trajectory.
    
    Args:
        relative_poses: [T, 7] array of relative poses (3 translation + 4 quaternion)
        
    Returns:
        trajectory: [T+1, 7] array of absolute poses
    """
    T = relative_poses.shape[0]
    trajectory = np.zeros((T + 1, 7))
    
    # Initialize with identity pose
    trajectory[0, :3] = [0, 0, 0]  # position
    trajectory[0, 3:] = [0, 0, 0, 1]  # quaternion [x, y, z, w]
    
    # Current pose
    current_pos = np.array([0.0, 0.0, 0.0])
    current_rot = np.eye(3)
    
    for i in range(T):
        # Extract relative pose
        rel_trans = relative_poses[i, :3]
        rel_quat = relative_poses[i, 3:]
        
        # Convert quaternion to rotation matrix
        rel_rot = quaternion_to_rotation_matrix(rel_quat)
        
        # Apply relative transformation
        current_pos = current_pos + current_rot @ rel_trans
        current_rot = current_rot @ rel_rot
        
        # Store absolute pose
        trajectory[i + 1, :3] = current_pos
        
        # Convert rotation matrix back to quaternion
        # Using simple method (can be improved with more robust conversion)
        trace = np.trace(current_rot)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (current_rot[2, 1] - current_rot[1, 2]) * s
            y = (current_rot[0, 2] - current_rot[2, 0]) * s
            z = (current_rot[1, 0] - current_rot[0, 1]) * s
        else:
            if current_rot[0, 0] > current_rot[1, 1] and current_rot[0, 0] > current_rot[2, 2]:
                s = 2.0 * np.sqrt(1.0 + current_rot[0, 0] - current_rot[1, 1] - current_rot[2, 2])
                w = (current_rot[2, 1] - current_rot[1, 2]) / s
                x = 0.25 * s
                y = (current_rot[0, 1] + current_rot[1, 0]) / s
                z = (current_rot[0, 2] + current_rot[2, 0]) / s
            elif current_rot[1, 1] > current_rot[2, 2]:
                s = 2.0 * np.sqrt(1.0 + current_rot[1, 1] - current_rot[0, 0] - current_rot[2, 2])
                w = (current_rot[0, 2] - current_rot[2, 0]) / s
                x = (current_rot[0, 1] + current_rot[1, 0]) / s
                y = 0.25 * s
                z = (current_rot[1, 2] + current_rot[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + current_rot[2, 2] - current_rot[0, 0] - current_rot[1, 1])
                w = (current_rot[1, 0] - current_rot[0, 1]) / s
                x = (current_rot[0, 2] + current_rot[2, 0]) / s
                y = (current_rot[1, 2] + current_rot[2, 1]) / s
                z = 0.25 * s
        
        trajectory[i + 1, 3:] = [x, y, z, w]
    
    return trajectory


def poses_to_trajectory(poses: np.ndarray) -> np.ndarray:
    """
    Convert pose sequence to trajectory points.
    
    Args:
        poses: [T, 7] array of poses
        
    Returns:
        trajectory: [T, 3] array of positions
    """
    return poses[:, :3]


def compute_relative_pose(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """
    Compute relative pose from pose1 to pose2.
    
    Args:
        pose1: [7] array (3 translation + 4 quaternion)
        pose2: [7] array (3 translation + 4 quaternion)
        
    Returns:
        relative_pose: [7] array
    """
    # Extract components
    t1, q1 = pose1[:3], pose1[3:]
    t2, q2 = pose2[:3], pose2[3:]
    
    # Convert to rotation matrices
    R1 = quaternion_to_rotation_matrix(q1)
    R2 = quaternion_to_rotation_matrix(q2)
    
    # Compute relative transformation
    R_rel = R1.T @ R2
    t_rel = R1.T @ (t2 - t1)
    
    # Convert back to quaternion
    # (Using same conversion as above)
    trace = np.trace(R_rel)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R_rel[2, 1] - R_rel[1, 2]) * s
        y = (R_rel[0, 2] - R_rel[2, 0]) * s
        z = (R_rel[1, 0] - R_rel[0, 1]) * s
    else:
        if R_rel[0, 0] > R_rel[1, 1] and R_rel[0, 0] > R_rel[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R_rel[0, 0] - R_rel[1, 1] - R_rel[2, 2])
            w = (R_rel[2, 1] - R_rel[1, 2]) / s
            x = 0.25 * s
            y = (R_rel[0, 1] + R_rel[1, 0]) / s
            z = (R_rel[0, 2] + R_rel[2, 0]) / s
        elif R_rel[1, 1] > R_rel[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R_rel[1, 1] - R_rel[0, 0] - R_rel[2, 2])
            w = (R_rel[0, 2] - R_rel[2, 0]) / s
            x = (R_rel[0, 1] + R_rel[1, 0]) / s
            y = 0.25 * s
            z = (R_rel[1, 2] + R_rel[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R_rel[2, 2] - R_rel[0, 0] - R_rel[1, 1])
            w = (R_rel[1, 0] - R_rel[0, 1]) / s
            x = (R_rel[0, 2] + R_rel[2, 0]) / s
            y = (R_rel[1, 2] + R_rel[2, 1]) / s
            z = 0.25 * s
    
    q_rel = np.array([x, y, z, w])
    
    return np.concatenate([t_rel, q_rel])


def align_trajectories(pred_traj: np.ndarray, gt_traj: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Align predicted trajectory to ground truth using Umeyama alignment.
    
    Args:
        pred_traj: [N, 3] predicted positions
        gt_traj: [N, 3] ground truth positions
        
    Returns:
        aligned_traj: [N, 3] aligned predicted trajectory
        scale: Scale factor used for alignment
    """
    # Center trajectories
    pred_mean = np.mean(pred_traj, axis=0)
    gt_mean = np.mean(gt_traj, axis=0)
    
    pred_centered = pred_traj - pred_mean
    gt_centered = gt_traj - gt_mean
    
    # Compute scale
    pred_norm = np.linalg.norm(pred_centered)
    gt_norm = np.linalg.norm(gt_centered)
    scale = gt_norm / pred_norm if pred_norm > 0 else 1.0
    
    # Apply SVD for rotation
    H = pred_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply transformation
    aligned_traj = scale * (pred_traj - pred_mean) @ R.T + gt_mean
    
    return aligned_traj, scale