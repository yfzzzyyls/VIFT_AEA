"""
Evaluation metrics for visual-inertial odometry.
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy.spatial.transform import Rotation as R
from .pose_utils import align_trajectories


def compute_absolute_trajectory_error(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray
) -> float:
    """
    Compute Absolute Trajectory Error (ATE).
    
    Args:
        pred_trajectory: Predicted trajectory [N, 3]
        gt_trajectory: Ground truth trajectory [N, 3]
        
    Returns:
        RMSE of trajectory positions
    """
    errors = np.linalg.norm(pred_trajectory - gt_trajectory, axis=1)
    return np.sqrt(np.mean(errors ** 2))


def compute_relative_pose_error(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    delta: int = 1
) -> Tuple[float, float]:
    """
    Compute Relative Pose Error (RPE) for translation and rotation.
    
    Args:
        pred_poses: Predicted poses [N, 7] (3 trans + 4 quat)
        gt_poses: Ground truth poses [N, 7]
        delta: Frame distance for computing relative poses
        
    Returns:
        (translation_rpe, rotation_rpe) in meters and degrees
    """
    trans_errors = []
    rot_errors = []
    
    for i in range(len(pred_poses) - delta):
        # Validate and normalize quaternions
        pred_q1 = pred_poses[i, 3:]
        pred_q2 = pred_poses[i + delta, 3:]
        gt_q1 = gt_poses[i, 3:]
        gt_q2 = gt_poses[i + delta, 3:]
        
        # Check for zero quaternions and replace with identity
        if np.linalg.norm(pred_q1) < 1e-8:
            pred_q1 = np.array([0, 0, 0, 1])
        else:
            pred_q1 = pred_q1 / np.linalg.norm(pred_q1)
            
        if np.linalg.norm(pred_q2) < 1e-8:
            pred_q2 = np.array([0, 0, 0, 1])
        else:
            pred_q2 = pred_q2 / np.linalg.norm(pred_q2)
            
        if np.linalg.norm(gt_q1) < 1e-8:
            gt_q1 = np.array([0, 0, 0, 1])
        else:
            gt_q1 = gt_q1 / np.linalg.norm(gt_q1)
            
        if np.linalg.norm(gt_q2) < 1e-8:
            gt_q2 = np.array([0, 0, 0, 1])
        else:
            gt_q2 = gt_q2 / np.linalg.norm(gt_q2)
        
        # Predicted relative pose
        pred_trans_rel = pred_poses[i + delta, :3] - pred_poses[i, :3]
        pred_rot1 = R.from_quat(pred_q1)
        pred_rot2 = R.from_quat(pred_q2)
        pred_rot_rel = pred_rot1.inv() * pred_rot2
        
        # Ground truth relative pose
        gt_trans_rel = gt_poses[i + delta, :3] - gt_poses[i, :3]
        gt_rot1 = R.from_quat(gt_q1)
        gt_rot2 = R.from_quat(gt_q2)
        gt_rot_rel = gt_rot1.inv() * gt_rot2
        
        # Translation error
        trans_error = np.linalg.norm(pred_trans_rel - gt_trans_rel)
        trans_errors.append(trans_error)
        
        # Rotation error (geodesic distance)
        rot_error = (pred_rot_rel.inv() * gt_rot_rel).magnitude()
        rot_errors.append(np.rad2deg(rot_error))
    
    return np.mean(trans_errors), np.mean(rot_errors)


def compute_trajectory_length(trajectory: np.ndarray) -> float:
    """
    Compute total trajectory length.
    
    Args:
        trajectory: Trajectory positions [N, 3]
        
    Returns:
        Total trajectory length in meters
    """
    segments = trajectory[1:] - trajectory[:-1]
    distances = np.linalg.norm(segments, axis=1)
    return np.sum(distances)


def compute_scale_error(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray
) -> float:
    """
    Compute scale error between predicted and ground truth trajectories.
    
    Args:
        pred_trajectory: Predicted trajectory [N, 3]
        gt_trajectory: Ground truth trajectory [N, 3]
        
    Returns:
        Scale error (1.0 means perfect scale)
    """
    pred_length = compute_trajectory_length(pred_trajectory)
    gt_length = compute_trajectory_length(gt_trajectory)
    
    if gt_length > 0:
        return pred_length / gt_length
    else:
        return 1.0


def integrate_poses_to_trajectory(poses: np.ndarray) -> np.ndarray:
    """
    Integrate relative poses to get absolute trajectory.
    
    Args:
        poses: Relative poses [N, 7] (3 trans + 4 quat)
        
    Returns:
        Absolute trajectory positions [N+1, 3]
    """
    trajectory = [np.zeros(3)]  # Start at origin
    current_pos = np.zeros(3)
    current_rot = R.from_quat([0, 0, 0, 1])  # Identity
    
    for pose in poses:
        # Extract relative transformation
        rel_trans = pose[:3]
        rel_rot = R.from_quat(pose[3:])
        
        # Update absolute pose
        current_pos = current_pos + current_rot.apply(rel_trans)
        current_rot = current_rot * rel_rot
        
        trajectory.append(current_pos.copy())
    
    return np.array(trajectory)


def compute_drift_rate(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray
) -> float:
    """
    Compute drift rate (final position error / trajectory length).
    
    Args:
        pred_trajectory: Predicted trajectory [N, 3]
        gt_trajectory: Ground truth trajectory [N, 3]
        
    Returns:
        Drift rate as percentage
    """
    final_error = np.linalg.norm(pred_trajectory[-1] - gt_trajectory[-1])
    trajectory_length = compute_trajectory_length(gt_trajectory)
    
    if trajectory_length > 0:
        return (final_error / trajectory_length) * 100
    else:
        return 0.0


def compute_trajectory_metrics(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive trajectory metrics.
    
    Args:
        pred_poses: Predicted relative poses [N, 7]
        gt_poses: Ground truth relative poses [N, 7]
        
    Returns:
        Dictionary of metrics
    """
    # Integrate to get absolute trajectories
    pred_traj = integrate_poses_to_trajectory(pred_poses)
    gt_traj = integrate_poses_to_trajectory(gt_poses)
    
    # Compute metrics
    ate = compute_absolute_trajectory_error(pred_traj, gt_traj)
    rpe_trans, rpe_rot = compute_relative_pose_error(pred_poses, gt_poses)
    scale_error = compute_scale_error(pred_traj, gt_traj)
    drift_rate = compute_drift_rate(pred_traj, gt_traj)
    
    # Additional RPE at different scales
    rpe_trans_10, rpe_rot_10 = compute_relative_pose_error(pred_poses, gt_poses, delta=10)
    rpe_trans_100, rpe_rot_100 = compute_relative_pose_error(pred_poses, gt_poses, delta=min(100, len(pred_poses)-1))
    
    return {
        'ate': ate,
        'rpe_trans_1': rpe_trans,
        'rpe_rot_1': rpe_rot,
        'rpe_trans_10': rpe_trans_10,
        'rpe_rot_10': rpe_rot_10,
        'rpe_trans_100': rpe_trans_100,
        'rpe_rot_100': rpe_rot_100,
        'scale_error': scale_error,
        'drift_rate': drift_rate,
        'trajectory_length': compute_trajectory_length(gt_traj)
    }


def evaluate_on_segments(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    segment_length: int = 100
) -> List[Dict[str, float]]:
    """
    Evaluate trajectory in segments for detailed analysis.
    
    Args:
        pred_poses: Predicted poses [N, 7]
        gt_poses: Ground truth poses [N, 7]
        segment_length: Length of each segment
        
    Returns:
        List of metrics for each segment
    """
    num_segments = len(pred_poses) // segment_length
    segment_metrics = []
    
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, len(pred_poses))
        
        segment_pred = pred_poses[start_idx:end_idx]
        segment_gt = gt_poses[start_idx:end_idx]
        
        metrics = compute_trajectory_metrics(segment_pred, segment_gt)
        metrics['segment_id'] = i
        metrics['start_frame'] = start_idx
        metrics['end_frame'] = end_idx
        
        segment_metrics.append(metrics)
    
    return segment_metrics


# Add aliases for compatibility with evaluation script
def compute_ate(pred_traj: np.ndarray, gt_traj: np.ndarray, align: bool = True) -> float:
    """Alias for compute_absolute_trajectory_error with optional alignment."""
    if align and len(pred_traj) == len(gt_traj) and len(pred_traj) > 3:
        pred_traj, _ = align_trajectories(pred_traj, gt_traj)
    return compute_absolute_trajectory_error(pred_traj, gt_traj)


def compute_rpe(pred_poses: np.ndarray, gt_poses: np.ndarray, delta: int = 1) -> Tuple[float, float]:
    """Alias for compute_relative_pose_error."""
    trans_error, rot_error = compute_relative_pose_error(pred_poses, gt_poses, delta)
    # Convert rotation error from degrees to radians for consistency
    return trans_error, np.deg2rad(rot_error)


# Also export integrate_poses for compatibility
integrate_poses = integrate_poses_to_trajectory