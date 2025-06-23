#!/usr/bin/env python3
"""
Umeyama alignment implementation for trajectory alignment and error computation.
Used for evaluating visual odometry results.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def umeyama_alignment(x, y, with_scale=True):
    """
    Computes similarity transformation between two point sets.
    
    Args:
        x: (N, 3) numpy array of source points
        y: (N, 3) numpy array of target points
        with_scale: if True, computes similarity transform with scale
    
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scale factor (1.0 if with_scale=False)
    """
    assert x.shape == y.shape
    n = x.shape[0]
    
    # Centroids
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    
    # Center the points
    xc = x - mx
    yc = y - my
    
    # Compute cross-covariance matrix
    H = xc.T @ yc / n
    
    # SVD
    U, D, Vt = np.linalg.svd(H)
    V = Vt.T
    
    # Rotation matrix
    R_mat = V @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R_mat) < 0:
        V[:, -1] *= -1
        R_mat = V @ U.T
    
    # Scale factor
    if with_scale:
        # Variance of x
        var_x = np.var(xc, axis=0).sum()
        # Scale - D is returned as 1D array from SVD
        s = np.sum(D) / var_x if var_x > 0 else 1.0
    else:
        s = 1.0
    
    # Translation
    t = my - s * R_mat @ mx
    
    return R_mat, t, s


def align_trajectory(pred_positions, gt_positions, with_scale=True):
    """
    Aligns predicted trajectory to ground truth using Umeyama alignment.
    
    Args:
        pred_positions: (N, 3) numpy array of predicted positions
        gt_positions: (N, 3) numpy array of ground truth positions
        with_scale: if True, allows scale correction
    
    Returns:
        aligned_pred: (N, 3) aligned predicted positions
        R: rotation matrix used for alignment
        t: translation vector used for alignment
        s: scale factor used for alignment
    """
    R_align, t_align, s_align = umeyama_alignment(pred_positions, gt_positions, with_scale)
    aligned_pred = s_align * (pred_positions @ R_align.T) + t_align
    return aligned_pred, R_align, t_align, s_align


def compute_ate(pred_positions, gt_positions):
    """
    Computes Absolute Trajectory Error (ATE) after alignment.
    
    Args:
        pred_positions: (N, 3) numpy array of predicted positions
        gt_positions: (N, 3) numpy array of ground truth positions
    
    Returns:
        ate: mean absolute trajectory error in meters
        aligned_pred: aligned predicted trajectory
    """
    # Align trajectories
    aligned_pred, _, _, _ = align_trajectory(pred_positions, gt_positions, with_scale=True)
    
    # Compute ATE
    errors = np.linalg.norm(aligned_pred - gt_positions, axis=1)
    ate = np.mean(errors)
    
    return ate, aligned_pred


def compute_rpe(pred_positions, gt_positions, pred_rotations=None, gt_rotations=None, delta=1):
    """
    Computes Relative Pose Error (RPE).
    
    Args:
        pred_positions: (N, 3) numpy array of predicted positions
        gt_positions: (N, 3) numpy array of ground truth positions
        pred_rotations: (N, 4) numpy array of predicted quaternions (optional)
        gt_rotations: (N, 4) numpy array of ground truth quaternions (optional)
        delta: frame delta for computing relative poses
    
    Returns:
        rpe_trans: mean relative translation error
        rpe_rot: mean relative rotation error (if rotations provided)
    """
    n = pred_positions.shape[0]
    
    trans_errors = []
    rot_errors = []
    
    for i in range(n - delta):
        # Relative translation
        pred_delta = pred_positions[i + delta] - pred_positions[i]
        gt_delta = gt_positions[i + delta] - gt_positions[i]
        trans_error = np.linalg.norm(pred_delta - gt_delta)
        trans_errors.append(trans_error)
        
        # Relative rotation (if provided)
        if pred_rotations is not None and gt_rotations is not None:
            pred_q1 = R.from_quat(pred_rotations[i])
            pred_q2 = R.from_quat(pred_rotations[i + delta])
            gt_q1 = R.from_quat(gt_rotations[i])
            gt_q2 = R.from_quat(gt_rotations[i + delta])
            
            pred_delta_rot = pred_q1.inv() * pred_q2
            gt_delta_rot = gt_q1.inv() * gt_q2
            
            error_rot = (pred_delta_rot * gt_delta_rot.inv()).magnitude()
            rot_errors.append(error_rot)
    
    rpe_trans = np.mean(trans_errors)
    rpe_rot = np.mean(rot_errors) if rot_errors else None
    
    return rpe_trans, rpe_rot


def compute_trajectory_metrics(pred_positions, gt_positions, pred_rotations=None, gt_rotations=None):
    """
    Computes comprehensive trajectory metrics.
    
    Args:
        pred_positions: (N, 3) numpy array of predicted positions
        gt_positions: (N, 3) numpy array of ground truth positions
        pred_rotations: (N, 4) numpy array of predicted quaternions (optional)
        gt_rotations: (N, 4) numpy array of ground truth quaternions (optional)
    
    Returns:
        metrics: dict containing various trajectory metrics
    """
    # ATE
    ate, aligned_pred = compute_ate(pred_positions, gt_positions)
    
    # RPE
    rpe_trans, rpe_rot = compute_rpe(pred_positions, gt_positions, pred_rotations, gt_rotations)
    
    # Scale drift
    pred_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1))
    gt_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1))
    scale_error = abs(pred_length - gt_length) / gt_length
    
    # Final drift
    final_drift = np.linalg.norm(pred_positions[-1] - gt_positions[-1])
    
    metrics = {
        'ate': ate,
        'rpe_trans': rpe_trans,
        'rpe_rot': rpe_rot,
        'scale_error': scale_error,
        'final_drift': final_drift,
        'trajectory_length': gt_length
    }
    
    return metrics