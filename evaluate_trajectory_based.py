#!/usr/bin/env python3
"""
Trajectory-based evaluation for VIO models using industry-standard metrics.
Implements ATE, RPE, and drift analysis for realistic AR/VR performance assessment.
"""

import torch
import numpy as np
import argparse
import sys
from pathlib import Path
import lightning as L
from typing import List, Tuple, Dict

# Add src to path
sys.path.append('src')

from src.data.simple_aria_datamodule import SimpleAriaDataModule
from src.models.multiscale_vio import MultiScaleTemporalVIO
from src.models.multihead_vio import MultiHeadVIOModel


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])
    return R


def pose_to_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Convert rotation (quaternion) and translation to 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(rotation)
    T[:3, 3] = translation
    return T


def accumulate_trajectory(relative_poses: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
    """
    Accumulate relative poses into global trajectory.
    
    Args:
        relative_poses: List of (rotation_quat, translation) tuples
        
    Returns:
        List of 4x4 transformation matrices representing global poses
    """
    trajectory = [np.eye(4)]  # Start at origin
    
    for rotation, translation in relative_poses:
        # Convert relative pose to transformation matrix
        T_rel = pose_to_matrix(rotation, translation)
        
        # Accumulate: T_global_new = T_global_prev @ T_rel
        T_global = trajectory[-1] @ T_rel
        trajectory.append(T_global)
    
    return trajectory


def compute_ate(traj_est: List[np.ndarray], traj_gt: List[np.ndarray]) -> float:
    """
    Compute Absolute Trajectory Error (ATE).
    
    Args:
        traj_est: Estimated trajectory (list of 4x4 matrices)
        traj_gt: Ground truth trajectory (list of 4x4 matrices)
        
    Returns:
        ATE in meters
    """
    assert len(traj_est) == len(traj_gt), "Trajectories must have same length"
    
    # Extract translations
    trans_est = np.array([T[:3, 3] for T in traj_est])
    trans_gt = np.array([T[:3, 3] for T in traj_gt])
    
    # Compute ATE
    errors = np.linalg.norm(trans_est - trans_gt, axis=1)
    ate = np.sqrt(np.mean(errors**2))
    
    return ate


def compute_rpe(traj_est: List[np.ndarray], traj_gt: List[np.ndarray], 
                delta_t: int = 1) -> Tuple[float, float]:
    """
    Compute Relative Pose Error (RPE).
    
    Args:
        traj_est: Estimated trajectory (list of 4x4 matrices)
        traj_gt: Ground truth trajectory (list of 4x4 matrices)
        delta_t: Time interval for relative pose computation
        
    Returns:
        (translation_rpe, rotation_rpe) in meters and degrees
    """
    trans_errors = []
    rot_errors = []
    
    for i in range(len(traj_est) - delta_t):
        # Compute relative poses
        T_rel_est = np.linalg.inv(traj_est[i]) @ traj_est[i + delta_t]
        T_rel_gt = np.linalg.inv(traj_gt[i]) @ traj_gt[i + delta_t]
        
        # Compute relative error
        T_error = np.linalg.inv(T_rel_gt) @ T_rel_est
        
        # Translation error
        trans_error = np.linalg.norm(T_error[:3, 3])
        trans_errors.append(trans_error)
        
        # Rotation error
        R_error = T_error[:3, :3]
        trace = np.clip((np.trace(R_error) - 1) / 2, -1, 1)
        rot_error = np.arccos(trace) * 180 / np.pi
        rot_errors.append(rot_error)
    
    trans_rpe = np.sqrt(np.mean(np.array(trans_errors)**2))
    rot_rpe = np.sqrt(np.mean(np.array(rot_errors)**2))
    
    return trans_rpe, rot_rpe


def compute_drift_rates(trajectory: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compute drift rates per unit distance traveled.
    
    Args:
        trajectory: List of 4x4 transformation matrices
        
    Returns:
        (distance_traveled, time_duration) in meters and frames
    """
    if len(trajectory) < 2:
        return 0.0, 0.0
    
    # Compute total distance traveled
    positions = np.array([T[:3, 3] for T in trajectory])
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_distance = np.sum(distances)
    
    # Time duration
    time_duration = len(trajectory) - 1
    
    return total_distance, time_duration


def evaluate_sequence_trajectory(model, dataloader, device) -> Dict:
    """
    Evaluate model on sequences and compute trajectory-based metrics.
    """
    model.eval()
    
    all_ates = []
    all_rpe_trans_1s = []
    all_rpe_rot_1s = []
    all_rpe_trans_5s = []
    all_rpe_rot_5s = []
    all_distances = []
    all_durations = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get full sequence predictions (not just last frame)
            batch_size = batch['poses'].shape[0]
            seq_len = batch['poses'].shape[1]
            
            # We need to predict for each frame in sequence
            predictions_list = []
            targets_list = []
            
            # For trajectory evaluation, we need all relative poses in sequence
            for i in range(1, seq_len):  # Start from 1 (relative to frame 0)
                # Create sliding window batch for frame i prediction
                window_end = i + 1
                window_start = max(0, window_end - 11)  # Model uses 11 frames
                
                if window_end <= seq_len:
                    window_batch = {
                        'images': batch['images'][:, window_start:window_end],
                        'imus': batch['imus'][:, window_start:window_end],
                        'poses': batch['poses'][:, window_start:window_end]
                    }
                    
                    pred = model(window_batch)
                    
                    predictions_list.append({
                        'rotation': pred['rotation'].cpu().numpy(),
                        'translation': pred['translation'].cpu().numpy()
                    })
                    
                    targets_list.append({
                        'rotation': batch['poses'][:, i, 3:7].cpu().numpy(),
                        'translation': batch['poses'][:, i, :3].cpu().numpy()
                    })
            
            # Compute trajectory metrics for each sequence in batch
            for b in range(batch_size):
                # Extract predictions and targets for this sequence
                pred_poses = [(pred['rotation'][b], pred['translation'][b]) 
                             for pred in predictions_list]
                gt_poses = [(target['rotation'][b], target['translation'][b]) 
                           for target in targets_list]
                
                if len(pred_poses) < 2:
                    continue
                
                # Accumulate trajectories
                traj_est = accumulate_trajectory(pred_poses)
                traj_gt = accumulate_trajectory(gt_poses)
                
                # Compute metrics
                ate = compute_ate(traj_est, traj_gt)
                all_ates.append(ate)
                
                # RPE at different time scales
                trans_rpe_1s, rot_rpe_1s = compute_rpe(traj_est, traj_gt, delta_t=1)
                all_rpe_trans_1s.append(trans_rpe_1s)
                all_rpe_rot_1s.append(rot_rpe_1s)
                
                if len(traj_est) > 5:
                    trans_rpe_5s, rot_rpe_5s = compute_rpe(traj_est, traj_gt, delta_t=5)
                    all_rpe_trans_5s.append(trans_rpe_5s)
                    all_rpe_rot_5s.append(rot_rpe_5s)
                
                # Drift analysis
                distance, duration = compute_drift_rates(traj_gt)
                all_distances.append(distance)
                all_durations.append(duration)
            
            if batch_idx % 5 == 0:
                print(f"  Processed batch {batch_idx}/{len(dataloader)}")
    
    # Aggregate results
    results = {
        'ate_mean': np.mean(all_ates),
        'ate_std': np.std(all_ates),
        'rpe_trans_1s_mean': np.mean(all_rpe_trans_1s),
        'rpe_trans_1s_std': np.std(all_rpe_trans_1s),
        'rpe_rot_1s_mean': np.mean(all_rpe_rot_1s),
        'rpe_rot_1s_std': np.std(all_rpe_rot_1s),
        'total_distance': np.sum(all_distances),
        'total_duration': np.sum(all_durations),
        'num_sequences': len(all_ates)
    }
    
    if all_rpe_trans_5s:
        results.update({
            'rpe_trans_5s_mean': np.mean(all_rpe_trans_5s),
            'rpe_trans_5s_std': np.std(all_rpe_trans_5s),
            'rpe_rot_5s_mean': np.mean(all_rpe_rot_5s),
            'rpe_rot_5s_std': np.std(all_rpe_rot_5s)
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Trajectory-based VIO evaluation')
    parser.add_argument('--multiscale_checkpoint', type=str,
                       default='logs/arvr_multiscale_vio/version_2/checkpoints/multiscale_epoch=11_val_total_loss=0.0000.ckpt',
                       help='Path to multi-scale model checkpoint')
    parser.add_argument('--multihead_checkpoint', type=str,
                       default='logs/arvr_multihead_vio/version_1/checkpoints/multihead_epoch=18_val_total_loss=0.0000.ckpt',
                       help='Path to multi-head model checkpoint')
    
    args = parser.parse_args()
    
    print("🚀 Trajectory-Based VIO Evaluation")
    print("=" * 60)
    print("Using industry-standard ATE, RPE, and drift analysis")
    print("=" * 60)
    
    # Setup data
    datamodule = SimpleAriaDataModule(
        batch_size=4,  # Smaller batch for trajectory evaluation
        num_workers=2,
        max_train_samples=None,
        max_val_samples=None
    )
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate models
    results = {}
    
    if Path(args.multihead_checkpoint).exists():
        print("\n🏆 Evaluating Multi-Head Model (Trajectory-Based)...")
        model = MultiHeadVIOModel.load_from_checkpoint(args.multihead_checkpoint)
        model = model.to(device)
        results['multihead'] = evaluate_sequence_trajectory(model, test_loader, device)
    
    if Path(args.multiscale_checkpoint).exists():
        print("\n🥈 Evaluating Multi-Scale Model (Trajectory-Based)...")
        model = MultiScaleTemporalVIO.load_from_checkpoint(args.multiscale_checkpoint)
        model = model.to(device)
        results['multiscale'] = evaluate_sequence_trajectory(model, test_loader, device)
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("📊 TRAJECTORY-BASED EVALUATION RESULTS")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n🚀 {model_name.upper()} Model (Industry Standard Metrics):")
        print(f"   📍 ATE (Absolute Trajectory Error): {result['ate_mean']:.4f}m ± {result['ate_std']:.4f}m")
        print(f"   🔄 RPE Translation (1s): {result['rpe_trans_1s_mean']:.4f}m ± {result['rpe_trans_1s_std']:.4f}m")
        print(f"   🔄 RPE Rotation (1s): {result['rpe_rot_1s_mean']:.2f}° ± {result['rpe_rot_1s_std']:.2f}°")
        
        if 'rpe_trans_5s_mean' in result:
            print(f"   🔄 RPE Translation (5s): {result['rpe_trans_5s_mean']:.4f}m ± {result['rpe_trans_5s_std']:.4f}m")
            print(f"   🔄 RPE Rotation (5s): {result['rpe_rot_5s_mean']:.2f}° ± {result['rpe_rot_5s_std']:.2f}°")
        
        print(f"   📏 Total Distance Traveled: {result['total_distance']:.2f}m")
        print(f"   ⏱️ Total Duration: {result['total_duration']} frames")
        print(f"   📊 Sequences Evaluated: {result['num_sequences']}")
        
        # Compute drift rates
        if result['total_distance'] > 0:
            trans_drift_rate = result['ate_mean'] / result['total_distance'] * 100  # Per 100m
            print(f"   📈 Translation Drift Rate: {trans_drift_rate:.2f}m per 100m traveled")
        
        if result['total_duration'] > 0:
            frame_error_rate = result['ate_mean'] / result['total_duration'] * 30  # Per second at 30fps
            print(f"   ⚡ Frame Error Rate: {frame_error_rate:.4f}m/s")
    
    print("\n" + "="*60)
    print("✅ Trajectory-Based Evaluation Complete!")
    print("💡 These metrics reflect real-world VIO performance with error accumulation")
    print("="*60)


if __name__ == "__main__":
    main()