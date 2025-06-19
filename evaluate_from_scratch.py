#!/usr/bin/env python3
"""
Evaluate VIFT model trained from scratch on test data.
Generates 3D plots and detailed metrics for sequences 016, 017, 018, 019.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
import pandas as pd
import json
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive HTML reports will be disabled.")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from train_aria_from_scratch import VIFTFromScratch
from src.data.components.aria_raw_dataset import AriaRawDataset
# Removed remove_gravity import - always use raw IMU


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = VIFTFromScratch().to(device)
    
    # Handle DataParallel state dict
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Handle missing keys for backward compatibility
    model_state = model.state_dict()
    for key in model_state.keys():
        if key not in new_state_dict:
            print(f"Warning: Missing key '{key}' in checkpoint, using default initialization")
            new_state_dict[key] = model_state[key]
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Disable anomaly detection for performance
    # torch.autograd.set_detect_anomaly(True)
    
    print(f"\n📊 Model Information:")
    print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'val_metrics' in checkpoint:
        metrics = checkpoint['val_metrics']
        print(f"   Validation loss: {metrics['loss']:.4f}")
        if 'trans_error' in metrics:
            print(f"   Val translation error: {metrics['trans_error']*100:.2f} cm")
        if 'rot_error' in metrics:
            print(f"   Val rotation error: {metrics['rot_error']:.2f}°")
    
    return model


def sim3_alignment(pred_translations, gt_translations):
    """Apply Sim(3) alignment to find optimal scale between predicted and GT translations
    
    Args:
        pred_translations: Predicted translations [N, 3]
        gt_translations: Ground truth translations [N, 3]
    
    Returns:
        scale: Optimal scale factor
        aligned_translations: Scaled predicted translations
    """
    # Compute centroids
    pred_centroid = np.mean(pred_translations, axis=0)
    gt_centroid = np.mean(gt_translations, axis=0)
    
    # Center the translations
    pred_centered = pred_translations - pred_centroid
    gt_centered = gt_translations - gt_centroid
    
    # Compute scale using Frobenius norm ratio
    pred_norm = np.linalg.norm(pred_centered, 'fro')
    gt_norm = np.linalg.norm(gt_centered, 'fro')
    
    if pred_norm > 1e-8:
        scale = gt_norm / pred_norm
    else:
        scale = 1.0
    
    # Apply scale
    aligned_translations = pred_centered * scale + gt_centroid
    
    return scale, aligned_translations


def compute_scale_drift(pred_poses, gt_poses):
    """Compute scale drift metric for relative poses
    
    Args:
        pred_poses: Predicted relative poses [N, 7]
        gt_poses: Ground truth relative poses [N, 7]
    
    Returns:
        scale_errors: Array of scale errors as percentages
    """
    scale_errors = []
    
    for i in range(len(pred_poses)):
        # Get translation magnitudes
        pred_trans_norm = np.linalg.norm(pred_poses[i, :3])
        gt_trans_norm = np.linalg.norm(gt_poses[i, :3])
        
        # Clamp denominator to avoid explosion on tiny steps (<1cm)
        # Matches TUM/KITTI evaluation scripts
        denominator = max(gt_trans_norm, 0.01)  # 1cm minimum
        scale_error = 100.0 * abs(pred_trans_norm - gt_trans_norm) / denominator
        scale_errors.append(scale_error)
    
    return np.array(scale_errors)


def integrate_trajectory(relative_poses, initial_pose=None):
    """Integrate relative poses to get absolute trajectory
    
    Args:
        relative_poses: Array of relative poses [N, 7] where each pose is [x,y,z,qx,qy,qz,qw]
        initial_pose: Optional initial pose [7]. If None, starts at origin.
    """
    N = relative_poses.shape[0]
    
    absolute_positions = np.zeros((N + 1, 3))
    absolute_rotations = np.zeros((N + 1, 4))
    
    # Set initial pose
    if initial_pose is not None:
        absolute_positions[0] = initial_pose[:3]
        absolute_rotations[0] = initial_pose[3:]
        current_position = initial_pose[:3].copy()
        current_rotation = R.from_quat(initial_pose[3:])
    else:
        absolute_positions[0] = [0, 0, 0]
        absolute_rotations[0] = [0, 0, 0, 1]
        current_position = np.zeros(3)
        current_rotation = R.from_quat([0, 0, 0, 1])
    
    for i in range(N):
        rel_trans = relative_poses[i, :3]
        
        # Quaternion format
        rel_quat = relative_poses[i, 3:]
        rel_quat = rel_quat / (np.linalg.norm(rel_quat) + 1e-8)
        
        # Check for valid quaternion
        if np.any(np.isnan(rel_quat)) or np.linalg.norm(rel_quat) < 0.5:
            rel_quat = np.array([0, 0, 0, 1])
        
        rel_rotation = R.from_quat(rel_quat)
        
        # Apply transformation
        # rel_trans comes from the **body/device frame**, so we must rotate
        # it into the current world frame before accumulating.
        world_trans = current_rotation.apply(rel_trans)  # body → world
        current_position += world_trans
        current_rotation = current_rotation * rel_rotation
        
        # Re-normalize quaternion to prevent numerical drift
        current_quat = current_rotation.as_quat()
        current_quat = current_quat / (np.linalg.norm(current_quat) + 1e-8)
        current_rotation = R.from_quat(current_quat)
        
        absolute_positions[i + 1] = current_position
        absolute_rotations[i + 1] = current_rotation.as_quat()
    
    return absolute_positions, absolute_rotations


def compute_ape(pred_positions, gt_positions):
    """Compute Absolute Pose Error (APE) metrics
    
    Args:
        pred_positions: Predicted absolute positions [N, 3]
        gt_positions: Ground truth absolute positions [N, 3]
    
    Returns:
        Dictionary with APE statistics
    """
    # Compute position errors
    position_errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
    
    ape_stats = {
        'mean': np.mean(position_errors),
        'std': np.std(position_errors),
        'median': np.median(position_errors),
        'min': np.min(position_errors),
        'max': np.max(position_errors),
        'rmse': np.sqrt(np.mean(position_errors**2))
    }
    
    return ape_stats


def compute_rpe(pred_poses, gt_poses, delta=1):
    """Compute Relative Pose Error (RPE) metrics
    
    Args:
        pred_poses: Predicted relative poses [N, 7] 
        gt_poses: Ground truth relative poses [N, 7]
        delta: Frame delta for computing RPE (default=1)
    
    Returns:
        Dictionary with RPE translation and rotation statistics
    """
    trans_errors = []
    rot_errors = []
    
    for i in range(0, len(pred_poses) - delta):
        # Translation error
        pred_trans = pred_poses[i:i+delta, :3].sum(axis=0)
        gt_trans = gt_poses[i:i+delta, :3].sum(axis=0)
        trans_error = np.linalg.norm(pred_trans - gt_trans)
        trans_errors.append(trans_error)
        
        # Rotation error - compose quaternions
        pred_rot = R.from_quat(pred_poses[i, 3:])
        gt_rot = R.from_quat(gt_poses[i, 3:])
        for j in range(1, delta):
            pred_rot = pred_rot * R.from_quat(pred_poses[i+j, 3:])
            gt_rot = gt_rot * R.from_quat(gt_poses[i+j, 3:])
        
        rel_rot = gt_rot.inv() * pred_rot
        angle_error = np.abs(rel_rot.magnitude())
        rot_errors.append(np.rad2deg(angle_error))
    
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    rpe_stats = {
        'trans': {
            'mean': np.mean(trans_errors),
            'std': np.std(trans_errors),
            'median': np.median(trans_errors),
            'rmse': np.sqrt(np.mean(trans_errors**2))
        },
        'rot': {
            'mean': np.mean(rot_errors),
            'std': np.std(rot_errors),
            'median': np.median(rot_errors),
            'rmse': np.sqrt(np.mean(rot_errors**2))
        }
    }
    
    return rpe_stats


def evaluate_model(model, test_loader, device, output_dir, test_sequences, use_sim3_alignment=False, scale_correction=1.0):
    """Evaluate model on test data
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: torch device
        output_dir: Directory to save results
        test_sequences: List of test sequence IDs
        use_sim3_alignment: If True, apply Sim(3) alignment to predictions
        remove_gravity: If True, remove gravity from IMU data before inference
        scale_correction: Global scale correction factor to apply to predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    sequence_results = {seq_id: [] for seq_id in test_sequences}
    
    # Track cumulative scale per sequence
    cumulative_scales = {}
    window_counters = {}
    scale_history = {seq_id: [] for seq_id in test_sequences}  # Track scale over time
    
    print("\n🔍 Evaluating on test sequences...")
    if scale_correction != 1.0:
        print(f"   Applying global scale correction: {scale_correction:.2f}x")
    if use_sim3_alignment:
        print("   Using Sim(3) alignment for scale correction")
    # Always use raw IMU to match training
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            # Move to device
            images = batch['images'].to(device)
            imu = batch['imu'].to(device)
            
            # Always use raw IMU to match training
            # No gravity removal - model was trained with raw IMU
            
            gt_poses = batch['gt_poses'].to(device)
            
            # Get predictions
            predictions = model({'images': images, 'imu': imu, 'gt_poses': gt_poses})  # {'poses': [B, 20, 7]}
            pred_poses_batch = predictions['poses']  # [B, 20, 7]
            
            # Process each sample in batch
            batch_size = pred_poses_batch.shape[0]
            for i in range(batch_size):
                # For multi-step prediction, we predict all 20 transitions
                pred_poses = pred_poses_batch[i].cpu().numpy()  # [20, 7]
                gt_poses_sample = gt_poses[i].cpu().numpy()  # [20, 7]
                
                # Apply global scale correction factor (model outputs are ~2.5x over-scaled)
                pred_poses[:, :3] *= scale_correction
                
                # Apply Y/Z swap to predictions to match ground truth coordinate convention
                # This is the inverse of the swap applied during training
                pred_poses_swapped = pred_poses.copy()
                pred_poses_swapped[:, 1] = pred_poses[:, 2]  # New Y = Old Z
                pred_poses_swapped[:, 2] = pred_poses[:, 1]  # New Z = Old Y
                pred_poses = pred_poses_swapped
                
                # Frame convention debug (only for first batch)
                if batch_idx == 0 and i == 0:
                    print("\n" + "="*50)
                    print("FRAME CONVENTION MICRO-EXPERIMENT")
                    print("="*50)
                    
                    # Test different axis transformations
                    transforms = {
                        'Original': np.eye(3),
                        'Flip X': np.diag([-1, 1, 1]),
                        'Flip Y': np.diag([1, -1, 1]),
                        'Flip Z': np.diag([1, 1, -1]),
                        'Swap XY': np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                        'Swap XZ': np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
                        'Swap YZ': np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                    }
                    
                    for name, T in transforms.items():
                        angles = []
                        for j in range(20):
                            pred_t = pred_poses[j, :3].copy()
                            gt_t = gt_poses_sample[j, :3]
                            
                            # Apply transform
                            pred_t_transformed = T @ pred_t
                            
                            pred_norm = np.linalg.norm(pred_t_transformed)
                            gt_norm = np.linalg.norm(gt_t)
                            
                            if pred_norm > 1e-6 and gt_norm > 1e-6:
                                cos_angle = np.dot(pred_t_transformed, gt_t) / (pred_norm * gt_norm)
                                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                                angles.append(angle)
                        
                        mean_angle = np.mean(angles)
                        print(f"{name:10s}: mean angle = {mean_angle:6.1f}°")
                    
                    print("="*50 + "\n")
                
                # Ensure quaternion sign continuity for predictions to avoid 180° jumps
                for k in range(1, len(pred_poses)):
                    if np.dot(pred_poses[k, 3:], pred_poses[k-1, 3:]) < 0:
                        pred_poses[k, 3:] *= -1
                
                # Apply Sim(3) alignment if requested
                if use_sim3_alignment:
                    # Extract translations
                    pred_trans = pred_poses[:, :3]
                    gt_trans = gt_poses_sample[:, :3]
                    
                    # Find optimal scale
                    scale, aligned_trans = sim3_alignment(pred_trans, gt_trans)
                    
                    # Apply scale to predictions
                    pred_poses[:, :3] = aligned_trans
                else:
                    # No scale clamping - let the model's learned scale show through
                    pass
                
                # Initialize or update cumulative scale tracking
                seq_id = batch['seq_name'][i]
                if seq_id not in window_counters:
                    window_counters[seq_id] = 0
                
                window_idx = window_counters[seq_id]
                
                # ─────────────────────────────────────────────────────────────
                # QUICK VISUAL / CSV CHECK (always on – no debug flag needed)
                # ─────────────────────────────────────────────────────────────
                # 1) print the 20 raw relative translations (m) for this window
                if window_idx < 5:  # Only print first 5 windows per sequence
                    np.set_printoptions(precision=4, suppress=True)
                    print(f"\n[Δt window {window_idx:03d} | Seq {seq_id}] "
                          f"pred (m):\n{pred_poses[:, :3]}")
                    print(f"[Δt window {window_idx:03d} | Seq {seq_id}] "
                          f"gt   (m):\n{gt_poses_sample[:, :3]}\n")
                
                # 2) dump the first 5 s of relative motion once per sequence
                #    (20 Hz → 100 steps; include *all* 7-D pose rows)
                if window_idx == 0:                         # only first window of the seq
                    first_5s = min(100, pred_poses.shape[0])
                    csv_path = os.path.join(
                        output_dir, f"rel_pose_first5s_seq{seq_id}.csv")
                    pd.DataFrame(
                        np.hstack([pred_poses[:first_5s],
                                   gt_poses_sample[:first_5s]]),
                        columns=[
                            'pred_x','pred_y','pred_z','pred_qx','pred_qy','pred_qz','pred_qw',
                            'gt_x',  'gt_y',  'gt_z',  'gt_qx', 'gt_qy', 'gt_qz', 'gt_qw'
                        ]).to_csv(csv_path, index=False)
                    print(f"[INFO] wrote first-5-s relative poses to {csv_path}")
                # ─────────────────────────────────────────────────────────────
                
                # Compute errors for all transitions
                trans_errors = []
                rot_errors = []
                
                # Track accumulated rotation for world frame conversion
                # Start with identity since predictions are relative to first frame
                accumulated_rotation = R.from_quat([0, 0, 0, 1])
                
                for j in range(20):
                    # Get translation vectors
                    pred_t = pred_poses[j, :3]
                    gt_t = gt_poses_sample[j, :3]
                    
                    # Debug prints disabled for cleaner output
                    # Uncomment the following block if you need detailed debugging:
                    # if window_idx < 30 and j == 0:
                    #     print(f"[DBG] Seq {seq_id} win {window_idx:03d}  Δt_pred={pred_t}  Δt_gt={gt_t}")
                    #     ...
                    pass
                    
                    # Both pred and gt translations are in body frame - compare directly
                    trans_error = np.linalg.norm(pred_t - gt_t)
                    trans_errors.append(trans_error)
                    
                    # Update accumulated rotation for next step using PREDICTED quaternion
                    # This ensures frame consistency - we should use pred_poses for integration
                    accumulated_rotation = accumulated_rotation * R.from_quat(pred_poses[j, 3:])
                    
                    # Rotation error
                    pred_q = pred_poses[j, 3:]
                    pred_q = pred_q / (np.linalg.norm(pred_q) + 1e-8)
                    gt_q = gt_poses_sample[j, 3:]
                    gt_q = gt_q / (np.linalg.norm(gt_q) + 1e-8)
                    
                    # Compute angle between quaternions
                    dot = np.dot(pred_q, gt_q)
                    dot = np.clip(dot, -1.0, 1.0)
                    angle = 2 * np.arccos(np.abs(dot))
                    rot_error = np.rad2deg(angle)
                    rot_errors.append(rot_error)
                
                # Compute scale drift
                scale_errors = compute_scale_drift(pred_poses, gt_poses_sample)
                
                # Calculate mean translation distances for this window
                pred_dist = np.mean(np.linalg.norm(pred_poses[:, :3], axis=1)) * 100  # to cm
                gt_dist = np.mean(np.linalg.norm(gt_poses_sample[:, :3], axis=1)) * 100  # to cm
                scale_factor = pred_dist / (gt_dist + 1e-8)
                
                # Track scale history for analysis
                if seq_id in scale_history:
                    scale_history[seq_id].append(scale_factor)
                
                window_counters[seq_id] += 1
                
                # Remove verbose window prints - they clutter the output
                # Statistics are already captured in scale_history and final summaries
                
                result = {
                    'pred_poses': pred_poses,  # [20, 7]
                    'gt_poses': gt_poses_sample,  # [20, 7]
                    'trans_errors': np.array(trans_errors),  # [20]
                    'rot_errors': np.array(rot_errors),  # [20]
                    'scale_errors': scale_errors,  # Scale drift percentages
                    'seq_name': batch['seq_name'][i],
                    'start_idx': batch['start_idx'][i].item()
                }
                all_results.append(result)
                
                # Group by sequence
                seq_id = batch['seq_name'][i]
                if seq_id in sequence_results:
                    sequence_results[seq_id].append(result)
    
    # Compute overall statistics
    all_trans_errors = np.concatenate([r['trans_errors'] for r in all_results])
    all_rot_errors = np.concatenate([r['rot_errors'] for r in all_results])
    all_scale_errors = np.concatenate([r['scale_errors'] for r in all_results if len(r['scale_errors']) > 0])
    
    # ---------- GLOBAL DIAGNOSTICS ----------
    total_gt_len = sum(np.linalg.norm(r['gt_poses'][:, :3], axis=1).sum()
                       for r in all_results)
    total_pred_len = sum(np.linalg.norm(r['pred_poses'][:, :3], axis=1).sum()
                         for r in all_results)
    print(f"\n🛣️  Path-length ratio (pred / GT): {total_pred_len/(total_gt_len+1e-8):.3f}")
    print(f"🔍 Global scale-drift  mean={np.mean(all_scale_errors):.2f}%  "
          f"max={np.max(all_scale_errors):.2f}%")
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"\n📏 Translation Error:")
    print(f"   Mean:   {np.mean(all_trans_errors) * 100:.2f} cm")
    print(f"   Std:    {np.std(all_trans_errors) * 100:.2f} cm")
    print(f"   Median: {np.median(all_trans_errors) * 100:.2f} cm")
    print(f"   95%:    {np.percentile(all_trans_errors, 95) * 100:.2f} cm")
    print(f"   Max:    {np.max(all_trans_errors) * 100:.2f} cm")
    
    print(f"\n🔄 Rotation Error (degrees):")
    print(f"   Mean:   {np.mean(all_rot_errors):.2f}")
    print(f"   Std:    {np.std(all_rot_errors):.2f}")
    print(f"   Median: {np.median(all_rot_errors):.2f}")
    print(f"   95%:    {np.percentile(all_rot_errors, 95):.2f}")
    print(f"   Max:    {np.max(all_rot_errors):.2f}")
    
    print(f"\n📐 Scale Drift (%):")
    print(f"   Mean:   {np.mean(all_scale_errors):.2f}")
    print(f"   Std:    {np.std(all_scale_errors):.2f}")
    print(f"   Median: {np.median(all_scale_errors):.2f}")
    print(f"   95%:    {np.percentile(all_scale_errors, 95):.2f}")
    print(f"   Max:    {np.max(all_scale_errors):.2f}")
    print("="*50)
    
    # Save sample trajectories
    plot_sample_trajectories(all_results[:8], output_dir)
    
    # Generate 3D plots and CSV files for each test sequence
    print("\n📊 Generating 3D trajectory plots and CSV files...")
    fps = 20  # Aria dataset is 20 FPS
    
    for seq_id in test_sequences:
        if seq_id not in sequence_results:
            print(f"\nSequence {seq_id}: Not found in results")
            continue
            
        if not sequence_results[seq_id]:
            print(f"\nSequence {seq_id}: No results available")
            continue
        
        # Sort by start index to maintain chronological order
        seq_results = sorted(sequence_results[seq_id], key=lambda x: x['start_idx'])
        
        print(f"\nSequence {seq_id}:")
        print(f"  Total windows: {len(seq_results)}")
        
        # Keep only non-overlapping windows to avoid double-counting
        # Use sequence_length - 1 to stay in sync with model configuration
        overlap_stride = 20  # Default for 21-frame sequences (20 transitions)
        if hasattr(model, 'config') and hasattr(model.config, 'seq_len'):
            overlap_stride = model.config.seq_len - 1
        seq_results_sampled = [r for r in seq_results if r['start_idx'] % overlap_stride == 0]
        print(f"  Non-overlapping windows: {len(seq_results_sampled)}")
        
        # If no non-overlapping windows found, show all start indices for debugging
        if len(seq_results_sampled) == 0:
            print(f"  WARNING: No non-overlapping windows found!")
            print(f"  Start indices: {[r['start_idx'] for r in seq_results[:10]]}...")
            continue
        
        # Load raw ground truth poses from file
        seq_dir = test_loader.dataset.data_dir / seq_id
        with open(seq_dir / 'poses_quaternion.json', 'r') as f:
            raw_poses = json.load(f)
        
        # Get ground truth absolute poses for the frames we're evaluating
        # We need to track which frames are covered by our predictions
        all_frame_indices = []
        for r in seq_results_sampled:
            start_idx = r['start_idx']
            # Each window covers 21 frames (20 transitions)
            # Include all frames starting from start_idx
            frame_indices = list(range(start_idx, start_idx + 21))
            all_frame_indices.extend(frame_indices)
        
        # Remove duplicates and sort
        all_frame_indices = sorted(list(set(all_frame_indices)))
        
        # Extract ground truth absolute poses
        gt_positions_full = np.array([raw_poses[i]['translation'] for i in all_frame_indices])
        gt_quaternions = np.array([raw_poses[i]['quaternion'] for i in all_frame_indices])
        
        # Make ground truth quaternion stream sign-continuous
        for k in range(1, len(gt_quaternions)):
            if np.dot(gt_quaternions[k], gt_quaternions[k-1]) < 0:
                gt_quaternions[k] *= -1
        
        # Build all_pred without double steps
        all_pred = []
        for n, r in enumerate(seq_results_sampled):
            steps = r['pred_poses']
            if n:                       # every window except the first
                steps = steps[1:]       # drop duplicate step 0
            all_pred.append(steps)
        all_pred = np.concatenate(all_pred, axis=0)
        
        # Make quaternion stream sign-continuous to avoid 180° jumps
        for k in range(1, len(all_pred)):
            if np.dot(all_pred[k, 3:], all_pred[k-1, 3:]) < 0:
                all_pred[k, 3:] *= -1
        
        # Integrate predictions starting from the first frame
        initial_frame_idx = all_frame_indices[0]
        initial_pose = np.concatenate([
            raw_poses[initial_frame_idx]['translation'],
            raw_poses[initial_frame_idx]['quaternion']
        ])
        pred_positions_full, pred_rotations_full = integrate_trajectory(all_pred, initial_pose)
        
        # For ground truth rotations, use the raw quaternions
        gt_rotations_full = gt_quaternions
        
        # Ensure arrays have matching shapes for APE computation
        min_len = min(len(pred_positions_full), len(gt_positions_full))
        pred_positions_aligned = pred_positions_full[:min_len]
        gt_positions_aligned = gt_positions_full[:min_len]
        
        # Compute APE for this sequence
        ape_stats = compute_ape(pred_positions_aligned, gt_positions_aligned)
        print(f"  APE - Mean: {ape_stats['mean']*100:.2f}cm, RMSE: {ape_stats['rmse']*100:.2f}cm")
        
        # Compute RPE for this sequence
        rpe_stats = compute_rpe(all_pred, np.concatenate([r['gt_poses'] for r in seq_results_sampled], axis=0))
        print(f"  RPE - Trans: {rpe_stats['trans']['mean']*100:.2f}cm, Rot: {rpe_stats['rot']['mean']:.2f}°")
        
        # Sequence-level path length diagnostic
        seq_gt_len = np.sum(np.linalg.norm(np.diff(gt_positions_full, axis=0), axis=1))
        seq_pred_len = np.sum(np.linalg.norm(np.diff(pred_positions_full, axis=0), axis=1))
        # Collect scale errors for this sequence
        seq_scale_errors = np.concatenate([r['scale_errors'] for r in seq_results_sampled if len(r['scale_errors']) > 0])
        print(f"  🛤️  Path-length ratio: {seq_pred_len/(seq_gt_len+1e-8):.3f}")
        print(f"  📏 Seq scale-drift  mean={np.mean(seq_scale_errors):.2f}%  "
              f"max={np.max(seq_scale_errors):.2f}%")
        
        # Save trajectories to CSV with global frame indices
        save_trajectory_csv(pred_positions_aligned, gt_positions_aligned, 
                           pred_rotations_full[:min_len], gt_rotations_full[:min_len],
                           seq_id, output_dir, frame_offset=all_frame_indices[0])
        
        # Save all relative poses for the entire sequence
        save_relative_poses_csv(all_pred, np.concatenate([r['gt_poses'] for r in seq_results_sampled], axis=0),
                               seq_id, output_dir)
        
        # Plot full trajectory
        plot_trajectory_3d(
            pred_positions_aligned,
            gt_positions_aligned,
            seq_id,
            os.path.join(output_dir, f'trajectory_3d_{seq_id}.png')
        )
        
        # Create interactive HTML plot
        create_interactive_html_plot(
            pred_positions_aligned,
            gt_positions_aligned,
            seq_id,
            os.path.join(output_dir, f'trajectory_3d_{seq_id}_interactive.html')
        )
        
        # Plot rotation trajectory
        plot_rotation_3d(
            pred_rotations_full[1:],
            gt_rotations_full[1:],
            seq_id,
            os.path.join(output_dir, f'rotation_3d_{seq_id}.png')
        )
        
        # Generate 1-second plots
        frames_1s = min(fps, len(all_pred))
        if frames_1s > 0:
            # For predictions, integrate only the first second
            pred_positions_1s, pred_rotations_1s = integrate_trajectory(all_pred[:frames_1s], initial_pose)
            
            # For ground truth, use raw poses for first second
            gt_positions_1s = gt_positions_full[:frames_1s + 1]  # +1 because we need initial pose too
            gt_rotations_1s = gt_rotations_full[:frames_1s + 1]
            
            plot_trajectory_3d(
                pred_positions_1s,
                gt_positions_1s,
                seq_id,
                os.path.join(output_dir, f'trajectory_3d_{seq_id}_1s.png'),
                time_window='1s'
            )
            
            plot_rotation_3d(
                pred_rotations_1s[1:],
                gt_rotations_1s[1:],
                seq_id,
                os.path.join(output_dir, f'rotation_3d_{seq_id}_1s.png'),
                time_window='1s'
            )
        
        # Generate 5-second plots
        frames_5s = min(5 * fps, len(all_pred))
        if frames_5s > 0:
            # For predictions, integrate only the first 5 seconds
            pred_positions_5s, pred_rotations_5s = integrate_trajectory(all_pred[:frames_5s], initial_pose)
            
            # For ground truth, use raw poses for first 5 seconds
            gt_positions_5s = gt_positions_full[:frames_5s + 1]  # +1 because we need initial pose too
            gt_rotations_5s = gt_rotations_full[:frames_5s + 1]
            
            plot_trajectory_3d(
                pred_positions_5s,
                gt_positions_5s,
                seq_id,
                os.path.join(output_dir, f'trajectory_3d_{seq_id}_5s.png'),
                time_window='5s'
            )
            
            plot_rotation_3d(
                pred_rotations_5s[1:],
                gt_rotations_5s[1:],
                seq_id,
                os.path.join(output_dir, f'rotation_3d_{seq_id}_5s.png'),
                time_window='5s'
            )
    
    # Save numerical results
    save_evaluation_metrics(all_trans_errors, all_rot_errors, all_scale_errors, output_dir)
    
    # Plot scale convergence
    if scale_history and any(len(scales) > 0 for scales in scale_history.values()):
        plot_scale_convergence(scale_history, output_dir)
    
    # Perform sanity checks
    print("\n" + "="*50)
    print("📋 SANITY CHECKS")
    print("="*50)
    
    # Check 1: Scale convergence
    scale_ratio = total_pred_len / (total_gt_len + 1e-8)
    print(f"\n✓ Scale Check:")
    print(f"  Total pred/gt length ratio: {scale_ratio:.3f}")
    if 0.95 <= scale_ratio <= 1.05:
        print("  ✅ PASS: Scale within 5% of ground truth")
    else:
        print(f"  ❌ FAIL: Scale off by {abs(1.0 - scale_ratio)*100:.1f}%")
    
    # Check 2: Mean translation angle
    if batch_idx > 0:  # Only if we have data
        print(f"\n✓ Coordinate Frame Check:")
        print(f"  Mean angle between pred and gt: <printed above>")
        print(f"  ✅ PASS if < 5° (after Y/Z swap fix)")
    
    # Check 3: First 5s median error
    first_5s_errors = all_trans_errors[:100] if len(all_trans_errors) > 100 else all_trans_errors
    median_5s_error = np.median(first_5s_errors) * 100
    print(f"\n✓ First 5s Accuracy Check:")
    print(f"  Median translation error: {median_5s_error:.2f} cm")
    if median_5s_error < 3.0:
        print("  ✅ PASS: Median error < 3 cm")
    else:
        print(f"  ❌ FAIL: Median error too high")
    
    # Note: Uncertainty weights check would require passing checkpoint data
    # For now, this check is informational only
    
    # Scale analysis summary
    print("\n" + "="*50)
    print("📊 SCALE ANALYSIS BY SEQUENCE")
    print("="*50)
    for seq_id in test_sequences:
        if seq_id in scale_history and scale_history[seq_id]:
            scales = scale_history[seq_id]
            print(f"\nSequence {seq_id}:")
            print(f"  Windows analyzed: {len(scales)}")
            print(f"  Mean scale: {np.mean(scales):.3f}")
            print(f"  Std scale: {np.std(scales):.3f}")
            print(f"  Range: [{np.min(scales):.3f}, {np.max(scales):.3f}]")
            
            # Check if problematic
            if np.mean(scales) > 1.15 or np.mean(scales) < 0.85:
                print(f"  ⚠️  SCALE DRIFT DETECTED - Mean significantly off from 1.0")
    
    print(f"\n📊 Generated outputs:")
    for seq_id in test_sequences:
        if seq_id in sequence_results and sequence_results[seq_id]:
            print(f"   Sequence {seq_id}:")
            print(f"   - trajectory_{seq_id}_gt.csv (ground truth)")
            print(f"   - trajectory_{seq_id}_pred.csv (predictions with errors)")
            print(f"   - trajectory_3d_{seq_id}.png, trajectory_3d_{seq_id}_1s.png, trajectory_3d_{seq_id}_5s.png")
            print(f"   - rotation_3d_{seq_id}.png, rotation_3d_{seq_id}_1s.png, rotation_3d_{seq_id}_5s.png")
            print(f"   - relative_poses_{seq_id}_gt.csv (ground truth relative poses)")
            print(f"   - relative_poses_{seq_id}_pred.csv (predicted relative poses with errors)")
            if PLOTLY_AVAILABLE:
                print(f"   - trajectory_3d_{seq_id}_interactive.html (interactive 3D plot)")
    
    return all_results


def save_trajectory_csv(pred_positions, gt_positions, pred_rotations, gt_rotations, seq_id, output_dir, frame_offset=0):
    """Save trajectory data to separate CSV files for ground truth and predictions"""
    
    # Save ground truth data
    gt_data = []
    for i in range(len(gt_positions)):
        row = {
            'frame': i + frame_offset,
            'x': gt_positions[i, 0],
            'y': gt_positions[i, 1],
            'z': gt_positions[i, 2]
        }
        
        if i < len(gt_rotations):
            # Store quaternions directly
            row.update({
                'qx': gt_rotations[i, 0],
                'qy': gt_rotations[i, 1],
                'qz': gt_rotations[i, 2],
                'qw': gt_rotations[i, 3]
            })
        
        gt_data.append(row)
    
    gt_df = pd.DataFrame(gt_data)
    gt_csv_path = os.path.join(output_dir, f'trajectory_{seq_id}_gt.csv')
    gt_df.to_csv(gt_csv_path, index=False)
    print(f"Saved ground truth to {gt_csv_path}")
    
    # Save prediction data
    pred_data = []
    for i in range(len(pred_positions)):
        row = {
            'frame': i + frame_offset,
            'x': pred_positions[i, 0],
            'y': pred_positions[i, 1],
            'z': pred_positions[i, 2]
        }
        
        if i < len(pred_rotations):
            # Store quaternions directly
            row.update({
                'qx': pred_rotations[i, 0],
                'qy': pred_rotations[i, 1],
                'qz': pred_rotations[i, 2],
                'qw': pred_rotations[i, 3]
            })
        
        # Add errors compared to ground truth
        if i < len(gt_positions):
            row['trans_error'] = np.linalg.norm(pred_positions[i] - gt_positions[i])
            
            if i < len(gt_rotations) and i < len(pred_rotations):
                # Compute rotation error
                gt_rot = R.from_quat(gt_rotations[i])
                pred_rot = R.from_quat(pred_rotations[i])
                rel_rot = gt_rot.inv() * pred_rot
                rot_error = np.abs(rel_rot.magnitude() * 180 / np.pi)
                row['rot_error_deg'] = rot_error
        
        pred_data.append(row)
    
    pred_df = pd.DataFrame(pred_data)
    pred_csv_path = os.path.join(output_dir, f'trajectory_{seq_id}_pred.csv')
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved predictions to {pred_csv_path}")


def create_interactive_html_plot(pred_positions, gt_positions, sequence_name, output_path):
    """Create interactive 3D trajectory plot using Plotly"""
    if not PLOTLY_AVAILABLE:
        return
    
    # Convert to centimeters
    pred_positions_cm = pred_positions * 100
    gt_positions_cm = gt_positions * 100
    
    # Create figure
    fig = go.Figure()
    
    # Optimize for large sequences
    if pred_positions.shape[0] > 10000:
        # Subsample for performance
        stride = max(1, pred_positions.shape[0] // 5000)
        pred_positions_cm = pred_positions_cm[::stride]
        gt_positions_cm = gt_positions_cm[::stride]
        print(f"Note: Subsampled trajectory to {len(pred_positions_cm)} points for performance")
    
    # Add ground truth trajectory
    fig.add_trace(go.Scatter3d(
        x=gt_positions_cm[:, 0],
        y=gt_positions_cm[:, 1],
        z=gt_positions_cm[:, 2],
        mode='lines',
        name='Ground Truth',
        line=dict(color='blue', width=4),
        hovertemplate='GT<br>X: %{x:.2f}cm<br>Y: %{y:.2f}cm<br>Z: %{z:.2f}cm<extra></extra>'
    ))
    
    # Add predicted trajectory
    fig.add_trace(go.Scatter3d(
        x=pred_positions_cm[:, 0],
        y=pred_positions_cm[:, 1],
        z=pred_positions_cm[:, 2],
        mode='lines',
        name='Prediction',
        line=dict(color='red', width=4, dash='dash'),
        hovertemplate='Pred<br>X: %{x:.2f}cm<br>Y: %{y:.2f}cm<br>Z: %{z:.2f}cm<extra></extra>'
    ))
    
    # Add start and end markers
    fig.add_trace(go.Scatter3d(
        x=[gt_positions_cm[0, 0]],
        y=[gt_positions_cm[0, 1]],
        z=[gt_positions_cm[0, 2]],
        mode='markers',
        name='Start',
        marker=dict(color='green', size=10, symbol='circle'),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[gt_positions_cm[-1, 0]],
        y=[gt_positions_cm[-1, 1]],
        z=[gt_positions_cm[-1, 2]],
        mode='markers',
        name='End',
        marker=dict(color='red', size=10, symbol='x'),
        showlegend=True
    ))
    
    # Calculate path lengths and errors
    gt_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)) * 100
    pred_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)) * 100
    ape_mean = np.mean(np.linalg.norm(pred_positions - gt_positions[:len(pred_positions)], axis=1)) * 100
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Sequence {sequence_name} - Interactive 3D Trajectory<br>' +
                 f'GT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm, APE: {ape_mean:.2f}cm',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)',
            zaxis_title='Z (cm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=800,
        hovermode='closest'
    )
    
    # Save HTML
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Saved interactive plot to {output_path}")


def plot_trajectory_3d(pred_positions, gt_positions, sequence_name, output_path, time_window=None):
    """Create 3D trajectory plot"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to centimeters for display
    pred_positions_cm = pred_positions * 100
    gt_positions_cm = gt_positions * 100
    
    # Plot trajectories
    ax.plot(gt_positions_cm[:, 0], gt_positions_cm[:, 1], gt_positions_cm[:, 2], 
            'b-', linewidth=2, label='Ground Truth')
    ax.plot(pred_positions_cm[:, 0], pred_positions_cm[:, 1], pred_positions_cm[:, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_positions_cm[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_positions_cm[-1], color='red', s=100, marker='x', label='End')
    
    # Calculate path lengths
    gt_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)) * 100
    pred_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)) * 100
    
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    
    # Set equal aspect ratio for all axes
    try:
        ax.set_box_aspect([1, 1, 1])  # Available in matplotlib >= 3.4
    except AttributeError:
        # Fallback for older matplotlib versions
        pass
    
    if time_window:
        ax.set_title(f'Sequence {sequence_name} - First {time_window}\nGT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm')
    else:
        ax.set_title(f'Sequence {sequence_name} - Full Trajectory\nGT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm')
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rotation_3d(pred_rotations, gt_rotations, sequence_name, output_path, time_window=None):
    """Create 3D rotation trajectory visualization"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert quaternions to axis-angle
    gt_axis_angles = np.array([R.from_quat(q).as_rotvec() for q in gt_rotations])
    pred_axis_angles = np.array([R.from_quat(q).as_rotvec() for q in pred_rotations])
    
    # Downsample for clarity if sequence is very long
    stride = max(1, len(gt_axis_angles) // 1000)
    
    # Plot rotation trajectories
    ax.plot(gt_axis_angles[::stride, 0], gt_axis_angles[::stride, 1], gt_axis_angles[::stride, 2], 
            'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(pred_axis_angles[::stride, 0], pred_axis_angles[::stride, 1], pred_axis_angles[::stride, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_axis_angles[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_axis_angles[-1], color='red', s=100, marker='x', label='End')
    
    # Calculate rotation statistics
    rot_errors = []
    for pred_q, gt_q in zip(pred_rotations, gt_rotations):
        gt_rot = R.from_quat(gt_q)
        pred_rot = R.from_quat(pred_q)
        rel_rot = gt_rot.inv() * pred_rot
        angle_error = np.abs(rel_rot.magnitude() * 180 / np.pi)
        rot_errors.append(angle_error)
    
    mean_error = np.mean(rot_errors)
    max_error = np.max(rot_errors)
    
    ax.set_xlabel('X (rad)')
    ax.set_ylabel('Y (rad)')
    ax.set_zlabel('Z (rad)')
    
    # Set equal aspect ratio for all axes
    try:
        ax.set_box_aspect([1, 1, 1])  # Available in matplotlib >= 3.4
    except AttributeError:
        # Fallback for older matplotlib versions
        pass
    
    if time_window:
        ax.set_title(f'Sequence {sequence_name} - Rotation First {time_window}\n'
                     f'Mean Error: {mean_error:.2f}°, Max Error: {max_error:.2f}°')
    else:
        ax.set_title(f'Sequence {sequence_name} - Full Rotation Trajectory\n'
                     f'Mean Error: {mean_error:.2f}°, Max Error: {max_error:.2f}°')
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_sample_trajectories(results, output_dir):
    """Plot sample trajectories"""
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot first 8 sequences
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(results[:8]):
        ax = axes[idx]
        
        # Integrate trajectories
        pred_positions, _ = integrate_trajectory(result['pred_poses'])
        gt_positions, _ = integrate_trajectory(result['gt_poses'])
        
        # Convert to cm
        gt_positions_cm = gt_positions * 100
        pred_positions_cm = pred_positions * 100
        
        # Plot 2D trajectories
        ax.plot(gt_positions_cm[:, 0], gt_positions_cm[:, 1], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(pred_positions_cm[:, 0], pred_positions_cm[:, 1], 'r--', label='Prediction', linewidth=2)
        
        ax.scatter(gt_positions_cm[0, 0], gt_positions_cm[0, 1], c='green', s=100, marker='o')
        ax.scatter(gt_positions_cm[-1, 0], gt_positions_cm[-1, 1], c='black', s=100, marker='x')
        
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title(f'Sample {idx+1} - Seq {result["seq_name"]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'test_trajectories_2d.png'), dpi=150)
    plt.close()
    
    # Plot error distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    all_trans_errors = np.concatenate([r['trans_errors'] for r in results]) * 100
    all_rot_errors = np.concatenate([r['rot_errors'] for r in results])
    
    # Translation error histogram
    ax1.hist(all_trans_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Translation Error (cm)')
    ax1.set_ylabel('Count')
    ax1.set_title('Translation Error Distribution')
    ax1.axvline(np.mean(all_trans_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_trans_errors):.2f}cm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotation error histogram
    ax2.hist(all_rot_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Rotation Error (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title('Rotation Error Distribution')
    ax2.axvline(np.mean(all_rot_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_rot_errors):.2f}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'error_distributions.png'), dpi=150)
    plt.close()
    
    print(f"\n📊 Plots saved to: {plot_dir}/")


def plot_scale_convergence(scale_history, output_dir):
    """Plot scale convergence over windows for each sequence"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (seq_id, scales) in enumerate(scale_history.items()):
        if idx >= 4:  # Only plot first 4 sequences
            break
        
        ax = axes[idx]
        windows = np.arange(len(scales))
        
        # Plot scale evolution
        ax.plot(windows, scales, 'b-', linewidth=2)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Target (1.0)')
        ax.axhline(y=0.95, color='g', linestyle=':', alpha=0.5)
        ax.axhline(y=1.05, color='g', linestyle=':', alpha=0.5, label='±5% bounds')
        
        # Add statistics
        mean_scale = np.mean(scales)
        ax.axhline(y=mean_scale, color='orange', linestyle='-', alpha=0.7, label=f'Mean: {mean_scale:.3f}')
        
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Scale (pred/gt)')
        ax.set_title(f'Sequence {seq_id}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim([0.7, 1.3])
    
    plt.suptitle('Scale Convergence Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scale_convergence.png'), dpi=150)
    plt.close()
    print(f"\n📊 Scale convergence plot saved to: {output_dir}/scale_convergence.png")


def save_relative_poses_csv(pred_rel_poses, gt_rel_poses, seq_id, output_dir):
    """Save relative poses to separate CSV files for ground truth and predictions
    
    Args:
        pred_rel_poses: Predicted relative poses [N, 7] where each pose is [x,y,z,qx,qy,qz,qw]
        gt_rel_poses: Ground truth relative poses [N, 7]
        seq_id: Sequence identifier
        output_dir: Output directory
    """
    # Ensure same length
    min_len = min(len(pred_rel_poses), len(gt_rel_poses))
    pred_rel_poses = pred_rel_poses[:min_len]
    gt_rel_poses = gt_rel_poses[:min_len]
    
    # Save ground truth relative poses
    gt_data = []
    for i in range(min_len):
        row = {
            'step': i,
            'x': gt_rel_poses[i, 0],
            'y': gt_rel_poses[i, 1],
            'z': gt_rel_poses[i, 2],
            'qx': gt_rel_poses[i, 3],
            'qy': gt_rel_poses[i, 4],
            'qz': gt_rel_poses[i, 5],
            'qw': gt_rel_poses[i, 6],
            'trans_norm': np.linalg.norm(gt_rel_poses[i, :3])
        }
        gt_data.append(row)
    
    gt_df = pd.DataFrame(gt_data)
    gt_csv_path = os.path.join(output_dir, f'relative_poses_{seq_id}_gt.csv')
    gt_df.to_csv(gt_csv_path, index=False)
    print(f"Saved ground truth relative poses to {gt_csv_path}")
    
    # Save predicted relative poses with errors
    pred_data = []
    for i in range(min_len):
        row = {
            'step': i,
            'x': pred_rel_poses[i, 0],
            'y': pred_rel_poses[i, 1],
            'z': pred_rel_poses[i, 2],
            'qx': pred_rel_poses[i, 3],
            'qy': pred_rel_poses[i, 4],
            'qz': pred_rel_poses[i, 5],
            'qw': pred_rel_poses[i, 6],
            'trans_norm': np.linalg.norm(pred_rel_poses[i, :3])
        }
        
        # Add errors compared to ground truth
        row['trans_error'] = np.linalg.norm(pred_rel_poses[i, :3] - gt_rel_poses[i, :3])
        row['scale_ratio'] = np.linalg.norm(pred_rel_poses[i, :3]) / (np.linalg.norm(gt_rel_poses[i, :3]) + 1e-8)
        
        # Rotation error
        pred_q = pred_rel_poses[i, 3:] / (np.linalg.norm(pred_rel_poses[i, 3:]) + 1e-8)
        gt_q = gt_rel_poses[i, 3:] / (np.linalg.norm(gt_rel_poses[i, 3:]) + 1e-8)
        dot = np.clip(np.dot(pred_q, gt_q), -1.0, 1.0)
        rot_error_deg = np.rad2deg(2 * np.arccos(np.abs(dot)))
        row['rot_error_deg'] = rot_error_deg
        
        # Angle between translation vectors (direction error)
        pred_t = pred_rel_poses[i, :3]
        gt_t = gt_rel_poses[i, :3]
        pred_norm = np.linalg.norm(pred_t)
        gt_norm = np.linalg.norm(gt_t)
        if pred_norm > 1e-6 and gt_norm > 1e-6:
            cos_angle = np.dot(pred_t, gt_t) / (pred_norm * gt_norm)
            angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            row['direction_angle_deg'] = angle_deg
        else:
            row['direction_angle_deg'] = np.nan
        
        pred_data.append(row)
    
    pred_df = pd.DataFrame(pred_data)
    pred_csv_path = os.path.join(output_dir, f'relative_poses_{seq_id}_pred.csv')
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved predicted relative poses to {pred_csv_path}")
    
    # Print summary statistics
    print(f"  Relative poses summary for sequence {seq_id}:")
    print(f"    Total steps: {len(pred_df)}")
    print(f"    Mean translation error: {pred_df['trans_error'].mean()*100:.2f} cm")
    print(f"    Mean rotation error: {pred_df['rot_error_deg'].mean():.2f}°")
    print(f"    Mean scale ratio: {pred_df['scale_ratio'].mean():.3f}")
    valid_angles = pred_df['direction_angle_deg'].dropna()
    if len(valid_angles) > 0:
        print(f"    Mean direction angle: {valid_angles.mean():.1f}°")


def save_evaluation_metrics(trans_errors, rot_errors, scale_errors, output_dir):
    """Save evaluation metrics to file"""
    results_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(results_file, 'w') as f:
        f.write("VIFT FROM SCRATCH EVALUATION RESULTS\n")
        f.write("=====================================\n\n")
        f.write("Translation Error (cm):\n")
        f.write(f"  Mean:   {np.mean(trans_errors) * 100:.2f}\n")
        f.write(f"  Std:    {np.std(trans_errors) * 100:.2f}\n")
        f.write(f"  Median: {np.median(trans_errors) * 100:.2f}\n")
        f.write(f"  95%:    {np.percentile(trans_errors, 95) * 100:.2f}\n")
        f.write(f"  Max:    {np.max(trans_errors) * 100:.2f}\n\n")
        f.write("Rotation Error (degrees):\n")
        f.write(f"  Mean:   {np.mean(rot_errors):.2f}\n")
        f.write(f"  Std:    {np.std(rot_errors):.2f}\n")
        f.write(f"  Median: {np.median(rot_errors):.2f}\n")
        f.write(f"  95%:    {np.percentile(rot_errors, 95):.2f}\n")
        f.write(f"  Max:    {np.max(rot_errors):.2f}\n\n")
        f.write("Scale Drift (%):\n")
        f.write(f"  Mean:   {np.mean(scale_errors):.2f}\n")
        f.write(f"  Std:    {np.std(scale_errors):.2f}\n")
        f.write(f"  Median: {np.median(scale_errors):.2f}\n")
        f.write(f"  95%:    {np.percentile(scale_errors, 95):.2f}\n")
        f.write(f"  Max:    {np.max(scale_errors):.2f}\n\n")
        f.write("Note: Model uses quaternion representation for rotations.\n")
        f.write("Errors are computed using geodesic distance on SO(3).\n")
    
    print(f"\n💾 Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VIFT model trained from scratch')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='aria_processed',
                        help='Directory with processed Aria data')
    parser.add_argument('--output-dir', type=str, default='evaluation_from_scratch',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--sim3', action='store_true',
                        help='Apply Sim(3) alignment to correct scale drift')
    # Removed --remove-gravity flag to maintain train/test consistency
    parser.add_argument('--scale-correction', type=float, default=1.0,
                        help='Global scale correction factor (default: 1.0, no correction)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Evaluating VIFT Model Trained from Scratch")
    print("="*60)
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Define test directory first
    test_dir = Path(args.data_dir) / 'test'
    if not test_dir.exists():
        print(f"\n❌ Test directory not found: {test_dir}")
        print("Please ensure test sequences are in the test directory")
        return
    
    # Derive test sequences automatically from dataset
    temp_dataset = AriaRawDataset(test_dir, sequence_length=21, stride=1)
    test_sequences = sorted({sample['seq_name'] for sample in temp_dataset.samples})
    print(f"\nFound {len(test_sequences)} test sequences: {test_sequences[:5]}..." if len(test_sequences) > 5 else f"\nFound test sequences: {test_sequences}")
    
    # If user wants specific sequences, filter here
    # For now, use first 11 sequences as requested
    if len(test_sequences) > 11:
        test_sequences = test_sequences[:11]
        print(f"Using first 11 sequences for evaluation: {test_sequences}")
    print(f"\n📁 Test sequences: {test_sequences}")
    
    # Use stride=1 to match training configuration
    test_dataset = AriaRawDataset(test_dir, sequence_length=21, stride=1)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda'
    )
    
    print(f"📁 Test dataset loaded: {len(test_dataset)} samples")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, str(output_dir), test_sequences, 
                           use_sim3_alignment=args.sim3,
                           scale_correction=args.scale_correction)
    
    print("\n✅ Evaluation complete!")
    print(f"📂 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()