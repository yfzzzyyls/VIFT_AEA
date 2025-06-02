#!/usr/bin/env python3
"""
Full sequence inference pipeline for VIFT-AEA with pretrained features.

Pipeline:
1. Load processed data from aria_processed/
2. Extract features using pretrained encoder
3. Run sliding window inference with trained model
4. Accumulate relative poses to build trajectory
5. Evaluate against ground truth
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from rich.console import Console
from rich.table import Table
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('src')

from src.models.components.vsvio import Encoder
from src.models.multihead_vio import MultiHeadVIOModel

console = Console()


class WrapperModel(nn.Module):
    """Wrapper for pretrained encoder."""
    def __init__(self):
        super().__init__()
        class Params:
            v_f_len = 512
            i_f_len = 256
            img_w = 512
            img_h = 256
            imu_dropout = 0.2
            
        self.Feature_net = Encoder(Params())
        
    def forward(self, imgs, imus):
        # The model outputs 10 frames for 11 input frames
        v_feat, i_feat = self.Feature_net(imgs, imus)
        # Pad to 11 frames by repeating the last frame
        v_feat_padded = torch.cat([v_feat, v_feat[:, -1:, :]], dim=1)
        i_feat_padded = torch.cat([i_feat, i_feat[:, -1:, :]], dim=1)
        return torch.cat([v_feat_padded, i_feat_padded], dim=-1)


def load_pretrained_encoder(model_path):
    """Load pretrained Visual-Selective-VIO encoder."""
    model = WrapperModel()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    encoder_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('Feature_net.'):
            new_k = k.replace('Feature_net.', '')
            encoder_dict[new_k] = v
    
    model.Feature_net.load_state_dict(encoder_dict, strict=False)
    console.print(f"✅ Loaded pretrained encoder with {len(encoder_dict)} parameters")
    return model


def quaternion_to_matrix(q):
    """Convert quaternion (XYZW) to rotation matrix."""
    x, y, z, w = q
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])


def accumulate_poses(relative_poses):
    """
    Accumulate relative poses to build absolute trajectory.
    
    Args:
        relative_poses: [N, 7] array of relative poses (tx, ty, tz, qx, qy, qz, qw)
    
    Returns:
        absolute_poses: [N, 7] array of absolute poses
    """
    N = len(relative_poses)
    absolute_poses = np.zeros((N, 7))
    
    # Start at origin
    current_pos = np.zeros(3)
    current_rot = np.eye(3)
    
    # First pose is always at origin
    absolute_poses[0, :3] = [0, 0, 0]
    absolute_poses[0, 3:] = [0, 0, 0, 1]
    
    for i in range(1, N):
        # Get relative transformation
        rel_trans = relative_poses[i, :3]
        rel_quat = relative_poses[i, 3:]
        rel_rot = quaternion_to_matrix(rel_quat)
        
        # Update absolute position
        current_pos = current_pos + current_rot @ rel_trans
        current_rot = current_rot @ rel_rot
        
        # Convert back to quaternion
        r = Rotation.from_matrix(current_rot)
        current_quat = r.as_quat()  # [x, y, z, w]
        
        absolute_poses[i, :3] = current_pos
        absolute_poses[i, 3:] = current_quat
    
    return absolute_poses


def sliding_window_inference(
    sequence_path: Path,
    encoder_model: nn.Module,
    vio_model: nn.Module,
    device: torch.device,
    window_size: int = 11,
    stride: int = 1,
    pose_scale: float = 100.0,
    mode: str = 'independent'
):
    """
    Run sliding window inference on a full sequence.
    
    Args:
        sequence_path: Path to processed sequence (e.g., aria_processed/020)
        encoder_model: Pretrained encoder
        vio_model: Trained VIO model
        device: torch device
        window_size: Window size (default 11)
        stride: Stride for sliding window (default 1)
        pose_scale: Scale factor for poses
        mode: 'independent' or 'history' - inference mode
    
    Returns:
        predictions: Dict with trajectory and metrics
    """
    console.print(f"\n[bold cyan]Processing sequence: {sequence_path.name}[/bold cyan]")
    
    # Load sequence data
    visual_data = torch.load(sequence_path / "visual_data.pt")  # [N, 3, H, W]
    imu_data = torch.load(sequence_path / "imu_data.pt")        # [N, 33, 6]
    
    # Load ground truth poses
    with open(sequence_path / "poses.json", 'r') as f:
        poses_data = json.load(f)
    
    num_frames = visual_data.shape[0]
    console.print(f"  Total frames: {num_frames}")
    console.print(f"  Window size: {window_size}")
    console.print(f"  Stride: {stride}")
    console.print(f"  Mode: {mode}")
    
    if mode == 'independent':
        console.print(f"  Aggregation: Middle-priority (frames near window center preferred)")
    
    # Store predictions for each window
    window_predictions = []
    window_indices = []
    
    # Move models to device
    encoder_model = encoder_model.to(device)
    vio_model = vio_model.to(device)
    encoder_model.eval()
    vio_model.eval()
    
    # For history mode, maintain a buffer of past features
    if mode == 'history':
        history_size = 5  # Number of past windows to consider
        feature_history = []
        pose_history = []
    
    # Process with sliding window
    num_windows = 0
    for start_idx in tqdm(range(0, num_frames - window_size + 1, stride), desc="Sliding window inference"):
        end_idx = start_idx + window_size
        
        # Extract window
        window_visual = visual_data[start_idx:end_idx]  # [11, 3, H, W]
        window_imu = imu_data[start_idx:end_idx]        # [11, 33, 6]
        
        # Preprocess visual data
        window_visual_resized = F.interpolate(
            window_visual, 
            size=(256, 512), 
            mode='bilinear', 
            align_corners=False
        )
        window_visual_normalized = window_visual_resized - 0.5
        
        # Prepare IMU data (take first 10 samples per frame)
        window_imu_110 = []
        for i in range(window_size):
            window_imu_110.append(window_imu[i, :10, :])
        window_imu_110 = torch.cat(window_imu_110, dim=0)  # [110, 6]
        
        # Add batch dimension and move to device
        batch_visual = window_visual_normalized.unsqueeze(0).to(device)  # [1, 11, 3, 256, 512]
        batch_imu = window_imu_110.unsqueeze(0).to(device)              # [1, 110, 6]
        
        # Generate features using encoder
        with torch.no_grad():
            features = encoder_model(batch_visual, batch_imu)  # [1, 11, 768]
        
        if mode == 'history' and len(feature_history) > 0:
            # Blend current features with history
            # Use weighted average giving more weight to recent features
            blended_features = features.clone()
            
            # Apply temporal smoothing to overlapping frames
            overlap_frames = min(stride, window_size)
            if overlap_frames > 0 and len(feature_history) > 0:
                # Get weights for blending (exponential decay)
                weights = torch.exp(-torch.arange(overlap_frames, dtype=torch.float32) / overlap_frames).to(device)
                weights = weights / weights.sum()
                
                # Blend overlapping frames with previous window
                for i in range(overlap_frames):
                    if i < features.shape[1]:
                        prev_idx = window_size - overlap_frames + i
                        if prev_idx < feature_history[-1].shape[1]:
                            blended_features[0, i] = (
                                weights[i] * features[0, i] + 
                                (1 - weights[i]) * feature_history[-1][0, prev_idx]
                            )
            
            features = blended_features
        
        # Prepare batch for VIO model
        batch = {
            'images': features,
            'imus': torch.zeros(1, 11, 6).to(device),  # Dummy
            'poses': None  # Not needed for inference
        }
        
        # Run VIO model
        with torch.no_grad():
            predictions = vio_model(batch)
            # predictions['rotation']: [1, 11, 4]
            # predictions['translation']: [1, 11, 3]
        
        # Extract predictions
        pred_poses = torch.cat([
            predictions['translation'][0],  # [11, 3]
            predictions['rotation'][0]      # [11, 4]
        ], dim=1).cpu().numpy()  # [11, 7]
        
        if mode == 'history':
            # Update history buffers (keep on same device)
            feature_history.append(features)  # Keep on GPU
            pose_history.append(pred_poses)
            
            # Keep only recent history
            if len(feature_history) > history_size:
                feature_history.pop(0)
                pose_history.pop(0)
            
            # Apply temporal consistency constraints
            if len(pose_history) > 1:
                # Smooth predictions based on history
                for i in range(min(overlap_frames, pred_poses.shape[0])):
                    if i > 0:  # Don't modify first frame
                        # Blend with corresponding frame from previous window
                        prev_window_idx = window_size - overlap_frames + i - stride
                        if 0 <= prev_window_idx < pose_history[-2].shape[0]:
                            # Weighted average for translation
                            alpha = 0.7  # Weight for current prediction
                            pred_poses[i, :3] = (
                                alpha * pred_poses[i, :3] + 
                                (1 - alpha) * pose_history[-2][prev_window_idx, :3]
                            )
                            
                            # SLERP for rotation quaternions
                            q1 = pred_poses[i, 3:] / np.linalg.norm(pred_poses[i, 3:])
                            q2 = pose_history[-2][prev_window_idx, 3:] / np.linalg.norm(pose_history[-2][prev_window_idx, 3:])
                            
                            # Simple quaternion interpolation
                            dot = np.dot(q1, q2)
                            if dot < 0:
                                q2 = -q2
                                dot = -dot
                            
                            if dot > 0.9995:
                                # Linear interpolation for close quaternions
                                pred_poses[i, 3:] = alpha * q1 + (1 - alpha) * q2
                            else:
                                # Spherical interpolation
                                theta = np.arccos(np.clip(dot, -1, 1))
                                sin_theta = np.sin(theta)
                                if sin_theta > 0.001:
                                    w1 = np.sin(alpha * theta) / sin_theta
                                    w2 = np.sin((1 - alpha) * theta) / sin_theta
                                    pred_poses[i, 3:] = w1 * q1 + w2 * q2
                            
                            # Normalize quaternion
                            pred_poses[i, 3:] /= np.linalg.norm(pred_poses[i, 3:])
        
        window_predictions.append(pred_poses)
        window_indices.append((start_idx, end_idx))
        num_windows += 1
    
    console.print(f"  Processed {num_windows} windows")
    
    # Aggregation based on mode
    aggregated_poses = np.zeros((num_frames, 7))
    aggregated_poses[:, 3:] = [0, 0, 0, 1]  # Initialize with identity quaternions
    
    frame_counts = np.zeros(num_frames)
    
    if mode == 'independent':
        # Mode 1: Middle-priority aggregation - prioritize predictions from window center
        quality_scores = np.full(num_frames, -1.0)  # Track best quality score for each frame
        middle_idx = window_size // 2  # Center of window (index 5 for window_size=11)
        
        for (start_idx, end_idx), pred_poses in zip(window_indices, window_predictions):
            for i in range(window_size):
                frame_idx = start_idx + i
                if frame_idx < num_frames:
                    # Quality score: 1.0 at center, decreases linearly to edges
                    # For window_size=11: center (i=5) gets 1.0, edges (i=0,10) get 0.0
                    quality = 1.0 - abs(i - middle_idx) / middle_idx
                    
                    # Use this prediction if it has higher quality than previous
                    if quality > quality_scores[frame_idx]:
                        aggregated_poses[frame_idx] = pred_poses[i]
                        quality_scores[frame_idx] = quality
                    frame_counts[frame_idx] += 1
    
    else:  # mode == 'history'
        # Mode 2: Use predictions that have been smoothed with history
        # Priority to more recent predictions as they have more context
        for (start_idx, end_idx), pred_poses in zip(window_indices, window_predictions):
            for i in range(window_size):
                frame_idx = start_idx + i
                if frame_idx < num_frames:
                    # Always use the latest prediction (which has been smoothed)
                    aggregated_poses[frame_idx] = pred_poses[i]
                    frame_counts[frame_idx] += 1
    
    # Build absolute trajectory from relative poses
    absolute_trajectory = accumulate_poses(aggregated_poses)
    
    # Convert ground truth to relative poses for fair comparison
    gt_absolute = []
    for pose in poses_data:
        t = pose['translation']
        euler = pose['rotation_euler']
        r = Rotation.from_euler('xyz', euler)
        q = r.as_quat()  # [x, y, z, w]
        gt_absolute.append([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
    gt_absolute = np.array(gt_absolute)
    
    # Convert absolute GT to relative poses
    gt_relative = np.zeros_like(gt_absolute)
    gt_relative[0, :3] = [0, 0, 0]
    gt_relative[0, 3:] = [0, 0, 0, 1]
    
    for i in range(1, len(gt_absolute)):
        # Relative translation
        prev_t = gt_absolute[i-1, :3]
        curr_t = gt_absolute[i, :3]
        
        # Get rotation matrices
        prev_r = Rotation.from_quat(gt_absolute[i-1, 3:])
        curr_r = Rotation.from_quat(gt_absolute[i, 3:])
        
        # Relative translation in previous frame's coordinates
        rel_trans = prev_r.inv().apply(curr_t - prev_t)
        
        # Relative rotation
        rel_rot = prev_r.inv() * curr_r
        rel_quat = rel_rot.as_quat()
        
        gt_relative[i, :3] = rel_trans
        gt_relative[i, 3:] = rel_quat
    
    # Scale after conversion to relative
    gt_relative[:, :3] *= pose_scale
    
    # Build absolute trajectory from relative GT for verification
    gt_absolute_from_relative = accumulate_poses(gt_relative)
    
    return {
        'relative_poses': aggregated_poses,
        'absolute_trajectory': absolute_trajectory,
        'ground_truth': gt_absolute_from_relative,  # Use GT built from relative poses
        'ground_truth_relative': gt_relative,  # Also save relative GT for comparison
        'num_frames': num_frames,
        'num_windows': num_windows,
        'frame_overlap': frame_counts
    }


def sliding_window_inference_batched(
    sequence_path: Path,
    encoder_model: nn.Module,
    vio_model: nn.Module,
    device: torch.device,
    window_size: int = 11,
    stride: int = 1,
    pose_scale: float = 100.0,
    mode: str = 'independent',
    batch_size: int = 32,
    num_gpus: int = 4
):
    """
    Run sliding window inference on a full sequence with batched processing.
    
    Args:
        sequence_path: Path to processed sequence
        encoder_model: Pretrained encoder
        vio_model: Trained VIO model
        device: torch device
        window_size: Window size (default 11)
        stride: Stride for sliding window (default 1)
        pose_scale: Scale factor for poses
        mode: 'independent' or 'history' - inference mode
        batch_size: Number of windows to process simultaneously
    
    Returns:
        predictions: Dict with trajectory and metrics
    """
    console.print(f"\n[bold cyan]Processing sequence: {sequence_path.name}[/bold cyan]")
    
    # Load sequence data
    visual_data = torch.load(sequence_path / "visual_data.pt")  # [N, 3, H, W]
    imu_data = torch.load(sequence_path / "imu_data.pt")        # [N, 33, 6]
    
    # Load ground truth poses
    with open(sequence_path / "poses.json", 'r') as f:
        poses_data = json.load(f)
    
    num_frames = visual_data.shape[0]
    console.print(f"  Total frames: {num_frames}")
    console.print(f"  Window size: {window_size}")
    console.print(f"  Stride: {stride}")
    console.print(f"  Mode: {mode}")
    console.print(f"  Batch size: {batch_size}")
    
    if mode == 'independent':
        console.print(f"  Aggregation: Middle-priority (frames near window center preferred)")
    
    # Store predictions for each window
    window_predictions = []
    window_indices = []
    
    # Setup multi-GPU if available
    if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
        console.print(f"  Using {num_gpus} GPUs for parallel processing")
        # Create model copies for each GPU
        encoder_models = []
        vio_models = []
        for gpu_id in range(num_gpus):
            device_gpu = torch.device(f'cuda:{gpu_id}')
            encoder_copy = WrapperModel()
            encoder_copy.load_state_dict(encoder_model.state_dict())
            encoder_copy = encoder_copy.to(device_gpu)
            encoder_copy.eval()
            encoder_models.append(encoder_copy)
            
            vio_copy = type(vio_model)()
            vio_copy.load_state_dict(vio_model.state_dict())
            vio_copy = vio_copy.to(device_gpu)
            vio_copy.eval()
            vio_models.append(vio_copy)
    else:
        console.print(f"  Using single GPU")
        num_gpus = 1
        encoder_models = [encoder_model.to(device)]
        vio_models = [vio_model.to(device)]
        encoder_models[0].eval()
        vio_models[0].eval()
    
    # Collect all windows first
    all_windows_visual = []
    all_windows_imu = []
    all_start_indices = []
    
    for start_idx in range(0, num_frames - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Extract window
        window_visual = visual_data[start_idx:end_idx]  # [11, 3, H, W]
        window_imu = imu_data[start_idx:end_idx]        # [11, 33, 6]
        
        # Preprocess visual data
        window_visual_resized = F.interpolate(
            window_visual, 
            size=(256, 512), 
            mode='bilinear', 
            align_corners=False
        )
        window_visual_normalized = window_visual_resized - 0.5
        
        # Prepare IMU data (take first 10 samples per frame)
        window_imu_110 = []
        for i in range(window_size):
            window_imu_110.append(window_imu[i, :10, :])
        window_imu_110 = torch.cat(window_imu_110, dim=0)  # [110, 6]
        
        all_windows_visual.append(window_visual_normalized)
        all_windows_imu.append(window_imu_110)
        all_start_indices.append(start_idx)
    
    num_windows = len(all_windows_visual)
    console.print(f"  Total windows: {num_windows}")
    
    # Process in batches with multi-GPU
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_windows, batch_size * num_gpus), desc="Batched inference"):
            batch_end = min(batch_start + batch_size * num_gpus, num_windows)
            
            # Split batch across GPUs
            gpu_predictions = []
            gpu_batch_sizes = []
            
            for gpu_id in range(num_gpus):
                gpu_batch_start = batch_start + gpu_id * batch_size
                gpu_batch_end = min(gpu_batch_start + batch_size, batch_end)
                
                if gpu_batch_start >= batch_end:
                    break
                
                # Stack batch for this GPU
                gpu_device = torch.device(f'cuda:{gpu_id}' if num_gpus > 1 else device)
                batch_visual = torch.stack(all_windows_visual[gpu_batch_start:gpu_batch_end]).to(gpu_device)
                batch_imu = torch.stack(all_windows_imu[gpu_batch_start:gpu_batch_end]).to(gpu_device)
                
                # Generate features using encoder on this GPU
                features = encoder_models[gpu_id](batch_visual, batch_imu)  # [B, 11, 768]
                
                # Prepare batch for VIO model
                batch = {
                    'images': features,
                    'imus': torch.zeros(features.shape[0], 11, 6).to(gpu_device),  # Dummy
                    'poses': None  # Not needed for inference
                }
                
                # Run VIO model on this GPU
                predictions = vio_models[gpu_id](batch)
                gpu_predictions.append(predictions)
                gpu_batch_sizes.append(gpu_batch_end - gpu_batch_start)
            
            # Collect predictions from all GPUs
            for gpu_id, (predictions, gpu_batch_size) in enumerate(zip(gpu_predictions, gpu_batch_sizes)):
                gpu_batch_start = batch_start + gpu_id * batch_size
                
                # Extract predictions for each window in this GPU's batch
                for i in range(gpu_batch_size):
                    pred_poses = torch.cat([
                        predictions['translation'][i],  # [11, 3]
                        predictions['rotation'][i]      # [11, 4]
                    ], dim=1).cpu().numpy()  # [11, 7]
                    
                    start_idx = all_start_indices[gpu_batch_start + i]
                    window_predictions.append(pred_poses)
                    window_indices.append((start_idx, start_idx + window_size))
    
    console.print(f"  Processed {num_windows} windows in {(num_windows + batch_size * num_gpus - 1) // (batch_size * num_gpus)} batches")
    
    # Aggregation (same as before)
    aggregated_poses = np.zeros((num_frames, 7))
    aggregated_poses[:, 3:] = [0, 0, 0, 1]  # Initialize with identity quaternions
    
    frame_counts = np.zeros(num_frames)
    
    if mode == 'independent':
        # Mode 1: Middle-priority aggregation
        quality_scores = np.full(num_frames, -1.0)
        middle_idx = window_size // 2
        
        for (start_idx, end_idx), pred_poses in zip(window_indices, window_predictions):
            for i in range(window_size):
                frame_idx = start_idx + i
                if frame_idx < num_frames:
                    quality = 1.0 - abs(i - middle_idx) / middle_idx
                    
                    if quality > quality_scores[frame_idx]:
                        aggregated_poses[frame_idx] = pred_poses[i]
                        quality_scores[frame_idx] = quality
                    frame_counts[frame_idx] += 1
    
    else:  # mode == 'history'
        # For history mode with batching, we need a different approach
        # For now, use simple last-write-wins
        for (start_idx, end_idx), pred_poses in zip(window_indices, window_predictions):
            for i in range(window_size):
                frame_idx = start_idx + i
                if frame_idx < num_frames:
                    aggregated_poses[frame_idx] = pred_poses[i]
                    frame_counts[frame_idx] += 1
    
    # Build absolute trajectory from relative poses
    absolute_trajectory = accumulate_poses(aggregated_poses)
    
    # Convert ground truth to relative poses (same as before)
    gt_absolute = []
    for pose in poses_data:
        t = pose['translation']
        euler = pose['rotation_euler']
        r = Rotation.from_euler('xyz', euler)
        q = r.as_quat()  # [x, y, z, w]
        gt_absolute.append([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
    gt_absolute = np.array(gt_absolute)
    
    # Convert absolute GT to relative poses
    gt_relative = np.zeros_like(gt_absolute)
    gt_relative[0, :3] = [0, 0, 0]
    gt_relative[0, 3:] = [0, 0, 0, 1]
    
    for i in range(1, len(gt_absolute)):
        # Relative translation
        prev_t = gt_absolute[i-1, :3]
        curr_t = gt_absolute[i, :3]
        
        # Get rotation matrices
        prev_r = Rotation.from_quat(gt_absolute[i-1, 3:])
        curr_r = Rotation.from_quat(gt_absolute[i, 3:])
        
        # Relative translation in previous frame's coordinates
        rel_trans = prev_r.inv().apply(curr_t - prev_t)
        
        # Relative rotation
        rel_rot = prev_r.inv() * curr_r
        rel_quat = rel_rot.as_quat()
        
        gt_relative[i, :3] = rel_trans
        gt_relative[i, 3:] = rel_quat
    
    # Scale after conversion to relative
    gt_relative[:, :3] *= pose_scale
    
    # Build absolute trajectory from relative GT for verification
    gt_absolute_from_relative = accumulate_poses(gt_relative)
    
    return {
        'relative_poses': aggregated_poses,
        'absolute_trajectory': absolute_trajectory,
        'ground_truth': gt_absolute_from_relative,
        'ground_truth_relative': gt_relative,
        'num_frames': num_frames,
        'num_windows': num_windows,
        'frame_overlap': frame_counts
    }


def calculate_metrics(results):
    """Calculate ATE, RPE and other metrics."""
    pred_traj = results['absolute_trajectory']
    gt_traj = results['ground_truth']
    pred_rel = results['relative_poses']
    gt_rel = results['ground_truth_relative']
    
    # Ensure same length
    min_len = min(len(pred_traj), len(gt_traj))
    pred_traj = pred_traj[:min_len]
    gt_traj = gt_traj[:min_len]
    pred_rel = pred_rel[:min_len]
    gt_rel = gt_rel[:min_len]
    
    # Calculate ATE (Absolute Trajectory Error)
    ate_errors = []
    for pred, gt in zip(pred_traj, gt_traj):
        error = np.linalg.norm(pred[:3] - gt[:3])
        ate_errors.append(error)
    
    ate_errors = np.array(ate_errors)
    
    # Calculate rotation errors for absolute poses
    rot_errors = []
    for pred, gt in zip(pred_traj[1:], gt_traj[1:]):  # Skip first frame (origin)
        pred_r = Rotation.from_quat(pred[3:])
        gt_r = Rotation.from_quat(gt[3:])
        rel_r = pred_r * gt_r.inv()
        angle = np.abs(rel_r.magnitude())
        rot_errors.append(np.degrees(angle))
    
    rot_errors = np.array(rot_errors)
    
    # Calculate RPE (Relative Pose Error) - frame-to-frame accuracy
    rpe_trans_1 = []
    rpe_rot_1 = []
    
    # Skip first frame (always origin)
    for i in range(1, len(pred_rel)):
        # Translation error
        trans_error = np.linalg.norm(pred_rel[i, :3] - gt_rel[i, :3])
        rpe_trans_1.append(trans_error)
        
        # Rotation error using quaternion distance
        pred_q = pred_rel[i, 3:]
        gt_q = gt_rel[i, 3:]
        
        # Normalize quaternions
        pred_q = pred_q / (np.linalg.norm(pred_q) + 1e-8)
        gt_q = gt_q / (np.linalg.norm(gt_q) + 1e-8)
        
        # Compute angle between quaternions
        dot = np.clip(np.abs(np.dot(pred_q, gt_q)), 0, 1)
        angle = 2 * np.arccos(dot)
        rpe_rot_1.append(np.degrees(angle))
    
    rpe_trans_1 = np.array(rpe_trans_1)
    rpe_rot_1 = np.array(rpe_rot_1)
    
    # Calculate RPE at 5 frames
    rpe_trans_5 = []
    rpe_rot_5 = []
    
    if len(pred_traj) > 5:
        for i in range(len(pred_traj) - 5):
            # Get 5-frame relative motion
            start_pos_pred = pred_traj[i, :3]
            end_pos_pred = pred_traj[i + 5, :3]
            start_pos_gt = gt_traj[i, :3]
            end_pos_gt = gt_traj[i + 5, :3]
            
            # Translation error over 5 frames
            pred_motion = end_pos_pred - start_pos_pred
            gt_motion = end_pos_gt - start_pos_gt
            trans_error = np.linalg.norm(pred_motion - gt_motion)
            rpe_trans_5.append(trans_error)
            
            # Rotation error over 5 frames
            start_rot_pred = Rotation.from_quat(pred_traj[i, 3:])
            end_rot_pred = Rotation.from_quat(pred_traj[i + 5, 3:])
            start_rot_gt = Rotation.from_quat(gt_traj[i, 3:])
            end_rot_gt = Rotation.from_quat(gt_traj[i + 5, 3:])
            
            rel_rot_pred = start_rot_pred.inv() * end_rot_pred
            rel_rot_gt = start_rot_gt.inv() * end_rot_gt
            rel_error = rel_rot_pred * rel_rot_gt.inv()
            angle = np.abs(rel_error.magnitude())
            rpe_rot_5.append(np.degrees(angle))
    
    rpe_trans_5 = np.array(rpe_trans_5) if rpe_trans_5 else np.array([0])
    rpe_rot_5 = np.array(rpe_rot_5) if rpe_rot_5 else np.array([0])
    
    return {
        'ate_mean': ate_errors.mean(),
        'ate_std': ate_errors.std(),
        'ate_median': np.median(ate_errors),
        'ate_95': np.percentile(ate_errors, 95),
        'rot_mean': rot_errors.mean(),
        'rot_std': rot_errors.std(),
        'rpe_trans_1_mean': rpe_trans_1.mean(),
        'rpe_trans_1_std': rpe_trans_1.std(),
        'rpe_rot_1_mean': rpe_rot_1.mean(),
        'rpe_rot_1_std': rpe_rot_1.std(),
        'rpe_trans_5_mean': rpe_trans_5.mean(),
        'rpe_trans_5_std': rpe_trans_5.std(),
        'rpe_rot_5_mean': rpe_rot_5.mean(),
        'rpe_rot_5_std': rpe_rot_5.std(),
        'total_frames': len(ate_errors)
    }


def main():
    parser = argparse.ArgumentParser(description='Full sequence inference for VIFT-AEA')
    parser.add_argument('--sequence-id', type=str, default='all',
                        help='Sequence ID from test set (114-142) or "all" for all test sequences')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--encoder-path', type=str, 
                        default='pretrained_models/vf_512_if_256_3e-05.model',
                        help='Path to pretrained encoder')
    parser.add_argument('--processed-dir', type=str, default='data/aria_processed',
                        help='Directory with processed sequences')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window')
    parser.add_argument('--mode', type=str, default='independent',
                        choices=['independent', 'history'],
                        help='Inference mode: independent or history-based')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs to use (default: 4)')
    
    args = parser.parse_args()
    
    console.rule("[bold cyan]Full Sequence Inference Pipeline[/bold cyan]")
    
    # Set device and check GPUs
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    console.print(f"Using device: {device}")
    if torch.cuda.is_available():
        console.print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            console.print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load models
    console.print("\n[bold]Loading models...[/bold]")
    encoder_model = load_pretrained_encoder(args.encoder_path)
    vio_model = MultiHeadVIOModel.load_from_checkpoint(args.checkpoint)
    
    # Determine which sequences to run
    if args.sequence_id.lower() == 'all':
        # Test sequences are typically 114-142 (last ~20% of data)
        # Get all numeric sequence directories
        all_seqs = sorted([d.name for d in Path(args.processed_dir).iterdir() 
                          if d.is_dir() and d.name.isdigit()])
        # Take last 20% as test set (matching train script split)
        num_test = max(1, int(len(all_seqs) * 0.2))
        test_sequences = all_seqs[-num_test:]
        console.print(f"\nRunning on {len(test_sequences)} test sequences: {test_sequences[0]} to {test_sequences[-1]}")
    else:
        test_sequences = [args.sequence_id]
    
    # Store metrics for all sequences
    all_metrics = []
    
    for seq_id in test_sequences:
        sequence_path = Path(args.processed_dir) / seq_id
        if not sequence_path.exists():
            console.print(f"[red]Warning: Sequence {sequence_path} not found, skipping![/red]")
            continue
        
        # Always use batched inference for efficiency
        results = sliding_window_inference_batched(
            sequence_path=sequence_path,
            encoder_model=encoder_model,
            vio_model=vio_model,
            device=device,
            stride=args.stride,
            mode=args.mode,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus
        )
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        metrics['sequence_id'] = seq_id
        all_metrics.append(metrics)
        
        # Save individual results
        if len(test_sequences) == 1:
            output_path = Path(f"inference_results_seq_{seq_id}_stride_{args.stride}_mode_{args.mode}.npz")
        else:
            output_dir = Path(f"inference_results_all_stride_{args.stride}_mode_{args.mode}")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"seq_{seq_id}.npz"
        
        np.savez(
            output_path,
            relative_poses=results['relative_poses'],
            absolute_trajectory=results['absolute_trajectory'],
            ground_truth=results['ground_truth'],
            metrics=metrics
        )
    
    # Display results
    if len(all_metrics) == 1:
        # Single sequence - show detailed results
        metrics = all_metrics[0]
        table = Table(title="Full Sequence Inference Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Sequence", metrics['sequence_id'])
        table.add_row("Total Frames", f"{metrics['total_frames']:,}")
        table.add_row("ATE Mean", f"{metrics['ate_mean']:.4f} cm")
        table.add_row("ATE Std", f"{metrics['ate_std']:.4f} cm")
        table.add_row("ATE Median", f"{metrics['ate_median']:.4f} cm")
        table.add_row("ATE 95%", f"{metrics['ate_95']:.4f} cm")
        table.add_row("Rotation Error Mean", f"{metrics['rot_mean']:.4f}°")
        table.add_row("Rotation Error Std", f"{metrics['rot_std']:.4f}°")
        
        console.print("\n")
        console.print(table)
    else:
        # Multiple sequences - show per-sequence and averaged results
        console.print("\n")
        per_seq_table = Table(title="Per-Sequence Results")
        per_seq_table.add_column("Sequence", style="cyan")
        per_seq_table.add_column("Frames", style="white")
        per_seq_table.add_column("ATE Mean", style="green")
        per_seq_table.add_column("RPE Trans(1)", style="green")
        per_seq_table.add_column("RPE Rot(1)", style="green")
        per_seq_table.add_column("Status", style="white")
        
        for m in all_metrics:
            status = "✅" if m['ate_mean'] < 1.0 else "❌"
            per_seq_table.add_row(
                m['sequence_id'],
                f"{m['total_frames']:,}",
                f"{m['ate_mean']:.4f} cm",
                f"{m['rpe_trans_1_mean']:.4f} cm",
                f"{m['rpe_rot_1_mean']:.4f}°",
                status
            )
        
        console.print(per_seq_table)
        
        # Calculate averaged metrics
        avg_metrics = {}
        for key in ['ate_mean', 'ate_std', 'ate_median', 'ate_95', 
                    'rot_mean', 'rot_std', 'rpe_trans_1_mean', 'rpe_trans_1_std',
                    'rpe_rot_1_mean', 'rpe_rot_1_std', 'rpe_trans_5_mean', 'rpe_trans_5_std',
                    'rpe_rot_5_mean', 'rpe_rot_5_std']:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[key + '_std_across_seqs'] = np.std(values)
        
        # Display averaged results
        console.print("\n")
        avg_table = Table(title=f"Averaged Results over {len(all_metrics)} Test Sequences")
        avg_table.add_column("Metric", style="cyan")
        avg_table.add_column("Mean ± Std", style="green")
        avg_table.add_column("Std Across Seqs", style="yellow")
        
        avg_table.add_row("ATE Mean", 
                         f"{avg_metrics['ate_mean']:.4f} ± {avg_metrics['ate_std']:.4f} cm",
                         f"{avg_metrics['ate_mean_std_across_seqs']:.4f} cm")
        avg_table.add_row("ATE Median", 
                         f"{avg_metrics['ate_median']:.4f} cm",
                         f"{avg_metrics['ate_median_std_across_seqs']:.4f} cm")
        avg_table.add_row("RPE Trans (1 frame)", 
                         f"{avg_metrics['rpe_trans_1_mean']:.4f} ± {avg_metrics['rpe_trans_1_std']:.4f} cm",
                         f"{avg_metrics['rpe_trans_1_mean_std_across_seqs']:.4f} cm")
        avg_table.add_row("RPE Rot (1 frame)", 
                         f"{avg_metrics['rpe_rot_1_mean']:.4f} ± {avg_metrics['rpe_rot_1_std']:.4f}°",
                         f"{avg_metrics['rpe_rot_1_mean_std_across_seqs']:.4f}°")
        avg_table.add_row("Absolute Rotation Error", 
                         f"{avg_metrics['rot_mean']:.4f} ± {avg_metrics['rot_std']:.4f}°",
                         f"{avg_metrics['rot_mean_std_across_seqs']:.4f}°")
        
        console.print(avg_table)
        
        # Use averaged metrics for performance comparison
        metrics = avg_metrics
    
    # Display performance vs industry standards
    console.print("\n")
    perf_table = Table(title="Performance vs AR/VR Industry Standards")
    perf_table.add_column("Metric", style="cyan", width=40)
    perf_table.add_column("Our Model", style="green", width=25)
    perf_table.add_column("AR/VR Target", style="yellow", width=15)
    perf_table.add_column("Status", style="white", width=15)
    
    # ATE - Accumulated error over entire trajectory
    ate_status = "✅ EXCEEDS" if metrics['ate_mean'] < 1.0 else "❌ FAILS"
    perf_table.add_row(
        "ATE (Absolute Trajectory Error)",
        f"{metrics['ate_mean']:.4f} ± {metrics['ate_std']:.4f} cm",
        "<1 cm",
        ate_status
    )
    perf_table.add_row(
        "  ├─ Median",
        f"{metrics['ate_median']:.4f} cm",
        "-",
        "-"
    )
    perf_table.add_row(
        "  └─ 95th percentile",
        f"{metrics['ate_95']:.4f} cm",
        "-",
        "-"
    )
    
    # RPE-1: Frame-to-frame accuracy (critical for AR/VR)
    rpe_trans_1_status = "✅ EXCEEDS" if metrics['rpe_trans_1_mean'] < 0.1 else "❌ FAILS"
    perf_table.add_row(
        "RPE-1 Translation (frame-to-frame)",
        f"{metrics['rpe_trans_1_mean']:.4f} ± {metrics['rpe_trans_1_std']:.4f} cm",
        "<0.1 cm",
        rpe_trans_1_status
    )
    
    # RPE-1 Rotation
    rpe_rot_1_status = "✅ EXCEEDS" if metrics['rpe_rot_1_mean'] < 0.1 else ("⚠️  CLOSE" if metrics['rpe_rot_1_mean'] < 0.5 else "❌ FAILS")
    perf_table.add_row(
        "RPE-1 Rotation (frame-to-frame)",
        f"{metrics['rpe_rot_1_mean']:.4f} ± {metrics['rpe_rot_1_std']:.4f}°",
        "<0.1°",
        rpe_rot_1_status
    )
    
    # RPE-5: Short-term accuracy (~167ms @ 30fps)
    rpe_trans_5_status = "✅ EXCEEDS" if metrics['rpe_trans_5_mean'] < 0.5 else "❌ FAILS"
    perf_table.add_row(
        "RPE-5 Translation (167ms window)",
        f"{metrics['rpe_trans_5_mean']:.4f} ± {metrics['rpe_trans_5_std']:.4f} cm",
        "<0.5 cm",
        rpe_trans_5_status
    )
    
    # RPE-5 Rotation
    rpe_rot_5_status = "✅ EXCEEDS" if metrics['rpe_rot_5_mean'] < 0.5 else "❌ FAILS"
    perf_table.add_row(
        "RPE-5 Rotation (167ms window)",
        f"{metrics['rpe_rot_5_mean']:.4f} ± {metrics['rpe_rot_5_std']:.4f}°",
        "<0.5°",
        rpe_rot_5_status
    )
    
    # Absolute Rotation Error (accumulated)
    rot_status = "✅ EXCEEDS" if metrics['rot_mean'] < 0.1 else ("⚠️  CLOSE" if metrics['rot_mean'] < 0.5 else "❌ FAILS")
    perf_table.add_row(
        "Absolute Rotation Error",
        f"{metrics['rot_mean']:.4f} ± {metrics['rot_std']:.4f}°",
        "<0.1°",
        rot_status
    )
    
    console.print(perf_table)
    
    # Add metric explanations
    console.print("\n[bold]Metric Explanations:[/bold]")
    console.print("• ATE (Absolute Trajectory Error): Cumulative position drift over entire sequence (500 frames)")
    console.print("• RPE-1 (Relative Pose Error @ 1 frame): Frame-to-frame accuracy (33ms @ 30fps)")
    console.print("• RPE-5 (Relative Pose Error @ 5 frames): Short-term accuracy (167ms @ 30fps)")
    console.print("• Absolute Rotation Error: Accumulated orientation drift from start to end of sequence")
    
    # Performance summary
    console.print("\n[bold]Performance Summary:[/bold]")
    if metrics['ate_mean'] < 0.01:
        console.print(f"[bold green]✅ EXCEPTIONAL! Sub-millimeter accuracy of {metrics['ate_mean']:.4f} cm![/bold green]")
    elif metrics['ate_mean'] < 0.1:
        console.print(f"[bold green]✅ EXCELLENT! ATE of {metrics['ate_mean']:.4f} cm far exceeds AR/VR requirements![/bold green]")
    elif metrics['ate_mean'] < 1.0:
        console.print(f"[bold yellow]⚠️  GOOD. ATE of {metrics['ate_mean']:.4f} cm meets AR/VR requirements.[/bold yellow]")
    else:
        console.print(f"[bold red]❌ NEEDS IMPROVEMENT. ATE of {metrics['ate_mean']:.4f} cm exceeds 1cm threshold.[/bold red]")
    
    if args.mode == 'independent':
        console.print("\n[dim]Note: Using middle-priority aggregation for improved accuracy[/dim]")
    
    if len(all_metrics) > 1:
        # Save averaged metrics
        avg_output_path = Path(f"inference_results_averaged_stride_{args.stride}_mode_{args.mode}.json")
        import json
        with open(avg_output_path, 'w') as f:
            json.dump({
                'num_sequences': len(all_metrics),
                'sequences': [m['sequence_id'] for m in all_metrics],
                'averaged_metrics': avg_metrics,
                'per_sequence_metrics': all_metrics
            }, f, indent=2)
        console.print(f"\n✅ Saved averaged results to {avg_output_path}")
        console.print(f"✅ Individual trajectories saved to inference_results_all_stride_{args.stride}_mode_{args.mode}/")
    else:
        console.print(f"\n✅ Saved results to {output_path}")
        console.print("\n[bold cyan]Visualize trajectory:[/bold cyan]")
        console.print(f"python visualize_trajectory.py --results {output_path}")


if __name__ == "__main__":
    main()