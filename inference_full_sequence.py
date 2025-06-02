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
    pose_scale: float = 100.0
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
    
    # Store predictions for each window
    window_predictions = []
    window_indices = []
    
    # Move models to device
    encoder_model = encoder_model.to(device)
    vio_model = vio_model.to(device)
    encoder_model.eval()
    vio_model.eval()
    
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
        
        window_predictions.append(pred_poses)
        window_indices.append((start_idx, end_idx))
        num_windows += 1
    
    console.print(f"  Processed {num_windows} windows")
    
    # Mode 1: Simple aggregation - use non-overlapping predictions
    # For overlapping frames, we'll use the first prediction
    aggregated_poses = np.zeros((num_frames, 7))
    aggregated_poses[:, 3:] = [0, 0, 0, 1]  # Initialize with identity quaternions
    
    frame_counts = np.zeros(num_frames)
    
    for (start_idx, end_idx), pred_poses in zip(window_indices, window_predictions):
        for i in range(window_size):
            frame_idx = start_idx + i
            if frame_idx < num_frames:
                if frame_counts[frame_idx] == 0:  # Use first prediction
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


def calculate_metrics(results):
    """Calculate ATE and other metrics."""
    pred_traj = results['absolute_trajectory']
    gt_traj = results['ground_truth']
    
    # Ensure same length
    min_len = min(len(pred_traj), len(gt_traj))
    pred_traj = pred_traj[:min_len]
    gt_traj = gt_traj[:min_len]
    
    # Calculate ATE (Absolute Trajectory Error)
    ate_errors = []
    for pred, gt in zip(pred_traj, gt_traj):
        error = np.linalg.norm(pred[:3] - gt[:3])
        ate_errors.append(error)
    
    ate_errors = np.array(ate_errors)
    
    # Calculate rotation errors
    rot_errors = []
    for pred, gt in zip(pred_traj[1:], gt_traj[1:]):  # Skip first frame (origin)
        pred_r = Rotation.from_quat(pred[3:])
        gt_r = Rotation.from_quat(gt[3:])
        rel_r = pred_r * gt_r.inv()
        angle = np.abs(rel_r.magnitude())
        rot_errors.append(np.degrees(angle))
    
    rot_errors = np.array(rot_errors)
    
    return {
        'ate_mean': ate_errors.mean(),
        'ate_std': ate_errors.std(),
        'ate_median': np.median(ate_errors),
        'ate_95': np.percentile(ate_errors, 95),
        'rot_mean': rot_errors.mean(),
        'rot_std': rot_errors.std(),
        'total_frames': len(ate_errors)
    }


def main():
    parser = argparse.ArgumentParser(description='Full sequence inference for VIFT-AEA')
    parser.add_argument('--sequence-id', type=str, default='114',
                        help='Sequence ID from test set (114-142)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--encoder-path', type=str, 
                        default='pretrained_models/vf_512_if_256_3e-05.model',
                        help='Path to pretrained encoder')
    parser.add_argument('--processed-dir', type=str, default='data/aria_processed',
                        help='Directory with processed sequences')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    console.rule("[bold cyan]Full Sequence Inference Pipeline[/bold cyan]")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    console.print(f"Using device: {device}")
    
    # Load models
    console.print("\n[bold]Loading models...[/bold]")
    encoder_model = load_pretrained_encoder(args.encoder_path)
    vio_model = MultiHeadVIOModel.load_from_checkpoint(args.checkpoint)
    
    # Run inference
    sequence_path = Path(args.processed_dir) / args.sequence_id
    if not sequence_path.exists():
        console.print(f"[red]Error: Sequence {sequence_path} not found![/red]")
        return
    
    results = sliding_window_inference(
        sequence_path=sequence_path,
        encoder_model=encoder_model,
        vio_model=vio_model,
        device=device,
        stride=args.stride
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Display results
    table = Table(title="Full Sequence Inference Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Sequence", args.sequence_id)
    table.add_row("Total Frames", f"{metrics['total_frames']:,}")
    table.add_row("ATE Mean", f"{metrics['ate_mean']:.4f} cm")
    table.add_row("ATE Std", f"{metrics['ate_std']:.4f} cm")
    table.add_row("ATE Median", f"{metrics['ate_median']:.4f} cm")
    table.add_row("ATE 95%", f"{metrics['ate_95']:.4f} cm")
    table.add_row("Rotation Error Mean", f"{metrics['rot_mean']:.4f}°")
    table.add_row("Rotation Error Std", f"{metrics['rot_std']:.4f}°")
    
    console.print("\n")
    console.print(table)
    
    # Save trajectory for visualization
    output_path = Path(f"inference_results_seq_{args.sequence_id}_stride_{args.stride}.npz")
    np.savez(
        output_path,
        relative_poses=results['relative_poses'],
        absolute_trajectory=results['absolute_trajectory'],
        ground_truth=results['ground_truth'],
        metrics=metrics
    )
    console.print(f"\n✅ Saved results to {output_path}")
    
    # Show usage example
    console.print("\n[bold cyan]Visualize trajectory:[/bold cyan]")
    console.print(f"python visualize_trajectory.py --results {output_path}")


if __name__ == "__main__":
    main()