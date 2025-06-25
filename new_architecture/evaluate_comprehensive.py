#!/usr/bin/env python3
"""
Comprehensive evaluation script for FlowNet-LSTM-Transformer architecture.
Generates outputs matching evaluation_from_scratch format.

Usage:
    python evaluate_comprehensive.py \
        --checkpoint checkpoints/flownet_cnn_rotation1000_stride5_first500frames/best_model.pt \
        --output-dir evaluation_comprehensive
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import math
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

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.flownet_lstm_transformer import FlowNetLSTMTransformer
from data.aria_dataset_mmap_shared import AriaDatasetMMapShared, collate_mmap_shared
from configs.flownet_lstm_transformer_config import Config, ModelConfig

# Import alignment functions from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umeyama_alignment import align_trajectory, compute_ate


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("Warning: No config found in checkpoint, using defaults")
        from configs.flownet_lstm_transformer_config import get_config
        config = get_config()
    
    # Create model
    if isinstance(config, dict):
        model_config = ModelConfig(**config.get('model', {}))
        model = FlowNetLSTMTransformer(model_config)
    else:
        model = FlowNetLSTMTransformer(config.model)
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    # For now, don't use DataParallel as it doesn't handle list inputs well
    # We'll use single GPU evaluation which is still fast enough
    model = model.to(device)
    model.eval()
    
    if torch.cuda.device_count() > 1:
        print(f"Note: {torch.cuda.device_count()} GPUs available, but using single GPU for compatibility")
    
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Print model architecture details
    print("\nModel Architecture:")
    # Handle DataParallel wrapper
    base_model = model.module if hasattr(model, 'module') else model
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Visual encoder: {base_model.visual_encoder.__class__.__name__}")
    print(f"  IMU encoder: {base_model.imu_encoder.__class__.__name__}")
    # The transformer is inside pose_predictor
    if hasattr(base_model, 'pose_predictor') and hasattr(base_model.pose_predictor, 'transformer'):
        print(f"  Transformer layers: {len(base_model.pose_predictor.transformer.layers)}")
    else:
        print(f"  Pose predictor: {base_model.pose_predictor.__class__.__name__}")
    
    # Print training loss from checkpoint
    if 'val_losses' in checkpoint:
        val_losses = checkpoint['val_losses']
        print(f"\nCheckpoint validation losses:")
        print(f"  Total: {val_losses['total_loss']:.4f}")
        print(f"  Translation: {val_losses['translation_loss']:.4f}")
        print(f"  Rotation: {val_losses['rotation_loss'] * 180 / math.pi:.2f}°")
        print(f"  Scale: {val_losses['scale_loss']:.4f}")
    
    return model, config


def integrate_trajectory(relative_poses, initial_pose=None):
    """Integrate relative poses to get absolute trajectory"""
    if initial_pose is None:
        initial_pose = np.array([0, 0, 0, 0, 0, 0, 1])  # x,y,z,qx,qy,qz,qw
    
    positions = [initial_pose[:3]]
    rotations = [initial_pose[3:]]
    
    current_position = initial_pose[:3].copy()
    current_rotation = R.from_quat(initial_pose[3:])
    
    for rel_pose in relative_poses:
        # Extract relative translation and rotation
        rel_trans = rel_pose[:3]
        rel_rot = R.from_quat(rel_pose[3:])
        
        # Update position in world frame
        current_position = current_position + current_rotation.apply(rel_trans)
        current_rotation = current_rotation * rel_rot
        
        positions.append(current_position.copy())
        rotations.append(current_rotation.as_quat())
    
    positions = np.array(positions)
    rotations = np.array(rotations)
    
    return positions, rotations


def plot_trajectory_3d(pred_positions, gt_positions, sequence_name, output_path, 
                      time_window=None, title_suffix=''):
    """Create 3D trajectory visualization"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to centimeters
    gt_positions_cm = gt_positions * 100
    pred_positions_cm = pred_positions * 100
    
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
    
    # Set equal aspect ratio
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass
    
    if time_window:
        ax.set_title(f'Sequence {sequence_name} - First {time_window}{title_suffix}\n'
                    f'GT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm')
    else:
        ax.set_title(f'Sequence {sequence_name} - Full Trajectory{title_suffix}\n'
                    f'GT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm')
    
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
    
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
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
        name='End (GT)',
        marker=dict(color='black', size=10, symbol='x'),
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Interactive 3D Trajectory - Sequence {sequence_name}',
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)',
            zaxis_title='Z (cm)',
            aspectmode='data'
        ),
        showlegend=True,
        height=800
    )
    
    # Save to HTML
    fig.write_html(output_path)
    print(f"Saved interactive plot to {output_path}")


def save_trajectory_csv(pred_positions, gt_positions, pred_rotations, gt_rotations, 
                       seq_id, output_dir, frame_offset=0):
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
            row.update({
                'qx': pred_rotations[i, 0],
                'qy': pred_rotations[i, 1],
                'qz': pred_rotations[i, 2],
                'qw': pred_rotations[i, 3]
            })
        
        # Add errors
        if i < len(gt_positions):
            row['trans_error'] = np.linalg.norm(pred_positions[i] - gt_positions[i])
            
            if i < len(gt_rotations) and i < len(pred_rotations):
                gt_rot = R.from_quat(gt_rotations[i])
                pred_rot = R.from_quat(pred_rotations[i])
                rel_rot = gt_rot.inv() * pred_rot
                rot_error = np.abs(rel_rot.magnitude() * 180 / np.pi)
                row['rot_error_deg'] = rot_error
        
        pred_data.append(row)
    
    pred_df = pd.DataFrame(pred_data)
    pred_csv_path = os.path.join(output_dir, f'trajectory_{seq_id}_pred.csv')
    pred_df.to_csv(pred_csv_path, index=False)


def plot_sample_trajectories(all_trajectories, output_dir):
    """Plot sample trajectories in 2D grid"""
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot first 8 sequences
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx in range(min(8, len(all_trajectories))):
        ax = axes[idx]
        traj = all_trajectories[idx]
        
        # Convert to cm
        gt_positions_cm = traj['gt_positions'] * 100
        pred_positions_cm = traj['pred_positions'] * 100
        
        # Plot 2D trajectories (X-Y plane)
        ax.plot(gt_positions_cm[:, 0], gt_positions_cm[:, 1], 'b-', 
                label='Ground Truth', linewidth=2)
        ax.plot(pred_positions_cm[:, 0], pred_positions_cm[:, 1], 'r--', 
                label='Prediction', linewidth=2)
        
        ax.scatter(gt_positions_cm[0, 0], gt_positions_cm[0, 1], 
                  c='green', s=100, marker='o')
        ax.scatter(gt_positions_cm[-1, 0], gt_positions_cm[-1, 1], 
                  c='black', s=100, marker='x')
        
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title(f'Sample {idx+1} - Seq {traj["seq_id"]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'test_trajectories_2d.png'), dpi=150)
    plt.close()


def plot_error_distributions(all_trans_errors, all_rot_errors, output_dir):
    """Plot error distributions"""
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convert to appropriate units
    trans_errors_cm = np.array(all_trans_errors) * 100
    rot_errors_deg = np.array(all_rot_errors)
    
    # Translation error histogram
    ax1.hist(trans_errors_cm, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Translation Error (cm)')
    ax1.set_ylabel('Count')
    ax1.set_title('Translation Error Distribution')
    ax1.axvline(np.mean(trans_errors_cm), color='red', linestyle='--', 
                label=f'Mean: {np.mean(trans_errors_cm):.2f}cm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotation error histogram
    ax2.hist(rot_errors_deg, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Rotation Error (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title('Rotation Error Distribution')
    ax2.axvline(np.mean(rot_errors_deg), color='red', linestyle='--', 
                label=f'Mean: {np.mean(rot_errors_deg):.2f}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'error_distributions.png'), dpi=150)
    plt.close()


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


def evaluate_sequence(model, dataloader, device, sequence_name, stride=5, sequence_length=21, debug=True):
    """Evaluate model on a single sequence with debugging"""
    all_predictions = []
    all_ground_truth = []
    per_frame_trans_errors = []
    per_frame_rot_errors = []
    
    # For stride=1, we only use the last prediction from each window
    use_last_only = (stride == 1)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {sequence_name}")):
            # Move data to device
            images = batch['images'].to(device)
            poses_gt = batch['poses'].to(device)
            
            # AriaDatasetMMapShared format
            imu_padded = batch['imu_padded'].to(device)
            imu_lengths = batch['imu_lengths'].to(device)
            # Convert to list format expected by model
            imu_sequences = []
            for b in range(imu_padded.shape[0]):
                seq_list = []
                for t in range(imu_padded.shape[1]):
                    actual_len = imu_lengths[b, t]
                    if actual_len > 0:
                        seq_list.append(imu_padded[b, t, :actual_len])
                imu_sequences.append(seq_list)
            
            # Forward pass
            outputs = model(images, imu_sequences)
            
            # Debug: Print first batch predictions
            if debug and batch_idx == 0:
                print(f"\n[DEBUG] First batch predictions:")
                pred_sample = outputs['poses'][0, :5].cpu().numpy()
                gt_sample = poses_gt[0, :5].cpu().numpy()
                for i in range(min(5, pred_sample.shape[0])):
                    pred_trans = pred_sample[i, :3]
                    pred_rot = pred_sample[i, 3:]
                    gt_trans = gt_sample[i, :3]
                    gt_rot = gt_sample[i, 3:]
                    
                    # Compute per-frame errors
                    trans_error = np.linalg.norm(pred_trans - gt_trans)
                    # Quaternion geodesic distance
                    dot_product = np.clip(np.abs(np.dot(pred_rot, gt_rot)), 0, 1)
                    rot_error = 2 * np.arccos(dot_product) * 180 / np.pi
                    
                    print(f"  Frame {i}: trans_err={trans_error*100:.2f}cm, rot_err={rot_error:.2f}°")
                    print(f"    Pred: trans={pred_trans}, rot={pred_rot}")
                    print(f"    GT:   trans={gt_trans}, rot={gt_rot}")
            
            # Compute per-frame errors for all predictions
            pred_poses = outputs['poses'].cpu().numpy()
            gt_poses = poses_gt.cpu().numpy()
            
            for b in range(pred_poses.shape[0]):
                seq_len = pred_poses.shape[1] if not use_last_only else 1
                start_idx = -1 if use_last_only else 0
                
                for t in range(start_idx, pred_poses.shape[1] if not use_last_only else 0):
                    pred_trans = pred_poses[b, t, :3]
                    pred_rot = pred_poses[b, t, 3:]
                    gt_trans = gt_poses[b, t, :3]
                    gt_rot = gt_poses[b, t, 3:]
                    
                    # Translation error
                    trans_error = np.linalg.norm(pred_trans - gt_trans)
                    per_frame_trans_errors.append(trans_error)
                    
                    # Rotation error (geodesic)
                    pred_rot = pred_rot / np.linalg.norm(pred_rot)
                    gt_rot = gt_rot / np.linalg.norm(gt_rot)
                    dot_product = np.clip(np.abs(np.dot(pred_rot, gt_rot)), 0, 1)
                    rot_error = 2 * np.arccos(dot_product) * 180 / np.pi
                    per_frame_rot_errors.append(rot_error)
                
                # Collect predictions based on stride
                if use_last_only:
                    all_predictions.append(pred_poses[b, -1:])
                    all_ground_truth.append(gt_poses[b, -1:])
                else:
                    all_predictions.append(pred_poses[b])
                    all_ground_truth.append(gt_poses[b])
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    # Print per-frame error statistics
    if debug and len(per_frame_trans_errors) > 0:
        print(f"\n[DEBUG] Per-frame error statistics:")
        print(f"  Translation: mean={np.mean(per_frame_trans_errors)*100:.2f}cm, "
              f"std={np.std(per_frame_trans_errors)*100:.2f}cm, "
              f"max={np.max(per_frame_trans_errors)*100:.2f}cm")
        print(f"  Rotation: mean={np.mean(per_frame_rot_errors):.2f}°, "
              f"std={np.std(per_frame_rot_errors):.2f}°, "
              f"max={np.max(per_frame_rot_errors):.2f}°")
    
    return all_predictions, all_ground_truth, per_frame_trans_errors, per_frame_rot_errors


def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation matching evaluation_from_scratch')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='../aria_processed',
                        help='Directory with processed Aria data')
    parser.add_argument('--output-dir', type=str, default='evaluation_comprehensive',
                        help='Output directory for results')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--stride', type=int, default=5,
                        help='Stride for evaluation')
    parser.add_argument('--sequence-length', type=int, default=21,
                        help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug prints')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Comprehensive Evaluation (matching evaluation_from_scratch)")
    print("="*60)
    
    # Load model
    model, config = load_checkpoint(args.checkpoint, device)
    
    # Test sequences
    test_sequences = ['016', '017', '018', '019']
    print(f"\n📁 Test sequences: {test_sequences}")
    
    all_trans_errors = []
    all_rot_errors = []
    all_scale_errors = []
    all_trajectories = []
    
    # Process each test sequence
    for seq_id in test_sequences:
        print(f"\n{'='*50}")
        print(f"Processing sequence {seq_id}")
        print(f"{'='*50}")
        
        # Create dataset for this sequence using AriaDatasetMMapShared
        print(f"Using AriaDatasetMMapShared (same as training)")
        dataset = AriaDatasetMMapShared(
            data_dir=args.data_dir,
            split=args.split,
            sequence_length=args.sequence_length,
            stride=args.stride,
            image_size=(704, 704),
            cache_dir=f"/tmp/aria_mmap_eval_{seq_id}",
            rank=0,
            world_size=1
        )
        # Filter samples for this sequence
        dataset.samples = [s for s in dataset.samples if s['seq_name'] == seq_id]
        collate_fn = collate_mmap_shared
        
        if len(dataset) == 0:
            print(f"Warning: No data found for sequence {seq_id}")
            continue
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn
        )
        
        print(f"Loaded {len(dataset)} samples for sequence {seq_id}")
        
        # Evaluate sequence
        pred_poses, gt_poses, per_frame_trans_errs, per_frame_rot_errs = evaluate_sequence(
            model, dataloader, device, seq_id, 
            stride=args.stride, sequence_length=args.sequence_length, debug=args.debug
        )
        
        # Integrate trajectory
        pred_positions, pred_rotations = integrate_trajectory(pred_poses)
        gt_positions, gt_rotations = integrate_trajectory(gt_poses)
        
        # Debug: Show how errors accumulate
        if args.debug:
            print(f"\n[DEBUG] Trajectory integration analysis:")
            print(f"  Number of relative poses: {len(pred_poses)}")
            print(f"  Number of integrated positions: {len(pred_positions)}")
            
            # Show error growth at different points
            checkpoints = [10, 20, 50, 100, len(pred_positions)-1]
            print(f"\n  Error growth over trajectory:")
            for idx in checkpoints:
                if idx < len(pred_positions):
                    pos_error = np.linalg.norm(pred_positions[idx] - gt_positions[idx])
                    print(f"    After {idx} frames: {pos_error*100:.2f}cm")
            
            # Per-frame statistics
            if len(per_frame_trans_errs) > 0:
                print(f"\n  Per-frame relative pose errors (before integration):")
                print(f"    Translation: mean={np.mean(per_frame_trans_errs)*100:.2f}cm")
                print(f"    Rotation: mean={np.mean(per_frame_rot_errs):.2f}°")
        
        # Store for later plotting
        all_trajectories.append({
            'seq_id': seq_id,
            'pred_positions': pred_positions,
            'gt_positions': gt_positions,
            'pred_rotations': pred_rotations,
            'gt_rotations': gt_rotations
        })
        
        # Save trajectory CSVs
        save_trajectory_csv(pred_positions, gt_positions, pred_rotations, gt_rotations, 
                           seq_id, output_dir)
        
        # Generate full trajectory plot
        plot_trajectory_3d(pred_positions, gt_positions, seq_id,
                          os.path.join(output_dir, f'trajectory_3d_{seq_id}.png'))
        
        # Generate aligned trajectory plot
        if len(pred_positions) > 1:
            pred_aligned, _, _, _ = align_trajectory(
                pred_positions[1:], gt_positions[1:len(pred_positions)], with_scale=True
            )
            pred_aligned_full = np.vstack([gt_positions[0], pred_aligned])
            plot_trajectory_3d(pred_aligned_full, gt_positions[:len(pred_aligned_full)], seq_id,
                              os.path.join(output_dir, f'trajectory_3d_{seq_id}_aligned.png'),
                              title_suffix=' (Aligned)')
        
        # Generate rotation plot
        plot_rotation_3d(pred_rotations[1:], gt_rotations[1:], seq_id,
                        os.path.join(output_dir, f'rotation_3d_{seq_id}.png'))
        
        # Generate 1s and 5s plots
        fps = 20  # Aria dataset is 20Hz
        for duration in [1, 5]:
            frames = min(duration * fps, len(pred_poses))
            if frames > 0:
                # Trajectory plots
                pred_pos_window, pred_rot_window = integrate_trajectory(pred_poses[:frames])
                gt_pos_window = gt_positions[:frames + 1]
                gt_rot_window = gt_rotations[:frames + 1]
                
                plot_trajectory_3d(pred_pos_window, gt_pos_window, seq_id,
                                  os.path.join(output_dir, f'trajectory_3d_{seq_id}_{duration}s.png'),
                                  time_window=f'{duration}s')
                
                # Aligned version
                if len(pred_pos_window) > 1:
                    pred_aligned, _, _, _ = align_trajectory(
                        pred_pos_window[1:], gt_pos_window[1:len(pred_pos_window)], with_scale=True
                    )
                    pred_aligned_full = np.vstack([gt_pos_window[0], pred_aligned])
                    plot_trajectory_3d(pred_aligned_full, gt_pos_window[:len(pred_aligned_full)], seq_id,
                                      os.path.join(output_dir, f'trajectory_3d_{seq_id}_{duration}s_aligned.png'),
                                      time_window=f'{duration}s', title_suffix=' (Aligned)')
                
                # Rotation plot
                plot_rotation_3d(pred_rot_window[1:], gt_rot_window[1:], seq_id,
                                os.path.join(output_dir, f'rotation_3d_{seq_id}_{duration}s.png'),
                                time_window=f'{duration}s')
        
        # Generate interactive HTML plot
        if PLOTLY_AVAILABLE:
            create_interactive_html_plot(pred_positions, gt_positions, seq_id,
                                       os.path.join(output_dir, f'trajectory_3d_{seq_id}_interactive.html'))
        
        # Compute errors
        trans_errors = np.linalg.norm(pred_positions[1:] - gt_positions[1:len(pred_positions)], axis=1)
        all_trans_errors.extend(trans_errors)
        
        rot_errors = []
        for i in range(1, min(len(pred_rotations), len(gt_rotations))):
            gt_rot = R.from_quat(gt_rotations[i])
            pred_rot = R.from_quat(pred_rotations[i])
            rel_rot = gt_rot.inv() * pred_rot
            angle_error = np.abs(rel_rot.magnitude() * 180 / np.pi)
            rot_errors.append(angle_error)
        all_rot_errors.extend(rot_errors)
        
        # Scale error
        pred_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1))
        gt_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1))
        scale_error = abs(pred_length - gt_length) / gt_length * 100
        all_scale_errors.append(scale_error)
        
        # Report both integrated and per-frame errors
        print(f"\nSequence {seq_id} Results:")
        print(f"  INTEGRATED trajectory errors (after {len(pred_positions)} frames):")
        print(f"    - Trans Error: {np.mean(trans_errors)*100:.2f}cm")
        print(f"    - Rot Error: {np.mean(rot_errors):.2f}°") 
        print(f"    - Scale Error: {scale_error:.2f}%")
        if len(per_frame_trans_errs) > 0:
            print(f"  PER-FRAME relative pose errors:")
            print(f"    - Trans Error: {np.mean(per_frame_trans_errs)*100:.2f}cm")
            print(f"    - Rot Error: {np.mean(per_frame_rot_errs):.2f}°")
    
    # Generate summary plots
    if len(all_trajectories) > 0:
        plot_sample_trajectories(all_trajectories, output_dir)
        plot_error_distributions(all_trans_errors, all_rot_errors, output_dir)
    
    # Save evaluation metrics
    if len(all_trans_errors) > 0:
        save_evaluation_metrics(all_trans_errors, all_rot_errors, all_scale_errors, output_dir)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📂 Results saved to: {output_dir}")
    print(f"\n📊 Generated outputs:")
    for seq_id in test_sequences:
        print(f"   Sequence {seq_id}:")
        print(f"   - trajectory_{seq_id}_gt.csv (ground truth)")
        print(f"   - trajectory_{seq_id}_pred.csv (predictions with errors)")
        print(f"   - trajectory_3d_{seq_id}.png, trajectory_3d_{seq_id}_1s.png, trajectory_3d_{seq_id}_5s.png")
        print(f"   - trajectory_3d_{seq_id}_aligned.png, trajectory_3d_{seq_id}_1s_aligned.png, trajectory_3d_{seq_id}_5s_aligned.png")
        print(f"   - rotation_3d_{seq_id}.png, rotation_3d_{seq_id}_1s.png, rotation_3d_{seq_id}_5s.png")
        if PLOTLY_AVAILABLE:
            print(f"   - trajectory_3d_{seq_id}_interactive.html (interactive 3D plot)")


if __name__ == "__main__":
    main()