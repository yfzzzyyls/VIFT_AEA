#!/usr/bin/env python3
"""
Run inference with full frame model and generate 3D trajectory plots
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import model components
from src.models.components.pose_transformer import PoseTransformer
from generate_all_pretrained_latents_fixed import process_sequence, load_pretrained_model


class VIFT(nn.Module):
    """VIFT model with 7DoF output (original VIFT architecture with quaternion output)"""
    
    def __init__(self, input_dim=768, embedding_dim=128, num_layers=2, 
                 nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # Use the existing PoseTransformer
        self.pose_transformer = PoseTransformer(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output embeddings (not poses directly)
        self.pose_transformer.fc2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        
        # Transition to pose projection (embeddings -> 7DoF poses)
        self.transition_to_pose = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 7)  # 3 translation + 4 quaternion
        )
        
    def forward(self, x, gt=None):
        # x shape: [batch_size, seq_len, input_dim]
        batch = (x, None, None)
        
        # Get embeddings from transformer
        embeddings = self.pose_transformer(batch, gt)
        
        # Compute transitions as differences between consecutive embeddings
        # This is the key part of VIFT architecture
        transitions = embeddings[:, 1:] - embeddings[:, :-1]
        
        # Project transitions to pose space
        pose_predictions = self.transition_to_pose(transitions)
        
        # Split and normalize quaternions
        translation = pose_predictions[..., :3]
        quaternion = pose_predictions[..., 3:7]
        quaternion = torch.nn.functional.normalize(quaternion, p=2, dim=-1)
        
        return torch.cat([translation, quaternion], dim=-1)


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration
    config = checkpoint.get('config', {})
    
    # Initialize model with VIFT architecture (transition-based)
    model = VIFT(
        input_dim=config.get('input_dim', 768),
        embedding_dim=config.get('embedding_dim', 128),
        num_layers=config.get('num_layers', 2),
        nhead=config.get('nhead', 8),
        dim_feedforward=config.get('dim_feedforward', 512),
        dropout=0.0  # No dropout for inference
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def run_inference_on_sequence(model, vift_model, sequence_path, window_size=11, stride=10, device='cuda'):
    """Run inference on a full sequence"""
    
    # Generate latent features for the sequence
    print(f"Generating latent features for {sequence_path}")
    features_list, poses_list = process_sequence(
        sequence_path, 
        vift_model,
        device,
        window_size=window_size,
        stride=stride,
        pose_scale=100.0
    )
    
    if not features_list:
        print(f"Failed to generate latent features for {sequence_path}")
        return None
    
    # Prepare for inference
    all_predictions = []
    all_ground_truth = []
    
    num_samples = len(features_list)
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Running inference"):
            # Extract features for this window
            v_feat, i_feat = features_list[i]
            gt_poses = poses_list[i]
            
            # Prepare input - VIFTQuaternion expects concatenated features
            # Concatenate visual and IMU features
            combined_features = torch.cat([v_feat, i_feat], dim=-1)
            input_batch = combined_features.unsqueeze(0).float().to(device)
            
            # Run model (pass None for gt during inference)
            output = model(input_batch, gt=None)
            
            # Get predictions (already in cm)
            # Model outputs seq_len-1 predictions due to transition computation
            pred_poses = output.squeeze(0).cpu().numpy()
            pred_trans = pred_poses[:, :3]
            pred_rot = pred_poses[:, 3:]
            
            # Get ground truth (already in cm from dataset)
            # Skip the first frame (which is always [0,0,0] and identity quaternion)
            # to align with model predictions
            gt_trans = gt_poses[1:, :3].numpy()  # Skip first frame
            gt_rot = gt_poses[1:, 3:].numpy()
            
            # Debug: Print detailed predictions for all windows
            print(f"\n{'='*80}")
            print(f"Window {i}/{num_samples-1}")
            print(f"{'='*80}")
            print(f"Overall statistics:")
            print(f"  Pred trans mean: {pred_trans.mean(axis=0)}")
            print(f"  Pred trans std: {pred_trans.std(axis=0)}")
            print(f"  Pred trans range: [{pred_trans.min():.3f}, {pred_trans.max():.3f}]")
            print(f"  GT trans mean: {gt_trans.mean(axis=0)}")
            print(f"  GT trans std: {gt_trans.std(axis=0)}")
            print(f"  GT trans range: [{gt_trans.min():.3f}, {gt_trans.max():.3f}]")
            
            print(f"\nFrame-by-frame comparison:")
            print(f"{'Frame':<6} {'Pred Trans (cm)':<30} {'GT Trans (cm)':<30} {'Error (cm)':<15} {'Pred Quat':<30} {'GT Quat':<30}")
            print(f"{'-'*6} {'-'*30} {'-'*30} {'-'*15} {'-'*30} {'-'*30}")
            
            for j in range(len(pred_trans)):
                error = np.linalg.norm(pred_trans[j] - gt_trans[j])
                pred_trans_str = f"[{pred_trans[j][0]:7.3f}, {pred_trans[j][1]:7.3f}, {pred_trans[j][2]:7.3f}]"
                gt_trans_str = f"[{gt_trans[j][0]:7.3f}, {gt_trans[j][1]:7.3f}, {gt_trans[j][2]:7.3f}]"
                pred_quat_str = f"[{pred_rot[j][0]:6.3f}, {pred_rot[j][1]:6.3f}, {pred_rot[j][2]:6.3f}, {pred_rot[j][3]:6.3f}]"
                gt_quat_str = f"[{gt_rot[j][0]:6.3f}, {gt_rot[j][1]:6.3f}, {gt_rot[j][2]:6.3f}, {gt_rot[j][3]:6.3f}]"
                print(f"{j:<6} {pred_trans_str:<30} {gt_trans_str:<30} {error:<15.3f} {pred_quat_str:<30} {gt_quat_str:<30}")
            
            # Calculate average error for this window
            window_errors = [np.linalg.norm(pred_trans[j] - gt_trans[j]) for j in range(len(pred_trans))]
            print(f"\nWindow average translation error: {np.mean(window_errors):.3f} cm")
            
            # Limit detailed output to first 5 windows to avoid too much output
            if i >= 5:
                # For remaining windows, just print summary
                if i % 50 == 0:  # Print every 50th window
                    print(f"\nWindow {i}: avg trans error = {np.mean(window_errors):.3f} cm")
            
            # Store predictions and ground truth
            pred_poses = np.concatenate([pred_trans, pred_rot], axis=-1)
            all_predictions.append(pred_poses)
            # Store aligned ground truth (already has first frame skipped)
            gt_poses_aligned = np.concatenate([gt_trans, gt_rot], axis=-1)
            all_ground_truth.append(gt_poses_aligned)
    
    return {
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'metadata': {
            'sequence_name': Path(sequence_path).name,
            'num_windows': len(all_predictions),
            'window_size': window_size,
            'stride': stride
        }
    }


def reconstruct_trajectory(relative_poses, start_pose=None):
    """Reconstruct absolute trajectory from relative poses"""
    if start_pose is None:
        # Start at origin with identity rotation
        start_pose = np.array([0, 0, 0, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]
    
    trajectory = [start_pose[:3].copy()]
    current_pos = start_pose[:3].copy()
    current_rot = start_pose[3:].copy()
    
    for rel_pose in relative_poses:
        # Update position (relative translation is already in world frame for this dataset)
        current_pos = current_pos + rel_pose[:3]
        trajectory.append(current_pos.copy())
        
        # Update rotation (simplified - not using quaternion multiplication for visualization)
        current_rot = rel_pose[3:]
    
    return np.array(trajectory)


def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in radians"""
    # q = [x, y, z, w]
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, 
                     np.copysign(np.pi / 2, sinp),  # use 90 degrees if out of range
                     np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.stack([roll, pitch, yaw], axis=-1)


def plot_relative_poses(pred_poses, gt_poses, output_path, title="Relative Pose Analysis"):
    """Plot relative poses over time to show prediction patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    frames = np.arange(len(pred_poses))
    
    # Translation components
    for i, (ax, label) in enumerate(zip(axes[0], ['X (cm)', 'Y (cm)'])):
        ax.plot(frames, gt_poses[:, i], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(frames, pred_poses[:, i], 'r--', linewidth=2, label='Prediction', alpha=0.8)
        ax.set_xlabel('Frame')
        ax.set_ylabel(label)
        ax.set_title(f'Relative Translation - {label[0]} Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add mean lines
        ax.axhline(y=gt_poses[:, i].mean(), color='b', linestyle=':', alpha=0.5)
        ax.axhline(y=pred_poses[:, i].mean(), color='r', linestyle=':', alpha=0.5)
    
    # Z component
    ax = axes[1, 0]
    ax.plot(frames, gt_poses[:, 2], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(frames, pred_poses[:, 2], 'r--', linewidth=2, label='Prediction', alpha=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Z (cm)')
    ax.set_title('Relative Translation - Z Component')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate statistics
    pred_mean = pred_poses[:, :3].mean(axis=0)
    pred_std = pred_poses[:, :3].std(axis=0)
    gt_mean = gt_poses[:, :3].mean(axis=0)
    gt_std = gt_poses[:, :3].std(axis=0)
    
    stats_text = f"Statistics:\n\n"
    stats_text += f"Predictions:\n"
    stats_text += f"  Mean: [{pred_mean[0]:.3f}, {pred_mean[1]:.3f}, {pred_mean[2]:.3f}]\n"
    stats_text += f"  Std:  [{pred_std[0]:.3f}, {pred_std[1]:.3f}, {pred_std[2]:.3f}]\n\n"
    stats_text += f"Ground Truth:\n"
    stats_text += f"  Mean: [{gt_mean[0]:.3f}, {gt_mean[1]:.3f}, {gt_mean[2]:.3f}]\n"
    stats_text += f"  Std:  [{gt_std[0]:.3f}, {gt_std[1]:.3f}, {gt_std[2]:.3f}]\n\n"
    stats_text += f"Note: Low std in predictions indicates\nthe model outputs nearly constant values"
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace')
    
    plt.suptitle(f'{title}\nRelative Poses Over Time', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved relative pose plot to {output_path}")


def plot_3d_trajectory_short(pred_trajectory, gt_trajectory, output_path, title="First 5 Seconds", 
                             frames_per_second=20, duration_seconds=5):
    """Create 3D trajectory plot for first N seconds only"""
    # Calculate number of frames for desired duration
    num_frames = min(frames_per_second * duration_seconds, len(pred_trajectory))
    
    # Slice trajectories
    pred_short = pred_trajectory[:num_frames]
    gt_short = gt_trajectory[:num_frames]
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(gt_short[:, 0], gt_short[:, 1], gt_short[:, 2], 
            'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(pred_short[:, 0], pred_short[:, 1], pred_short[:, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark every second with dots
    for i in range(0, num_frames, frames_per_second):
        ax.scatter(gt_short[i, 0], gt_short[i, 1], gt_short[i, 2], 
                  color='blue', s=50, alpha=0.5)
        ax.scatter(pred_short[i, 0], pred_short[i, 1], pred_short[i, 2], 
                  color='red', s=50, alpha=0.5)
        # Add time labels
        ax.text(gt_short[i, 0], gt_short[i, 1], gt_short[i, 2], f'{i//frames_per_second}s', 
                fontsize=8, color='blue', alpha=0.7)
    
    # Mark start and end
    ax.scatter(*gt_short[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_short[-1], color='purple', s=100, marker='x', label='End')
    
    # Calculate trajectory length
    gt_length = np.sum(np.linalg.norm(np.diff(gt_short, axis=0), axis=1))
    pred_length = np.sum(np.linalg.norm(np.diff(pred_short, axis=0), axis=1))
    
    # Calculate final position error
    final_error = np.linalg.norm(pred_short[-1] - gt_short[-1])
    
    # Set labels and title
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'{title} - {duration_seconds} seconds ({num_frames} frames)\n'
                 f'GT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm, '
                 f'Final Error: {final_error:.1f}cm')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        gt_short[:, 0].max() - gt_short[:, 0].min(),
        gt_short[:, 1].max() - gt_short[:, 1].min(),
        gt_short[:, 2].max() - gt_short[:, 2].min()
    ]).max() / 2.0
    
    if max_range > 0:
        mid_x = (gt_short[:, 0].max() + gt_short[:, 0].min()) * 0.5
        mid_y = (gt_short[:, 1].max() + gt_short[:, 1].min()) * 0.5
        mid_z = (gt_short[:, 2].max() + gt_short[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved short trajectory plot to {output_path}")


def plot_3d_trajectory(pred_trajectory, gt_trajectory, output_path, title="Trajectory Comparison"):
    """Create 3D trajectory plot"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
            'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_trajectory[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_trajectory[-1], color='red', s=100, marker='x', label='End')
    
    # Calculate trajectory length
    gt_length = np.sum(np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1))
    pred_length = np.sum(np.linalg.norm(np.diff(pred_trajectory, axis=0), axis=0))
    
    # Set labels and title
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'{title}\nGT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        gt_trajectory[:, 0].max() - gt_trajectory[:, 0].min(),
        gt_trajectory[:, 1].max() - gt_trajectory[:, 1].min(),
        gt_trajectory[:, 2].max() - gt_trajectory[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (gt_trajectory[:, 0].max() + gt_trajectory[:, 0].min()) * 0.5
    mid_y = (gt_trajectory[:, 1].max() + gt_trajectory[:, 1].min()) * 0.5
    mid_z = (gt_trajectory[:, 2].max() + gt_trajectory[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved plot to {output_path}")


def plot_rotation_trajectory(pred_rotations, gt_rotations, output_path, title="Rotation Comparison"):
    """Create rotation trajectory plots showing Euler angles over time"""
    # Convert quaternions to Euler angles (in degrees for better readability)
    pred_euler = quaternion_to_euler(pred_rotations) * 180 / np.pi
    gt_euler = quaternion_to_euler(gt_rotations) * 180 / np.pi
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    labels = ['Roll', 'Pitch', 'Yaw']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        frames = np.arange(len(pred_euler))
        
        # Plot ground truth and predictions
        ax.plot(frames, gt_euler[:, i], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(frames, pred_euler[:, i], 'r--', linewidth=2, label='Prediction', alpha=0.8)
        
        # Calculate error
        error = np.abs(pred_euler[:, i] - gt_euler[:, i])
        mean_error = np.mean(error)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel(f'{label} (degrees)')
        ax.set_title(f'{label} Angle - Mean Error: {mean_error:.2f}°')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add zero line for reference
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.suptitle(f'{title}\nRotation Angles Over Time', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved rotation plot to {output_path}")


def plot_rotation_3d_trajectory(pred_rotations, gt_rotations, output_path, title="3D Rotation Visualization"):
    """Create 3D visualization of rotation using orientation vectors"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample every N frames to avoid clutter
    step = max(1, len(pred_rotations) // 50)
    indices = np.arange(0, len(pred_rotations), step)
    
    # Define a reference vector (pointing forward)
    ref_vector = np.array([0, 0, 1])
    
    # For each sampled frame, show the orientation
    for idx in indices:
        # Apply quaternion rotation to reference vector
        # For ground truth
        gt_q = gt_rotations[idx]
        gt_vec = rotate_vector_by_quaternion(ref_vector, gt_q)
        
        # For prediction
        pred_q = pred_rotations[idx]
        pred_vec = rotate_vector_by_quaternion(ref_vector, pred_q)
        
        # Normalize time for coloring
        t = idx / len(pred_rotations)
        
        # Plot orientation vectors at different heights to separate them
        height_offset = idx * 0.1
        
        # Ground truth in blue shades
        ax.quiver(0, 0, height_offset, gt_vec[0], gt_vec[1], gt_vec[2], 
                 color=(0, 0, 1-t*0.5), alpha=0.8, arrow_length_ratio=0.1)
        
        # Prediction in red shades
        ax.quiver(1, 0, height_offset, pred_vec[0], pred_vec[1], pred_vec[2], 
                 color=(1-t*0.5, 0, 0), alpha=0.8, arrow_length_ratio=0.1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time (frames)')
    ax.set_title(f'{title}\nBlue: Ground Truth, Red: Prediction')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Ground Truth'),
                      Patch(facecolor='red', label='Prediction')]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved 3D rotation plot to {output_path}")


def rotate_vector_by_quaternion(v, q):
    """Rotate vector v by quaternion q"""
    # q = [x, y, z, w]
    qx, qy, qz, qw = q
    
    # Convert vector to quaternion form: [vx, vy, vz, 0]
    vx, vy, vz = v
    
    # Quaternion multiplication: q * v * q^(-1)
    # For unit quaternions, q^(-1) = q* (conjugate)
    # This is a simplified version for rotation
    t = 2 * np.cross([qx, qy, qz], v)
    rotated = v + qw * t + np.cross([qx, qy, qz], t)
    
    return rotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='full_frames_checkpoints/20250611_161413/best_model_epoch_3.pt')
    parser.add_argument('--test-sequences', type=str,
                       default='/home/external/VIFT_AEA/aria_latent_full_frames/test_sequences.txt')
    parser.add_argument('--processed-dir', type=str,
                       default='/home/external/VIFT_AEA/aria_processed_full_frames')
    parser.add_argument('--output-dir', type=str,
                       default='full_frames_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--window-size', type=int, default=11)
    parser.add_argument('--stride', type=int, default=10)
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint)
    model = model.to(device)
    
    # Load pretrained VIFT encoder
    print("Loading pretrained VIFT encoder...")
    vift_model = load_pretrained_model('pretrained_models/vf_512_if_256_3e-05.model')
    vift_model = vift_model.to(device)
    
    # Load test sequences
    with open(args.test_sequences, 'r') as f:
        test_sequences = [line.strip() for line in f if line.strip()]
    
    print(f"Test sequences: {test_sequences}")
    
    # Run inference on each test sequence
    results = {}
    
    for seq_id in test_sequences:
        print(f"\nProcessing sequence {seq_id}")
        sequence_path = Path(args.processed_dir) / seq_id
        
        if not sequence_path.exists():
            print(f"Sequence path {sequence_path} not found")
            continue
        
        # Run inference
        result = run_inference_on_sequence(
            model, vift_model, sequence_path, 
            window_size=args.window_size,
            stride=args.stride,
            device=device
        )
        
        if result is None:
            continue
        
        results[seq_id] = result
        
        # Reconstruct full trajectory from predictions
        print("Reconstructing trajectories...")
        
        # Combine all window predictions (with overlap handling)
        all_pred_poses = []
        all_gt_poses = []
        
        for i, (pred, gt) in enumerate(zip(result['predictions'], result['ground_truth'])):
            if i == 0:
                # First window - use all frames
                all_pred_poses.extend(pred)
                all_gt_poses.extend(gt)
            else:
                # Subsequent windows - skip overlapping frames
                # Since stride=10 and window=10, no overlap
                all_pred_poses.extend(pred)
                all_gt_poses.extend(gt)
        
        # Convert to numpy arrays
        all_pred_poses = np.array(all_pred_poses)
        all_gt_poses = np.array(all_gt_poses)
        
        # Reconstruct absolute trajectories
        pred_trajectory = reconstruct_trajectory(all_pred_poses)
        gt_trajectory = reconstruct_trajectory(all_gt_poses)
        
        # Print trajectory reconstruction summary
        print(f"\n{'='*80}")
        print(f"TRAJECTORY RECONSTRUCTION SUMMARY for sequence {seq_id}")
        print(f"{'='*80}")
        print(f"Number of poses: {len(all_pred_poses)}")
        print(f"Total predicted trajectory length: {np.sum(np.linalg.norm(np.diff(pred_trajectory, axis=0), axis=1)):.1f} cm")
        print(f"Total ground truth trajectory length: {np.sum(np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1)):.1f} cm")
        
        # Check cumulative sum of translations
        print(f"\nCumulative translation analysis:")
        pred_trans_cumsum = np.cumsum(all_pred_poses[:, :3], axis=0)
        gt_trans_cumsum = np.cumsum(all_gt_poses[:, :3], axis=0)
        print(f"Predicted final position: {pred_trans_cumsum[-1]}")
        print(f"GT final position: {gt_trans_cumsum[-1]}")
        print(f"Final position error: {np.linalg.norm(pred_trans_cumsum[-1] - gt_trans_cumsum[-1]):.1f} cm")
        
        # Generate relative pose analysis plot
        relative_pose_plot_path = output_dir / f'relative_poses_{seq_id}.png'
        plot_relative_poses(
            all_pred_poses,
            all_gt_poses,
            relative_pose_plot_path,
            title=f"Sequence {seq_id}"
        )
        
        # Generate relative pose plot for first 5 seconds
        relative_pose_short_path = output_dir / f'relative_poses_{seq_id}_5sec.png'
        plot_relative_poses(
            all_pred_poses[:100],  # First 100 frames = 5 seconds at 20 FPS
            all_gt_poses[:100],
            relative_pose_short_path,
            title=f"Sequence {seq_id} - First 5 Seconds"
        )
        
        # Generate full trajectory plot
        plot_path = output_dir / f'trajectory_3d_{seq_id}.png'
        plot_3d_trajectory(
            pred_trajectory, 
            gt_trajectory, 
            plot_path,
            title=f"Sequence {seq_id} - Full Frame Model"
        )
        
        # Generate short (5 second) trajectory plot for detailed analysis
        short_plot_path = output_dir / f'trajectory_3d_{seq_id}_5sec.png'
        plot_3d_trajectory_short(
            pred_trajectory, 
            gt_trajectory, 
            short_plot_path,
            title=f"Sequence {seq_id} - First 5 Seconds",
            frames_per_second=20,  # Aria camera FPS
            duration_seconds=5
        )
        
        # Generate rotation plots
        rotation_plot_path = output_dir / f'rotation_euler_{seq_id}.png'
        plot_rotation_trajectory(
            all_pred_poses[:, 3:],  # Extract quaternions
            all_gt_poses[:, 3:],
            rotation_plot_path,
            title=f"Sequence {seq_id} - Rotation Angles"
        )
        
        # Generate 3D rotation visualization
        rotation_3d_plot_path = output_dir / f'rotation_3d_{seq_id}.png'
        plot_rotation_3d_trajectory(
            all_pred_poses[:, 3:],  # Extract quaternions
            all_gt_poses[:, 3:],
            rotation_3d_plot_path,
            title=f"Sequence {seq_id} - 3D Rotation Visualization"
        )
        
        # Calculate metrics
        frame_errors = np.linalg.norm(all_pred_poses[:, :3] - all_gt_poses[:, :3], axis=1)
        mean_error = np.mean(frame_errors)
        std_error = np.std(frame_errors)
        
        print(f"Sequence {seq_id} - Mean error: {mean_error:.2f}cm (±{std_error:.2f}cm)")
        
        # Save detailed results
        np.savez(
            output_dir / f'results_{seq_id}.npz',
            pred_poses=all_pred_poses,
            gt_poses=all_gt_poses,
            pred_trajectory=pred_trajectory,
            gt_trajectory=gt_trajectory,
            frame_errors=frame_errors,
            metadata=result['metadata']
        )
    
    # Generate summary
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Processed {len(results)} sequences")
    print(f"Results saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the 3D trajectory plots")
    print("2. Analyze alignment between predictions and ground truth")
    print("3. Compare with previous results to see improvements")


if __name__ == "__main__":
    main()