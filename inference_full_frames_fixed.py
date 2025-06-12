#!/usr/bin/env python3
"""
Fixed inference script using the correct VIFTTransition model from training
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the SAME model used in training
from train_vift_aria import VIFTTransition
from generate_all_pretrained_latents_fixed import process_sequence, load_pretrained_model


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model configuration
    config = checkpoint.get('config', {})
    
    # Initialize model with SAME architecture as training
    model = VIFTTransition(
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
            
            # Prepare input as DICT - same format as training!
            batch = {
                'visual_features': v_feat.unsqueeze(0).float().to(device),
                'imu_features': i_feat.unsqueeze(0).float().to(device),
                'poses': gt_poses.unsqueeze(0).float().to(device)  # Not used during inference
            }
            
            # Run model with correct input format
            output = model(batch)
            
            # Extract predictions from the output dict
            pred_poses = output['poses'].squeeze(0).cpu().numpy()  # [seq_len-1, 7]
            pred_trans = pred_poses[:, :3]
            pred_rot = pred_poses[:, 3:]
            
            # Get ground truth (already in cm from dataset)
            # Skip the first frame to align with model predictions
            gt_trans = gt_poses[1:, :3].numpy()
            gt_rot = gt_poses[1:, 3:].numpy()
            
            # Debug: Print predictions for first few windows
            if i < 3:
                print(f"\n{'='*80}")
                print(f"Window {i}/{num_samples-1}")
                print(f"{'='*80}")
                print(f"Prediction shape: {pred_poses.shape}")
                print(f"Ground truth shape: {gt_poses[1:].shape}")
                
                print(f"\nFirst 5 frame predictions:")
                for j in range(min(5, len(pred_trans))):
                    print(f"  Frame {j}: Pred=[{pred_trans[j,0]:.3f}, {pred_trans[j,1]:.3f}, {pred_trans[j,2]:.3f}] cm, "
                          f"GT=[{gt_trans[j,0]:.3f}, {gt_trans[j,1]:.3f}, {gt_trans[j,2]:.3f}] cm")
                
                # Check embedding statistics from model output
                if 'embeddings' in output:
                    embeddings = output['embeddings'].squeeze(0).cpu().numpy()
                    embed_std = embeddings.std(axis=0).mean()
                    print(f"\nEmbedding temporal std: {embed_std:.4f}")
                
                if 'transitions' in output:
                    transitions = output['transitions'].squeeze(0).cpu().numpy()
                    trans_norms = np.linalg.norm(transitions, axis=-1)
                    print(f"Transition norms: mean={trans_norms.mean():.4f}, std={trans_norms.std():.4f}")
            
            all_predictions.append(pred_poses)
            all_ground_truth.append(np.concatenate([gt_trans, gt_rot], axis=-1))
    
    return all_predictions, all_ground_truth


def integrate_poses(relative_poses):
    """Integrate relative poses to get absolute trajectory"""
    trajectory = []
    current_pos = np.zeros(3)
    current_rot = np.array([0, 0, 0, 1])  # Identity quaternion
    
    trajectory.append(current_pos.copy())
    
    for rel_pose in relative_poses:
        # Update position
        current_pos += rel_pose[:3]
        trajectory.append(current_pos.copy())
        
        # For now, ignore rotation integration
    
    return np.array(trajectory)


def plot_trajectory_3d(pred_trajectory, gt_trajectory, sequence_name, output_path):
    """Create 3D trajectory plot"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
            'b-', linewidth=2, label='Ground Truth')
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_trajectory[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_trajectory[-1], color='red', s=100, marker='x', label='End')
    
    # Calculate path lengths
    gt_length = np.sum(np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1))
    pred_length = np.sum(np.linalg.norm(np.diff(pred_trajectory, axis=0), axis=1))
    
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'Sequence {sequence_name} - Full Frame Model (Fixed)\nGT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved 3D trajectory plot to {output_path}")


def plot_relative_poses(pred_poses, gt_poses, output_path, title="Relative Pose Analysis"):
    """Plot relative poses over time"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    frames = np.arange(len(pred_poses))
    
    # Translation components
    for i, (ax, label) in enumerate(zip(axes[0], ['X (cm)', 'Y (cm)'])):
        ax.plot(frames, gt_poses[:, i], 'b-', linewidth=2, label='Ground Truth')
        ax.plot(frames, pred_poses[:, i], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('Frame')
        ax.set_ylabel(label)
        ax.set_title(f'Relative Translation - {label[0]} Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Z component
    ax = axes[1, 0]
    ax.plot(frames, gt_poses[:, 2], 'b-', linewidth=2, label='Ground Truth')
    ax.plot(frames, pred_poses[:, 2], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Z (cm)')
    ax.set_title('Relative Translation - Z Component')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
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
    stats_text += f"  Std:  [{gt_std[0]:.3f}, {gt_std[1]:.3f}, {gt_std[2]:.3f}]\n"
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace')
    
    plt.suptitle(f'{title}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved relative poses plot to {output_path}")


def plot_rotation_3d(pred_rotations, gt_rotations, sequence_name, output_path):
    """Create 3D rotation trajectory visualization in axis-angle space"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert quaternions to axis-angle representation
    # This maps rotations to 3D space where direction = rotation axis, magnitude = rotation angle
    gt_axis_angles = np.array([R.from_quat(q).as_rotvec() for q in gt_rotations])
    pred_axis_angles = np.array([R.from_quat(q).as_rotvec() for q in pred_rotations])
    
    # Downsample for clarity if sequence is very long
    stride = max(1, len(gt_axis_angles) // 1000)
    
    # Plot rotation trajectories
    ax.plot(gt_axis_angles[::stride, 0], gt_axis_angles[::stride, 1], gt_axis_angles[::stride, 2], 
            'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(pred_axis_angles[::stride, 0], pred_axis_angles[::stride, 1], pred_axis_angles[::stride, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end points
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
    
    # Set labels and title
    ax.set_xlabel('X (rad)')
    ax.set_ylabel('Y (rad)')
    ax.set_zlabel('Z (rad)')
    ax.set_title(f'Sequence {sequence_name} - Rotation Trajectory\n'
                 f'Mean Error: {mean_error:.2f}°, Max Error: {max_error:.2f}°')
    ax.legend()
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([
        gt_axis_angles[:, 0].max() - gt_axis_angles[:, 0].min(),
        gt_axis_angles[:, 1].max() - gt_axis_angles[:, 1].min(),
        gt_axis_angles[:, 2].max() - gt_axis_angles[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (gt_axis_angles[:, 0].max() + gt_axis_angles[:, 0].min()) * 0.5
    mid_y = (gt_axis_angles[:, 1].max() + gt_axis_angles[:, 1].min()) * 0.5
    mid_z = (gt_axis_angles[:, 2].max() + gt_axis_angles[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved 3D rotation plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--processed-dir', type=str, 
                       default='aria_processed_full_frames',
                       help='Directory with processed sequences')
    parser.add_argument('--output-dir', type=str, 
                       default='full_frames_results_fixed',
                       help='Output directory for results')
    parser.add_argument('--window-size', type=int, default=11,
                       help='Window size for features')
    parser.add_argument('--stride', type=int, default=10,
                       help='Stride for sliding window')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint)
    model = model.to(args.device)
    
    # Load pretrained VIFT model for feature extraction
    pretrained_path = "pretrained_models/vf_512_if_256_3e-05.model"
    vift_model = load_pretrained_model(pretrained_path)
    vift_model = vift_model.to(args.device)
    vift_model.eval()
    
    # Load test sequences from file
    test_sequences_file = Path(args.processed_dir).parent / 'aria_latent_full_frames' / 'test_sequences.txt'
    if test_sequences_file.exists():
        with open(test_sequences_file, 'r') as f:
            test_sequences = [line.strip() for line in f if line.strip()]
        print(f"Test sequences from {test_sequences_file}: {test_sequences}")
    else:
        # Fallback to default test sequences
        test_sequences = ['016', '017', '018', '019']
        print(f"Using default test sequences: {test_sequences}")
    
    # Process test sequences
    processed_dir = Path(args.processed_dir)
    
    for seq_id in test_sequences:
        seq_path = processed_dir / seq_id
        if not seq_path.exists() or not seq_path.is_dir():
            print(f"Sequence path {seq_path} not found, skipping")
            continue
            
        seq_name = seq_path.name
        print(f"\nProcessing sequence: {seq_name}")
        
        # Run inference
        results = run_inference_on_sequence(
            model, vift_model, seq_path, 
            window_size=args.window_size,
            stride=args.stride,
            device=args.device
        )
        
        if results is None:
            continue
            
        predictions, ground_truth = results
        
        # Concatenate all predictions
        all_pred = np.concatenate(predictions, axis=0)
        all_gt = np.concatenate(ground_truth, axis=0)
        
        # Integrate to get absolute trajectories
        pred_trajectory = integrate_poses(all_pred)
        gt_trajectory = integrate_poses(all_gt)
        
        # Save results
        np.savez(output_dir / f'results_{seq_name}.npz',
                predictions=all_pred,
                ground_truth=all_gt,
                pred_trajectory=pred_trajectory,
                gt_trajectory=gt_trajectory)
        
        # Plot trajectory
        plot_trajectory_3d(
            pred_trajectory, 
            gt_trajectory,
            seq_name,
            output_dir / f'trajectory_3d_{seq_name}.png'
        )
        
        # Plot relative poses
        plot_relative_poses(
            all_pred,
            all_gt,
            output_dir / f'relative_poses_{seq_name}.png',
            title=f'Sequence {seq_name} - Relative Poses'
        )
        
        # Plot rotation analysis
        # Extract rotations (quaternions) from the full poses
        pred_rotations = all_pred[:, 3:7]  # Quaternions are in columns 3-6
        gt_rotations = all_gt[:, 3:7]
        
        plot_rotation_3d(
            pred_rotations,
            gt_rotations,
            seq_name,
            output_dir / f'rotation_3d_{seq_name}.png'
        )
        
        print(f"Results saved for {seq_name}")


if __name__ == "__main__":
    main()